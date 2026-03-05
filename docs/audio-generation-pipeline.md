# Audio Generation Pipeline

Complete step-by-step breakdown of how a TTS request becomes audio, from HTTP request to PCM bytes.

## Architecture Overview

```
HTTP POST /tts
    |
    v
server.py: audio_stream()
    |
    v
tts.py: generate_stream() -> generate_stream_with_conds()
    |
    +---> [1] Conditioning: get_audio_conditionals()
    |         VoiceEncoder -> speaker embedding
    |         S3Tokenizer  -> reference speech tokens
    |         T3CondEnc    -> conditioning tensor for vLLM
    |
    +---> [2] T3 Token Generation (vLLM AsyncLLMEngine)
    |         Text -> speech tokens (25 tokens/sec of audio)
    |         Streams tokens incrementally (DELTA mode)
    |
    +---> [3] Token Buffering
    |         Accumulate chunk_size (15) tokens before vocoding
    |
    +---> [4] VocoderBatcher
    |         Queue + batch worker for concurrent request batching
    |
    +---> [5] S3Gen Vocoding
    |         flow_inference(): speech tokens -> mel-spectrogram (CFM diffusion)
    |         hift_inference(): mel-spectrogram -> waveform (HiFiGAN)
    |
    +---> [6] Post-processing
              Context cropping, fade-in, float32 -> int16 PCM
```

---

## Step 1: HTTP Request

**File:** `server.py`

### 1.1 Endpoint

The `POST /tts` endpoint (line 170) accepts:

| Parameter | Default | Description |
|---|---|---|
| `text` | required | Text to synthesize |
| `language_id` | `"en"` | Language code |
| `exaggeration` | `0.5` | Emotion intensity (0-1) |
| `temperature` | `0.5` | Sampling temperature |
| `chunk_size` | `15` | Speech tokens per streaming chunk |
| `diffusion_steps` | `5` | CFM solver steps |
| `format` | `"wav"` | Output format: `"wav"` or `"pcm"` |

### 1.2 audio_stream() Generator (line 54)

This async generator:
1. Optionally yields a 44-byte WAV RIFF header (`make_wav_header()`, line 30) with `data_size=0xFFFFFFFF` (unknown length for streaming)
2. Iterates over `model.generate_stream()`, which yields `(audio_chunk, metrics)` tuples
3. For each chunk:
   - Converts torch tensor to numpy
   - Sanitizes: `nan -> 0`, clips to `[-1.0, 1.0]`
   - Converts float32 to int16 PCM: `(audio * 32767).astype("<i2")`
   - Yields raw PCM bytes

**Output sample rate:** 24000 Hz, mono, 16-bit signed little-endian.

---

## Step 2: Text Preprocessing

**File:** `src/chatterbox_vllm/text_utils.py`

### 2.1 punc_norm() (line 2)

Before any model sees the text, `punc_norm()` normalizes punctuation:
- Capitalizes first letter
- Replaces `...` / `...` with `, `
- Replaces `:` `;` ` - ` with `, `
- Normalizes dashes and smart quotes
- Adds a period if the text doesn't end with sentence-ending punctuation

### 2.2 Token Wrapping

In `generate_stream_with_conds()` (tts.py, line 770):
```python
text = "[START]" + punc_norm(text) + "[STOP]"
# For multilingual:
text = f"<{language_id.lower()}>{text}"
```

The `[START]` and `[STOP]` tokens are special markers for the T3 model. The language tag (e.g., `<en>`) is prepended for multilingual models.

---

## Step 3: Audio Conditioning

**File:** `src/chatterbox_vllm/tts.py`, method `get_audio_conditionals()` (line 393)

This step encodes the reference voice (or default voice) into conditioning tensors that tell both the T3 token generator and S3Gen vocoder *what voice to use*.

### 3.1 If Using Default Voice

Pre-computed tensors are loaded from `conds.pt` at model init time. No work needed per-request.

### 3.2 If Voice Cloning (custom audio_prompt_path)

Three things are extracted from the reference audio:

#### a) S3Gen Reference Dict
**Method:** `s3gen.embed_ref()` (s3gen.py, line 122)

The reference waveform (up to 10 seconds) is processed into:
```python
ref_dict = {
    "prompt_token":     [1, T],       # S3-tokenized reference speech
    "prompt_token_len": [1],          # Length of above
    "prompt_feat":      [1, F, 80],   # Mel-spectrogram of reference (80 mel bins)
    "prompt_feat_len":  [1],          # Length of above
    "embedding":        [1, 80],      # Speaker identity vector (CAMPPlus)
}
```

Sub-steps inside `embed_ref()`:
1. Resample to 24 kHz -> extract mel-spectrogram (80 bins)
2. Resample to 16 kHz -> run through CAMPPlus speaker encoder -> x-vector
3. Resample to 16 kHz -> run through S3Tokenizer -> discrete speech tokens
4. Verify alignment: mel frames = 2 * token count

#### b) T3 Conditioning Prompt Tokens
**Method:** `s3gen.tokenizer.forward()` (s3tokenizer.py)

The first 6 seconds of reference audio (at 16 kHz) is tokenized into speech tokens at 25 tokens/sec. These tokens become the "conditioning prompt" that tells T3 what speaking style to generate.

#### c) Speaker Embedding
**Method:** `ve.embeds_from_wavs()` (voice_encoder.py, line 246)

A 3-layer LSTM processes mel-spectrograms of the reference audio into a 256-dim L2-normalized speaker embedding. This captures speaker identity (timbre, pitch characteristics).

### 3.3 T3 Conditioning Encoding

**File:** `src/chatterbox_vllm/models/t3/modules/cond_enc.py`

All three signals are combined by `T3CondEnc.forward()` (line 80):

1. Speaker embedding: `Linear(256, 1024)` -> `[1, 1024]`
2. Conditioning prompt speech: token embeddings + positional embeddings -> `[~32, 1024]`
3. Emotion exaggeration: `Linear(1, 1024)` -> `[1, 1024]`

These are concatenated into a single conditioning tensor `[~34, 1024]` that is passed to vLLM as `multi_modal_data`.

---

## Step 4: T3 Speech Token Generation (vLLM)

**File:** `src/chatterbox_vllm/tts.py`, method `generate_stream_with_conds()` (line 738)

### 4.1 What T3 Does

T3 is an autoregressive transformer that generates discrete speech tokens from text. It runs inside vLLM as a custom model (`T3VllmModel` in `src/chatterbox_vllm/models/t3/t3.py`).

- **Input:** Text tokens (from the tokenizer) + conditioning tensor (speaker/emotion)
- **Output:** Speech token IDs, one at a time, in the range `[0, 6561)`
- **Token rate:** ~19-20 tokens/sec on typical hardware
- **Each token represents 40ms of audio** (token rate = 25 tokens/sec of audio)

### 4.2 SPEECH_TOKEN_OFFSET

**File:** `src/chatterbox_vllm/models/t3/t3.py`, line 50

```python
SPEECH_TOKEN_OFFSET = 2500
```

vLLM sees token IDs offset by 2500. This separates the text vocabulary (IDs < 2500) from the speech vocabulary (IDs >= 2500). When tokens come back from vLLM, the offset is subtracted:

```python
speech_token = vllm_token_id - SPEECH_TOKEN_OFFSET
```

### 4.3 Sampling Parameters (line 782)

```python
SamplingParams(
    temperature=0.5,
    stop_token_ids=[6562 + 2500],  # = 9062 (stop_speech_token + offset)
    max_tokens=1000,
    top_p=1.0,
    min_p=0.05,
    repetition_penalty=2.0,
)
```

`output_kind = RequestOutputKind.DELTA` means each iteration yields only the *new* tokens since last yield, not the full sequence.

### 4.4 vLLM Prompt

```python
prompt = {
    "prompt": "[START]Hello world.[STOP]",
    "multi_modal_data": {
        "conditionals": [cond_emb],  # [34, 1024] conditioning tensor
    },
}
```

The `T3MultiModalProcessor` (t3.py) injects the conditioning tensor into the model's input embeddings during the prefill phase.

### 4.5 Generation Loop (line 805)

```python
async for output in self.async_engine.generate(prompt, sampling_params, request_id):
    for completion in output.outputs:
        token_buffer.extend(completion.token_ids)  # accumulate speech tokens
```

vLLM generates tokens asynchronously. The async generator yields outputs as tokens are produced. Each output typically contains 1 token (DELTA mode).

---

## Step 5: Token Buffering

**File:** `src/chatterbox_vllm/tts.py`, lines 818-849

### 5.1 Chunk Decision

```python
should_process = len(token_buffer) >= chunk_size or output.finished
```

Tokens accumulate in `token_buffer` until either:
- `chunk_size` tokens have been collected (default: 15 = 600ms of audio)
- The T3 model has finished generating (hit stop token or max_tokens)

### 5.2 Token Cleaning

Before vocoding, tokens are cleaned:
```python
new_speech_tokens = [t - SPEECH_TOKEN_OFFSET for t in token_buffer]  # de-offset
new_speech_tokens = drop_invalid_tokens(new_speech_tokens)            # remove >= 6561
new_speech_tokens = new_speech_tokens[new_speech_tokens < 6561]       # extra safety
```

`drop_invalid_tokens()` (s3tokenizer, line 27) filters out any tokens outside the valid speech vocabulary `[0, 6561)`.

---

## Step 6: Context Windowing

**File:** `src/chatterbox_vllm/tts.py`, method `_process_token_buffer_batched()` (line 628)

### 6.1 Why Context Is Needed

S3Gen converts speech tokens to audio independently per call. Without context, each chunk would start "cold" — the vocoder wouldn't know what came before, causing audible discontinuities at chunk boundaries.

### 6.2 How It Works

```python
context_tokens = all_tokens_so_far[-context_window:]  # last 50 tokens
tokens_to_process = torch.cat([context_tokens, new_tokens])  # 50 + 15 = 65 tokens
```

The vocoder processes 65 tokens but only the audio for the **last 15** (new tokens) is kept. The first 50 tokens' audio is discarded — it was only there to give S3Gen acoustic context.

```python
# After vocoding:
samples_per_token = len(wav) / len(clean_tokens)
skip_samples = int(context_length * samples_per_token)
audio_chunk = wav[skip_samples:]  # discard context audio, keep new audio
```

### 6.3 Fade-In Smoothing

A 20ms linear fade-in is applied to each chunk to prevent clicking at boundaries:

```python
fade_samples = int(0.02 * 24000)  # 480 samples
fade_in = np.linspace(0.0, 1.0, fade_samples)
audio_chunk[:fade_samples] *= fade_in
```

---

## Step 7: VocoderBatcher

**File:** `src/chatterbox_vllm/tts.py`, class `VocoderBatcher` (line 135)

### 7.1 Purpose

When multiple streaming requests are in-flight simultaneously, their vocoding calls would serialize on the GPU (one at a time). The VocoderBatcher collects pending vocoding requests and runs them as a single batched S3Gen call.

**Single request:** no batching benefit, but adds `max_wait_ms` (50ms) of latency waiting for other requests that never come.

**10 concurrent requests:** instead of 10 x 280ms = 2.8s serialized, one ~300ms batched call.

### 7.2 Flow

```
vocode() called by streaming coroutine
    |
    v
Enqueue: (speech_tokens, ref_dict, n_timesteps, future, submit_time)
    |                                                  ^-- for timing
    v
_batch_worker() (background asyncio task)
    |
    +-- Wait for first item (blocks until something arrives)
    +-- Collect more items for up to max_wait_ms (50ms)
    +-- Group items by ref_dict identity (same speaker)
    +-- For each group:
    |     s3gen.batch_inference(tokens_list, ref_dict, n_timesteps)
    |     -> (results, timing)
    |     Set each future's result: (audio_tensor, item_timing)
    |
    v
vocode() resumes, returns (audio_tensor, timing_dict)
```

### 7.3 Grouping

Items are grouped by `id(ref_dict)` — Python object identity. Requests using the same pre-computed voice reference share a batch. Different voices are processed in separate batches (different speaker embeddings can't be mixed in a single forward pass).

---

## Step 8: S3Gen Vocoding

**File:** `src/chatterbox_vllm/models/s3gen/s3gen.py`

S3Gen converts discrete speech tokens into audio waveforms in two stages:

```
Speech Tokens [B, T]
       |
       v
  flow_inference()          <-- ~461ms (the bottleneck)
  (Conditional Flow Matching)
       |
       v
  Mel-Spectrogram [B, 80, mel_frames]
       |
       v
  hift_inference()          <-- ~30ms
  (HiFiGAN vocoder)
       |
       v
  Waveform [B, num_samples]  (float32, 24 kHz)
```

### 8.1 flow_inference() — Tokens to Mel-Spectrogram

**File:** `s3gen.py` line 192, calls into `flow.py`

This is the most expensive step (~461ms per chunk, ~94% of vocode time).

#### What Happens Inside

1. **Token embedding** (flow.py, line 265): Speech tokens are embedded and positionally encoded
2. **Encoder** (flow.py, line 268): A conformer encoder processes the token sequence into hidden states, which are projected to mel-spectrogram dimensions (80 bins)
3. **CFM diffusion** (flow_matching.py, line 47): A Conditional Flow Matching solver iteratively denoises random noise into a mel-spectrogram

#### CFM Solver Detail (flow_matching.py)

```
For each of n_timesteps (default 5):
    1. Prepare doubled batch: [conditioned; unconditioned] (for CFG)
    2. Run estimator network (UNet-like) to predict velocity field
    3. Apply Classifier-Free Guidance:
       dphi = 1.7 * conditioned_velocity - 0.7 * unconditioned_velocity
    4. Euler step: x = x + dt * dphi
```

- **n_timesteps=5:** 5 Euler steps, fast but lower quality
- **n_timesteps=10:** 10 steps, better quality, ~2x slower
- **CFG rate:** 0.7 (inference_cfg_rate in flow_matching.py, line 25)
- **Scheduler:** Cosine time steps (not linear)

Each step requires a full forward pass of the estimator network, and the batch is doubled for CFG (conditioned + unconditioned). So `n_timesteps=5` actually runs 10 network forward passes.

### 8.2 hift_inference() — Mel-Spectrogram to Waveform

**File:** `s3gen.py` line 294, calls HiFiGAN in `hifigan.py`

HiFiGAN is a convolutional neural network that upsamples the mel-spectrogram to a waveform:

1. **Upsampling:** Three transposed convolution layers with rates [8, 5, 3] = 120x total upsampling
2. **Residual refinement:** Multiple ResBlock layers with Snake activation functions
3. **Source generation:** An F0 (fundamental frequency) predictor generates a source excitation signal
4. **Output:** Float32 waveform at 24 kHz

This step is fast (~30ms) because HiFiGAN is a feedforward conv network with no iterative solving.

### 8.3 batch_inference() — Batched Vocoding

**File:** `s3gen.py` line 338

For batched requests:
1. Pad all token sequences to the same length
2. Expand the reference dict (speaker embedding, mel prompt) to batch size
3. Run flow_inference with the full batch `[B, max_len]`
4. Run hift_inference with the full batch
5. Crop each result back to its original length based on token count ratio

Single-item batches fall back to the regular `inference()` method.

---

## Step 9: Post-Processing and Output

### 9.1 Trim Fade (s3gen.py, line 332)

A pre-computed fade is applied to the start of the waveform to reduce "spillover" artifacts from the reference clip:
```python
output_wavs[:, :len(self.trim_fade)] *= self.trim_fade
```

### 9.2 Audio Chunk Assembly (tts.py, line 683)

The waveform (after context cropping and fade-in) is wrapped as a torch tensor:
```python
audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0)  # shape [1, num_samples]
```

### 9.3 PCM Encoding (server.py, line 89)

```python
audio_np = audio_chunk.squeeze().cpu().numpy()       # [num_samples], float32
audio_np = np.clip(audio_np, -1.0, 1.0)             # safety clamp
pcm_data = (audio_np * 32767).astype("<i2").tobytes() # float32 -> int16 LE
```

The client receives a stream of raw PCM chunks (or WAV with a header prepended).

---

## Key Constants

| Constant | Value | File | Description |
|---|---|---|---|
| `S3_SR` | 16000 | s3tokenizer.py | S3 tokenizer sample rate |
| `S3GEN_SR` | 24000 | s3gen/const.py | S3Gen vocoder sample rate (= output rate) |
| `S3_TOKEN_RATE` | 25 | s3tokenizer.py | Speech tokens per second of audio |
| `SPEECH_VOCAB_SIZE` | 6561 | s3tokenizer.py | Number of valid speech token IDs |
| `SPEECH_TOKEN_OFFSET` | 2500 | t3/t3.py | Offset added to speech tokens for vLLM |
| `start_speech_token` | 6561 | t3_config.py | T3 start-of-speech marker |
| `stop_speech_token` | 6562 | t3_config.py | T3 end-of-speech marker |
| `ENC_COND_LEN` | 96000 | tts.py | Max reference audio for tokenizer (6s at 16kHz) |
| `DEC_COND_LEN` | 240000 | tts.py | Max reference audio for S3Gen (10s at 24kHz) |

## Streaming Defaults

| Parameter | Default | Effect |
|---|---|---|
| `chunk_size` | 10 tokens | 400ms of audio per chunk |
| `context_window` | 50 tokens | 2s of lookback for boundary smoothness |
| `fade_duration` | 0.02s | 20ms fade-in per chunk |
| `diffusion_steps` | 4 | CFM solver iterations (4 = fast, 10 = quality) |
| `max_wait_ms` | 50ms | VocoderBatcher collection window |
| `max_batch_size` | 32 | Max concurrent vocoding requests per batch |

## File Reference

```
server.py                              HTTP endpoint, PCM encoding, WAV header
src/chatterbox_vllm/
  tts.py                               Main orchestration, streaming, VocoderBatcher
  text_utils.py                        Text normalization (punc_norm)
  models/
    t3/
      t3.py                            T3 vLLM model wrapper, SPEECH_TOKEN_OFFSET
      modules/
        t3_config.py                   T3Config (vocab sizes, token IDs)
        cond_enc.py                    T3CondEnc (conditioning encoder), T3Cond dataclass
        learned_pos_emb.py             Positional embeddings for speech tokens
    s3gen/
      s3gen.py                         S3Gen vocoder (flow_inference + hift_inference)
      flow.py                          CausalMaskedDiffWithXvec (CFM decoder)
      flow_matching.py                 ConditionalCFM solver (Euler ODE)
      hifigan.py                       HiFTGenerator (mel -> waveform)
      const.py                         S3GEN_SR = 24000
    s3tokenizer/
      s3tokenizer.py                   S3Tokenizer (audio -> discrete tokens)
    voice_encoder/
      voice_encoder.py                 VoiceEncoder (LSTM speaker embedding)
```
