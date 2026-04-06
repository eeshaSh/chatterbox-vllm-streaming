# Classifier-Free Guidance (CFG) and Trailing Noise in vLLM Streaming

## What is CFG?

Classifier-Free Guidance is an inference-time technique that improves the quality of conditional generation. It works by running the model in two modes simultaneously and combining the results.

### Training

During training, 20% of the time (`training_cfg_rate = 0.2`), the text conditioning is randomly zeroed out. This teaches the model two behaviors:

- **Conditioned mode:** "Generate speech that matches this text"
- **Unconditioned mode:** "Generate generic speech patterns without any text guidance"

The model learns to operate in both modes within the same weights.

### Inference

At inference time, the model runs two forward passes per token:

1. **Conditioned pass:** Full text embeddings present → produces `cond_logits`
2. **Unconditioned pass:** Text embeddings zeroed out → produces `uncond_logits`

The final logits are computed as:

```
logits = cond_logits + cfg_scale * (cond_logits - uncond_logits)
```

With `cfg_scale = 0.5`, this expands to:

```
logits = 1.5 * cond_logits - 0.5 * uncond_logits
```

The term `(cond_logits - uncond_logits)` isolates the *effect* of the text conditioning — the direction the text is pushing the model's predictions. Multiplying by `cfg_scale` and adding it back extrapolates further in that direction.

### Why CFG Helps with Trailing Noise

When speech content is finished:

- `cond_logits` for EOS → **high** (the model sees the text is done, wants to stop)
- `uncond_logits` for EOS → **low** (no text signal, doesn't know when to stop)
- Result: `high + 0.5 * (high - low)` = **even higher** → EOS probability is amplified

When speech is mid-sentence:

- `cond_logits` for EOS → **low** (more text to generate)
- `uncond_logits` for EOS → some baseline
- Result: the "don't stop yet" signal is amplified too

CFG sharpens the model's conditioning signal. Without it, the model runs on raw conditioned logits alone, making it more likely to drift past the natural endpoint of speech and generate junk tokens that get vocoded into audible noise.

### Two Levels of CFG in Chatterbox

Chatterbox uses CFG at two separate stages:

1. **T3 (text-to-tokens):** `cfg_scale = 0.5` — guides speech token generation toward text-faithful output
2. **S3Gen (tokens-to-audio, flow matching/diffusion):** `inference_cfg_rate = 0.7` — guides mel-spectrogram generation toward conditioning-faithful output

The S3Gen CFG works correctly in all versions (it operates within the vocoder, not through vLLM). The problem is exclusively with T3 CFG.

---

## How CFG is Implemented Across Codebases

### Original HuggingFace Streaming (Working CFG)

**File:** `chatterbox-streaming/src/chatterbox/models/t3/t3.py`

This is the reference implementation with full CFG:

- **Line 96:** `text_emb[1].zero_()` — Zeroes text embeddings for the unconditioned branch
- **Lines 297-298:** `bos_embed = torch.cat([bos_embed, bos_embed])` — Maintains batch-of-2 (cond + uncond)
- **Lines 327-331:** CFG formula applied in the generation loop:
  ```python
  logits_cond = logits[0:1]
  logits_uncond = logits[1:2]
  logits = logits_cond + cfg_weight * (logits_cond - logits_uncond)
  ```
- **Lines 357-358:** `next_token_embed = torch.cat([next_token_embed, next_token_embed])` — Each new speech token is duplicated for both branches

The transformer runs with batch size 2 throughout. The KV cache maintains separate cond and uncond states because the prefill produced different hidden states (one with text, one without). During decode, even though the input speech token is the same for both branches, the transformer attends over divergent KV caches and produces genuinely different hidden states.

### Non-Streaming vLLM (Working CFG)

**File:** `chatterbox-vllm/src/chatterbox_vllm/models/t3/t3.py`

This version packs cond and uncond into a single tensor by doubling along `dim=0`:

- **Lines 635-652:**
  ```python
  cond_embeds, uncond_embeds = inputs_embeds.split([self.dim, self.dim], dim=1)
  hidden_states = self.tfmr(
      input_ids=None,
      positions=torch.cat([positions, positions], dim=0),
      inputs_embeds=torch.cat([cond_embeds, uncond_embeds], dim=0)
  )
  hidden_state_1, hidden_state_2 = hidden_states.split(
      [len(cond_embeds), len(uncond_embeds)], dim=0
  )
  return torch.cat([hidden_state_1, hidden_state_2], dim=1)
  ```

This works because the transformer sees 2N tokens (N cond + N uncond) in a single forward pass, producing divergent hidden states. However, this approach has a latent issue: the attention metadata (`slot_mapping`, `block_tables`, `seq_lens_tensor`) is built for N tokens, not 2N. It works when there's a single sequence but fails with concurrent batching.

### Streaming vLLM (CFG Disabled)

**File:** `chatterbox-vllm-streaming/src/chatterbox_vllm/models/t3/t3.py`

CFG is completely disabled in both prefill and decode:

- **Prefill (lines 730-771):** Only conditioned embeddings are used. The uncond embeddings (with zeroed text) are discarded:
  ```python
  cond_embeds = inputs_embeds[:, :self.dim].contiguous()
  cond_hidden = self.tfmr(...)
  return torch.cat([cond_hidden, cond_hidden], dim=1)  # duplicate
  ```

- **Decode (lines 706-728):** The transformer runs once and the output is duplicated:
  ```python
  hidden_states = self.tfmr(...)
  return torch.cat([hidden_states, hidden_states], dim=1)  # duplicate
  ```

- **compute_logits (lines 602-619):** The CFG formula is still applied, but since `cond_hidden == uncond_hidden`, it degenerates to: `logits = cond_logits + 0.5 * 0 = cond_logits`

The comments at lines 745-750 explain why:

> We cannot call self.tfmr() twice in a single forward() because vLLM's attention metadata (forward context) is consumed by the first call, causing assertion failures when concurrent requests are batched together.

---

## Why CFG Cannot Be Restored in vLLM Streaming

### The Fundamental Constraint

vLLM manages attention state globally. Each model `forward()` call must process exactly the tokens the engine scheduled. The architecture has three constraints that collectively make CFG impossible:

#### 1. Forward Context is Consumed Once

vLLM sets a global `ForwardContext` containing `attn_metadata` before each `model_executable()` call (`forward_context.py:27-34`, `model_runner.py:1722`). This metadata includes `slot_mapping`, `block_tables`, `seq_lens_tensor`, `query_start_loc` — all sized for exactly N tokens/sequences.

The transformer's attention layers read this metadata during their forward pass. Calling `self.tfmr()` a second time within the same `forward()` would attempt to reuse consumed metadata, causing assertion failures.

#### 2. One KV Cache Per Sequence

vLLM allocates one set of KV cache blocks per sequence via its block manager. There is no concept of "two KV caches for the same sequence" (one conditioned, one unconditioned). Proper CFG requires divergent KV caches — the conditioned cache built from prefill with text, the unconditioned cache built from prefill with zeroed text. These diverge at prefill and remain divergent throughout decode.

#### 3. Cannot Double Along dim=0

Passing `torch.cat([cond_embeds, uncond_embeds], dim=0)` with `torch.cat([positions, positions], dim=0)` produces 2N tokens, but the attention metadata only has N entries for `slot_mapping`. This causes:

- `reshape_and_cache_flash` (`flash_attn.py:714`): Tries to write 2N K/V entries into N cache slots
- `flash_attn_with_kvcache` (`flash_attn.py:824`): Gets 2N queries but `seq_lens` has N entries
- `advance_step` (`flash_attn.py:342`): `assert self.num_decode_tokens == num_seqs` fails (2N ≠ N)

### Approaches Evaluated

| Approach | Why It Fails |
|----------|-------------|
| **Two separate vLLM requests** | Both sequences must produce the same token at every step. vLLM's async engine has no cross-sequence sampling coordination. |
| **LoRA / Prompt adapters** | Wrong mechanism — these modify weights or prepend tokens, not zero out embeddings mid-sequence. |
| **Fixed CFG bias vector from prefill** | Can't call `tfmr()` twice during prefill either (same constraint). And a fixed bias goes stale during decode as KV cache grows. |
| **Modify vLLM for "virtual sequences"** | Requires deep changes to block manager, scheduler, attention metadata builder, and sampler. Essentially a vLLM feature request. |
| **Post-hoc uncond forward without KV cache** | Implementing custom attention outside flash attention is complex, and quality degrades without proper KV cache. |

### The Circular Problem

Even if we found a way to run the transformer twice during decode, it still wouldn't work because **prefill also has CFG disabled**. The KV cache is built entirely from conditioned embeddings. Both "cond" and "uncond" decode branches would attend over the same conditioned KV cache, producing identical hidden states. CFG would still be a no-op.

Restoring CFG requires fixing both prefill AND decode, which means maintaining two separate KV caches per sequence — a fundamental change to vLLM's memory management.

---

## Current Mitigations (and Their Limitations)

Since CFG cannot be restored, the codebase uses several compensating mechanisms:

### 1. AlignmentState (Token-Count Heuristics)

**File:** `src/chatterbox_vllm/models/t3/alignment.py`

A custom replacement for the original codebase's `AlignmentStreamAnalyzer` (which used attention weights from layer 9 — not available in vLLM). Uses simple ratio-based rules:

- `MIN_SPEECH_PER_TEXT = 1.5`: Suppress EOS before this speech-to-text ratio
- `SOFT_EOS_START = 2.5`: Begin linearly boosting EOS logit
- `MAX_SPEECH_PER_TEXT = 4`: Force EOS above this ratio
- `SOFT_EOS_MAX_BOOST = 10.0`: Maximum logit boost
- `REPEAT_THRESHOLD = 3`: Force EOS if same token appears 3+ times consecutively

**Limitation:** These are fixed ratios applied uniformly across languages. Turkish has a different speech-to-text token ratio than English due to agglutinative morphology and character-level BPE fragmentation. The thresholds on the `fix-turkish-trailing-noise` branch (`SOFT_EOS_START = 2.5`, `MAX_SPEECH_PER_TEXT = 4`) were found to cut speech off too early for some inputs while still not preventing all trailing noise.

### 2. Trailing Silence Trimmer

**File:** `src/chatterbox_vllm/tts.py`, method `_trim_trailing_silence()`

Post-processing that detects and removes junk audio from the final vocoded chunk using RMS energy gap detection and zero-crossing rate spectral analysis.

**Limitation:** Only operates on the final streaming chunk. If junk spans chunk boundaries, the trimmer sees only part of the pattern. Also, the "quiet gap" between speech and junk may not be quiet enough for threshold-based detection.

### 3. Mel-Spectrogram Spectral Gating

**File:** `src/chatterbox_vllm/models/s3gen/s3gen.py`, method `_gate_mel_silence()`

Attenuates low-frequency mel bins (0-11, ~0-300Hz) during detected silence frames.

**Limitation:** The junk burst often has enough total energy to pass the silence threshold. The gating was designed for quiet groans at pause boundaries, not loud post-speech rumble.

---

## Summary

CFG is the primary mechanism the original Chatterbox uses to maintain text-faithful speech generation and clean stopping behavior. The vLLM streaming version cannot use CFG because vLLM's attention backend manages KV cache globally with one cache per sequence, and the model cannot run the transformer twice per forward call. The `AlignmentState` heuristics, trailing silence trimmer, and spectral gating are practical band-aids, but none fully replaces the continuous per-token guidance that CFG provides. The trailing noise problem — particularly in Turkish — is a direct consequence of this architectural limitation.
