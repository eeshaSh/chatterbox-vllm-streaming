# Strategy 5: Reduce Chunk Context Overlap

## Overview
In streaming mode, the system vocodes each chunk of new speech tokens together with a window of previous tokens (the "context") to ensure smooth audio at chunk boundaries. The current context window is 50 tokens (2000ms of audio), which means the system vocodes 65 tokens (50 context + 15 new) to produce only 15 tokens (600ms) of output audio. The overlap ratio is 76.9%. Reducing this context window decreases per-chunk vocoder cost proportionally.

## Current Implementation

### Token Flow (tts.py, lines 570-606)
1. New tokens arrive from T3 in buffer of chunk_size (default: 15)
2. Context is assembled from previous token history:
   ```
   context_tokens = all_tokens_so_far[-context_window:]  # last 50 tokens
   tokens_to_process = cat([context_tokens, new_tokens])  # 50 + 15 = 65 tokens
   ```
3. All 65 tokens are sent to S3Gen for vocoding
4. The resulting audio is cropped to remove the context portion:
   ```
   samples_per_token = len(wav) / len(clean_tokens)  # ~1920 samples/token
   skip_samples = int(context_length * samples_per_token)  # 96,000 samples
   audio_chunk = wav[skip_samples:]  # keep only new-token audio (~28,800 samples)
   ```
5. A 20ms linear fade-in is applied to the cropped audio chunk
6. The chunk is yielded as PCM bytes

### Key Parameters
| Parameter | Value | Duration | Samples (24kHz) |
|-----------|-------|----------|-----------------|
| chunk_size | 15 tokens | 600ms | 28,800 |
| context_window | 50 tokens | 2000ms | 96,000 |
| Total vocoded | 65 tokens | 2600ms | 124,800 |
| Overlap ratio | 76.9% | — | — |
| Fade-in | 20ms | 20ms | 480 |
| Token rate | 25/sec | 40ms/token | 1,920 samples/token |

### Boundary Handling Techniques (Three Layers)
1. Primary: Context overlap — 50 previous tokens give CFM decoder full acoustic context
2. Secondary: Linear fade-in — 20ms ramp from 0 to 1 amplitude on each chunk
3. Tertiary: HiFiGAN trim/fade — 20ms cosine fade at vocoder output start (s3gen.py, lines 250-253)

No crossfading between adjacent chunks is implemented. Chunks are hard-cropped at token boundaries.

## Key Files
- `src/chatterbox_vllm/tts.py` — Context assembly (lines 570-576), audio cropping (lines 593-598), fade-in (lines 603-606), streaming loop (lines 618-773)
- `src/chatterbox_vllm/models/s3gen/s3gen.py` — Vocoder inference, HiFiGAN trim/fade (lines 250-253)
- `src/chatterbox_vllm/models/s3gen/flow.py` — pre_lookahead_len=3 tokens (unused in streaming)
- `server.py` — chunk_size parameter exposed at API level

## Why Context Matters
The CFM decoder generates mel-spectrograms conditioned on surrounding token context. Without context:
- The model starts from silence/noise at each chunk boundary
- Prosody (pitch, rhythm) has no continuity between chunks
- Energy levels can jump at boundaries
- Spectral characteristics shift abruptly

With sufficient context, the model generates audio that naturally flows from the previous chunk, and the overlapping portion is discarded in favor of the previous chunk's output for that region.

## Impact of Reducing Context

### 50 tokens (current, 2000ms) — Excellent quality
- Very stable generation
- Minimal boundary artifacts
- Conservative, likely oversized

### 30-35 tokens (1200-1400ms) — Expected good quality
- Still provides substantial context
- ~35% less vocoding work per chunk
- Likely sufficient for CFM to maintain continuity
- Low risk, recommended as first test point

### 20 tokens (800ms) — Moderate risk
- Overlap ratio drops to 57%
- Some boundary artifacts possible
- May need crossfading to compensate

### 10 tokens (400ms) — High risk
- Overlap ratio 40%
- Expected: clicks, energy drops, prosody discontinuities, timbre shifts
- Not recommended without additional boundary techniques

### 0 tokens — Unusable
- Each chunk generated independently
- Severe artifacts at every boundary

## Changes Required

### 1. Adjust context_window default
In tts.py, change the default parameter:
```python
context_window: int = 35  # was 50
```
Locations: lines 505, 566, 626 (and any other generate/stream function signatures)

### 2. Fix missing finalize parameter (BONUS)
Location: tts.py, line 744 (and similar in _process_token_buffer_batched)

Currently, the vocoder is always called with finalize=True (or it defaults to True). The S3Gen flow model has a pre_lookahead_len=3 that trims the last 3 tokens when finalize=False, which is designed for streaming but is not being used.

Fix: Pass finalize=output.finished so that non-final chunks use pre_lookahead trimming for better streaming quality.

### 3. Optional: Implement crossfading
Currently, chunks are hard-cropped and only a fade-in is applied. Adding crossfading between the tail of the previous chunk and the head of the current chunk could allow reducing context further while maintaining quality.

Implementation approach:
- Store the last N ms of the previous chunk's audio
- Blend it with the first N ms of the current chunk using a linear or cosine crossfade
- Output the blended region followed by the rest of the current chunk
- Typical crossfade duration: 10-30ms

### 4. Fix rounding in sample cropping
Location: tts.py, line 594
```python
samples_per_token = len(wav) / len(clean_tokens)  # float division
skip_samples = int(context_length * samples_per_token)  # truncates
```
This can lose 1-2 samples per chunk. Not audible, but could be made precise by using the known constant (1920 samples/token at 24kHz, 25 tokens/sec).

## Expected Gains
- Reducing context from 50 to 35 tokens: vocoder processes 50 tokens instead of 65 per chunk = ~23% less vocoder compute per chunk
- Reducing context from 50 to 30 tokens: vocoder processes 45 tokens instead of 65 = ~31% less compute
- These savings apply to every chunk in every streaming request
- S3Gen compute scales roughly linearly with sequence length (both flow matching and HiFiGAN)

## Risks
- Audio quality degradation at chunk boundaries (clicks, pops, energy drops)
- Prosody discontinuities (unnatural rhythm/pitch changes)
- Spectral discontinuities (timbre shifts)
- Language-dependent sensitivity (some languages/phonemes may need more context)
- Speaker-dependent sensitivity (some voice profiles may be more sensitive)

## Validation
- Generate test audio at context values: 50, 40, 35, 30, 25, 20
- Compare using:
  - Automated: PESQ, STOI, or mel-spectrogram visual inspection at boundaries
  - Subjective: MOS listening tests (is boundary audible?)
- Test across multiple languages and speakers
- Test with varying chunk sizes (10, 15, 20, 25) since chunk size interacts with context adequacy
- Focus listening on chunk boundaries specifically, not overall quality

## Recommendation
Test at 30-35 tokens as the first candidate. This provides a meaningful compute reduction (~25-30%) with low risk given that 1.2-1.4 seconds of acoustic context is still substantial for the CFM model. If quality holds, this is a reliable win that compounds across every chunk of every request. Also fix the finalize parameter regardless of context changes — it is a correctness improvement that the architecture was designed for but is not currently using.
