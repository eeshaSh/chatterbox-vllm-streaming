# Strategy 4: VocoderBatcher Tuning

## Overview
The VocoderBatcher (tts.py, lines 77-175) collects S3Gen vocoding requests from concurrent streaming connections and submits them as batched GPU operations. This amortizes kernel launch overhead and improves GPU utilization under concurrency. The current implementation has several tunable parameters and at least one correctness issue.

## Current Implementation

### Parameters
- max_batch_size: 32 (maximum items per batch)
- max_wait_ms: 50 (milliseconds to wait for additional requests before flushing)
- Queue: asyncio.Queue, lazily initialized per event loop
- Worker: Single background asyncio task running _batch_worker()

### Batching Logic (lines 115-167)
1. Block waiting for at least one request from the queue
2. Start a 50ms deadline timer
3. Collect additional requests up to max_batch_size or until deadline expires
4. Group collected requests by voice identity using id(ref_dict)
5. For each voice group: run s3gen.batch_inference() under torch.autocast(bfloat16)
6. Set futures with results, returning audio to waiting coroutines

### Request Format
Each request is a tuple: (speech_tokens, ref_dict, n_timesteps, future)
- speech_tokens: 1D tensor of speech token IDs
- ref_dict: dict containing speaker embedding, reference mel, prompt tokens
- n_timesteps: diffusion step count
- future: asyncio.Future for returning result to caller

## Issues Found

### Issue 1: Voice Grouping by Object Identity (BUG)
Location: tts.py, line 141
```python
key = id(item[1])  # Groups by Python object identity
```

This uses Python's `id()` function, which returns the memory address of the ref_dict object. Two requests for the same speaker but with different ref_dict objects (even if contents are identical) will NOT be batched together.

This is a problem because:
- `get_audio_conditionals()` has `@lru_cache(maxsize=10)` but the cache key is the file path -- concurrent requests for the same voice should hit the cache and get the same object. However, if the cache evicts or if conditionals are constructed differently, separate objects with identical content will be treated as different voices.
- Any code path that creates ref_dict without going through the cache will produce unbatchable requests.

Fix: Replace `id(ref_dict)` with a content-based key. Options:
- Hash the speaker embedding tensor: `key = ref_dict['ref_spk_emb'].data_ptr()` (fast, unique per tensor allocation)
- Or use a voice ID string passed through the pipeline
- Or compare tensor contents with a cached fingerprint

### Issue 2: No Adaptive Batch Firing
The batcher always waits the full 50ms timeout even if it has collected 30 of 32 possible items in the first 5ms. An adaptive strategy would fire early when the batch is nearly full.

Fix: Add an early-fire condition:
```python
if len(batch) >= self.max_batch_size * 0.8:
    break  # Fire at 80% capacity
```

### Issue 3: Single-User Overhead
When only one user is active, every vocoding request still goes through the queue and waits up to 50ms for additional requests that will never arrive. This adds unnecessary latency.

Fix: Short-circuit path -- if the queue is empty after the first item is dequeued and no more arrive within 1-2ms, skip the full timeout and process immediately. Or expose a bypass mode for single-user deployments.

### Issue 4: No Observability
Only one log statement exists (line 152): prints batch size when > 1 item. No metrics on queue depth, wait time, batch utilization, or timeout frequency.

## Tuning Recommendations

### Timeout (max_wait_ms)
| Scenario | Recommended | Rationale |
|----------|-------------|-----------|
| Single user, low latency | 10-20ms | Minimize unnecessary waiting |
| 2-5 concurrent users | 30-50ms | Balance batching vs latency |
| 10+ concurrent users | 50-100ms | Maximize batch utilization |
| Throughput-first (offline) | 100-200ms | Maximize GPU efficiency |

The current 50ms is reasonable for moderate concurrency. For latency-sensitive single-user deployments, reducing to 20ms would eliminate ~30ms of wasted waiting per chunk.

### Batch Size (max_batch_size)
GPU memory scaling for vocoder batches (approximate):
- batch_size=1: ~50ms vocoding, ~40ms kernel launch overhead (80% overhead)
- batch_size=8: ~70ms vocoding (amortized ~8.75ms per item)
- batch_size=32: ~150ms vocoding (amortized ~4.7ms per item)
- batch_size=64: ~280ms vocoding (diminishing returns begin)

The current 32 is conservative and appropriate for 8-24 GB GPUs. Note that CFG internally doubles the batch (32 becomes 64 in the flow estimator), so actual GPU batch is 2x the configured value.

Keep 32 as default but expose as a constructor parameter for deployment-specific tuning.

### Padding Efficiency
batch_inference() pads all sequences to the longest in the batch. A batch with one 100-token sequence and 31 ten-token sequences wastes significant compute on padding.

Potential improvement: Sort by sequence length before batching, or split into sub-batches with similar lengths. This is a secondary optimization.

## Changes Required

### 1. Fix voice grouping (correctness)
Replace `id(ref_dict)` with content-based comparison. Simplest approach:
```python
key = ref_dict['ref_spk_emb'].data_ptr()
```
This is fast (pointer comparison), unique per unique tensor allocation, and correctly identifies when two requests share the same pre-computed speaker embedding.

### 2. Add adaptive batch firing
In the collection loop, add:
```python
if len(batch) >= int(self.max_batch_size * 0.8):
    break
```

### 3. Add instrumentation
Track and expose:
- batch_sizes: list of actual batch sizes formed (for histogramming)
- queue_wait_times: time from request submission to batch start
- timeout_count vs early_fire_count: how often batches fire due to timeout vs filling up
- total_batches_processed: running counter

Add a method like `get_stats()` that returns these metrics for monitoring.

### 4. Make parameters configurable
Expose max_batch_size and max_wait_ms as constructor parameters (they already are) but also through server.py command-line arguments or environment variables for deployment tuning without code changes.

### 5. Optional: Add single-user fast path
If queue is empty after dequeuing first item and remains empty after 2ms, process immediately without waiting for the full timeout.

## Expected Gains
- Fix voice grouping bug: Correct batching behavior when multiple requests use same voice but different object refs. Impact depends on how often this occurs.
- Reduced timeout (50->20ms): ~30ms latency reduction per chunk in single-user scenarios. Over a 10-chunk stream, this saves ~300ms total latency.
- Adaptive firing: Reduces latency under high concurrency when batches fill quickly.
- Better observability: Enables data-driven tuning of all other parameters.

## Risks
- Low. These are application-level logic changes, not model changes.
- Voice grouping fix: if data_ptr() is not stable across requests, could cause incorrect grouping. Validate with logging.
- Reducing timeout too aggressively: smaller batches, less GPU efficiency under concurrency.

## Validation
- Log batch sizes before and after changes to confirm batching behavior
- Measure TTFC (time to first chunk) in single-user and multi-user scenarios
- Compare S3Gen batch inference time at various batch sizes to confirm scaling assumptions
- Load test with 5-10 concurrent streaming requests to measure throughput impact
- Verify voice grouping correctness by logging group keys

## Recommendation
The voice grouping bug fix (Issue 1) and observability (Issue 4) should be done regardless of other optimization work. They are low-risk and foundational for understanding system behavior. Timeout tuning (Issue 2-3) can be done after observability is in place, informed by actual batch size and wait time data.
