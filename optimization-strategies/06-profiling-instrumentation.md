# Strategy 6: Profiling and Instrumentation

## Overview
Before pursuing any optimization, the system needs proper instrumentation to identify actual bottlenecks. The current benchmarking (benchmark.py) and metrics (StreamingMetrics) only measure coarse pipeline stages. There is no per-component timing, no GPU utilization tracking, no VocoderBatcher observability, and no token generation rate measurement. This strategy establishes the measurement foundation that all other optimizations depend on.

## Current Instrumentation

### What IS measured
| Metric | Location | Method |
|--------|----------|--------|
| Model load time | benchmark.py:79 | time.time() |
| Total generation time (batch) | benchmark.py:92 | time.time() |
| T3 generation time (batch) | tts.py:465 | time.time() |
| S3Gen time (batch) | tts.py:491 | time.time() |
| TTFC (time to first chunk) | tts.py:551 | time.time() |
| RTF (real-time factor) | tts.py:768 | generation_time / audio_duration |
| Chunk count | tts.py:554 | Counter |
| TTFB (server, time to first byte) | server.py:90 | time.time() |
| GPU memory snapshots | example-tts-min-vram.py | torch.cuda.memory_allocated() (manual, 3 points) |
| VocoderBatcher batch size | tts.py:152 | print() when batch > 1 |

### What is NOT measured (critical gaps)
1. Per-component breakdown: flow inference time vs HiFiGAN time vs post-processing
2. VocoderBatcher queue dynamics: wait time, queue depth, batch utilization
3. Token generation rate from vLLM (tokens/sec)
4. Time to first token (prefill latency)
5. GPU utilization percentage
6. Peak memory per phase
7. Per-chunk latency in streaming mode
8. Inter-chunk delay distribution

## Key Files
- `benchmark.py` -- Existing benchmark script (coarse-grained)
- `src/chatterbox_vllm/tts.py` -- StreamingMetrics dataclass (lines 68-75), VocoderBatcher (lines 77-175), generation logic
- `server.py` -- TTFB measurement
- `src/chatterbox_vllm/models/s3gen/s3gen.py` -- S3Gen inference methods (needs timing)
- `example-tts.py`, `example-tts-multilingual.py` -- Examples with no metrics

## Instrumentation Plan

### Priority 1: S3Gen Component Breakdown

The most critical unknown is where S3Gen time is spent. Two candidates: flow inference (CFM diffusion) vs HiFiGAN (mel to waveform).

Add CUDA events in s3gen.py around the two main operations:

Instrumentation points in S3Token2Wav:
- Before/after flow.inference() in flow_inference() method (~line 290)
- Before/after mel2wav.inference() in hift_inference() method (~line 300)
- Before/after the full batch_inference() call (~lines 380-420)

Use torch.cuda.Event for precise GPU-side timing:
```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# ... operation ...
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

This answers: "Should we optimize flow matching (reduce steps, compile decoder) or HiFiGAN (compile decode, quantize)?"

### Priority 2: VocoderBatcher Observability

Add tracking to the batch worker in tts.py:

Metrics to collect:
- submit_time: timestamp when request enters queue (set in vocode() method)
- batch_start_time: when batch processing begins
- batch_end_time: when batch processing completes
- batch_size: number of items in each batch
- queue_depth: queue.qsize() sampled at batch start
- wait_time: batch_start_time - submit_time (per request)
- timeout_fired: whether the batch was triggered by timeout vs reaching max_batch_size

Store as running statistics (mean, p50, p95, max) or append to a list for later analysis.

This answers: "Is the batcher helping or hurting? Are batches well-utilized? Is the timeout too long?"

### Priority 3: Token Generation Metrics

Add timing around vLLM token generation in tts.py streaming loop:

Instrumentation points:
- Record time of first token received (lines ~731): gives time-to-first-token (TTFT)
- Count tokens per second across the generation (total tokens / total time)
- Track time between consecutive buffer fills (inter-chunk interval)

This answers: "Is T3 (vLLM) or S3Gen the bottleneck in the streaming pipeline?"

### Priority 4: GPU Resource Monitoring

Add periodic GPU sampling using pynvml:

```python
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_gpu_stats():
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return {
        'mem_used_gb': mem.used / 1e9,
        'mem_total_gb': mem.total / 1e9,
        'gpu_util_pct': util.gpu,
        'mem_util_pct': util.memory,
    }
```

Sample at key points:
- After model loading
- Before/after T3 generation
- Before/after S3Gen vocoding
- Peak memory via torch.cuda.max_memory_allocated()

This answers: "Is memory the constraint? How much headroom exists for larger batches?"

### Priority 5: End-to-End Request Profiling

Enhance server.py to track per-request lifecycle:

- Request arrival time
- TTFB (already measured)
- Per-chunk timestamps (when each chunk is yielded)
- Total request duration
- Total audio duration generated
- RTF per request
- Concurrent request count at time of request

This answers: "What is the user-facing latency? How does it degrade under concurrency?"

## Profiling Tools

### CUDA Events (recommended for production)
- Overhead: < 1%
- Precision: microsecond on GPU
- Use for: per-component timing in S3Gen
- Limitation: requires torch.cuda.synchronize() which blocks CPU

### torch.profiler (recommended for deep dives)
- Overhead: 10-20%
- Captures: kernel-level timing, memory allocation, operator breakdown
- Output: TensorBoard trace files
- Use for: one-time investigation of specific bottlenecks
- Not for production

### pynvml (recommended for resource monitoring)
- Overhead: negligible
- Captures: GPU memory, utilization percentage
- Use for: continuous monitoring, deployment dashboards
- Limitation: sampling-based, may miss short bursts

### cProfile (use sparingly)
- Overhead: moderate
- Captures: Python function-level CPU time
- Use for: identifying unexpected Python-side bottlenecks (data conversion, preprocessing)
- Not useful for GPU timing

## Implementation Approach

### Option A: Lightweight Always-On Metrics
Add a metrics collection class that accumulates statistics during normal operation with minimal overhead. Expose via a /metrics endpoint on the server or periodic log output.

Tracked metrics:
- S3Gen flow_time_ms, hifigan_time_ms (CUDA events, per batch)
- VocoderBatcher batch_size, wait_time_ms, queue_depth
- T3 tokens_per_second, time_to_first_token_ms
- GPU memory_used_gb, peak_memory_gb
- Per-request TTFB, total_time, rtf

### Option B: Opt-In Detailed Profiling
Add a --profile flag to server.py and benchmark.py that enables torch.profiler tracing for a configurable number of requests. Saves trace files for offline analysis in TensorBoard.

### Recommendation: Both
Option A for always-on production monitoring (answer "what is happening"). Option B for occasional deep investigation (answer "why is it slow").

## Specific Instrumentation Points

### s3gen.py -- S3Token2Wav
1. flow_inference(): CUDA event around self.flow.inference() call
2. hift_inference(): CUDA event around self.mel2wav.inference() call
3. batch_inference(): CUDA events around flow.inference() and mel2wav batch calls
4. batch_inference(): Log batch size and sequence lengths

### tts.py -- VocoderBatcher._batch_worker()
1. After dequeuing first item: record batch_start_time
2. After collection loop: record batch_ready_time, log batch size and queue depth
3. After s3gen call: record batch_end_time
4. Per-request: compute wait_time = batch_start_time - submit_time (submit_time added to queue item)

### tts.py -- generate_stream_with_conds()
1. Before vLLM generate: record gen_start_time
2. On first token received: record ttft = time.time() - gen_start_time
3. On each chunk process: record chunk_start_time, chunk_end_time
4. After stream complete: compute tokens_per_second, total chunks, total audio duration

### server.py -- audio_stream()
1. On request arrival: record request_start (already done)
2. On each chunk yield: record chunk_timestamp
3. On stream end: log total_time, total_chunks, rtf
4. Track concurrent_requests counter (increment on start, decrement on end)

## Expected Output
After instrumentation, a single streaming request should produce a log or metrics payload like:

```
request_id: abc123
concurrent_requests: 3
ttft_ms: 45.2
ttfb_ms: 312.5
total_time_ms: 4523.1
total_audio_ms: 8400.0
rtf: 0.54
chunks: 14
tokens_per_second: 32.1
gpu_memory_gb: 4.2
avg_flow_time_ms: 128.3
avg_hifigan_time_ms: 22.1
avg_batcher_wait_ms: 31.4
avg_batch_size: 2.8
p95_chunk_latency_ms: 215.0
```

This data directly informs which of the other five strategies will have the most impact.

## Risks
- CUDA event synchronization (torch.cuda.synchronize()) blocks the CPU. Use sparingly in production -- profile in development, switch to async-safe alternatives for production.
- pynvml initialization can fail if NVIDIA drivers are not properly installed.
- Metrics collection adds small memory overhead for storing statistics.
- Logging overhead in hot paths (VocoderBatcher) should use buffered/sampling approach, not per-item I/O.

## Validation
- Confirm instrumentation overhead is < 2% by comparing throughput with and without metrics enabled
- Verify CUDA event timing matches wall-clock expectations (sanity check)
- Run under load and confirm metrics are consistent and not showing data races
- Compare TTFB reported by server vs client-measured TTFB to validate server-side measurement

## Recommendation
This is the prerequisite for all other strategies. Implement Priority 1 (S3Gen breakdown) and Priority 2 (VocoderBatcher observability) first. These two measurements will immediately reveal whether optimization effort should focus on the vocoder (strategies 1-3, 5) or the batching/orchestration layer (strategy 4). Do this before writing any optimization code.
