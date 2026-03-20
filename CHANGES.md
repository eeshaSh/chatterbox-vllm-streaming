# Recent Changes

## 1. Prometheus Metrics (commit `b659b89`)

### What is Prometheus?

Prometheus is a monitoring system. Your server exposes numbers (metrics) at an HTTP endpoint, and Prometheus scrapes that endpoint on a schedule to collect them. You then query/graph them in Grafana or similar.

### What was added

**Endpoint:** `GET /metrics` on port 4123 (same server, mounted automatically).

**Metrics exposed:**

| Metric | Type | What it tells you |
|--------|------|-------------------|
| `tts_requests_total` | Counter | Total requests, labeled by `status` ("success"/"error") |
| `tts_request_duration_seconds` | Histogram | End-to-end request duration |
| `tts_time_to_first_byte_seconds` | Histogram | Time until the first audio chunk is sent (TTFB) |
| `tts_active_requests` | Gauge | How many requests are in-flight right now |
| `tts_t3_tokens_generated_total` | Counter | Total speech tokens generated |
| `tts_t3_time_to_first_token_seconds` | Histogram | Time to first T3 token |
| `tts_t3_tokens_per_second` | Histogram | Token generation throughput |
| `tts_vocoding_flow_duration_seconds` | Histogram | CFM flow step duration per chunk |
| `tts_vocoding_hifigan_duration_seconds` | Histogram | HiFiGAN vocoder duration per chunk |
| `tts_vocoding_chunk_latency_seconds` | Histogram | Total per-chunk latency |
| `tts_batcher_batches_total` | Counter | Total batches the VocoderBatcher has processed |
| `tts_batcher_timeout_fires_total` | Counter | Batches that fired because the 50ms wait expired |
| `tts_batcher_early_fires_total` | Counter | Batches that fired early (hit 80% capacity) |
| `tts_batcher_fast_path_fires_total` | Counter | Batches that fired via fast path (single user, no wait) |
| `tts_batcher_avg_batch_size` | Gauge | Average batch size |
| `tts_batcher_avg_wait_seconds` | Gauge | Average queue wait time |

### How to use it

```bash
# Quick check - hit the endpoint directly
curl http://your-server:4123/metrics

# In Prometheus config (prometheus.yml), add a scrape target:
scrape_configs:
  - job_name: 'chatterbox-tts'
    scrape_interval: 15s
    static_configs:
      - targets: ['your-server:4123']
    metrics_path: /metrics
```

### Where the code lives

- `src/chatterbox_vllm/metrics.py` — all metric definitions and the batcher collector
- `server.py:36-38` — mounts the `/metrics` endpoint and registers the batcher collector
- `server.py:84,112,121-124` — request-level metrics are recorded in the `audio_stream()` generator

---

## 2. Batch Size Fix (commit `88067b7`)

### The problem

When 13+ requests arrived concurrently, the VocoderBatcher grouped them into one batch. The CFM decoder uses **Classifier-Free Guidance (CFG)**, which internally doubles the batch size (conditioned + unconditioned pass). So 13 requests became a batch of 26 — but the TensorRT engine was built to accept a max batch of 20.

```
13 requests × 2 (CFG) = 26 > 20 (TRT max) → error
```

### What changed

| File | Before | After | Why |
|------|--------|-------|-----|
| `scripts/build_trt_engine.py` | `batch_max=20` | `batch_max=40` | TRT engine now accepts up to 40 (= 20 real requests × 2 CFG) |
| `src/chatterbox_vllm/tts.py` | `max_batch_size=32` | `max_batch_size=20` | Batcher won't exceed 20 requests per batch, staying within TRT's 40 limit |

### Deployment note

The TRT engine is built at container startup and cached. If a cached engine already exists from a previous build, **it won't be rebuilt automatically**. You need to delete the old one:

```bash
# Delete cached engine so the entrypoint rebuilds with the new batch_max=40
rm -f ~/.cache/huggingface/hub/*/conditional_decoder.engine
```

Then rebuild and restart the container. The entrypoint (`entrypoint.sh`) will rebuild the engine on next startup.
