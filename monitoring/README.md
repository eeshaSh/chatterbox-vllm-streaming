# Monitoring Setup (Prometheus + Grafana)

Prometheus scrapes the `/metrics` endpoint exposed by the TTS server on port 4123.
Grafana provides dashboards on top of the collected metrics.

## Quick Start

```bash
sudo ./monitoring/setup.sh
```

This installs Prometheus and Grafana as systemd services and provisions the
Prometheus datasource in Grafana automatically.

| Service    | URL                        | Credentials   |
|------------|----------------------------|---------------|
| Prometheus | http://localhost:9090       | —             |
| Grafana    | http://localhost:3000       | admin / admin |
| TTS metrics| http://localhost:4123/metrics | —           |

## Verify

```bash
# Check services are running
systemctl status prometheus grafana-server

# Check Prometheus is scraping the TTS server
curl -s http://localhost:9090/api/v1/targets | python3 -m json.tool

# Check TTS metrics directly
curl -s http://localhost:4123/metrics | grep "^# TYPE tts_"
```

## Service Management

```bash
# Stop / start / restart
sudo systemctl stop prometheus
sudo systemctl start prometheus
sudo systemctl restart grafana-server

# View logs
journalctl -u prometheus -f
journalctl -u grafana-server -f
```

## Config Changes

Edit `/etc/prometheus/prometheus.yml`, then hot-reload without restart:

```bash
curl -X POST http://localhost:9090/-/reload
```

Or copy the repo version and reload:

```bash
sudo cp monitoring/prometheus.yml /etc/prometheus/prometheus.yml
curl -X POST http://localhost:9090/-/reload
```

## Available Metrics

### Request-level
| Metric | Type | Description |
|--------|------|-------------|
| `tts_requests_total{status}` | Counter | Total requests (success/error) |
| `tts_request_duration_seconds` | Histogram | End-to-end request duration |
| `tts_time_to_first_byte_seconds` | Histogram | Time to first audio byte |
| `tts_active_requests` | Gauge | Currently in-flight requests |

### T3 Token Generation
| Metric | Type | Description |
|--------|------|-------------|
| `tts_t3_tokens_generated_total` | Counter | Total speech tokens generated |
| `tts_t3_time_to_first_token_seconds` | Histogram | Time to first T3 token |
| `tts_t3_tokens_per_second` | Histogram | Token generation throughput |

### Vocoding
| Metric | Type | Description |
|--------|------|-------------|
| `tts_vocoding_flow_duration_seconds` | Histogram | CFM flow step duration |
| `tts_vocoding_hifigan_duration_seconds` | Histogram | HiFiGAN duration |
| `tts_vocoding_chunk_latency_seconds` | Histogram | Total chunk latency |

### VocoderBatcher
| Metric | Type | Description |
|--------|------|-------------|
| `tts_batcher_batches_total` | Counter | Total batches processed |
| `tts_batcher_timeout_fires_total` | Counter | Batches fired on timeout |
| `tts_batcher_early_fires_total` | Counter | Batches fired at 80% capacity |
| `tts_batcher_fast_path_fires_total` | Counter | Fast-path fires (single user) |
| `tts_batcher_avg_batch_size` | Gauge | Average batch size |
| `tts_batcher_avg_wait_seconds` | Gauge | Average queue wait time |

## Useful PromQL Queries

```promql
# Requests per second
rate(tts_requests_total[5m])

# p95 request duration
histogram_quantile(0.95, rate(tts_request_duration_seconds_bucket[5m]))

# p95 time to first byte
histogram_quantile(0.95, rate(tts_time_to_first_byte_seconds_bucket[5m]))

# Error rate
rate(tts_requests_total{status="error"}[5m])

# Average tokens per second
rate(tts_t3_tokens_generated_total[5m])

# Current active requests
tts_active_requests
```

## Uninstall

```bash
sudo systemctl stop prometheus grafana-server
sudo systemctl disable prometheus grafana-server
sudo rm /etc/systemd/system/prometheus.service
sudo rm /usr/local/bin/prometheus /usr/local/bin/promtool
sudo apt-get remove --purge grafana
sudo rm -rf /var/lib/prometheus /etc/prometheus
sudo systemctl daemon-reload
```
