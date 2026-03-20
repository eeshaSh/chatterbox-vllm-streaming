# Monitoring Setup (Prometheus + Grafana)

Prometheus scrapes the `/metrics` endpoint exposed by the TTS server on port 4123.
Grafana provides dashboards on top of the collected metrics.

## How It Works

This is **not** log ingestion. Prometheus makes an HTTP GET request to your TTS
server's `/metrics` endpoint every 10 seconds. The `prometheus_client` library in
the Python code exposes counters, histograms, and gauges at that endpoint.
Prometheus stores the time series data and Grafana visualizes it.

```
TTS Server (:4123/metrics)  <──── scrapes ────  Prometheus (:9090)  <──── queries ────  Grafana (:3000)
```

## Step-by-Step Setup (Ubuntu/Debian)

### 1. Clone the repo on your Linux machine

```bash
git clone <your-repo-url>
cd chatterbox-vllm-streaming
```

### 2. Install Prometheus

```bash
# Download and install the binary
PROMETHEUS_VERSION="3.2.1"
cd /tmp
curl -fsSL -O "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz"
tar xzf "prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz"
sudo cp "prometheus-${PROMETHEUS_VERSION}.linux-amd64/prometheus" /usr/local/bin/
sudo cp "prometheus-${PROMETHEUS_VERSION}.linux-amd64/promtool" /usr/local/bin/
rm -rf "prometheus-${PROMETHEUS_VERSION}.linux-amd64" "prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz"
cd -

# Verify
prometheus --version
```

### 3. Configure Prometheus

```bash
# Create config directory and data directory
sudo mkdir -p /etc/prometheus /var/lib/prometheus

# Copy the config, replacing the placeholder with your TTS server address
# Replace <TTS_IP:PORT> with your actual TTS server address (e.g. 20.163.2.63:4123)
sudo sed 's|TTS_TARGET_PLACEHOLDER|<TTS_IP:PORT>|g' \
    monitoring/prometheus.yml > /tmp/prometheus.yml
sudo mv /tmp/prometheus.yml /etc/prometheus/prometheus.yml

# Verify the config looks right
cat /etc/prometheus/prometheus.yml
```

### 4. Create Prometheus systemd service

```bash
sudo tee /etc/systemd/system/prometheus.service > /dev/null << 'EOF'
[Unit]
Description=Prometheus Monitoring
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/prometheus \
    --config.file=/etc/prometheus/prometheus.yml \
    --storage.tsdb.path=/var/lib/prometheus \
    --storage.tsdb.retention.time=30d \
    --web.enable-lifecycle
ExecReload=/bin/kill -HUP $MAINPID
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable prometheus
sudo systemctl start prometheus
```

### 5. Install Grafana

```bash
# Import GPG key
curl -fsSL https://apt.grafana.com/gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/grafana-archive-keyring.gpg --yes

# Add repo
echo "deb [signed-by=/usr/share/keyrings/grafana-archive-keyring.gpg] https://apt.grafana.com stable main" \
    | sudo tee /etc/apt/sources.list.d/grafana.list

# Install
sudo apt-get update && sudo apt-get install -y grafana
```

### 6. Provision the Prometheus datasource in Grafana

```bash
sudo cp monitoring/grafana/provisioning/datasources/prometheus.yml \
    /etc/grafana/provisioning/datasources/prometheus.yml
```

### 7. Start Grafana

```bash
sudo systemctl enable grafana-server
sudo systemctl start grafana-server
```

### 8. Verify everything is running

```bash
# Check services
systemctl status prometheus grafana-server

# Check Prometheus is scraping your TTS server
# (target should show health: "up")
curl -s http://localhost:9090/api/v1/targets | python3 -m json.tool

# Check TTS metrics directly
curl -s http://<TTS_IP:PORT>/metrics | grep "^# TYPE tts_"
```

| Service    | URL                        | Credentials   |
|------------|----------------------------|---------------|
| Prometheus | http://localhost:9090       | --            |
| Grafana    | http://localhost:3000       | admin / admin |
| TTS metrics| http://\<TTS_IP\>:4123/metrics | --        |

## Automated Setup (alternative)

If the manual steps above work, you can also use the setup script which does
all of the above in one command:

```bash
sudo ./monitoring/setup.sh <TTS_IP:PORT>
# e.g. sudo ./monitoring/setup.sh 20.163.2.63:4123
```

If you omit the argument, it will prompt you (defaults to `localhost:4123`).

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
sudo sed 's|TTS_TARGET_PLACEHOLDER|<TTS_IP:PORT>|g' \
    monitoring/prometheus.yml > /tmp/prometheus.yml
sudo mv /tmp/prometheus.yml /etc/prometheus/prometheus.yml
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
