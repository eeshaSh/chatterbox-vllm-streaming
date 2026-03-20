#!/usr/bin/env bash
# Setup script for Prometheus + Grafana on Linux.
# Run as root or with sudo.
set -euo pipefail

PROMETHEUS_VERSION="3.2.1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------------------------------------
# Prompt for TTS server address
# -----------------------------------------------------------------------
TTS_TARGET="${1:-}"
if [ -z "$TTS_TARGET" ]; then
    read -rp "TTS server address [localhost:4123]: " TTS_TARGET
    TTS_TARGET="${TTS_TARGET:-localhost:4123}"
fi
echo "Will scrape metrics from: ${TTS_TARGET}/metrics"

# -----------------------------------------------------------------------
# Detect architecture
# -----------------------------------------------------------------------
ARCH=$(uname -m)
case "$ARCH" in
    x86_64)  PROM_ARCH="amd64" ;;
    aarch64) PROM_ARCH="arm64" ;;
    *)       echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

# -----------------------------------------------------------------------
# Install Prometheus
# -----------------------------------------------------------------------
install_prometheus() {
    if command -v prometheus &>/dev/null; then
        echo "Prometheus already installed: $(prometheus --version 2>&1 | head -1)"
        echo "Skipping download. To reinstall, remove /usr/local/bin/prometheus first."
    else
        echo "Installing Prometheus ${PROMETHEUS_VERSION} (${PROM_ARCH})..."
        local tarball="prometheus-${PROMETHEUS_VERSION}.linux-${PROM_ARCH}.tar.gz"
        local url="https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/${tarball}"

        cd /tmp
        curl -fsSL -O "$url"
        tar xzf "$tarball"
        cp "prometheus-${PROMETHEUS_VERSION}.linux-${PROM_ARCH}/prometheus" /usr/local/bin/
        cp "prometheus-${PROMETHEUS_VERSION}.linux-${PROM_ARCH}/promtool"   /usr/local/bin/
        rm -rf "prometheus-${PROMETHEUS_VERSION}.linux-${PROM_ARCH}" "$tarball"
        echo "Prometheus installed to /usr/local/bin/prometheus"
    fi

    # Create data directory
    mkdir -p /var/lib/prometheus

    # Copy config, substituting the TTS target address
    mkdir -p /etc/prometheus
    sed "s|TTS_TARGET_PLACEHOLDER|${TTS_TARGET}|g" \
        "${SCRIPT_DIR}/prometheus.yml" > /etc/prometheus/prometheus.yml
    echo "Prometheus config written to /etc/prometheus/prometheus.yml (target: ${TTS_TARGET})"

    # Create systemd service
    cat > /etc/systemd/system/prometheus.service << 'EOF'
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

    systemctl daemon-reload
    systemctl enable prometheus
    systemctl start prometheus
    echo "Prometheus service started on :9090"
}

# -----------------------------------------------------------------------
# Install Grafana
# -----------------------------------------------------------------------
install_grafana() {
    if command -v grafana-server &>/dev/null; then
        echo "Grafana already installed: $(grafana-server -v 2>&1 | head -1)"
        echo "Skipping install. To reinstall, remove the package first."
    else
        echo "Installing Grafana (OSS)..."
        # Add Grafana APT repo
        apt-get install -y apt-transport-https software-properties-common curl gnupg
        curl -fsSL https://apt.grafana.com/gpg.key | gpg --dearmor -o /usr/share/keyrings/grafana-archive-keyring.gpg
        echo "deb [signed-by=/usr/share/keyrings/grafana-archive-keyring.gpg] https://apt.grafana.com stable main" \
            > /etc/apt/sources.list.d/grafana.list
        apt-get update
        apt-get install -y grafana
        echo "Grafana installed."
    fi

    # Provision Prometheus datasource so it's available on first boot
    mkdir -p /etc/grafana/provisioning/datasources
    cp "${SCRIPT_DIR}/grafana/provisioning/datasources/prometheus.yml" \
       /etc/grafana/provisioning/datasources/prometheus.yml
    chown grafana:grafana /etc/grafana/provisioning/datasources/prometheus.yml
    chmod 644 /etc/grafana/provisioning/datasources/prometheus.yml
    echo "Grafana datasource provisioned (Prometheus at localhost:9090)"

    # Provision dashboard so it loads automatically
    mkdir -p /etc/grafana/provisioning/dashboards
    cp "${SCRIPT_DIR}/grafana/provisioning/dashboards/dashboards.yml" \
       /etc/grafana/provisioning/dashboards/dashboards.yml
    chown grafana:grafana /etc/grafana/provisioning/dashboards/dashboards.yml
    chmod 644 /etc/grafana/provisioning/dashboards/dashboards.yml

    mkdir -p /var/lib/grafana/dashboards
    cp "${SCRIPT_DIR}/grafana/dashboard.json" \
       /var/lib/grafana/dashboards/chatterbox-tts.json
    chown grafana:grafana /var/lib/grafana/dashboards/chatterbox-tts.json
    chmod 644 /var/lib/grafana/dashboards/chatterbox-tts.json
    echo "Grafana dashboard provisioned (Chatterbox TTS)"

    # Set Grafana to port 8000
    sed -i 's/;http_port = 3000/http_port = 8000/' /etc/grafana/grafana.ini

    systemctl daemon-reload
    systemctl enable grafana-server
    systemctl start grafana-server
    echo "Grafana service started on :8000 (login: admin / admin)"
}

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root (use sudo)."
    exit 1
fi

echo "=== Chatterbox TTS Monitoring Setup ==="
echo ""

install_prometheus
echo ""
install_grafana

echo ""
echo "=== Setup complete ==="
echo ""
echo "  Prometheus:  http://localhost:9090"
echo "  Grafana:     http://localhost:8000  (admin / admin)"
echo "  TTS metrics: http://localhost:4123/metrics"
echo ""
echo "Verify targets are being scraped:"
echo "  curl -s http://localhost:9090/api/v1/targets | python3 -m json.tool"
echo ""
echo "To update Prometheus config:"
echo "  Edit /etc/prometheus/prometheus.yml, then:"
echo "  curl -X POST http://localhost:9090/-/reload"
