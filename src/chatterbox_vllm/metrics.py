"""Prometheus metric definitions for Chatterbox TTS.

All metrics are module-level singletons, auto-registered with the default
prometheus_client registry.  Import the objects you need and call
.observe() / .inc() / .dec() at the appropriate points.
"""

from typing import Callable, Optional

from prometheus_client import Counter, Gauge, Histogram
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily, REGISTRY
from prometheus_client.registry import Collector


# ---------------------------------------------------------------------------
# Request-level metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "tts_requests_total",
    "Total TTS requests",
    ["status"],
)

REQUEST_DURATION = Histogram(
    "tts_request_duration_seconds",
    "End-to-end TTS request duration",
)

TTFB_HISTOGRAM = Histogram(
    "tts_time_to_first_byte_seconds",
    "Time from request start to first audio byte sent",
)

ACTIVE_REQUESTS = Gauge(
    "tts_active_requests",
    "Number of TTS requests currently in progress",
)

# ---------------------------------------------------------------------------
# T3 token generation metrics
# ---------------------------------------------------------------------------
T3_TOKENS_GENERATED = Counter(
    "tts_t3_tokens_generated_total",
    "Total T3 speech tokens generated",
)

T3_TIME_TO_FIRST_TOKEN = Histogram(
    "tts_t3_time_to_first_token_seconds",
    "Time to first T3 token",
)

T3_TOKENS_PER_SECOND = Histogram(
    "tts_t3_tokens_per_second",
    "T3 token generation throughput (tokens/sec)",
)

# ---------------------------------------------------------------------------
# Vocoding (per-chunk) metrics
# ---------------------------------------------------------------------------
VOCODING_FLOW_DURATION = Histogram(
    "tts_vocoding_flow_duration_seconds",
    "CFM flow step duration per vocoder chunk",
)

VOCODING_HIFIGAN_DURATION = Histogram(
    "tts_vocoding_hifigan_duration_seconds",
    "HiFiGAN duration per vocoder chunk",
)

VOCODING_CHUNK_LATENCY = Histogram(
    "tts_vocoding_chunk_latency_seconds",
    "Total chunk latency (token prep + vocode + post-process)",
)


# ---------------------------------------------------------------------------
# VocoderBatcher custom collector
# ---------------------------------------------------------------------------
class VocoderBatcherCollector(Collector):
    """Scrape-time collector that reads live stats from the VocoderBatcher."""

    def __init__(self, batcher_getter: Callable):
        self._batcher_getter = batcher_getter

    def collect(self):
        batcher = self._batcher_getter()
        if batcher is None:
            return

        stats = batcher.get_stats()

        c = CounterMetricFamily(
            "tts_batcher_batches_total",
            "Total vocoder batches processed",
        )
        c.add_metric([], stats["total_batches"])
        yield c

        c = CounterMetricFamily(
            "tts_batcher_timeout_fires_total",
            "Batches fired due to timeout",
        )
        c.add_metric([], stats["timeout_fires"])
        yield c

        c = CounterMetricFamily(
            "tts_batcher_early_fires_total",
            "Batches fired early (80% capacity)",
        )
        c.add_metric([], stats["early_fires"])
        yield c

        c = CounterMetricFamily(
            "tts_batcher_fast_path_fires_total",
            "Batches fired via fast path (single user)",
        )
        c.add_metric([], stats["fast_path_fires"])
        yield c

        g = GaugeMetricFamily(
            "tts_batcher_avg_batch_size",
            "Average vocoder batch size",
        )
        g.add_metric([], stats["avg_batch_size"])
        yield g

        g = GaugeMetricFamily(
            "tts_batcher_avg_wait_seconds",
            "Average queue wait time in seconds",
        )
        g.add_metric([], stats["avg_wait_ms"] / 1000.0)
        yield g


def register_batcher_collector(batcher_getter: Callable) -> None:
    """Register the VocoderBatcher custom collector with the default registry."""
    REGISTRY.register(VocoderBatcherCollector(batcher_getter))
