"""Performance profiling and metrics collection.

Provides detailed performance profiling, metrics collection,
and Prometheus-compatible metrics export for RAG systems.
"""

import asyncio
import gc
import os
import sys
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, TypeVar

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class MetricType(str, Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricSample:
    """A single metric sample."""

    value: float
    timestamp: datetime
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class HistogramBucket:
    """A histogram bucket."""

    le: float  # Less than or equal
    count: int = 0


@dataclass
class Metric:
    """A metric with samples and metadata."""

    name: str
    metric_type: MetricType
    help_text: str = ""
    labels: list[str] = field(default_factory=list)

    # For counters and gauges
    value: float = 0.0
    labeled_values: dict[tuple, float] = field(default_factory=dict)

    # For histograms
    buckets: list[HistogramBucket] = field(default_factory=list)
    histogram_sum: float = 0.0
    histogram_count: int = 0
    labeled_histograms: dict[tuple, dict[str, Any]] = field(default_factory=dict)

    # For summaries
    quantiles: dict[float, float] = field(default_factory=dict)
    samples: list[float] = field(default_factory=list)
    max_samples: int = 1000


@dataclass
class TimingResult:
    """Result of a timing measurement."""

    name: str
    duration_ms: float
    start_time: datetime
    end_time: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfilerSpan:
    """A profiler span for tracking nested operations."""

    name: str
    start_time: float
    end_time: float | None = None
    parent: "ProfilerSpan | None" = None
    children: list["ProfilerSpan"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "children": [c.to_dict() for c in self.children],
            "metadata": self.metadata,
        }


class MetricsCollector:
    """Collect and aggregate metrics.

    Provides:
    - Counter, gauge, histogram, and summary metrics
    - Labels support
    - Thread-safe operations
    """

    # Default histogram buckets (in milliseconds)
    DEFAULT_BUCKETS = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._metrics: dict[str, Metric] = {}
        self._lock = threading.Lock()

    def counter(
        self,
        name: str,
        help_text: str = "",
        labels: list[str] | None = None,
    ) -> "CounterMetric":
        """Create or get a counter metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Metric(
                    name=name,
                    metric_type=MetricType.COUNTER,
                    help_text=help_text,
                    labels=labels or [],
                )
            return CounterMetric(self._metrics[name], self._lock)

    def gauge(
        self,
        name: str,
        help_text: str = "",
        labels: list[str] | None = None,
    ) -> "GaugeMetric":
        """Create or get a gauge metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Metric(
                    name=name,
                    metric_type=MetricType.GAUGE,
                    help_text=help_text,
                    labels=labels or [],
                )
            return GaugeMetric(self._metrics[name], self._lock)

    def histogram(
        self,
        name: str,
        help_text: str = "",
        labels: list[str] | None = None,
        buckets: list[float] | None = None,
    ) -> "HistogramMetric":
        """Create or get a histogram metric."""
        with self._lock:
            if name not in self._metrics:
                bucket_values = buckets or self.DEFAULT_BUCKETS
                self._metrics[name] = Metric(
                    name=name,
                    metric_type=MetricType.HISTOGRAM,
                    help_text=help_text,
                    labels=labels or [],
                    buckets=[HistogramBucket(le=b) for b in bucket_values] + [HistogramBucket(le=float("inf"))],
                )
            return HistogramMetric(self._metrics[name], self._lock)

    def summary(
        self,
        name: str,
        help_text: str = "",
        labels: list[str] | None = None,
        max_samples: int = 1000,
    ) -> "SummaryMetric":
        """Create or get a summary metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Metric(
                    name=name,
                    metric_type=MetricType.SUMMARY,
                    help_text=help_text,
                    labels=labels or [],
                    max_samples=max_samples,
                )
            return SummaryMetric(self._metrics[name], self._lock)

    def get_all_metrics(self) -> dict[str, Metric]:
        """Get all registered metrics."""
        with self._lock:
            return dict(self._metrics)

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()


class CounterMetric:
    """Counter metric wrapper."""

    def __init__(self, metric: Metric, lock: threading.Lock):
        self._metric = metric
        self._lock = lock

    def inc(self, value: float = 1, **labels) -> None:
        """Increment counter."""
        with self._lock:
            if labels:
                key = tuple(sorted(labels.items()))
                self._metric.labeled_values[key] = self._metric.labeled_values.get(key, 0) + value
            else:
                self._metric.value += value

    def labels(self, **labels) -> "LabeledCounter":
        """Get counter with labels."""
        return LabeledCounter(self, labels)


class LabeledCounter:
    """Labeled counter helper."""

    def __init__(self, counter: CounterMetric, labels: dict[str, str]):
        self._counter = counter
        self._labels = labels

    def inc(self, value: float = 1) -> None:
        """Increment counter."""
        self._counter.inc(value, **self._labels)


class GaugeMetric:
    """Gauge metric wrapper."""

    def __init__(self, metric: Metric, lock: threading.Lock):
        self._metric = metric
        self._lock = lock

    def set(self, value: float, **labels) -> None:
        """Set gauge value."""
        with self._lock:
            if labels:
                key = tuple(sorted(labels.items()))
                self._metric.labeled_values[key] = value
            else:
                self._metric.value = value

    def inc(self, value: float = 1, **labels) -> None:
        """Increment gauge."""
        with self._lock:
            if labels:
                key = tuple(sorted(labels.items()))
                self._metric.labeled_values[key] = self._metric.labeled_values.get(key, 0) + value
            else:
                self._metric.value += value

    def dec(self, value: float = 1, **labels) -> None:
        """Decrement gauge."""
        self.inc(-value, **labels)

    def labels(self, **labels) -> "LabeledGauge":
        """Get gauge with labels."""
        return LabeledGauge(self, labels)


class LabeledGauge:
    """Labeled gauge helper."""

    def __init__(self, gauge: GaugeMetric, labels: dict[str, str]):
        self._gauge = gauge
        self._labels = labels

    def set(self, value: float) -> None:
        """Set gauge value."""
        self._gauge.set(value, **self._labels)

    def inc(self, value: float = 1) -> None:
        """Increment gauge."""
        self._gauge.inc(value, **self._labels)

    def dec(self, value: float = 1) -> None:
        """Decrement gauge."""
        self._gauge.dec(value, **self._labels)


class HistogramMetric:
    """Histogram metric wrapper."""

    def __init__(self, metric: Metric, lock: threading.Lock):
        self._metric = metric
        self._lock = lock

    def observe(self, value: float, **labels) -> None:
        """Observe a value."""
        with self._lock:
            if labels:
                key = tuple(sorted(labels.items()))
                if key not in self._metric.labeled_histograms:
                    self._metric.labeled_histograms[key] = {
                        "buckets": [HistogramBucket(le=b.le) for b in self._metric.buckets],
                        "sum": 0.0,
                        "count": 0,
                    }
                hist = self._metric.labeled_histograms[key]
                for bucket in hist["buckets"]:
                    if value <= bucket.le:
                        bucket.count += 1
                hist["sum"] += value
                hist["count"] += 1
            else:
                for bucket in self._metric.buckets:
                    if value <= bucket.le:
                        bucket.count += 1
                self._metric.histogram_sum += value
                self._metric.histogram_count += 1

    def labels(self, **labels) -> "LabeledHistogram":
        """Get histogram with labels."""
        return LabeledHistogram(self, labels)

    @contextmanager
    def time(self, **labels):
        """Time a block and observe duration in milliseconds."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.observe(duration_ms, **labels)


class LabeledHistogram:
    """Labeled histogram helper."""

    def __init__(self, histogram: HistogramMetric, labels: dict[str, str]):
        self._histogram = histogram
        self._labels = labels

    def observe(self, value: float) -> None:
        """Observe a value."""
        self._histogram.observe(value, **self._labels)

    @contextmanager
    def time(self):
        """Time a block."""
        with self._histogram.time(**self._labels):
            yield


class SummaryMetric:
    """Summary metric wrapper."""

    def __init__(self, metric: Metric, lock: threading.Lock):
        self._metric = metric
        self._lock = lock

    def observe(self, value: float) -> None:
        """Observe a value."""
        with self._lock:
            self._metric.samples.append(value)
            if len(self._metric.samples) > self._metric.max_samples:
                self._metric.samples = self._metric.samples[-self._metric.max_samples:]

    def get_quantile(self, q: float) -> float:
        """Get quantile value."""
        with self._lock:
            if not self._metric.samples:
                return 0.0
            sorted_samples = sorted(self._metric.samples)
            idx = int(q * len(sorted_samples))
            return sorted_samples[min(idx, len(sorted_samples) - 1)]


class PerformanceProfiler:
    """Track latencies per component.

    Provides:
    - Hierarchical timing with spans
    - Automatic aggregation
    - Component-level latency tracking
    """

    def __init__(
        self,
        metrics_collector: MetricsCollector | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.metrics = metrics_collector or MetricsCollector(settings)
        self._current_span: dict[int, ProfilerSpan | None] = {}  # Per-task spans
        self._results: list[TimingResult] = []
        self._lock = asyncio.Lock()

        # Pre-create common metrics
        self._latency_histogram = self.metrics.histogram(
            "rag_component_latency_ms",
            "Latency of RAG components in milliseconds",
            labels=["component"],
        )
        self._request_counter = self.metrics.counter(
            "rag_requests_total",
            "Total number of RAG requests",
            labels=["endpoint", "status"],
        )
        self._active_requests = self.metrics.gauge(
            "rag_active_requests",
            "Number of active requests",
        )

    @asynccontextmanager
    async def span(self, name: str, **metadata):
        """Create a profiler span for async code."""
        task_id = id(asyncio.current_task())

        span = ProfilerSpan(
            name=name,
            start_time=time.perf_counter(),
            parent=self._current_span.get(task_id),
            metadata=metadata,
        )

        if span.parent:
            span.parent.children.append(span)

        self._current_span[task_id] = span

        try:
            yield span
        finally:
            span.end_time = time.perf_counter()
            self._current_span[task_id] = span.parent

            # Record metric
            self._latency_histogram.observe(span.duration_ms, component=name)

            # Store result
            result = TimingResult(
                name=name,
                duration_ms=span.duration_ms,
                start_time=datetime.utcnow() - timedelta(milliseconds=span.duration_ms),
                end_time=datetime.utcnow(),
                metadata=metadata,
            )
            async with self._lock:
                self._results.append(result)
                if len(self._results) > 10000:
                    self._results = self._results[-10000:]

    @contextmanager
    def span_sync(self, name: str, **metadata):
        """Create a profiler span for sync code."""
        span = ProfilerSpan(
            name=name,
            start_time=time.perf_counter(),
            metadata=metadata,
        )

        try:
            yield span
        finally:
            span.end_time = time.perf_counter()
            self._latency_histogram.observe(span.duration_ms, component=name)

    def profile(self, name: str | None = None):
        """Decorator to profile a function."""
        def decorator(func: F) -> F:
            func_name = name or func.__name__

            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    async with self.span(func_name):
                        return await func(*args, **kwargs)
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with self.span_sync(func_name):
                        return func(*args, **kwargs)
                return sync_wrapper

        return decorator

    async def get_component_stats(
        self,
        component: str | None = None,
        window_minutes: int = 5,
    ) -> dict[str, Any]:
        """Get statistics for components."""
        async with self._lock:
            cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
            recent = [r for r in self._results if r.start_time >= cutoff]

            if component:
                recent = [r for r in recent if r.name == component]

            if not recent:
                return {"count": 0}

            durations = [r.duration_ms for r in recent]
            durations.sort()

            return {
                "count": len(recent),
                "mean_ms": sum(durations) / len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "p50_ms": durations[len(durations) // 2],
                "p95_ms": durations[int(len(durations) * 0.95)],
                "p99_ms": durations[int(len(durations) * 0.99)],
            }

    async def get_all_stats(self, window_minutes: int = 5) -> dict[str, Any]:
        """Get statistics for all components."""
        async with self._lock:
            cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
            recent = [r for r in self._results if r.start_time >= cutoff]

            # Group by component
            by_component: dict[str, list[float]] = defaultdict(list)
            for r in recent:
                by_component[r.name].append(r.duration_ms)

            stats = {}
            for component, durations in by_component.items():
                durations.sort()
                stats[component] = {
                    "count": len(durations),
                    "mean_ms": sum(durations) / len(durations),
                    "p50_ms": durations[len(durations) // 2],
                    "p95_ms": durations[int(len(durations) * 0.95)],
                    "p99_ms": durations[int(len(durations) * 0.99)],
                }

            return stats


class ProfilerMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for request profiling.

    Adds timing headers and collects request metrics.
    """

    def __init__(
        self,
        app,
        profiler: PerformanceProfiler | None = None,
        exclude_paths: list[str] | None = None,
    ):
        super().__init__(app)
        self.profiler = profiler or PerformanceProfiler()
        self.exclude_paths = exclude_paths or ["/metrics", "/health"]

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with profiling."""
        # Skip excluded paths
        if any(request.url.path.startswith(p) for p in self.exclude_paths):
            return await call_next(request)

        self.profiler._active_requests.inc()
        start = time.perf_counter()

        try:
            async with self.profiler.span(
                f"http_{request.method.lower()}",
                path=request.url.path,
                method=request.method,
            ):
                response = await call_next(request)

            duration_ms = (time.perf_counter() - start) * 1000

            # Add timing header
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

            # Record metrics
            self.profiler._request_counter.inc(
                endpoint=request.url.path,
                status=str(response.status_code),
            )

            return response

        finally:
            self.profiler._active_requests.dec()


class PrometheusExporter:
    """Export metrics in Prometheus format.

    Generates Prometheus-compatible text format for scraping.
    """

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        settings: Settings | None = None,
    ):
        self.metrics = metrics_collector
        self.settings = settings or get_settings()

    def export(self) -> str:
        """Export all metrics in Prometheus format."""
        lines: list[str] = []
        metrics = self.metrics.get_all_metrics()

        for name, metric in metrics.items():
            # Add help text
            if metric.help_text:
                lines.append(f"# HELP {name} {metric.help_text}")
            lines.append(f"# TYPE {name} {metric.metric_type.value}")

            if metric.metric_type == MetricType.COUNTER:
                lines.extend(self._export_counter(name, metric))
            elif metric.metric_type == MetricType.GAUGE:
                lines.extend(self._export_gauge(name, metric))
            elif metric.metric_type == MetricType.HISTOGRAM:
                lines.extend(self._export_histogram(name, metric))
            elif metric.metric_type == MetricType.SUMMARY:
                lines.extend(self._export_summary(name, metric))

            lines.append("")  # Empty line between metrics

        # Add process metrics
        lines.extend(self._export_process_metrics())

        return "\n".join(lines)

    def _export_counter(self, name: str, metric: Metric) -> list[str]:
        """Export counter metric."""
        lines = []

        # Base value
        if metric.value > 0 or not metric.labeled_values:
            lines.append(f"{name} {metric.value}")

        # Labeled values
        for labels, value in metric.labeled_values.items():
            label_str = self._format_labels(dict(labels))
            lines.append(f"{name}{label_str} {value}")

        return lines

    def _export_gauge(self, name: str, metric: Metric) -> list[str]:
        """Export gauge metric."""
        lines = []

        if not metric.labeled_values:
            lines.append(f"{name} {metric.value}")

        for labels, value in metric.labeled_values.items():
            label_str = self._format_labels(dict(labels))
            lines.append(f"{name}{label_str} {value}")

        return lines

    def _export_histogram(self, name: str, metric: Metric) -> list[str]:
        """Export histogram metric."""
        lines = []

        # Base histogram
        if metric.histogram_count > 0:
            cumulative = 0
            for bucket in metric.buckets:
                cumulative += bucket.count
                le = "+Inf" if bucket.le == float("inf") else str(bucket.le)
                lines.append(f'{name}_bucket{{le="{le}"}} {cumulative}')
            lines.append(f"{name}_sum {metric.histogram_sum}")
            lines.append(f"{name}_count {metric.histogram_count}")

        # Labeled histograms
        for labels, hist in metric.labeled_histograms.items():
            label_dict = dict(labels)
            cumulative = 0
            for bucket in hist["buckets"]:
                cumulative += bucket.count
                le = "+Inf" if bucket.le == float("inf") else str(bucket.le)
                label_str = self._format_labels({**label_dict, "le": le})
                lines.append(f"{name}_bucket{label_str} {cumulative}")
            label_str = self._format_labels(label_dict)
            lines.append(f"{name}_sum{label_str} {hist['sum']}")
            lines.append(f"{name}_count{label_str} {hist['count']}")

        return lines

    def _export_summary(self, name: str, metric: Metric) -> list[str]:
        """Export summary metric."""
        lines = []

        if metric.samples:
            sorted_samples = sorted(metric.samples)
            for q in [0.5, 0.9, 0.99]:
                idx = int(q * len(sorted_samples))
                value = sorted_samples[min(idx, len(sorted_samples) - 1)]
                lines.append(f'{name}{{quantile="{q}"}} {value}')
            lines.append(f"{name}_sum {sum(metric.samples)}")
            lines.append(f"{name}_count {len(metric.samples)}")

        return lines

    def _format_labels(self, labels: dict[str, Any]) -> str:
        """Format labels for Prometheus."""
        if not labels:
            return ""
        parts = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ",".join(parts) + "}"

    def _export_process_metrics(self) -> list[str]:
        """Export process-level metrics."""
        lines = [
            "# HELP process_cpu_seconds_total Total user and system CPU time spent in seconds",
            "# TYPE process_cpu_seconds_total counter",
        ]

        try:
            import resource
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            cpu_time = rusage.ru_utime + rusage.ru_stime
            lines.append(f"process_cpu_seconds_total {cpu_time}")
        except (ImportError, AttributeError):
            pass

        lines.extend([
            "",
            "# HELP process_resident_memory_bytes Resident memory size in bytes",
            "# TYPE process_resident_memory_bytes gauge",
        ])

        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            lines.append(f"process_resident_memory_bytes {mem_info.rss}")
        except ImportError:
            # Fallback for when psutil is not available
            pass

        lines.extend([
            "",
            "# HELP python_gc_objects_collected_total Objects collected during gc",
            "# TYPE python_gc_objects_collected_total counter",
        ])

        for gen, stats in enumerate(gc.get_stats()):
            lines.append(f'python_gc_objects_collected_total{{generation="{gen}"}} {stats["collected"]}')

        return lines


# Global instances
_metrics_collector: MetricsCollector | None = None
_profiler: PerformanceProfiler | None = None
_exporter: PrometheusExporter | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_profiler() -> PerformanceProfiler:
    """Get or create performance profiler."""
    global _profiler
    if _profiler is None:
        _profiler = PerformanceProfiler(get_metrics_collector())
    return _profiler


def get_prometheus_exporter() -> PrometheusExporter:
    """Get or create Prometheus exporter."""
    global _exporter
    if _exporter is None:
        _exporter = PrometheusExporter(get_metrics_collector())
    return _exporter
