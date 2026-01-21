"""Latency benchmarks for RAG components.

Provides comprehensive benchmarking capabilities for measuring
latency across all RAG pipeline components.
"""

import asyncio
import gc
import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import structlog

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class BenchmarkPhase(str, Enum):
    """Phases in the RAG pipeline."""

    EMBEDDING = "embedding"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    LLM_GENERATION = "llm_generation"
    TOTAL_E2E = "total_e2e"
    QUERY_PARSING = "query_parsing"
    VECTOR_SEARCH = "vector_search"
    GRAPH_EXPANSION = "graph_expansion"
    CONTEXT_BUILDING = "context_building"
    CACHE_LOOKUP = "cache_lookup"


@dataclass
class BenchmarkSample:
    """A single benchmark sample."""

    phase: BenchmarkPhase
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    input_size: int | None = None  # Tokens, chars, or items
    output_size: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Latency statistics."""

    count: int
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p75_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    p999_ms: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p75_ms": round(self.p75_ms, 3),
            "p90_ms": round(self.p90_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "p999_ms": round(self.p999_ms, 3),
        }


class PercentileCalculator:
    """Calculate percentile latencies from samples."""

    @staticmethod
    def calculate_percentile(sorted_values: list[float], percentile: float) -> float:
        """Calculate a specific percentile from sorted values."""
        if not sorted_values:
            return 0.0

        n = len(sorted_values)
        k = (n - 1) * percentile / 100.0
        f = int(k)
        c = f + 1 if f + 1 < n else f

        if f == c:
            return sorted_values[f]

        d0 = sorted_values[f] * (c - k)
        d1 = sorted_values[c] * (k - f)
        return d0 + d1

    @staticmethod
    def calculate_all_percentiles(values: list[float]) -> dict[str, float]:
        """Calculate all common percentiles."""
        if not values:
            return {
                "p50": 0.0, "p75": 0.0, "p90": 0.0,
                "p95": 0.0, "p99": 0.0, "p999": 0.0,
            }

        sorted_values = sorted(values)
        return {
            "p50": PercentileCalculator.calculate_percentile(sorted_values, 50),
            "p75": PercentileCalculator.calculate_percentile(sorted_values, 75),
            "p90": PercentileCalculator.calculate_percentile(sorted_values, 90),
            "p95": PercentileCalculator.calculate_percentile(sorted_values, 95),
            "p99": PercentileCalculator.calculate_percentile(sorted_values, 99),
            "p999": PercentileCalculator.calculate_percentile(sorted_values, 99.9),
        }

    @staticmethod
    def calculate_stats(values: list[float]) -> LatencyStats:
        """Calculate full latency statistics."""
        if not values:
            return LatencyStats(
                count=0, mean_ms=0, median_ms=0, std_ms=0,
                min_ms=0, max_ms=0,
                p50_ms=0, p75_ms=0, p90_ms=0,
                p95_ms=0, p99_ms=0, p999_ms=0,
            )

        sorted_values = sorted(values)
        percentiles = PercentileCalculator.calculate_all_percentiles(values)

        return LatencyStats(
            count=len(values),
            mean_ms=statistics.mean(values),
            median_ms=statistics.median(values),
            std_ms=statistics.stdev(values) if len(values) > 1 else 0.0,
            min_ms=min(values),
            max_ms=max(values),
            p50_ms=percentiles["p50"],
            p75_ms=percentiles["p75"],
            p90_ms=percentiles["p90"],
            p95_ms=percentiles["p95"],
            p99_ms=percentiles["p99"],
            p999_ms=percentiles["p999"],
        )


@dataclass
class ComponentBenchmarkResult:
    """Result of benchmarking a component."""

    component: str
    phase: BenchmarkPhase
    stats: LatencyStats
    samples: list[BenchmarkSample]
    throughput_per_second: float
    input_stats: LatencyStats | None = None  # Stats on input sizes
    output_stats: LatencyStats | None = None  # Stats on output sizes
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "phase": self.phase.value,
            "latency_stats": self.stats.to_dict(),
            "throughput_per_second": round(self.throughput_per_second, 2),
            "sample_count": len(self.samples),
            "metadata": self.metadata,
        }


class ComponentBenchmark:
    """Benchmark individual RAG components."""

    def __init__(
        self,
        component_name: str,
        phase: BenchmarkPhase,
        settings: Settings | None = None,
    ):
        self.component_name = component_name
        self.phase = phase
        self.settings = settings or get_settings()
        self.samples: list[BenchmarkSample] = []
        self._start_time: float | None = None

    def start(self) -> None:
        """Start timing."""
        self._start_time = time.perf_counter()

    def stop(
        self,
        input_size: int | None = None,
        output_size: int | None = None,
        **metadata,
    ) -> BenchmarkSample:
        """Stop timing and record sample."""
        if self._start_time is None:
            raise ValueError("Benchmark not started")

        latency_ms = (time.perf_counter() - self._start_time) * 1000
        sample = BenchmarkSample(
            phase=self.phase,
            latency_ms=latency_ms,
            input_size=input_size,
            output_size=output_size,
            metadata=metadata,
        )
        self.samples.append(sample)
        self._start_time = None
        return sample

    async def measure(
        self,
        func: Callable,
        *args,
        input_size: int | None = None,
        **kwargs,
    ) -> tuple[Any, BenchmarkSample]:
        """Measure a function call."""
        self.start()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Try to determine output size
            output_size = None
            if isinstance(result, (list, tuple)):
                output_size = len(result)
            elif isinstance(result, str):
                output_size = len(result)
            elif isinstance(result, dict) and "chunks" in result:
                output_size = len(result["chunks"])

            sample = self.stop(input_size=input_size, output_size=output_size)
            return result, sample

        except Exception:
            self.stop(input_size=input_size)
            raise

    def get_result(self) -> ComponentBenchmarkResult:
        """Get benchmark result with statistics."""
        latencies = [s.latency_ms for s in self.samples]
        stats = PercentileCalculator.calculate_stats(latencies)

        # Calculate throughput
        if latencies:
            total_time_s = sum(latencies) / 1000
            throughput = len(latencies) / total_time_s if total_time_s > 0 else 0
        else:
            throughput = 0

        # Input size stats
        input_sizes = [s.input_size for s in self.samples if s.input_size is not None]
        input_stats = PercentileCalculator.calculate_stats(input_sizes) if input_sizes else None

        # Output size stats
        output_sizes = [s.output_size for s in self.samples if s.output_size is not None]
        output_stats = PercentileCalculator.calculate_stats(output_sizes) if output_sizes else None

        return ComponentBenchmarkResult(
            component=self.component_name,
            phase=self.phase,
            stats=stats,
            samples=self.samples,
            throughput_per_second=throughput,
            input_stats=input_stats,
            output_stats=output_stats,
        )

    def clear(self) -> None:
        """Clear all samples."""
        self.samples.clear()


@dataclass
class BenchmarkTest:
    """A benchmark test definition."""

    name: str
    description: str
    phase: BenchmarkPhase
    func: Callable
    inputs: list[Any]
    iterations: int = 10
    warmup_iterations: int = 2
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkSuite:
    """Collection of benchmark tests."""

    def __init__(
        self,
        name: str,
        description: str = "",
        settings: Settings | None = None,
    ):
        self.name = name
        self.description = description
        self.settings = settings or get_settings()
        self.tests: list[BenchmarkTest] = []

    def add_test(
        self,
        name: str,
        func: Callable,
        inputs: list[Any],
        phase: BenchmarkPhase,
        description: str = "",
        iterations: int = 10,
        warmup_iterations: int = 2,
        **metadata,
    ) -> None:
        """Add a benchmark test."""
        test = BenchmarkTest(
            name=name,
            description=description,
            phase=phase,
            func=func,
            inputs=inputs,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            metadata=metadata,
        )
        self.tests.append(test)

    def add_embedding_test(
        self,
        embedding_service: Any,
        texts: list[str],
        **kwargs,
    ) -> None:
        """Add embedding benchmark test."""
        self.add_test(
            name="embedding",
            func=embedding_service.embed_texts,
            inputs=[texts],
            phase=BenchmarkPhase.EMBEDDING,
            description="Batch embedding generation",
            **kwargs,
        )

    def add_retrieval_test(
        self,
        retrieval_func: Callable,
        queries: list[str],
        **kwargs,
    ) -> None:
        """Add retrieval benchmark test."""
        for i, query in enumerate(queries):
            self.add_test(
                name=f"retrieval_{i}",
                func=retrieval_func,
                inputs=[query],
                phase=BenchmarkPhase.RETRIEVAL,
                description=f"Retrieval for: {query[:50]}...",
                **kwargs,
            )

    def add_reranking_test(
        self,
        reranker: Any,
        query_docs_pairs: list[tuple[str, list[str]]],
        **kwargs,
    ) -> None:
        """Add reranking benchmark test."""
        for i, (query, docs) in enumerate(query_docs_pairs):
            self.add_test(
                name=f"reranking_{i}",
                func=reranker.rerank,
                inputs=[query, docs],
                phase=BenchmarkPhase.RERANKING,
                description=f"Reranking {len(docs)} docs",
                **kwargs,
            )


@dataclass
class LatencyReport:
    """Complete latency benchmark report."""

    suite_name: str
    timestamp: datetime
    duration_seconds: float
    results: dict[str, ComponentBenchmarkResult]
    summary: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suite_name": self.suite_name,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": round(self.duration_seconds, 2),
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "summary": self.summary,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str | Path) -> None:
        """Save report to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())
        logger.info("Benchmark report saved", path=str(path))


class BenchmarkRunner:
    """Run latency benchmarks.

    Executes benchmark tests and generates reports.
    """

    def __init__(
        self,
        suite: BenchmarkSuite | None = None,
        settings: Settings | None = None,
    ):
        self.suite = suite
        self.settings = settings or get_settings()
        self.results: dict[str, ComponentBenchmarkResult] = {}

    async def run_suite(
        self,
        suite: BenchmarkSuite | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> LatencyReport:
        """Run a benchmark suite.

        Args:
            suite: Benchmark suite to run (or use suite from init)
            progress_callback: Optional callback for progress updates

        Returns:
            Complete latency report
        """
        suite = suite or self.suite
        if not suite:
            raise ValueError("No benchmark suite provided")

        start_time = time.perf_counter()
        total_tests = len(suite.tests)
        self.results.clear()

        logger.info(
            "Starting benchmark suite",
            suite=suite.name,
            test_count=total_tests,
        )

        for i, test in enumerate(suite.tests):
            if progress_callback:
                progress_callback(test.name, i + 1, total_tests)

            result = await self._run_test(test)
            self.results[test.name] = result

            logger.info(
                "Benchmark test complete",
                test=test.name,
                mean_ms=result.stats.mean_ms,
                p95_ms=result.stats.p95_ms,
            )

        duration = time.perf_counter() - start_time

        # Generate summary
        summary = self._generate_summary()

        report = LatencyReport(
            suite_name=suite.name,
            timestamp=datetime.utcnow(),
            duration_seconds=duration,
            results=self.results,
            summary=summary,
            metadata={
                "description": suite.description,
                "test_count": total_tests,
            },
        )

        logger.info(
            "Benchmark suite complete",
            suite=suite.name,
            duration_seconds=round(duration, 2),
        )

        return report

    async def _run_test(self, test: BenchmarkTest) -> ComponentBenchmarkResult:
        """Run a single benchmark test."""
        benchmark = ComponentBenchmark(test.name, test.phase, self.settings)

        # Warmup
        for _ in range(test.warmup_iterations):
            try:
                if asyncio.iscoroutinefunction(test.func):
                    await test.func(*test.inputs)
                else:
                    test.func(*test.inputs)
            except Exception as e:
                logger.warning("Warmup iteration failed", test=test.name, error=str(e))

        # Force garbage collection between tests
        gc.collect()

        # Actual benchmark
        for _ in range(test.iterations):
            try:
                input_size = None
                if test.inputs and isinstance(test.inputs[0], (str, list)):
                    input_size = len(test.inputs[0])

                benchmark.start()
                if asyncio.iscoroutinefunction(test.func):
                    result = await test.func(*test.inputs)
                else:
                    result = test.func(*test.inputs)

                output_size = None
                if isinstance(result, (list, tuple)):
                    output_size = len(result)

                benchmark.stop(input_size=input_size, output_size=output_size)

            except Exception as e:
                logger.error(
                    "Benchmark iteration failed",
                    test=test.name,
                    error=str(e),
                )
                benchmark._start_time = None  # Reset on error

        return benchmark.get_result()

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            "by_phase": {},
            "overall": {},
        }

        # Aggregate by phase
        phase_latencies: dict[BenchmarkPhase, list[float]] = {}
        for result in self.results.values():
            if result.phase not in phase_latencies:
                phase_latencies[result.phase] = []
            phase_latencies[result.phase].extend(
                [s.latency_ms for s in result.samples]
            )

        for phase, latencies in phase_latencies.items():
            stats = PercentileCalculator.calculate_stats(latencies)
            summary["by_phase"][phase.value] = stats.to_dict()

        # Overall statistics
        all_latencies = []
        for result in self.results.values():
            all_latencies.extend([s.latency_ms for s in result.samples])

        if all_latencies:
            overall_stats = PercentileCalculator.calculate_stats(all_latencies)
            summary["overall"] = overall_stats.to_dict()

        return summary

    async def run_quick_benchmark(
        self,
        embedding_service: Any = None,
        retrieval_func: Callable | None = None,
        test_queries: list[str] | None = None,
        iterations: int = 5,
    ) -> LatencyReport:
        """Run a quick benchmark with minimal configuration.

        Args:
            embedding_service: Optional embedding service
            retrieval_func: Optional retrieval function
            test_queries: Test queries to use
            iterations: Number of iterations per test

        Returns:
            Latency report
        """
        test_queries = test_queries or [
            "What is machine learning?",
            "Explain neural networks",
            "How does RAG work?",
        ]

        suite = BenchmarkSuite(
            name="quick_benchmark",
            description="Quick latency benchmark",
        )

        # Add embedding test if service provided
        if embedding_service:
            suite.add_embedding_test(
                embedding_service,
                test_queries,
                iterations=iterations,
            )

        # Add retrieval test if function provided
        if retrieval_func:
            suite.add_retrieval_test(
                retrieval_func,
                test_queries,
                iterations=iterations,
            )

        return await self.run_suite(suite)


class ContinuousBenchmark:
    """Run continuous benchmarks for monitoring."""

    def __init__(
        self,
        runner: BenchmarkRunner,
        interval_seconds: int = 60,
        settings: Settings | None = None,
    ):
        self.runner = runner
        self.interval = interval_seconds
        self.settings = settings or get_settings()
        self._running = False
        self._task: asyncio.Task | None = None
        self._history: list[LatencyReport] = []
        self._max_history = 100

    async def start(self, suite: BenchmarkSuite) -> None:
        """Start continuous benchmarking."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop(suite))
        logger.info("Continuous benchmark started", interval=self.interval)

    async def stop(self) -> None:
        """Stop continuous benchmarking."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Continuous benchmark stopped")

    async def _run_loop(self, suite: BenchmarkSuite) -> None:
        """Main benchmark loop."""
        while self._running:
            try:
                report = await self.runner.run_suite(suite)
                self._history.append(report)

                # Trim history
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history:]

            except Exception as e:
                logger.error("Continuous benchmark iteration failed", error=str(e))

            await asyncio.sleep(self.interval)

    def get_history(
        self,
        limit: int | None = None,
    ) -> list[LatencyReport]:
        """Get benchmark history."""
        if limit:
            return self._history[-limit:]
        return list(self._history)

    def get_trend(self, phase: BenchmarkPhase | None = None) -> dict[str, Any]:
        """Get latency trend over time."""
        if not self._history:
            return {"error": "No data"}

        trend = {
            "timestamps": [],
            "mean_ms": [],
            "p95_ms": [],
            "p99_ms": [],
        }

        for report in self._history:
            trend["timestamps"].append(report.timestamp.isoformat())

            if phase:
                phase_data = report.summary.get("by_phase", {}).get(phase.value, {})
            else:
                phase_data = report.summary.get("overall", {})

            trend["mean_ms"].append(phase_data.get("mean_ms", 0))
            trend["p95_ms"].append(phase_data.get("p95_ms", 0))
            trend["p99_ms"].append(phase_data.get("p99_ms", 0))

        return trend


# Module-level singleton
_benchmark_runner: BenchmarkRunner | None = None


def get_benchmark_runner() -> BenchmarkRunner:
    """Get or create benchmark runner."""
    global _benchmark_runner
    if _benchmark_runner is None:
        _benchmark_runner = BenchmarkRunner()
    return _benchmark_runner
