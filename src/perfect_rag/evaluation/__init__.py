"""
Evaluation module for Perfect RAG system.

Provides RAGAS (Retrieval Augmented Generation Assessment) metrics,
traditional information retrieval metrics, benchmarks, and evaluation datasets.
"""

from perfect_rag.evaluation.ragas_metrics import (
    RAGASResult,
    RAGASEvaluator,
    RetrievalMetrics,
)
from perfect_rag.evaluation.benchmarks import (
    BenchmarkRunner,
    BenchmarkSuite,
    BenchmarkTest,
    LatencyReport,
    LatencyStats,
    PercentileCalculator,
    ComponentBenchmark,
    ComponentBenchmarkResult,
    get_benchmark_runner,
)
from perfect_rag.evaluation.dataset import (
    EvaluationSample,
    EvaluationDataset,
    DatasetGenerator,
    GoldenAnswer,
    GoldenAnswerStore,
    DatasetLoader,
    DatasetSplitter,
    DatasetSplit,
)

__all__ = [
    # RAGAS
    "RAGASResult",
    "RAGASEvaluator",
    "RetrievalMetrics",
    # Benchmarks
    "BenchmarkRunner",
    "BenchmarkSuite",
    "BenchmarkTest",
    "LatencyReport",
    "LatencyStats",
    "PercentileCalculator",
    "ComponentBenchmark",
    "ComponentBenchmarkResult",
    "get_benchmark_runner",
    # Dataset
    "EvaluationSample",
    "EvaluationDataset",
    "DatasetGenerator",
    "GoldenAnswer",
    "GoldenAnswerStore",
    "DatasetLoader",
    "DatasetSplitter",
    "DatasetSplit",
]
