"""Core RAG components."""

from perfect_rag.core.embedding import EmbeddingService, get_embedding_service
from perfect_rag.core.resilience import (
    CircuitBreaker,
    CircuitState,
    RetryConfig,
    RetryWithBackoff,
    ResilientService,
    with_retry,
    with_circuit_breaker,
)
from perfect_rag.core.profiling import (
    PerformanceProfiler,
    MetricsCollector,
    ProfilerMiddleware,
    PrometheusExporter,
    MetricType,
    get_profiler,
    get_metrics_collector,
)
from perfect_rag.core.personalization import (
    UserProfile,
    UserInteraction,
    PersonalizationEngine,
    PreferenceTracker,
    PersonalizedReranker,
    ProfileStore,
)

__all__ = [
    # Embedding
    "EmbeddingService",
    "get_embedding_service",
    # Resilience
    "CircuitBreaker",
    "CircuitState",
    "RetryConfig",
    "RetryWithBackoff",
    "ResilientService",
    "with_retry",
    "with_circuit_breaker",
    # Profiling
    "PerformanceProfiler",
    "MetricsCollector",
    "ProfilerMiddleware",
    "PrometheusExporter",
    "MetricType",
    "get_profiler",
    "get_metrics_collector",
    # Personalization
    "UserProfile",
    "UserInteraction",
    "PersonalizationEngine",
    "PreferenceTracker",
    "PersonalizedReranker",
    "ProfileStore",
]
