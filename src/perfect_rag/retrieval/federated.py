"""Federated RAG for querying multiple RAG sources.

Provides federated retrieval across multiple RAG endpoints,
with intelligent routing, result aggregation, and deduplication.
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx
import structlog

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class SourceStatus(str, Enum):
    """Status of a RAG source."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class SourceType(str, Enum):
    """Type of RAG source."""

    PERFECT_RAG = "perfect_rag"  # This system
    OPENAI_RETRIEVAL = "openai_retrieval"
    ELASTICSEARCH = "elasticsearch"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    CUSTOM = "custom"


@dataclass
class SourceConfig:
    """Configuration for a RAG source."""

    source_id: str
    name: str
    endpoint: str
    source_type: SourceType = SourceType.CUSTOM
    api_key: str | None = None
    timeout_seconds: float = 30.0
    max_results: int = 20
    weight: float = 1.0  # For result weighting
    enabled: bool = True
    headers: dict[str, str] = field(default_factory=dict)
    query_params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Routing hints
    topics: list[str] = field(default_factory=list)  # Topics this source is good for
    document_types: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "name": self.name,
            "endpoint": self.endpoint,
            "source_type": self.source_type.value,
            "timeout_seconds": self.timeout_seconds,
            "max_results": self.max_results,
            "weight": self.weight,
            "enabled": self.enabled,
            "topics": self.topics,
            "document_types": self.document_types,
            "languages": self.languages,
        }


@dataclass
class SourceHealth:
    """Health status of a source."""

    source_id: str
    status: SourceStatus
    latency_ms: float | None = None
    last_check: datetime = field(default_factory=datetime.utcnow)
    error_count: int = 0
    success_count: int = 0
    last_error: str | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.error_count + self.success_count
        if total == 0:
            return 0.0
        return self.success_count / total


@dataclass
class FederatedResult:
    """Result from a federated source."""

    chunk_id: str
    content: str
    score: float
    source_id: str
    source_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str | None = None
    doc_title: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "metadata": self.metadata,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
        }


class SourceAdapter(Protocol):
    """Protocol for source adapters."""

    async def search(
        self,
        query: str,
        config: SourceConfig,
        top_k: int,
    ) -> list[FederatedResult]:
        """Search the source."""
        ...


class DefaultSourceAdapter:
    """Default adapter for Perfect RAG compatible endpoints."""

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def search(
        self,
        query: str,
        config: SourceConfig,
        top_k: int,
    ) -> list[FederatedResult]:
        """Search a Perfect RAG compatible endpoint."""
        client = await self._get_client()

        headers = {"Content-Type": "application/json"}
        headers.update(config.headers)
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"

        payload = {
            "query": query,
            "top_k": min(top_k, config.max_results),
            **config.query_params,
        }

        try:
            response = await client.post(
                f"{config.endpoint}/api/v1/query",
                json=payload,
                headers=headers,
                timeout=config.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            # Handle different response formats
            chunks = data.get("chunks", data.get("results", []))
            if isinstance(data.get("retrieval_result"), dict):
                chunks = data["retrieval_result"].get("chunks", chunks)

            for chunk in chunks:
                results.append(FederatedResult(
                    chunk_id=chunk.get("chunk_id", chunk.get("id", "")),
                    content=chunk.get("content", chunk.get("text", "")),
                    score=chunk.get("score", 0.0),
                    source_id=config.source_id,
                    source_name=config.name,
                    metadata=chunk.get("metadata", {}),
                    doc_id=chunk.get("doc_id"),
                    doc_title=chunk.get("doc_title"),
                ))

            return results

        except Exception as e:
            logger.error(
                "Source search failed",
                source_id=config.source_id,
                error=str(e),
            )
            raise

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class ElasticsearchAdapter:
    """Adapter for Elasticsearch sources."""

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def search(
        self,
        query: str,
        config: SourceConfig,
        top_k: int,
    ) -> list[FederatedResult]:
        """Search an Elasticsearch endpoint."""
        client = await self._get_client()

        headers = {"Content-Type": "application/json"}
        headers.update(config.headers)
        if config.api_key:
            headers["Authorization"] = f"ApiKey {config.api_key}"

        # Build Elasticsearch query
        es_query = {
            "size": min(top_k, config.max_results),
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "title", "text"],
                }
            },
        }

        try:
            response = await client.post(
                f"{config.endpoint}/_search",
                json=es_query,
                headers=headers,
                timeout=config.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for hit in data.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                results.append(FederatedResult(
                    chunk_id=hit.get("_id", ""),
                    content=source.get("content", source.get("text", "")),
                    score=hit.get("_score", 0.0) / 10.0,  # Normalize score
                    source_id=config.source_id,
                    source_name=config.name,
                    metadata=source.get("metadata", {}),
                    doc_id=source.get("doc_id"),
                    doc_title=source.get("title"),
                ))

            return results

        except Exception as e:
            logger.error(
                "Elasticsearch search failed",
                source_id=config.source_id,
                error=str(e),
            )
            raise


class SourceRegistry:
    """Registry for managing RAG sources.

    Handles source registration, health monitoring, and configuration.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._sources: dict[str, SourceConfig] = {}
        self._health: dict[str, SourceHealth] = {}
        self._adapters: dict[SourceType, SourceAdapter] = {
            SourceType.PERFECT_RAG: DefaultSourceAdapter(),
            SourceType.CUSTOM: DefaultSourceAdapter(),
            SourceType.ELASTICSEARCH: ElasticsearchAdapter(),
        }
        self._lock = asyncio.Lock()

    async def register_source(self, config: SourceConfig) -> None:
        """Register a new RAG source."""
        async with self._lock:
            self._sources[config.source_id] = config
            self._health[config.source_id] = SourceHealth(
                source_id=config.source_id,
                status=SourceStatus.UNKNOWN,
            )

            logger.info(
                "Registered RAG source",
                source_id=config.source_id,
                name=config.name,
                endpoint=config.endpoint,
            )

    async def unregister_source(self, source_id: str) -> bool:
        """Unregister a RAG source."""
        async with self._lock:
            if source_id in self._sources:
                del self._sources[source_id]
                del self._health[source_id]
                return True
            return False

    async def get_source(self, source_id: str) -> SourceConfig | None:
        """Get source configuration."""
        return self._sources.get(source_id)

    async def list_sources(
        self,
        enabled_only: bool = True,
        healthy_only: bool = False,
    ) -> list[SourceConfig]:
        """List registered sources."""
        sources = list(self._sources.values())

        if enabled_only:
            sources = [s for s in sources if s.enabled]

        if healthy_only:
            sources = [
                s for s in sources
                if self._health.get(s.source_id, SourceHealth(s.source_id, SourceStatus.UNKNOWN)).status
                in (SourceStatus.HEALTHY, SourceStatus.DEGRADED)
            ]

        return sources

    async def update_health(
        self,
        source_id: str,
        status: SourceStatus,
        latency_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        """Update source health status."""
        async with self._lock:
            if source_id not in self._health:
                self._health[source_id] = SourceHealth(
                    source_id=source_id,
                    status=status,
                )

            health = self._health[source_id]
            health.status = status
            health.latency_ms = latency_ms
            health.last_check = datetime.utcnow()

            if error:
                health.error_count += 1
                health.last_error = error
            else:
                health.success_count += 1

    async def get_health(self, source_id: str) -> SourceHealth | None:
        """Get source health status."""
        return self._health.get(source_id)

    async def check_source_health(self, source_id: str) -> SourceHealth:
        """Check health of a specific source."""
        config = self._sources.get(source_id)
        if not config:
            return SourceHealth(
                source_id=source_id,
                status=SourceStatus.UNKNOWN,
                last_error="Source not found",
            )

        adapter = self._adapters.get(config.source_type, self._adapters[SourceType.CUSTOM])

        start = time.time()
        try:
            # Simple health check - search for empty string with 1 result
            await adapter.search("health check", config, top_k=1)
            latency = (time.time() - start) * 1000

            status = SourceStatus.HEALTHY
            if latency > 5000:  # More than 5 seconds
                status = SourceStatus.DEGRADED

            await self.update_health(source_id, status, latency)

        except Exception as e:
            await self.update_health(
                source_id,
                SourceStatus.UNHEALTHY,
                error=str(e),
            )

        return self._health[source_id]

    async def check_all_health(self) -> dict[str, SourceHealth]:
        """Check health of all sources."""
        tasks = [
            self.check_source_health(source_id)
            for source_id in self._sources.keys()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        return dict(self._health)

    def get_adapter(self, source_type: SourceType) -> SourceAdapter:
        """Get adapter for source type."""
        return self._adapters.get(source_type, self._adapters[SourceType.CUSTOM])

    def register_adapter(self, source_type: SourceType, adapter: SourceAdapter) -> None:
        """Register a custom adapter for a source type."""
        self._adapters[source_type] = adapter


class SourceRouter:
    """Route queries to appropriate sources.

    Uses query analysis and source metadata to determine
    which sources are most appropriate for a query.
    """

    def __init__(
        self,
        registry: SourceRegistry,
        settings: Settings | None = None,
    ):
        self.registry = registry
        self.settings = settings or get_settings()

    async def route(
        self,
        query: str,
        topics: list[str] | None = None,
        document_types: list[str] | None = None,
        language: str | None = None,
        max_sources: int | None = None,
    ) -> list[SourceConfig]:
        """Route query to appropriate sources.

        Args:
            query: User query
            topics: Optional topic hints
            document_types: Optional document type filters
            language: Optional language preference
            max_sources: Maximum number of sources to query

        Returns:
            List of source configurations to query
        """
        all_sources = await self.registry.list_sources(
            enabled_only=True,
            healthy_only=True,
        )

        if not all_sources:
            # Fall back to all enabled sources if none are healthy
            all_sources = await self.registry.list_sources(enabled_only=True)

        # Score each source
        scored_sources = []
        for source in all_sources:
            score = await self._score_source(
                source,
                query,
                topics,
                document_types,
                language,
            )
            scored_sources.append((source, score))

        # Sort by score
        scored_sources.sort(key=lambda x: x[1], reverse=True)

        # Take top sources
        max_sources = max_sources or len(scored_sources)
        selected = [s for s, _ in scored_sources[:max_sources]]

        logger.debug(
            "Routed query to sources",
            query=query[:50],
            selected_sources=[s.source_id for s in selected],
        )

        return selected

    async def _score_source(
        self,
        source: SourceConfig,
        query: str,
        topics: list[str] | None,
        document_types: list[str] | None,
        language: str | None,
    ) -> float:
        """Score a source for a query."""
        score = source.weight

        # Topic matching
        if topics and source.topics:
            matching_topics = set(topics) & set(source.topics)
            if matching_topics:
                score *= 1.0 + 0.2 * len(matching_topics)

        # Document type matching
        if document_types and source.document_types:
            matching_types = set(document_types) & set(source.document_types)
            if matching_types:
                score *= 1.0 + 0.1 * len(matching_types)

        # Language matching
        if language and source.languages:
            if language in source.languages:
                score *= 1.2

        # Health factor
        health = await self.registry.get_health(source.source_id)
        if health:
            if health.status == SourceStatus.DEGRADED:
                score *= 0.8
            elif health.status == SourceStatus.UNHEALTHY:
                score *= 0.3

            # Latency factor
            if health.latency_ms and health.latency_ms > 2000:
                score *= 0.9

        return score


class ResultAggregator:
    """Aggregate and deduplicate results from multiple sources.

    Handles:
    - Score normalization across sources
    - Deduplication using content similarity
    - Source diversity balancing
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.similarity_threshold = 0.9  # For deduplication

    async def aggregate(
        self,
        results_by_source: dict[str, list[FederatedResult]],
        top_k: int,
        normalize_scores: bool = True,
        deduplicate: bool = True,
        source_diversity: float = 0.3,  # Encourage diversity
    ) -> list[FederatedResult]:
        """Aggregate results from multiple sources.

        Args:
            results_by_source: Results keyed by source ID
            top_k: Number of results to return
            normalize_scores: Whether to normalize scores across sources
            deduplicate: Whether to remove duplicates
            source_diversity: Weight for source diversity (0-1)

        Returns:
            Aggregated and ranked results
        """
        all_results: list[FederatedResult] = []

        # Collect all results
        for source_id, results in results_by_source.items():
            # Normalize scores per source
            if normalize_scores and results:
                max_score = max(r.score for r in results)
                min_score = min(r.score for r in results)
                score_range = max_score - min_score if max_score > min_score else 1.0

                for result in results:
                    result.score = (result.score - min_score) / score_range

            all_results.extend(results)

        # Deduplicate
        if deduplicate:
            all_results = await self._deduplicate(all_results)

        # Apply source diversity
        if source_diversity > 0:
            all_results = await self._apply_diversity(
                all_results,
                source_diversity,
            )

        # Sort by score and take top_k
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]

    async def _deduplicate(
        self,
        results: list[FederatedResult],
    ) -> list[FederatedResult]:
        """Remove duplicate results based on content similarity."""
        if not results:
            return results

        seen_hashes: set[str] = set()
        deduplicated: list[FederatedResult] = []

        for result in results:
            # Create content hash
            content_hash = hashlib.md5(
                result.content.lower().strip().encode()
            ).hexdigest()

            # Also check for similar content using shingles
            if content_hash not in seen_hashes:
                # Check against existing results
                is_duplicate = False
                for existing in deduplicated:
                    similarity = self._jaccard_similarity(
                        result.content,
                        existing.content,
                    )
                    if similarity > self.similarity_threshold:
                        is_duplicate = True
                        # Keep the one with higher score
                        if result.score > existing.score:
                            deduplicated.remove(existing)
                            deduplicated.append(result)
                        break

                if not is_duplicate:
                    seen_hashes.add(content_hash)
                    deduplicated.append(result)

        logger.debug(
            "Deduplicated results",
            original_count=len(results),
            deduplicated_count=len(deduplicated),
        )

        return deduplicated

    def _jaccard_similarity(self, text1: str, text2: str, n: int = 3) -> float:
        """Calculate Jaccard similarity using n-gram shingles."""
        def get_shingles(text: str, n: int) -> set[str]:
            text = text.lower()
            return {text[i:i+n] for i in range(len(text) - n + 1)}

        shingles1 = get_shingles(text1, n)
        shingles2 = get_shingles(text2, n)

        if not shingles1 or not shingles2:
            return 0.0

        intersection = len(shingles1 & shingles2)
        union = len(shingles1 | shingles2)

        return intersection / union if union > 0 else 0.0

    async def _apply_diversity(
        self,
        results: list[FederatedResult],
        diversity_weight: float,
    ) -> list[FederatedResult]:
        """Apply source diversity to results.

        Penalizes results from sources that already have many results
        in the top positions.
        """
        if not results:
            return results

        # Count results per source
        source_counts: dict[str, int] = {}
        for result in results:
            source_counts[result.source_id] = source_counts.get(result.source_id, 0) + 1

        total_results = len(results)
        num_sources = len(source_counts)

        if num_sources <= 1:
            return results

        # Target distribution (equal across sources)
        target_per_source = total_results / num_sources

        # Adjust scores
        adjusted_results = []
        position_source_counts: dict[str, int] = {}

        for result in sorted(results, key=lambda x: x.score, reverse=True):
            source_id = result.source_id
            current_count = position_source_counts.get(source_id, 0)

            # Calculate diversity penalty
            if current_count > target_per_source:
                excess = current_count - target_per_source
                penalty = diversity_weight * (excess / target_per_source)
                result.score *= (1 - min(penalty, 0.5))

            position_source_counts[source_id] = current_count + 1
            adjusted_results.append(result)

        return adjusted_results


class FederatedRetriever:
    """Main federated retrieval orchestrator.

    Coordinates querying multiple RAG sources and aggregating results.
    """

    def __init__(
        self,
        registry: SourceRegistry | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.registry = registry or SourceRegistry(settings)
        self.router = SourceRouter(self.registry, settings)
        self.aggregator = ResultAggregator(settings)

    async def search(
        self,
        query: str,
        top_k: int = 10,
        topics: list[str] | None = None,
        document_types: list[str] | None = None,
        language: str | None = None,
        source_ids: list[str] | None = None,
        timeout: float | None = None,
    ) -> list[FederatedResult]:
        """Search across federated sources.

        Args:
            query: User query
            top_k: Number of results to return
            topics: Optional topic hints for routing
            document_types: Optional document type filters
            language: Optional language preference
            source_ids: Specific source IDs to query (bypasses routing)
            timeout: Overall timeout in seconds

        Returns:
            Aggregated search results
        """
        timeout = timeout or 30.0
        start = time.time()

        # Determine which sources to query
        if source_ids:
            sources = []
            for sid in source_ids:
                source = await self.registry.get_source(sid)
                if source and source.enabled:
                    sources.append(source)
        else:
            sources = await self.router.route(
                query,
                topics=topics,
                document_types=document_types,
                language=language,
            )

        if not sources:
            logger.warning("No sources available for federated search")
            return []

        logger.info(
            "Starting federated search",
            query=query[:50],
            source_count=len(sources),
        )

        # Query all sources in parallel
        results_by_source: dict[str, list[FederatedResult]] = {}
        tasks = []

        for source in sources:
            task = asyncio.create_task(
                self._search_source(source, query, top_k)
            )
            tasks.append((source.source_id, task))

        # Wait with timeout
        remaining_time = timeout - (time.time() - start)
        for source_id, task in tasks:
            try:
                results = await asyncio.wait_for(task, timeout=remaining_time)
                results_by_source[source_id] = results
            except asyncio.TimeoutError:
                logger.warning(
                    "Source search timed out",
                    source_id=source_id,
                )
                await self.registry.update_health(
                    source_id,
                    SourceStatus.DEGRADED,
                    error="Timeout",
                )
            except Exception as e:
                logger.error(
                    "Source search failed",
                    source_id=source_id,
                    error=str(e),
                )
                await self.registry.update_health(
                    source_id,
                    SourceStatus.UNHEALTHY,
                    error=str(e),
                )

        # Aggregate results
        aggregated = await self.aggregator.aggregate(
            results_by_source,
            top_k=top_k,
        )

        logger.info(
            "Federated search complete",
            total_results=len(aggregated),
            sources_responded=len(results_by_source),
            latency_ms=(time.time() - start) * 1000,
        )

        return aggregated

    async def _search_source(
        self,
        source: SourceConfig,
        query: str,
        top_k: int,
    ) -> list[FederatedResult]:
        """Search a single source."""
        adapter = self.registry.get_adapter(source.source_type)

        start = time.time()
        try:
            results = await adapter.search(query, source, top_k)
            latency = (time.time() - start) * 1000

            await self.registry.update_health(
                source.source_id,
                SourceStatus.HEALTHY,
                latency_ms=latency,
            )

            # Apply source weight to scores
            for result in results:
                result.score *= source.weight

            return results

        except Exception as e:
            await self.registry.update_health(
                source.source_id,
                SourceStatus.UNHEALTHY,
                error=str(e),
            )
            raise

    async def add_source(
        self,
        source_id: str,
        name: str,
        endpoint: str,
        source_type: SourceType = SourceType.CUSTOM,
        **kwargs,
    ) -> None:
        """Add a new source to the registry."""
        config = SourceConfig(
            source_id=source_id,
            name=name,
            endpoint=endpoint,
            source_type=source_type,
            **kwargs,
        )
        await self.registry.register_source(config)

    async def remove_source(self, source_id: str) -> bool:
        """Remove a source from the registry."""
        return await self.registry.unregister_source(source_id)

    async def list_sources(self) -> list[dict[str, Any]]:
        """List all registered sources with health status."""
        sources = await self.registry.list_sources(enabled_only=False)
        result = []

        for source in sources:
            health = await self.registry.get_health(source.source_id)
            info = source.to_dict()
            info["health"] = {
                "status": health.status.value if health else "unknown",
                "latency_ms": health.latency_ms if health else None,
                "success_rate": health.success_rate if health else 0.0,
            }
            result.append(info)

        return result

    async def health_check(self) -> dict[str, Any]:
        """Check health of all sources."""
        health = await self.registry.check_all_health()
        return {
            source_id: {
                "status": h.status.value,
                "latency_ms": h.latency_ms,
                "success_rate": h.success_rate,
                "last_error": h.last_error,
            }
            for source_id, h in health.items()
        }


# Module-level singleton
_federated_retriever: FederatedRetriever | None = None


async def get_federated_retriever() -> FederatedRetriever:
    """Get or create federated retriever."""
    global _federated_retriever
    if _federated_retriever is None:
        _federated_retriever = FederatedRetriever()
    return _federated_retriever
