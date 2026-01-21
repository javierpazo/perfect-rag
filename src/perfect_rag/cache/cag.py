"""Cache-Augmented Generation (CAG) implementation.

CAG pre-loads frequently used context into the prompt to reduce retrieval latency.
"""

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import structlog

from perfect_rag.config import Settings, get_settings
from perfect_rag.core.embedding import EmbeddingService
from perfect_rag.db.surrealdb import SurrealDBClient

logger = structlog.get_logger(__name__)


@dataclass
class CacheEntry:
    """A cached context entry."""

    key: str  # Query embedding hash or semantic key
    query_text: str
    query_embedding: list[float]
    chunk_ids: list[str]
    chunk_contents: list[str]
    entity_ids: list[str]
    subgraph: dict[str, Any]  # Mini knowledge graph
    response: str | None = None  # Cached response (optional)
    confidence: float = 0.0
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)
    ttl: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CAGCache:
    """Cache-Augmented Generation cache.

    Features:
    - Semantic similarity-based cache lookup
    - LRU + importance hybrid eviction
    - Automatic cache warming from feedback
    - Configurable TTL
    - Memory-efficient storage
    """

    def __init__(
        self,
        surrealdb: SurrealDBClient,
        embedding_service: EmbeddingService,
        settings: Settings | None = None,
    ):
        self.surrealdb = surrealdb
        self.embedding = embedding_service
        self.settings = settings or get_settings()

        # In-memory cache for fast lookup
        self._cache: dict[str, CacheEntry] = {}
        self._embeddings: dict[str, np.ndarray] = {}  # key -> embedding array

        # Cache configuration
        self.max_entries = self.settings.cag_max_entries
        self.similarity_threshold = self.settings.cag_similarity_threshold
        self.default_ttl_hours = self.settings.cag_ttl_hours
        self.max_context_tokens = self.settings.cag_max_context_tokens

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Load cache from database."""
        logger.info("Initializing CAG cache...")

        result = await self.surrealdb.client.query(
            """
            SELECT * FROM cag_cache
            WHERE ttl IS NULL OR ttl > time::now()
            ORDER BY access_count DESC
            LIMIT $limit
            """,
            {"limit": self.max_entries},
        )

        if result and result[0].get("result"):
            for row in result[0]["result"]:
                entry = CacheEntry(
                    key=row["key"],
                    query_text=row["query_text"],
                    query_embedding=row["query_embedding"],
                    chunk_ids=row.get("chunk_ids", []),
                    chunk_contents=row.get("chunk_contents", []),
                    entity_ids=row.get("entity_ids", []),
                    subgraph=row.get("subgraph", {}),
                    response=row.get("response"),
                    confidence=row.get("confidence", 0.0),
                    access_count=row.get("access_count", 0),
                    last_accessed=datetime.fromisoformat(row["last_accessed"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                    ttl=datetime.fromisoformat(row["ttl"]) if row.get("ttl") else None,
                    metadata=row.get("metadata", {}),
                )
                self._cache[entry.key] = entry
                self._embeddings[entry.key] = np.array(entry.query_embedding)

        logger.info("CAG cache initialized", entries=len(self._cache))

    async def lookup(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        similarity_threshold: float | None = None,
    ) -> CacheEntry | None:
        """Look up a query in the cache.

        Args:
            query: Query text
            query_embedding: Pre-computed embedding (optional)
            similarity_threshold: Custom threshold (optional)

        Returns:
            CacheEntry if found, None otherwise
        """
        if not self._cache:
            return None

        threshold = similarity_threshold or self.similarity_threshold

        # Get query embedding
        if query_embedding is None:
            query_embedding = await self.embedding.embed_query(query)

        query_vec = np.array(query_embedding)

        # Find most similar cached query
        best_key = None
        best_similarity = 0.0

        for key, cached_vec in self._embeddings.items():
            # Cosine similarity (vectors are normalized)
            similarity = float(np.dot(query_vec, cached_vec))

            if similarity > best_similarity:
                best_similarity = similarity
                best_key = key

        if best_key and best_similarity >= threshold:
            entry = self._cache[best_key]

            # Update access stats
            async with self._lock:
                entry.access_count += 1
                entry.last_accessed = datetime.utcnow()

            logger.debug(
                "CAG cache hit",
                similarity=best_similarity,
                cached_query=entry.query_text[:50],
            )

            return entry

        return None

    async def store(
        self,
        query: str,
        query_embedding: list[float],
        chunk_ids: list[str],
        chunk_contents: list[str],
        entity_ids: list[str] | None = None,
        subgraph: dict[str, Any] | None = None,
        response: str | None = None,
        confidence: float = 0.0,
        ttl_hours: int | None = None,
    ) -> CacheEntry:
        """Store a query result in cache.

        Args:
            query: Query text
            query_embedding: Query embedding
            chunk_ids: Retrieved chunk IDs
            chunk_contents: Chunk text contents
            entity_ids: Related entity IDs
            subgraph: Mini knowledge graph
            response: Cached response (optional)
            confidence: Result confidence
            ttl_hours: Custom TTL (optional)

        Returns:
            Created CacheEntry
        """
        # Generate cache key
        key = self._generate_key(query_embedding)

        # Calculate TTL
        ttl = None
        if ttl_hours or self.default_ttl_hours:
            hours = ttl_hours or self.default_ttl_hours
            ttl = datetime.utcnow() + timedelta(hours=hours)

        entry = CacheEntry(
            key=key,
            query_text=query,
            query_embedding=query_embedding,
            chunk_ids=chunk_ids,
            chunk_contents=chunk_contents,
            entity_ids=entity_ids or [],
            subgraph=subgraph or {},
            response=response,
            confidence=confidence,
            ttl=ttl,
        )

        async with self._lock:
            # Evict if necessary
            if len(self._cache) >= self.max_entries:
                await self._evict()

            # Store in memory
            self._cache[key] = entry
            self._embeddings[key] = np.array(query_embedding)

        # Persist to database
        await self._persist_entry(entry)

        logger.debug("CAG cache stored", key=key[:16], query=query[:50])

        return entry

    def _generate_key(self, embedding: list[float]) -> str:
        """Generate cache key from embedding."""
        # Hash the embedding for a compact key
        embedding_bytes = np.array(embedding).tobytes()
        return hashlib.sha256(embedding_bytes).hexdigest()[:32]

    async def _evict(self) -> None:
        """Evict entries using LRU + importance hybrid strategy."""
        if not self._cache:
            return

        # Score entries: lower is more evictable
        # Score = access_count * recency_factor
        now = datetime.utcnow()
        scores = {}

        for key, entry in self._cache.items():
            # Check TTL
            if entry.ttl and entry.ttl < now:
                scores[key] = -1  # Expired, evict first
                continue

            # Calculate recency (0-1, higher is more recent)
            age_hours = (now - entry.last_accessed).total_seconds() / 3600
            recency = max(0, 1 - (age_hours / 168))  # Decay over 1 week

            # Combined score
            scores[key] = entry.access_count * (0.5 + 0.5 * recency)

        # Remove lowest scoring entries (keep top 80%)
        target_size = int(self.max_entries * 0.8)
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k])
        to_remove = sorted_keys[:len(self._cache) - target_size]

        for key in to_remove:
            del self._cache[key]
            if key in self._embeddings:
                del self._embeddings[key]

            # Remove from database
            await self.surrealdb.client.query(
                "DELETE FROM cag_cache WHERE key = $key",
                {"key": key},
            )

        logger.info("CAG cache eviction", removed=len(to_remove))

    async def _persist_entry(self, entry: CacheEntry) -> None:
        """Persist cache entry to database."""
        data = {
            "key": entry.key,
            "query_text": entry.query_text,
            "query_embedding": entry.query_embedding,
            "chunk_ids": entry.chunk_ids,
            "chunk_contents": entry.chunk_contents[:10],  # Limit stored content
            "entity_ids": entry.entity_ids,
            "subgraph": entry.subgraph,
            "response": entry.response,
            "confidence": entry.confidence,
            "access_count": entry.access_count,
            "last_accessed": entry.last_accessed.isoformat(),
            "created_at": entry.created_at.isoformat(),
            "ttl": entry.ttl.isoformat() if entry.ttl else None,
            "metadata": entry.metadata,
        }

        await self.surrealdb.client.query(
            "UPSERT cag_cache:$key CONTENT $data",
            {"key": entry.key, "data": data},
        )

    async def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._embeddings:
                    del self._embeddings[key]

                await self.surrealdb.client.query(
                    "DELETE FROM cag_cache WHERE key = $key",
                    {"key": key},
                )
                return True
        return False

    async def invalidate_by_chunk(self, chunk_id: str) -> int:
        """Invalidate all cache entries containing a chunk."""
        count = 0
        keys_to_remove = []

        async with self._lock:
            for key, entry in self._cache.items():
                if chunk_id in entry.chunk_ids:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._cache[key]
                if key in self._embeddings:
                    del self._embeddings[key]
                count += 1

        if keys_to_remove:
            await self.surrealdb.client.query(
                "DELETE FROM cag_cache WHERE key IN $keys",
                {"keys": keys_to_remove},
            )

        return count

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_chunks = sum(len(e.chunk_ids) for e in self._cache.values())
        total_access = sum(e.access_count for e in self._cache.values())

        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "total_chunks_cached": total_chunks,
            "total_accesses": total_access,
            "avg_access_count": total_access / len(self._cache) if self._cache else 0,
            "similarity_threshold": self.similarity_threshold,
        }

    def build_context_from_cache(
        self,
        entry: CacheEntry,
        max_tokens: int | None = None,
    ) -> str:
        """Build context string from cache entry.

        Returns formatted context suitable for LLM prompt.
        """
        max_tokens = max_tokens or self.max_context_tokens
        context_parts = []
        current_tokens = 0

        # Add chunk contents
        for i, content in enumerate(entry.chunk_contents):
            chunk_tokens = len(content) // 4  # Rough estimate

            if current_tokens + chunk_tokens > max_tokens:
                break

            context_parts.append(f"[{i + 1}] {content}")
            current_tokens += chunk_tokens

        # Add subgraph facts if space allows
        if entry.subgraph and current_tokens < max_tokens * 0.9:
            facts = entry.subgraph.get("facts", [])
            for fact in facts[:5]:
                fact_str = f"Fact: {fact}"
                fact_tokens = len(fact_str) // 4

                if current_tokens + fact_tokens > max_tokens:
                    break

                context_parts.append(fact_str)
                current_tokens += fact_tokens

        return "\n\n".join(context_parts)
