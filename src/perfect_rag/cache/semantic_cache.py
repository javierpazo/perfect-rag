"""Semantic query cache for exact and near-duplicate queries."""

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import structlog

from perfect_rag.config import Settings, get_settings
from perfect_rag.core.embedding import EmbeddingService

logger = structlog.get_logger(__name__)


@dataclass
class SemanticCacheEntry:
    """Cached response entry."""

    key: str
    query: str
    query_embedding: list[float]
    response: str
    citations: list[dict[str, Any]]
    confidence: float
    model: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    ttl: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SemanticCache:
    """Semantic query cache with embedding-based lookup.

    Features:
    - Exact match via query hash
    - Near-match via embedding similarity
    - Response adaptation for similar queries
    - Configurable TTL and eviction
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        settings: Settings | None = None,
    ):
        self.embedding = embedding_service
        self.settings = settings or get_settings()

        # In-memory cache
        self._cache: dict[str, SemanticCacheEntry] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._exact_keys: dict[str, str] = {}  # query_hash -> cache_key

        # Configuration
        self.max_entries = 1000
        self.exact_threshold = 0.98  # Very similar = use cached response directly
        self.similar_threshold = 0.85  # Similar enough to adapt
        self.default_ttl_minutes = 60

        self._lock = asyncio.Lock()

    def _hash_query(self, query: str) -> str:
        """Generate hash for exact match lookup."""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    async def lookup(
        self,
        query: str,
        query_embedding: list[float] | None = None,
    ) -> tuple[SemanticCacheEntry | None, float]:
        """Look up query in cache.

        Returns:
            Tuple of (entry, similarity_score)
            Returns (None, 0) if not found
        """
        # Try exact match first
        query_hash = self._hash_query(query)
        if query_hash in self._exact_keys:
            key = self._exact_keys[query_hash]
            if key in self._cache:
                entry = self._cache[key]

                # Check TTL
                if entry.ttl and entry.ttl < datetime.utcnow():
                    await self._remove_entry(key)
                    return None, 0.0

                # Update stats
                entry.access_count += 1
                entry.last_accessed = datetime.utcnow()

                logger.debug("Semantic cache exact hit", query=query[:50])
                return entry, 1.0

        # Try semantic match
        if query_embedding is None:
            query_embedding = await self.embedding.embed_query(query)

        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return None, 0.0
        query_vec_normalized = query_vec / query_norm

        best_entry = None
        best_similarity = 0.0

        for key, cached_vec in self._embeddings.items():
            # Cosine similarity: dot product of normalized vectors
            cached_norm = np.linalg.norm(cached_vec)
            if cached_norm == 0:
                continue
            cached_vec_normalized = cached_vec / cached_norm
            similarity = float(np.dot(query_vec_normalized, cached_vec_normalized))

            if similarity > best_similarity:
                entry = self._cache.get(key)
                if entry and (not entry.ttl or entry.ttl > datetime.utcnow()):
                    best_similarity = similarity
                    best_entry = entry

        if best_entry and best_similarity >= self.similar_threshold:
            best_entry.access_count += 1
            best_entry.last_accessed = datetime.utcnow()

            logger.debug(
                "Semantic cache similar hit",
                similarity=best_similarity,
                cached_query=best_entry.query[:50],
            )
            return best_entry, best_similarity

        return None, 0.0

    async def store(
        self,
        query: str,
        query_embedding: list[float],
        response: str,
        citations: list[dict[str, Any]] | None = None,
        confidence: float = 0.0,
        model: str = "",
        ttl_minutes: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticCacheEntry:
        """Store a response in cache."""
        key = hashlib.sha256(
            np.array(query_embedding).tobytes()
        ).hexdigest()[:32]

        ttl = None
        if ttl_minutes or self.default_ttl_minutes:
            minutes = ttl_minutes or self.default_ttl_minutes
            ttl = datetime.utcnow() + timedelta(minutes=minutes)

        entry = SemanticCacheEntry(
            key=key,
            query=query,
            query_embedding=query_embedding,
            response=response,
            citations=citations or [],
            confidence=confidence,
            model=model,
            ttl=ttl,
            metadata=metadata or {},
        )

        async with self._lock:
            # Evict if needed
            if len(self._cache) >= self.max_entries:
                await self._evict()

            self._cache[key] = entry
            self._embeddings[key] = np.array(query_embedding)
            self._exact_keys[self._hash_query(query)] = key

        return entry

    def _remove_entry_unlocked(self, key: str) -> None:
        """Remove an entry from cache (internal use, no lock)."""
        if key in self._cache:
            entry = self._cache[key]
            query_hash = self._hash_query(entry.query)

            del self._cache[key]
            if key in self._embeddings:
                del self._embeddings[key]
            if query_hash in self._exact_keys:
                del self._exact_keys[query_hash]

    async def _remove_entry(self, key: str) -> None:
        """Remove an entry from cache."""
        async with self._lock:
            self._remove_entry_unlocked(key)

    async def _evict(self) -> None:
        """Evict old/unused entries (called with lock held)."""
        now = datetime.utcnow()

        # First remove expired entries
        expired = [
            k for k, e in self._cache.items()
            if e.ttl and e.ttl < now
        ]
        for key in expired:
            self._remove_entry_unlocked(key)

        # If still over limit, remove least recently used
        if len(self._cache) >= self.max_entries:
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].last_accessed,
            )

            # Remove oldest 20%
            to_remove = len(self._cache) - int(self.max_entries * 0.8)
            for key, _ in sorted_entries[:to_remove]:
                self._remove_entry_unlocked(key)

    def adapt_response(
        self,
        cached_entry: SemanticCacheEntry,
        new_query: str,
        similarity: float,
    ) -> str:
        """Adapt a cached response for a similar but different query.

        For very similar queries (>0.98), return as-is.
        For somewhat similar queries, add a note about adaptation.
        """
        if similarity >= self.exact_threshold:
            return cached_entry.response

        # For lower similarity, we could use LLM to adapt
        # For now, just return with metadata
        return cached_entry.response

    async def invalidate_by_content(self, content_substring: str) -> int:
        """Invalidate entries containing specific content."""
        count = 0
        keys_to_remove = []

        for key, entry in self._cache.items():
            if content_substring.lower() in entry.response.lower():
                keys_to_remove.append(key)

        for key in keys_to_remove:
            await self._remove_entry(key)
            count += 1

        return count

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        now = datetime.utcnow()
        active = sum(1 for e in self._cache.values() if not e.ttl or e.ttl > now)

        return {
            "total_entries": len(self._cache),
            "active_entries": active,
            "total_accesses": sum(e.access_count for e in self._cache.values()),
            "max_entries": self.max_entries,
            "exact_threshold": self.exact_threshold,
            "similar_threshold": self.similar_threshold,
        }

    async def clear(self) -> int:
        """Clear all cache entries."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._embeddings.clear()
            self._exact_keys.clear()
            return count
