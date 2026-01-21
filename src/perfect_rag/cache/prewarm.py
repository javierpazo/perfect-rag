"""Cache prewarming strategies.

All 5 strategies are maintained for maximum quality:
1. Centrality - Most connected entities (graph importance)
2. Coverage - Representative chunks from each document (document coverage)
3. Recency - Recent documents (fresh information)
4. Frequency - Most queried content (usage patterns)
5. Feedback - Content with positive feedback (quality signals)

Each strategy serves a unique purpose and contributes to cache quality.
"""

from typing import Any

import structlog

from perfect_rag.cache.cag import CAGCache
from perfect_rag.config import Settings, get_settings
from perfect_rag.core.embedding import EmbeddingService
from perfect_rag.db.surrealdb import SurrealDBClient

logger = structlog.get_logger(__name__)

# All 5 strategies for maximum quality
DEFAULT_PREWARM_STRATEGIES = ["centrality", "coverage", "recency", "frequency", "feedback"]


class CachePrewarmer:
    """Prewarm the CAG cache with important content.

    Strategies (all 5 for maximum quality):
    1. Centrality-based: High PageRank/degree entities
    2. Coverage-based: Representative chunks from clusters
    3. Recency-based: Recent documents
    4. Frequency-based: Frequently accessed queries
    5. Feedback-based: Chunks with positive feedback
    """

    def __init__(
        self,
        cache: CAGCache,
        surrealdb: SurrealDBClient,
        embedding_service: EmbeddingService,
        settings: Settings | None = None,
    ):
        self.cache = cache
        self.surrealdb = surrealdb
        self.embedding = embedding_service
        self.settings = settings or get_settings()

    async def prewarm(
        self,
        strategies: list[str] | None = None,
        max_entries: int = 100,
    ) -> dict[str, Any]:
        """Run cache prewarming.

        Args:
            strategies: List of strategies to use (default: all 5)
            max_entries: Maximum entries to add

        Returns:
            Summary of prewarming results
        """
        strategies = strategies or DEFAULT_PREWARM_STRATEGIES
        results = {
            "total_added": 0,
            "by_strategy": {},
        }

        entries_per_strategy = max_entries // len(strategies)

        for strategy in strategies:
            try:
                if strategy == "centrality":
                    added = await self._prewarm_centrality(entries_per_strategy)
                elif strategy == "coverage":
                    added = await self._prewarm_coverage(entries_per_strategy)
                elif strategy == "recency":
                    added = await self._prewarm_recency(entries_per_strategy)
                elif strategy == "frequency":
                    added = await self._prewarm_frequency(entries_per_strategy)
                elif strategy == "feedback":
                    added = await self._prewarm_feedback(entries_per_strategy)
                else:
                    logger.warning("Unknown prewarm strategy", strategy=strategy)
                    continue

                results["by_strategy"][strategy] = added
                results["total_added"] += added

            except Exception as e:
                logger.error("Prewarm strategy failed", strategy=strategy, error=str(e))
                results["by_strategy"][strategy] = 0

        logger.info("Cache prewarming complete", **results)
        return results

    async def _prewarm_centrality(self, limit: int) -> int:
        """Prewarm based on entity centrality (high-degree nodes)."""
        # Get entities with most relations
        result = await self.surrealdb.query(
            """
            SELECT
                id AS entity_id,
                name,
                source_chunks AS chunks,
                count(->related_to) + count(<-related_to) AS degree
            FROM entity
            ORDER BY degree DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )

        added = 0
        rows = result[0] if result else []
        for row in rows:
            entity_name = row.get("name", "")
            chunk_ids = row.get("chunks", [])[:5]  # Limit chunks per entity

            if not chunk_ids:
                continue

            # Get chunk contents
            chunks = await self._get_chunk_contents(chunk_ids)
            if not chunks:
                continue

            # Generate synthetic query
            query = f"What is {entity_name}?"
            embedding = await self.embedding.embed_query(query)

            # Build mini subgraph
            subgraph = await self._build_entity_subgraph(row.get("entity_id"))

            await self.cache.store(
                query=query,
                query_embedding=embedding,
                chunk_ids=chunk_ids,
                chunk_contents=[c["content"] for c in chunks],
                entity_ids=[row.get("entity_id")],
                subgraph=subgraph,
                confidence=0.8,
            )
            added += 1

        return added

    async def _prewarm_coverage(self, limit: int) -> int:
        """Prewarm based on document coverage (one entry per document).

        Ensures all documents have representation in the cache,
        preventing some documents from being completely ignored.
        """
        # Get one representative chunk per document
        result = await self.surrealdb.query(
            """
            SELECT
                doc_id,
                array::first(array::sort(id)) AS first_chunk_id
            FROM chunk
            GROUP BY doc_id
            LIMIT $limit
            """,
            {"limit": limit},
        )

        added = 0
        rows = result[0] if result else []
        for row in rows:
            doc_id = row.get("doc_id")
            chunk_id = row.get("first_chunk_id")

            if not chunk_id:
                continue

            # Get chunk and document info
            chunks = await self._get_chunk_contents([chunk_id])
            if not chunks:
                continue

            chunk = chunks[0]
            doc_title = chunk.get("metadata", {}).get("title", "Document")

            # Generate synthetic query
            query = f"Tell me about {doc_title}"
            embedding = await self.embedding.embed_query(query)

            await self.cache.store(
                query=query,
                query_embedding=embedding,
                chunk_ids=[chunk_id],
                chunk_contents=[chunk["content"]],
                entity_ids=[],
                subgraph={},
                confidence=0.7,
            )
            added += 1

        return added

    async def _prewarm_recency(self, limit: int) -> int:
        """Prewarm based on recent documents.

        Ensures recently added information is quickly available,
        important for keeping cache fresh with new content.
        """
        result = await self.surrealdb.query(
            """
            SELECT
                id,
                metadata.title AS title,
                content
            FROM document
            ORDER BY created_at DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )

        added = 0
        rows = result[0] if result else []
        for row in rows:
            doc_id = row.get("id")
            title = row.get("title", "Document")

            # Get first few chunks
            chunk_result = await self.surrealdb.query(
                """
                SELECT id, content FROM chunk
                WHERE doc_id = $doc_id
                ORDER BY chunk_index
                LIMIT 3
                """,
                {"doc_id": doc_id},
            )

            chunk_rows = chunk_result[0] if chunk_result else []
            if not chunk_rows:
                continue

            chunk_ids = [c["id"] for c in chunk_rows]
            contents = [c["content"] for c in chunk_rows]

            query = f"What's new in {title}?"
            embedding = await self.embedding.embed_query(query)

            await self.cache.store(
                query=query,
                query_embedding=embedding,
                chunk_ids=chunk_ids,
                chunk_contents=contents,
                entity_ids=[],
                subgraph={},
                confidence=0.6,
            )
            added += 1

        return added

    async def _prewarm_frequency(self, limit: int) -> int:
        """Prewarm based on frequent queries.

        Optimizes for common use cases by caching
        frequently asked questions.
        """
        # Get most common queries from logs
        result = await self.surrealdb.query(
            """
            SELECT
                query,
                retrieved_chunk_ids,
                count() AS freq
            FROM query_log
            GROUP BY query
            ORDER BY freq DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )

        added = 0
        rows = result[0] if result else []
        for row in rows:
            query = row.get("query")
            chunk_ids = row.get("retrieved_chunk_ids", [])[:5]

            if not query or not chunk_ids:
                continue

            # Check if already cached
            embedding = await self.embedding.embed_query(query)
            existing = await self.cache.lookup(query, embedding)
            if existing:
                continue

            # Get chunk contents
            chunks = await self._get_chunk_contents(chunk_ids)
            if not chunks:
                continue

            await self.cache.store(
                query=query,
                query_embedding=embedding,
                chunk_ids=chunk_ids,
                chunk_contents=[c["content"] for c in chunks],
                entity_ids=[],
                subgraph={},
                confidence=0.9,
            )
            added += 1

        return added

    async def _prewarm_feedback(self, limit: int) -> int:
        """Prewarm based on positive feedback.

        Uses user validation signals to cache high-quality
        content that has been explicitly approved.
        """
        # Get queries with positive feedback
        result = await self.surrealdb.query(
            """
            SELECT
                q.query,
                q.retrieved_chunk_ids,
                q.response,
                count() AS positive_count
            FROM query_log AS q
            JOIN feedback AS f ON f.query_id = q.id
            WHERE f.feedback_type IN ['thumbs_up', 'citation_click']
            GROUP BY q.query
            ORDER BY positive_count DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )

        added = 0
        rows = result[0] if result else []
        for row in rows:
            query = row.get("query")
            chunk_ids = row.get("retrieved_chunk_ids", [])[:5]
            response = row.get("response")

            if not query or not chunk_ids:
                continue

            embedding = await self.embedding.embed_query(query)

            # Check if already cached
            existing = await self.cache.lookup(query, embedding)
            if existing:
                continue

            chunks = await self._get_chunk_contents(chunk_ids)
            if not chunks:
                continue

            await self.cache.store(
                query=query,
                query_embedding=embedding,
                chunk_ids=chunk_ids,
                chunk_contents=[c["content"] for c in chunks],
                entity_ids=[],
                subgraph={},
                response=response,  # Cache the good response too
                confidence=0.95,
            )
            added += 1

        return added

    async def _get_chunk_contents(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """Get chunk contents from database."""
        if not chunk_ids:
            return []

        result = await self.surrealdb.query(
            "SELECT id, content, doc_id, metadata FROM chunk WHERE id IN $ids",
            {"ids": chunk_ids},
        )

        return result[0] if result else []

    async def _build_entity_subgraph(self, entity_id: str) -> dict[str, Any]:
        """Build a mini subgraph around an entity."""
        if not entity_id:
            return {}

        # Get entity relations
        result = await self.surrealdb.query(
            """
            SELECT
                ->related_to.out.name AS targets,
                ->related_to.relation_type AS relations
            FROM entity:$id
            """,
            {"id": entity_id},
        )

        facts = []
        rows = result[0] if result else []
        if rows:
            data = rows[0] if rows else {}
            targets = data.get("targets", []) or []
            relations = data.get("relations", []) or []

            for target, rel in zip(targets, relations):
                if target and rel:
                    facts.append(f"{entity_id} {rel} {target}")

        return {"facts": facts[:10]}
