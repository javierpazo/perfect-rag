"""Qdrant vector database client wrapper."""

import uuid
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    ScoredPoint,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)


def string_to_uuid(s: str) -> str:
    """Convert a string to a deterministic UUID string.

    Qdrant requires UUIDs or integers as point IDs.
    This creates a deterministic UUID from any string ID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))


class QdrantVectorClient:
    """Async Qdrant client wrapper."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client: AsyncQdrantClient | None = None
        # Get collection names from settings
        self.chunks_collection = self.settings.qdrant_chunks_collection
        self.entities_collection = self.settings.qdrant_entities_collection

    async def connect(self) -> None:
        """Initialize Qdrant client."""
        self._client = AsyncQdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
        )
        logger.info("Connected to Qdrant", url=self.settings.qdrant_url)

    async def disconnect(self) -> None:
        """Close client connection."""
        if self._client:
            await self._client.close()
            logger.info("Disconnected from Qdrant")

    @property
    def client(self) -> AsyncQdrantClient:
        """Get client, raising if not connected."""
        if self._client is None:
            raise RuntimeError("Qdrant client not connected. Call connect() first.")
        return self._client

    async def initialize_collections(self) -> None:
        """Create collections if they don't exist."""
        # Chunks collection (dense + sparse vectors)
        try:
            await self.client.get_collection(self.chunks_collection)
            logger.info("Collection exists", collection=self.chunks_collection)
        except UnexpectedResponse:
            await self.client.create_collection(
                collection_name=self.chunks_collection,
                vectors_config={
                    "dense": VectorParams(
                        size=1024,  # BGE-M3 dimension
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(),
                },
            )
            logger.info("Created collection", collection=self.chunks_collection)

        # Entities collection (for structural embeddings)
        try:
            await self.client.get_collection(self.entities_collection)
            logger.info("Collection exists", collection=self.entities_collection)
        except UnexpectedResponse:
            await self.client.create_collection(
                collection_name=self.entities_collection,
                vectors_config=VectorParams(
                    size=1024,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created collection", collection=self.entities_collection)

    # =========================================================================
    # Chunk Operations
    # =========================================================================

    async def upsert_chunk(
        self,
        chunk_id: str,
        dense_vector: list[float],
        sparse_vector: dict[int, float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Upsert a single chunk with vectors."""
        vectors = {"dense": dense_vector}
        if sparse_vector:
            vectors["sparse"] = SparseVector(
                indices=list(sparse_vector.keys()),
                values=list(sparse_vector.values()),
            )

        # Store original chunk_id in payload, use UUID for Qdrant point ID
        point_payload = payload or {}
        point_payload["chunk_id"] = chunk_id

        point = PointStruct(
            id=string_to_uuid(chunk_id),
            vector=vectors,
            payload=point_payload,
        )

        await self.client.upsert(
            collection_name=self.chunks_collection,
            points=[point],
        )

    async def upsert_chunks_batch(
        self,
        chunks: list[dict[str, Any]],
    ) -> None:
        """Upsert multiple chunks in batch.

        Each chunk dict should have:
        - id: str
        - dense_vector: list[float]
        - sparse_vector: dict[int, float] | None
        - payload: dict[str, Any]
        """
        points = []
        for chunk in chunks:
            vectors = {"dense": chunk["dense_vector"]}
            if chunk.get("sparse_vector"):
                vectors["sparse"] = SparseVector(
                    indices=list(chunk["sparse_vector"].keys()),
                    values=list(chunk["sparse_vector"].values()),
                )

            # Store original chunk_id in payload, use UUID for Qdrant point ID
            point_payload = chunk.get("payload", {})
            point_payload["chunk_id"] = chunk["id"]

            points.append(
                PointStruct(
                    id=string_to_uuid(chunk["id"]),
                    vector=vectors,
                    payload=point_payload,
                )
            )

        await self.client.upsert(
            collection_name=self.chunks_collection,
            points=points,
        )
        logger.info("Upserted chunks batch", count=len(points))

    async def search_chunks_dense(
        self,
        query_vector: list[float],
        limit: int = 20,
        acl_filter: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[ScoredPoint]:
        """Search chunks using dense vectors."""
        filter_conditions = self._build_filter(acl_filter, metadata_filter)

        # Use query_points with the dense vector (qdrant-client v1.16+)
        results = await self.client.query_points(
            collection_name=self.chunks_collection,
            query=query_vector,
            using="dense",
            query_filter=filter_conditions,
            limit=limit,
            with_payload=True,
        )
        return results.points

    async def search_chunks_sparse(
        self,
        query_sparse: dict[int, float],
        limit: int = 20,
        acl_filter: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[ScoredPoint]:
        """Search chunks using sparse vectors (BM25-style)."""
        filter_conditions = self._build_filter(acl_filter, metadata_filter)

        sparse_vector = SparseVector(
            indices=list(query_sparse.keys()),
            values=list(query_sparse.values()),
        )

        # Use query_points with the sparse vector (qdrant-client v1.16+)
        results = await self.client.query_points(
            collection_name=self.chunks_collection,
            query=sparse_vector,
            using="sparse",
            query_filter=filter_conditions,
            limit=limit,
            with_payload=True,
        )
        return results.points

    async def search_chunks_hybrid(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        limit: int = 20,
        dense_weight: float = 0.5,
        acl_filter: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining dense and sparse, using RRF fusion."""
        # Get both result sets
        dense_results = await self.search_chunks_dense(
            dense_vector, limit=limit * 2, acl_filter=acl_filter, metadata_filter=metadata_filter
        )
        sparse_results = await self.search_chunks_sparse(
            sparse_vector, limit=limit * 2, acl_filter=acl_filter, metadata_filter=metadata_filter
        )

        # RRF fusion
        fused = self._rrf_fusion(dense_results, sparse_results, k=60)

        return fused[:limit]

    def _rrf_fusion(
        self,
        dense_results: list[ScoredPoint],
        sparse_results: list[ScoredPoint],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion of two result sets.

        Also preserves original dense scores for better confidence calculation
        when sparse results are empty (common with migrated collections).
        """
        rrf_scores: dict[str, float] = {}
        dense_scores: dict[str, float] = {}  # Preserve original cosine similarity
        payloads: dict[str, dict[str, Any]] = {}

        # Score from dense results
        for rank, point in enumerate(dense_results):
            point_id = str(point.id)
            rrf_scores[point_id] = rrf_scores.get(point_id, 0) + 1 / (k + rank + 1)
            dense_scores[point_id] = point.score  # Original cosine similarity
            payloads[point_id] = point.payload or {}

        # Score from sparse results
        for rank, point in enumerate(sparse_results):
            point_id = str(point.id)
            rrf_scores[point_id] = rrf_scores.get(point_id, 0) + 1 / (k + rank + 1)
            payloads[point_id] = point.payload or {}

        # Sort by fused score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # If no sparse results, use dense scores directly for better interpretability
        # Otherwise normalize RRF scores to 0-1 range
        use_dense_scores = len(sparse_results) == 0

        return [
            {
                "id": pid,
                # Use original dense score if no sparse results, otherwise use RRF
                "score": dense_scores.get(pid, 0) if use_dense_scores else rrf_scores[pid],
                "rrf_score": rrf_scores[pid],  # Always include RRF for reference
                "dense_score": dense_scores.get(pid, 0),  # Always include dense for reference
                "payload": payloads[pid],
            }
            for pid in sorted_ids
        ]

    async def delete_chunks_by_doc(self, doc_id: str) -> None:
        """Delete all chunks for a document."""
        await self.client.delete(
            collection_name=self.chunks_collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id),
                    )
                ]
            ),
        )
        logger.info("Deleted chunks for document", doc_id=doc_id)

    # =========================================================================
    # Entity Operations
    # =========================================================================

    async def upsert_entity(
        self,
        entity_id: str,
        vector: list[float],
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Upsert an entity embedding."""
        point = PointStruct(
            id=entity_id,
            vector=vector,
            payload=payload or {},
        )

        await self.client.upsert(
            collection_name=self.entities_collection,
            points=[point],
        )

    async def search_entities(
        self,
        query_vector: list[float],
        limit: int = 10,
        entity_type: str | None = None,
    ) -> list[ScoredPoint]:
        """Search similar entities."""
        filter_conditions = None
        if entity_type:
            filter_conditions = Filter(
                must=[
                    FieldCondition(
                        key="entity_type",
                        match=MatchValue(value=entity_type),
                    )
                ]
            )

        results = await self.client.search(
            collection_name=self.entities_collection,
            query_vector=query_vector,
            query_filter=filter_conditions,
            limit=limit,
            with_payload=True,
        )
        return results

    # =========================================================================
    # Helpers
    # =========================================================================

    def _build_filter(
        self,
        acl_filter: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> Filter | None:
        """Build Qdrant filter from ACL and metadata."""
        conditions = []

        # ACL filter: user must have access (acl contains user role or "*")
        if acl_filter:
            # Include documents accessible by any of the user's roles or public (*)
            acl_with_public = acl_filter + ["*"]
            conditions.append(
                FieldCondition(
                    key="acl",
                    match=MatchAny(any=acl_with_public),
                )
            )

        # Metadata filters
        if metadata_filter:
            for key, value in metadata_filter.items():
                if isinstance(value, list):
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchAny(any=value),
                        )
                    )
                else:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value),
                        )
                    )

        if conditions:
            return Filter(must=conditions)
        return None

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> bool:
        """Check Qdrant connectivity."""
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False


# Global client instance
_client: QdrantVectorClient | None = None


async def get_qdrant() -> QdrantVectorClient:
    """Get or create Qdrant client."""
    global _client
    if _client is None:
        _client = QdrantVectorClient()
        await _client.connect()
        await _client.initialize_collections()
    return _client
