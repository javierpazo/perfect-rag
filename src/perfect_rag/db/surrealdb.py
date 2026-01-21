"""SurrealDB client wrapper."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from surrealdb import AsyncSurreal

from perfect_rag.config import Settings, get_settings
from perfect_rag.models.chunk import Chunk
from perfect_rag.models.document import Document
from perfect_rag.models.entity import Entity
from perfect_rag.models.relation import Relation

logger = structlog.get_logger(__name__)


class SurrealDBClient:
    """Async SurrealDB client with connection management."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client: Any | None = None
        self._connected = False

    @property
    def client(self) -> Any:
        """Get the underlying SurrealDB client."""
        if self._client is None:
            raise RuntimeError("SurrealDB client not connected. Call connect() first.")
        return self._client

    async def connect(self) -> None:
        """Establish connection to SurrealDB."""
        if self._connected:
            return

        try:
            self._client = AsyncSurreal(self.settings.surrealdb_url)
            await self._client.connect()
            await self._client.signin({
                "username": self.settings.surrealdb_user,
                "password": self.settings.surrealdb_pass,
            })
            await self._client.use(
                self.settings.surrealdb_namespace,
                self.settings.surrealdb_database,
            )
            self._connected = True
            logger.info(
                "Connected to SurrealDB",
                url=self.settings.surrealdb_url,
                namespace=self.settings.surrealdb_namespace,
                database=self.settings.surrealdb_database,
            )
        except Exception as e:
            logger.error("Failed to connect to SurrealDB", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Close connection."""
        if self._client and self._connected:
            await self._client.close()
            self._connected = False
            logger.info("Disconnected from SurrealDB")

    async def initialize_schema(self, schema_path: str | None = None) -> None:
        """Initialize database schema from SQL file."""
        if schema_path is None:
            schema_path = Path(__file__).parent.parent.parent.parent / "config" / "surrealdb" / "init.surql"
        else:
            schema_path = Path(schema_path)

        if not schema_path.exists():
            logger.warning("Schema file not found", path=str(schema_path))
            return

        schema_sql = schema_path.read_text()
        await self.query(schema_sql)
        logger.info("Schema initialized", path=str(schema_path))

    @asynccontextmanager
    async def session(self):
        """Context manager for database session."""
        await self.connect()
        try:
            yield self
        finally:
            pass  # Keep connection open for reuse

    async def query(self, sql: str, params: dict[str, Any] | None = None) -> list[Any]:
        """Execute a SurrealQL query."""
        if not self._connected:
            await self.connect()

        try:
            result = await self._client.query(sql, params or {})
            return result
        except Exception as e:
            logger.error("Query failed", sql=sql[:100], error=str(e))
            raise

    # =========================================================================
    # Document Operations
    # =========================================================================

    async def create_document(self, doc: Document) -> Document:
        """Create a new document."""
        result = await self.query(
            """
            CREATE document CONTENT {
                id: $id,
                source: $source,
                format: $format,
                content: $content,
                metadata: $metadata,
                acl: $acl,
                created_at: $created_at,
                updated_at: $updated_at,
                content_hash: $content_hash,
                chunk_count: $chunk_count
            }
            """,
            doc.model_dump(mode="json"),
        )
        logger.info("Created document", doc_id=doc.id)
        return doc

    async def get_document(self, doc_id: str) -> Document | None:
        """Get document by ID."""
        result = await self.query(
            "SELECT * FROM document WHERE id = $id",
            {"id": doc_id},
        )
        if result and result[0]:
            return Document(**result[0][0])
        return None

    async def get_document_by_hash(self, content_hash: str) -> Document | None:
        """Get document by content hash (for deduplication)."""
        result = await self.query(
            "SELECT * FROM document WHERE content_hash = $hash",
            {"hash": content_hash},
        )
        if result and result[0]:
            return Document(**result[0][0])
        return None

    async def update_document_chunk_count(self, doc_id: str, count: int) -> None:
        """Update document's chunk count."""
        await self.query(
            "UPDATE document SET chunk_count = $count, updated_at = time::now() WHERE id = $id",
            {"id": doc_id, "count": count},
        )

    async def update_document(self, doc: Document) -> None:
        """Update document with all fields."""
        await self.query(
            """
            UPDATE document SET
                status = $status,
                chunk_count = $chunk_count,
                entity_count = $entity_count,
                updated_at = time::now()
            WHERE id = $id
            """,
            {
                "id": doc.id,
                "status": doc.status.value if hasattr(doc.status, 'value') else doc.status,
                "chunk_count": doc.chunk_count,
                "entity_count": doc.entity_count,
            },
        )

    async def delete_document(self, doc_id: str) -> None:
        """Delete document and all related data."""
        # Delete chunks first
        await self.query("DELETE chunk WHERE doc_id = $id", {"id": doc_id})
        # Delete document
        await self.query("DELETE document WHERE id = $id", {"id": doc_id})
        logger.info("Deleted document", doc_id=doc_id)

    async def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        format_filter: str | None = None,
    ) -> list[Document]:
        """List documents with pagination."""
        if format_filter:
            result = await self.query(
                """
                SELECT * FROM document
                WHERE format = $format
                ORDER BY created_at DESC
                LIMIT $limit START $offset
                """,
                {"format": format_filter, "limit": limit, "offset": offset},
            )
        else:
            result = await self.query(
                """
                SELECT * FROM document
                ORDER BY created_at DESC
                LIMIT $limit START $offset
                """,
                {"limit": limit, "offset": offset},
            )
        rows = result[0] if result else []
        documents = []
        for r in rows:
            if isinstance(r, dict):
                documents.append(Document(**r))
            elif hasattr(r, '__dict__'):
                documents.append(Document(**r.__dict__))
        return documents

    # =========================================================================
    # Chunk Operations
    # =========================================================================

    async def create_chunk(self, chunk: Chunk) -> Chunk:
        """Create a single chunk."""
        await self.query(
            """
            CREATE chunk CONTENT {
                id: $id,
                doc_id: $doc_id,
                content: $content,
                offset_start: $offset_start,
                offset_end: $offset_end,
                token_count: $token_count,
                chunk_index: $chunk_index,
                metadata: $metadata,
                acl: $acl
            }
            """,
            chunk.model_dump(mode="json"),
        )
        return chunk

    async def create_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Create multiple chunks in batch."""
        for chunk in chunks:
            await self.query(
                """
                CREATE chunk CONTENT {
                    id: $id,
                    doc_id: $doc_id,
                    content: $content,
                    offset_start: $offset_start,
                    offset_end: $offset_end,
                    token_count: $token_count,
                    chunk_index: $chunk_index,
                    metadata: $metadata,
                    acl: $acl
                }
                """,
                chunk.model_dump(mode="json"),
            )
        logger.info("Created chunks", count=len(chunks))
        return chunks

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Get chunk by ID."""
        result = await self.query(
            "SELECT * FROM chunk WHERE id = $id",
            {"id": chunk_id},
        )
        if result and result[0]:
            return Chunk(**result[0][0])
        return None

    async def get_chunks_by_doc(self, doc_id: str) -> list[Chunk]:
        """Get all chunks for a document."""
        result = await self.query(
            "SELECT * FROM chunk WHERE doc_id = $id ORDER BY chunk_index",
            {"id": doc_id},
        )
        return [Chunk(**r) for r in (result[0] if result else [])]

    async def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        """Get multiple chunks by IDs."""
        result = await self.query(
            "SELECT * FROM chunk WHERE id IN $ids",
            {"ids": chunk_ids},
        )
        return [Chunk(**r) for r in (result[0] if result else [])]

    # =========================================================================
    # Entity Operations
    # =========================================================================

    async def create_entity(self, entity: Entity) -> Entity:
        """Create or update an entity."""
        # Check if entity with same normalized name and type exists
        existing = await self.query(
            """
            SELECT * FROM entity
            WHERE normalized_name = $name AND entity_type = $type
            """,
            {"name": entity.normalized_name, "type": entity.entity_type.value},
        )

        if existing and existing[0]:
            # Merge with existing entity
            existing_entity = Entity(**existing[0][0])
            merged = existing_entity.merge_with(entity)
            await self.query(
                """
                UPDATE entity SET
                    aliases = $aliases,
                    confidence = $confidence,
                    source_chunks = $source_chunks,
                    mention_count = $mention_count,
                    metadata = $metadata
                WHERE id = $id
                """,
                merged.model_dump(mode="json"),
            )
            return merged
        else:
            await self.query(
                """
                CREATE entity CONTENT {
                    id: $id,
                    name: $name,
                    normalized_name: $normalized_name,
                    entity_type: $entity_type,
                    aliases: $aliases,
                    confidence: $confidence,
                    source_chunks: $source_chunks,
                    mention_count: $mention_count,
                    metadata: $metadata
                }
                """,
                {
                    **entity.model_dump(mode="json"),
                    "entity_type": entity.entity_type.value,
                },
            )
            return entity

    async def get_entity(self, entity_id: str) -> Entity | None:
        """Get entity by ID."""
        result = await self.query(
            "SELECT * FROM entity WHERE id = $id",
            {"id": entity_id},
        )
        if result and result[0]:
            return Entity(**result[0][0])
        return None

    async def get_entity_by_name(
        self, name: str, entity_type: str | None = None
    ) -> Entity | None:
        """Get entity by normalized name."""
        normalized = name.lower().strip()
        if entity_type:
            result = await self.query(
                """
                SELECT * FROM entity
                WHERE normalized_name = $name AND entity_type = $type
                """,
                {"name": normalized, "type": entity_type},
            )
        else:
            result = await self.query(
                "SELECT * FROM entity WHERE normalized_name = $name",
                {"name": normalized},
            )
        if result and result[0]:
            return Entity(**result[0][0])
        return None

    async def search_entities(
        self,
        query: str,
        entity_type: str | None = None,
        limit: int = 10,
    ) -> list[Entity]:
        """Search entities by name (partial match)."""
        normalized = query.lower().strip()
        if entity_type:
            result = await self.query(
                """
                SELECT * FROM entity
                WHERE normalized_name CONTAINS $query AND entity_type = $type
                ORDER BY mention_count DESC
                LIMIT $limit
                """,
                {"query": normalized, "type": entity_type, "limit": limit},
            )
        else:
            result = await self.query(
                """
                SELECT * FROM entity
                WHERE normalized_name CONTAINS $query
                ORDER BY mention_count DESC
                LIMIT $limit
                """,
                {"query": normalized, "limit": limit},
            )
        return [Entity(**r) for r in (result[0] if result else [])]

    # =========================================================================
    # Relation Operations
    # =========================================================================

    async def create_relation(self, relation: Relation) -> Relation:
        """Create a relation between entities."""
        await self.query(
            """
            RELATE entity:$head_id -> related_to -> entity:$tail_id SET
                relation_type = $relation_type,
                confidence = $confidence,
                evidence_chunk = $evidence_chunk_id,
                span_start = $span_start,
                span_end = $span_end,
                metadata = $metadata
            """,
            {
                **relation.model_dump(mode="json"),
                "relation_type": relation.relation_type.value,
            },
        )
        logger.debug(
            "Created relation",
            head=relation.head_id,
            type=relation.relation_type.value,
            tail=relation.tail_id,
        )
        return relation

    async def create_mention(
        self,
        entity_id: str,
        chunk_id: str,
        confidence: float = 1.0,
        span: tuple[int, int] | None = None,
    ) -> None:
        """Create mentioned_in relation."""
        await self.query(
            """
            RELATE entity:$entity_id -> mentioned_in -> chunk:$chunk_id SET
                confidence = $confidence,
                span_start = $span_start,
                span_end = $span_end
            """,
            {
                "entity_id": entity_id,
                "chunk_id": chunk_id,
                "confidence": confidence,
                "span_start": span[0] if span else None,
                "span_end": span[1] if span else None,
            },
        )

    async def get_entity_relations(
        self,
        entity_id: str,
        direction: str = "both",
        relation_types: list[str] | None = None,
        min_confidence: float = 0.0,
    ) -> list[Relation]:
        """Get relations for an entity."""
        relations = []

        if direction in ("out", "both"):
            result = await self.query(
                """
                SELECT
                    out.id AS tail_id,
                    out.name AS tail_name,
                    relation_type,
                    confidence,
                    evidence_chunk,
                    span_start,
                    span_end,
                    metadata
                FROM related_to
                WHERE in.id = $id AND confidence >= $min_conf
                """,
                {"id": entity_id, "min_conf": min_confidence},
            )
            for r in (result[0] if result else []):
                if relation_types is None or r["relation_type"] in relation_types:
                    relations.append(
                        Relation(
                            id=f"{entity_id}:{r['relation_type']}:{r['tail_id']}",
                            head_id=entity_id,
                            relation_type=r["relation_type"],
                            tail_id=r["tail_id"],
                            tail_name=r.get("tail_name"),
                            confidence=r["confidence"],
                            evidence_chunk_id=r.get("evidence_chunk"),
                            span_start=r.get("span_start"),
                            span_end=r.get("span_end"),
                            metadata=r.get("metadata", {}),
                        )
                    )

        if direction in ("in", "both"):
            result = await self.query(
                """
                SELECT
                    in.id AS head_id,
                    in.name AS head_name,
                    relation_type,
                    confidence,
                    evidence_chunk,
                    span_start,
                    span_end,
                    metadata
                FROM related_to
                WHERE out.id = $id AND confidence >= $min_conf
                """,
                {"id": entity_id, "min_conf": min_confidence},
            )
            for r in (result[0] if result else []):
                if relation_types is None or r["relation_type"] in relation_types:
                    relations.append(
                        Relation(
                            id=f"{r['head_id']}:{r['relation_type']}:{entity_id}",
                            head_id=r["head_id"],
                            head_name=r.get("head_name"),
                            relation_type=r["relation_type"],
                            tail_id=entity_id,
                            confidence=r["confidence"],
                            evidence_chunk_id=r.get("evidence_chunk"),
                            span_start=r.get("span_start"),
                            span_end=r.get("span_end"),
                            metadata=r.get("metadata", {}),
                        )
                    )

        return relations

    # =========================================================================
    # Graph Expansion
    # =========================================================================

    async def expand_subgraph(
        self,
        seed_entity_ids: list[str],
        max_hops: int = 2,
        relation_types: list[str] | None = None,
        min_confidence: float = 0.7,
    ) -> dict[str, Any]:
        """Expand subgraph from seed entities."""
        entities: dict[str, Entity] = {}
        relations: list[Relation] = []
        evidence_chunk_ids: set[str] = set()
        visited: set[str] = set()
        frontier = set(seed_entity_ids)

        for hop in range(max_hops):
            next_frontier: set[str] = set()

            for entity_id in frontier:
                if entity_id in visited:
                    continue
                visited.add(entity_id)

                # Get entity
                entity = await self.get_entity(entity_id)
                if entity:
                    entities[entity_id] = entity

                # Get relations
                rels = await self.get_entity_relations(
                    entity_id,
                    relation_types=relation_types,
                    min_confidence=min_confidence,
                )

                for rel in rels:
                    relations.append(rel)
                    if rel.evidence_chunk_id:
                        evidence_chunk_ids.add(rel.evidence_chunk_id)

                    # Add connected entities to next frontier
                    other_id = rel.tail_id if rel.head_id == entity_id else rel.head_id
                    if other_id not in visited:
                        next_frontier.add(other_id)

            frontier = next_frontier

        # Get evidence chunks
        evidence_chunks = await self.get_chunks_by_ids(list(evidence_chunk_ids))

        return {
            "entities": list(entities.values()),
            "relations": relations,
            "evidence_chunks": evidence_chunks,
        }

    # =========================================================================
    # Cache Operations
    # =========================================================================

    async def get_cache(self, key: str) -> dict[str, Any] | None:
        """Get cache entry."""
        result = await self.query(
            """
            UPDATE cache SET
                access_count = access_count + 1,
                last_accessed = time::now()
            WHERE key = $key AND ttl > time::now()
            RETURN AFTER
            """,
            {"key": key},
        )
        if result and result[0]:
            return result[0][0]
        return None

    async def set_cache(
        self,
        key: str,
        chunk_ids: list[str],
        entity_ids: list[str],
        subgraph: dict[str, Any],
        context_tokens: int,
        ttl_hours: int = 24,
    ) -> None:
        """Set cache entry."""
        await self.query(
            """
            CREATE cache CONTENT {
                key: $key,
                chunk_ids: $chunk_ids,
                entity_ids: $entity_ids,
                subgraph: $subgraph,
                context_tokens: $context_tokens,
                access_count: 1,
                last_accessed: time::now(),
                created_at: time::now(),
                ttl: time::now() + $ttl
            }
            """,
            {
                "key": key,
                "chunk_ids": chunk_ids,
                "entity_ids": entity_ids,
                "subgraph": subgraph,
                "context_tokens": context_tokens,
                "ttl": f"{ttl_hours}h",
            },
        )

    async def clean_expired_cache(self) -> int:
        """Remove expired cache entries."""
        result = await self.query("DELETE cache WHERE ttl < time::now()")
        return len(result[0]) if result else 0

    # =========================================================================
    # Query Log Operations
    # =========================================================================

    async def log_query(self, log: dict[str, Any]) -> None:
        """Log a query for learning."""
        await self.query(
            """
            CREATE query_log CONTENT $log
            """,
            {"log": log},
        )

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            await self.query("SELECT * FROM document LIMIT 1")
            return True
        except Exception:
            return False


# Global client instance
_client: SurrealDBClient | None = None


async def get_surrealdb() -> SurrealDBClient:
    """Get or create SurrealDB client."""
    global _client
    if _client is None:
        _client = SurrealDBClient()
        await _client.connect()
    return _client
