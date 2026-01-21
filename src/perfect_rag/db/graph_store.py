"""
Unified graph store interface.

Provides graph operations using SurrealDB as primary store,
with optional Oxigraph backend for advanced SPARQL reasoning.
"""

from abc import ABC, abstractmethod
from typing import Any

import structlog

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class GraphStore(ABC):
    """Abstract graph store interface."""

    @abstractmethod
    async def add_entity(self, entity: dict[str, Any]) -> str:
        """Add entity to graph."""
        pass

    @abstractmethod
    async def add_relation(
        self,
        head_id: str,
        relation_type: str,
        tail_id: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Add relation between entities."""
        pass

    @abstractmethod
    async def get_neighbors(
        self,
        entity_id: str,
        max_hops: int = 1,
        relation_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get neighboring entities."""
        pass

    @abstractmethod
    async def find_paths(
        self,
        start_id: str,
        end_id: str,
        max_hops: int = 3,
    ) -> list[list[dict[str, str]]]:
        """Find paths between entities."""
        pass

    @abstractmethod
    async def expand_subgraph(
        self,
        seed_ids: list[str],
        max_hops: int = 2,
        min_confidence: float = 0.7,
    ) -> dict[str, Any]:
        """Expand subgraph from seed entities."""
        pass


class SurrealDBGraphStore(GraphStore):
    """
    Graph store using SurrealDB's native graph features.

    SurrealDB supports:
    - RELATE for creating edges
    - Graph traversal with -> and <-
    - Path finding with recursive queries
    """

    def __init__(self, surrealdb_client):
        self.db = surrealdb_client

    async def add_entity(self, entity: dict[str, Any]) -> str:
        """Add entity using SurrealDB."""
        entity_id = entity.get("id")
        await self.db.query(
            """
            CREATE entity CONTENT $entity
            ON DUPLICATE KEY UPDATE
                aliases = array::union(aliases, $entity.aliases),
                source_chunks = array::union(source_chunks, $entity.source_chunks),
                mention_count = mention_count + 1
            """,
            {"entity": entity},
        )
        return entity_id

    async def add_relation(
        self,
        head_id: str,
        relation_type: str,
        tail_id: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Add relation using RELATE."""
        props = properties or {}
        await self.db.query(
            f"""
            RELATE entity:{head_id} -> {relation_type} -> entity:{tail_id}
            CONTENT $props
            """,
            {"props": props},
        )

    async def get_neighbors(
        self,
        entity_id: str,
        max_hops: int = 1,
        relation_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get neighbors using SurrealDB graph traversal."""
        # Build relation filter
        rel_filter = ""
        if relation_types:
            rel_names = ", ".join(f'"{r}"' for r in relation_types)
            rel_filter = f"WHERE type::thing(edge).tb IN [{rel_names}]"

        # Single hop
        if max_hops == 1:
            result = await self.db.query(
                f"""
                SELECT
                    id,
                    name,
                    entity_type,
                    <->(* {rel_filter}) AS connections
                FROM entity:{entity_id}
                """,
                {},
            )
        else:
            # Multi-hop using recursive-like pattern
            result = await self.db.query(
                f"""
                SELECT
                    id,
                    name,
                    entity_type,
                    ->(*..{max_hops} {rel_filter}) AS outgoing,
                    <-(*..{max_hops} {rel_filter}) AS incoming
                FROM entity:{entity_id}
                """,
                {},
            )

        if result and result[0]:
            return result[0][0] if result[0] else {}
        return {}

    async def find_paths(
        self,
        start_id: str,
        end_id: str,
        max_hops: int = 3,
    ) -> list[list[dict[str, str]]]:
        """Find paths using SurrealDB."""
        paths = []

        # Direct connection (1 hop)
        result = await self.db.query(
            """
            SELECT
                type::thing(edge).tb AS relation
            FROM entity:$start
            WHERE ->*.id CONTAINS entity:$end
            """,
            {"start": start_id, "end": end_id},
        )

        if result and result[0]:
            for r in result[0]:
                paths.append([{
                    "from": start_id,
                    "relation": r.get("relation"),
                    "to": end_id,
                }])

        # 2-hop paths
        if max_hops >= 2 and not paths:
            result = await self.db.query(
                """
                SELECT
                    ->*->*.id AS path,
                    type::thing(edge).tb AS relations
                FROM entity:$start
                WHERE ->*->*.id CONTAINS entity:$end
                LIMIT 10
                """,
                {"start": start_id, "end": end_id},
            )

            # Process results into path format
            if result and result[0]:
                for r in result[0]:
                    path_nodes = r.get("path", [])
                    if path_nodes:
                        paths.append(self._format_path(start_id, path_nodes, end_id))

        return paths

    def _format_path(
        self,
        start: str,
        intermediates: list[str],
        end: str,
    ) -> list[dict[str, str]]:
        """Format path nodes into structured format."""
        path = []
        current = start
        for next_node in intermediates + [end]:
            path.append({
                "from": current,
                "relation": "related_to",  # Simplified
                "to": next_node,
            })
            current = next_node
        return path

    async def expand_subgraph(
        self,
        seed_ids: list[str],
        max_hops: int = 2,
        min_confidence: float = 0.7,
    ) -> dict[str, Any]:
        """Expand subgraph from seeds using SurrealDB graph traversal."""
        entities = {}
        relations = []
        evidence_chunks = set()

        for entity_id in seed_ids:
            # Get entity with relations
            result = await self.db.query(
                """
                SELECT
                    *,
                    ->related_to.* AS out_rels,
                    <-related_to.* AS in_rels
                FROM entity:$id
                """,
                {"id": entity_id},
            )

            if result and result[0]:
                entity_data = result[0][0] if result[0] else {}
                entities[entity_id] = entity_data

                # Process outgoing relations
                for rel in entity_data.get("out_rels", []) or []:
                    if rel.get("confidence", 1.0) >= min_confidence:
                        relations.append(rel)
                        if rel.get("evidence_chunk_id"):
                            evidence_chunks.add(rel["evidence_chunk_id"])

                # Process incoming relations
                for rel in entity_data.get("in_rels", []) or []:
                    if rel.get("confidence", 1.0) >= min_confidence:
                        relations.append(rel)
                        if rel.get("evidence_chunk_id"):
                            evidence_chunks.add(rel["evidence_chunk_id"])

        # Get evidence chunks
        chunks = []
        if evidence_chunks:
            chunk_result = await self.db.query(
                "SELECT * FROM chunk WHERE id IN $ids",
                {"ids": list(evidence_chunks)},
            )
            chunks = chunk_result[0] if chunk_result else []

        return {
            "entities": list(entities.values()),
            "relations": relations,
            "evidence_chunks": chunks,
        }

    async def get_entity_neighborhood(
        self,
        entity_id: str,
        max_hops: int = 1,
    ) -> dict[str, Any]:
        """Get entity and its immediate neighborhood (compatible with Oxigraph interface)."""
        # Get outgoing relations
        outgoing_result = await self.db.query(
            """
            SELECT
                out.id AS tail_id,
                out.name AS tail_name,
                relation_type,
                confidence
            FROM related_to
            WHERE in.id = $id
            """,
            {"id": f"entity:{entity_id}"},
        )

        # Get incoming relations
        incoming_result = await self.db.query(
            """
            SELECT
                in.id AS head_id,
                in.name AS head_name,
                relation_type,
                confidence
            FROM related_to
            WHERE out.id = $id
            """,
            {"id": f"entity:{entity_id}"},
        )

        outgoing = []
        if outgoing_result and outgoing_result[0]:
            for r in outgoing_result[0]:
                outgoing.append({
                    "relation": r.get("relation_type", "related_to"),
                    "target_id": str(r.get("tail_id", "")).replace("entity:", ""),
                    "target_name": r.get("tail_name"),
                })

        incoming = []
        if incoming_result and incoming_result[0]:
            for r in incoming_result[0]:
                incoming.append({
                    "source_id": str(r.get("head_id", "")).replace("entity:", ""),
                    "source_name": r.get("head_name"),
                    "relation": r.get("relation_type", "related_to"),
                })

        return {
            "entity_id": entity_id,
            "outgoing": outgoing,
            "incoming": incoming,
        }


class HybridGraphStore(GraphStore):
    """
    Hybrid graph store using SurrealDB primary with optional Oxigraph.

    Uses SurrealDB for:
    - Entity/relation storage
    - Basic graph traversal
    - Most queries

    Uses Oxigraph (optional) for:
    - Complex SPARQL reasoning
    - Ontology inference
    - Path finding with constraints
    """

    def __init__(
        self,
        surrealdb_client,
        oxigraph_client=None,
        use_oxigraph_for_reasoning: bool = False,
    ):
        self.surreal = SurrealDBGraphStore(surrealdb_client)
        self.oxigraph = oxigraph_client
        self.use_reasoning = use_oxigraph_for_reasoning and oxigraph_client is not None
        self._surrealdb_client = surrealdb_client

    async def add_entity(self, entity: dict[str, Any]) -> str:
        """Add to SurrealDB, optionally sync to Oxigraph."""
        entity_id = await self.surreal.add_entity(entity)

        if self.use_reasoning and self.oxigraph:
            # Sync to Oxigraph for reasoning
            try:
                from perfect_rag.models.entity import Entity
                await self.oxigraph.sync_entity(Entity(**entity))
            except Exception as e:
                logger.warning("Failed to sync entity to Oxigraph", error=str(e))

        return entity_id

    async def add_relation(
        self,
        head_id: str,
        relation_type: str,
        tail_id: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Add to SurrealDB, optionally sync to Oxigraph."""
        await self.surreal.add_relation(head_id, relation_type, tail_id, properties)

        if self.use_reasoning and self.oxigraph:
            try:
                from perfect_rag.models.relation import Relation, RelationType
                rel = Relation(
                    id=f"{head_id}:{relation_type}:{tail_id}",
                    head_id=head_id,
                    relation_type=RelationType(relation_type),
                    tail_id=tail_id,
                    **(properties or {}),
                )
                await self.oxigraph.sync_relation(rel)
            except Exception as e:
                logger.warning("Failed to sync relation to Oxigraph", error=str(e))

    async def get_neighbors(
        self,
        entity_id: str,
        max_hops: int = 1,
        relation_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get neighbors from SurrealDB."""
        return await self.surreal.get_neighbors(entity_id, max_hops, relation_types)

    async def find_paths(
        self,
        start_id: str,
        end_id: str,
        max_hops: int = 3,
    ) -> list[list[dict[str, str]]]:
        """Find paths - use Oxigraph if available for complex queries."""
        if self.use_reasoning and self.oxigraph and max_hops > 2:
            # Use Oxigraph for complex path finding
            try:
                return await self.oxigraph.find_paths(start_id, end_id, max_length=max_hops)
            except Exception:
                pass  # Fall back to SurrealDB

        return await self.surreal.find_paths(start_id, end_id, max_hops)

    async def expand_subgraph(
        self,
        seed_ids: list[str],
        max_hops: int = 2,
        min_confidence: float = 0.7,
    ) -> dict[str, Any]:
        """Expand subgraph using SurrealDB."""
        return await self.surreal.expand_subgraph(seed_ids, max_hops, min_confidence)

    async def get_entity_neighborhood(
        self,
        entity_id: str,
        max_hops: int = 1,
    ) -> dict[str, Any]:
        """Get entity neighborhood - prefer Oxigraph for reasoning if available."""
        if self.use_reasoning and self.oxigraph:
            try:
                return await self.oxigraph.get_entity_neighborhood(entity_id, max_hops)
            except Exception:
                pass  # Fall back to SurrealDB

        return await self.surreal.get_entity_neighborhood(entity_id, max_hops)

    async def sparql_query(self, sparql: str) -> list[dict[str, Any]]:
        """Execute SPARQL query (requires Oxigraph)."""
        if not self.oxigraph:
            raise RuntimeError("Oxigraph not configured for SPARQL queries")
        return await self.oxigraph.query(sparql)

    async def get_entities_by_type(
        self,
        entity_type: str,
        limit: int = 100,
    ) -> list[dict[str, str]]:
        """Get entities by type - use Oxigraph if available, else SurrealDB."""
        if self.use_reasoning and self.oxigraph:
            try:
                return await self.oxigraph.get_entities_by_type(entity_type, limit)
            except Exception:
                pass

        # Fallback to SurrealDB
        result = await self._surrealdb_client.query(
            """
            SELECT id, name FROM entity
            WHERE entity_type = $type
            LIMIT $limit
            """,
            {"type": entity_type, "limit": limit},
        )

        if result and result[0]:
            return [
                {"id": r.get("id", "").replace("entity:", ""), "name": r.get("name")}
                for r in result[0]
            ]
        return []


# Factory function for creating graph stores
async def create_graph_store(
    surrealdb_client,
    oxigraph_client=None,
    settings: Settings | None = None,
) -> GraphStore:
    """
    Create appropriate graph store based on configuration.

    Args:
        surrealdb_client: SurrealDB client instance
        oxigraph_client: Optional Oxigraph client instance
        settings: Application settings

    Returns:
        GraphStore instance (SurrealDB-only or Hybrid)
    """
    settings = settings or get_settings()

    # Check if Oxigraph is configured and enabled
    use_oxigraph = getattr(settings, "use_oxigraph", False)

    if use_oxigraph and oxigraph_client is not None:
        logger.info("Creating hybrid graph store with Oxigraph for SPARQL reasoning")
        return HybridGraphStore(
            surrealdb_client=surrealdb_client,
            oxigraph_client=oxigraph_client,
            use_oxigraph_for_reasoning=True,
        )
    else:
        logger.info("Creating SurrealDB-only graph store")
        return SurrealDBGraphStore(surrealdb_client)


# Global graph store instance
_graph_store: GraphStore | None = None


async def get_graph_store() -> GraphStore:
    """Get or create the graph store instance."""
    global _graph_store
    if _graph_store is None:
        from perfect_rag.db.surrealdb import get_surrealdb

        surrealdb = await get_surrealdb()

        settings = get_settings()
        oxigraph = None

        if getattr(settings, "use_oxigraph", False):
            try:
                from perfect_rag.db.oxigraph import get_oxigraph
                oxigraph = await get_oxigraph()
            except Exception as e:
                logger.warning("Oxigraph not available, using SurrealDB only", error=str(e))

        _graph_store = await create_graph_store(surrealdb, oxigraph, settings)

    return _graph_store
