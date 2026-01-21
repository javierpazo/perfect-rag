"""GraphRAG expansion for knowledge graph-enhanced retrieval.

Updated to use unified graph store interface with SurrealDB as primary
and optional Oxigraph for advanced SPARQL reasoning.
"""

from typing import Any

import structlog

from perfect_rag.config import Settings, get_settings
from perfect_rag.db.graph_store import GraphStore, HybridGraphStore, get_graph_store
from perfect_rag.db.surrealdb import SurrealDBClient

logger = structlog.get_logger(__name__)


class GraphRAGExpander:
    """Expand retrieval results using knowledge graph.

    GraphRAG techniques:
    1. Entity-based expansion: Find chunks mentioning related entities
    2. Path-based expansion: Find chunks along relationship paths
    3. Community expansion: Find chunks in same topic community
    4. SPARQL reasoning: Use RDF inference for semantic expansion (optional, requires Oxigraph)

    Uses unified GraphStore interface with SurrealDB as primary storage.
    Oxigraph is optional and used only when configured for SPARQL reasoning.
    """

    def __init__(
        self,
        surrealdb: SurrealDBClient,
        graph_store: GraphStore | None = None,
        settings: Settings | None = None,
    ):
        self.surrealdb = surrealdb
        self.graph_store = graph_store
        self.settings = settings or get_settings()
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure graph store is initialized."""
        if not self._initialized:
            if self.graph_store is None:
                self.graph_store = await get_graph_store()
            self._initialized = True

    async def expand(
        self,
        initial_chunks: list[dict[str, Any]],
        query_entities: list[str] | None = None,
        max_hops: int = 2,
        expansion_limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Expand initial retrieval results using knowledge graph.

        Args:
            initial_chunks: Chunks from initial vector search
            query_entities: Entity IDs mentioned in query (if extracted)
            max_hops: Maximum graph traversal hops
            expansion_limit: Maximum additional chunks to add

        Returns:
            Expanded list of chunks with graph-derived relevance
        """
        await self._ensure_initialized()

        # Collect entity IDs from initial chunks
        chunk_entities = set()
        for chunk in initial_chunks:
            entities = chunk.get("payload", {}).get("entities", [])
            chunk_entities.update(entities)

        # Add query entities
        if query_entities:
            chunk_entities.update(query_entities)

        if not chunk_entities:
            return initial_chunks

        # Expand through graph
        expanded_chunk_ids = set(c.get("id") for c in initial_chunks)
        expansion_chunks = []

        for entity_id in list(chunk_entities)[:10]:  # Limit entity expansion
            # Get entity neighborhood using unified graph store
            neighborhood = await self._get_entity_neighborhood(entity_id, max_hops)

            # Find chunks mentioning related entities
            for related_entity_id in neighborhood:
                related_chunks = await self._get_chunks_by_entity(related_entity_id)

                for chunk_id in related_chunks:
                    if chunk_id not in expanded_chunk_ids and len(expansion_chunks) < expansion_limit:
                        # Get chunk data
                        chunk_data = await self.surrealdb.get_chunk(chunk_id)
                        if chunk_data:
                            expansion_chunks.append({
                                "id": chunk_id,
                                "payload": {
                                    "text": chunk_data.get("content", ""),
                                    "doc_id": chunk_data.get("doc_id", ""),
                                    **chunk_data.get("metadata", {}),
                                },
                                "score": 0.5,  # Graph-derived score
                                "source": "graph_expansion",
                                "expansion_path": f"via {entity_id}",
                            })
                            expanded_chunk_ids.add(chunk_id)

        # Combine initial and expansion chunks
        all_chunks = initial_chunks + expansion_chunks

        return all_chunks

    async def _get_entity_neighborhood(
        self,
        entity_id: str,
        max_hops: int = 2,
    ) -> list[str]:
        """Get related entity IDs within hop distance using unified graph store."""
        related = set()

        try:
            # Get neighborhood from unified graph store (SurrealDB or Hybrid)
            if hasattr(self.graph_store, 'get_entity_neighborhood'):
                neighborhood = await self.graph_store.get_entity_neighborhood(
                    entity_id, max_hops=max_hops
                )
            else:
                # Fallback: use get_neighbors
                neighborhood = await self.graph_store.get_neighbors(
                    entity_id, max_hops=max_hops
                )

            # Extract entity IDs from outgoing relations
            for rel in neighborhood.get("outgoing", []):
                target_id = rel.get("target_id")
                if target_id:
                    related.add(target_id)

            # Extract entity IDs from incoming relations
            for rel in neighborhood.get("incoming", []):
                source_id = rel.get("source_id")
                if source_id:
                    related.add(source_id)

        except Exception as e:
            logger.warning("Failed to get entity neighborhood", entity_id=entity_id, error=str(e))

        return list(related)

    async def _get_chunks_by_entity(self, entity_id: str) -> list[str]:
        """Get chunk IDs that mention an entity."""
        try:
            # Query SurrealDB for chunks mentioning this entity
            entity_data = await self.surrealdb.get_entity(entity_id)
            if entity_data:
                return entity_data.get("source_chunks", [])
        except Exception as e:
            logger.warning("Failed to get chunks for entity", entity_id=entity_id, error=str(e))

        return []

    async def find_connecting_paths(
        self,
        entities_in_query: list[str],
        entities_in_results: list[str],
        max_path_length: int = 3,
    ) -> list[dict[str, Any]]:
        """Find relationship paths connecting query and result entities.

        Useful for explaining why certain results are relevant.
        Uses unified graph store for path finding.
        """
        await self._ensure_initialized()
        paths = []

        for query_ent in entities_in_query[:3]:  # Limit combinations
            for result_ent in entities_in_results[:5]:
                if query_ent != result_ent:
                    try:
                        # Use unified graph store for path finding
                        found_paths = await self.graph_store.find_paths(
                            query_ent, result_ent, max_hops=max_path_length
                        )
                        for path in found_paths:
                            paths.append({
                                "start": query_ent,
                                "end": result_ent,
                                "path": path,
                            })
                    except Exception:
                        pass

        return paths

    async def expand_with_sparql(
        self,
        query: str,
        initial_entities: list[str],
    ) -> list[str]:
        """Use SPARQL queries for semantic expansion.

        Examples:
        - Find all entities of same type
        - Find entities with specific relationships
        - Infer new relationships through reasoning

        Note: Requires Oxigraph to be configured via HybridGraphStore.
        Falls back to SurrealDB-based expansion if SPARQL is not available.
        """
        await self._ensure_initialized()
        expanded_entities = set(initial_entities)

        # Check if SPARQL is available (HybridGraphStore with Oxigraph)
        if isinstance(self.graph_store, HybridGraphStore) and hasattr(self.graph_store, 'sparql_query'):
            # Example: Find same-type entities using SPARQL
            for entity_id in initial_entities[:5]:
                try:
                    # Get entity type via SPARQL
                    sparql = f"""
                    SELECT ?type WHERE {{
                        <http://perfect-rag.local/entity/{entity_id}> a ?type .
                        FILTER(STRSTARTS(STR(?type), "http://perfect-rag.local/ontology#"))
                    }}
                    LIMIT 1
                    """
                    results = await self.graph_store.sparql_query(sparql)

                    if results:
                        entity_type = results[0].get("type", "").split("#")[-1]

                        # Find other entities of same type
                        same_type = await self.graph_store.get_entities_by_type(
                            entity_type, limit=10
                        )

                        for ent in same_type:
                            expanded_entities.add(ent["id"])

                except Exception as e:
                    logger.debug("SPARQL expansion failed", error=str(e))
        else:
            # Fallback: use graph store's subgraph expansion
            logger.debug("SPARQL not available, using SurrealDB-based expansion")
            try:
                subgraph = await self.graph_store.expand_subgraph(
                    seed_ids=initial_entities[:5],
                    max_hops=1,
                    min_confidence=0.7,
                )
                for entity in subgraph.get("entities", []):
                    entity_id = entity.get("id", "")
                    if entity_id:
                        expanded_entities.add(entity_id.replace("entity:", ""))
            except Exception as e:
                logger.debug("Subgraph expansion failed", error=str(e))

        return list(expanded_entities)


class CommunityDetector:
    """Detect and use communities in the knowledge graph.

    Communities are clusters of densely connected entities.
    They can represent topics, domains, or related concepts.
    """

    def __init__(
        self,
        surrealdb: SurrealDBClient,
        settings: Settings | None = None,
    ):
        self.surrealdb = surrealdb
        self.settings = settings or get_settings()
        self._communities: dict[str, list[str]] | None = None

    async def detect_communities(self) -> dict[str, list[str]]:
        """Detect communities in the knowledge graph.

        Uses a simple connected components approach.
        For production, consider using more sophisticated algorithms
        like Louvain or Label Propagation.
        """
        # Get all entities and relations
        # This is a simplified version - real implementation would use
        # graph algorithms from networkx or similar

        # For now, return cached communities or empty
        if self._communities is not None:
            return self._communities

        # TODO: Implement community detection
        self._communities = {}
        return self._communities

    async def get_community_for_entity(self, entity_id: str) -> str | None:
        """Get community ID for an entity."""
        communities = await self.detect_communities()

        for comm_id, members in communities.items():
            if entity_id in members:
                return comm_id

        return None

    async def get_community_summary(self, community_id: str) -> str | None:
        """Get or generate a summary for a community.

        These summaries can be used as additional context for generation.
        """
        # TODO: Implement community summarization
        return None


class StructuralEmbedder:
    """Generate structural embeddings from knowledge graph.

    Combines entity relationships into vector representations
    that capture graph structure, not just text content.
    """

    def __init__(
        self,
        surrealdb: SurrealDBClient,
        embedding_service: Any,
    ):
        self.surrealdb = surrealdb
        self.embedding = embedding_service

    async def embed_entity(self, entity_id: str) -> list[float]:
        """Generate embedding for entity based on graph context.

        Combines:
        - Entity name embedding
        - Neighbor entity embeddings
        - Relation type information
        """
        # Get entity data
        entity = await self.surrealdb.get_entity(entity_id)
        if not entity:
            return []

        # Build context from entity and relations
        context_parts = [entity.get("name", "")]

        # Get related entities (simplified)
        # In practice, you'd traverse the graph to build richer context

        # Embed the combined context
        context_text = " ".join(context_parts)
        embedding = await self.embedding.embed_text(context_text)

        return embedding

    async def embed_entities_batch(
        self,
        entity_ids: list[str],
    ) -> dict[str, list[float]]:
        """Embed multiple entities."""
        embeddings = {}

        for entity_id in entity_ids:
            emb = await self.embed_entity(entity_id)
            if emb:
                embeddings[entity_id] = emb

        return embeddings
