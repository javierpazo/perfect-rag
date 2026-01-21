"""Complete retrieval pipeline orchestrator."""

from typing import Any

import structlog

from perfect_rag.config import Settings, get_settings
from perfect_rag.core.embedding import EmbeddingService
from perfect_rag.db.oxigraph import OxigraphClient
from perfect_rag.db.qdrant import QdrantVectorClient
from perfect_rag.db.surrealdb import SurrealDBClient
from perfect_rag.models.query import QueryContext, RetrievalResult, SourceChunk
from perfect_rag.retrieval.graphrag import GraphRAGExpander
from perfect_rag.retrieval.query_rewriter import ContextAwarenessGate, QueryRewriter

logger = structlog.get_logger(__name__)

# Lazy import for ColBERT to avoid loading heavy dependencies when not needed
_colbert_reranker = None

# Lazy import for LLM reranker
_llm_reranker = None


class RetrievalPipeline:
    """Complete retrieval pipeline with hybrid search and GraphRAG.

    Pipeline steps:
    1. Context awareness gate (determine if retrieval needed)
    2. Query rewriting (expansion, HyDE, decomposition)
    3. Hybrid search (dense + sparse vectors)
    4. GraphRAG expansion (knowledge graph traversal)
    5. Cross-encoder reranking
    6. ColBERT late interaction reranking (optional, for improved accuracy)
    7. LLM-based reranking (optional, for semantic understanding)
    8. Result aggregation and deduplication
    """

    def __init__(
        self,
        qdrant: QdrantVectorClient,
        surrealdb: SurrealDBClient,
        oxigraph: OxigraphClient,
        embedding_service: EmbeddingService,
        llm_gateway: Any = None,
        settings: Settings | None = None,
    ):
        self.qdrant = qdrant
        self.surrealdb = surrealdb
        self.oxigraph = oxigraph
        self.embedding = embedding_service
        self.llm = llm_gateway
        self.settings = settings or get_settings()

        # Initialize components
        self.query_rewriter = QueryRewriter(llm_gateway=llm_gateway, settings=settings)
        self.context_gate = ContextAwarenessGate(llm_gateway=llm_gateway)
        self.graph_expander = GraphRAGExpander(
            surrealdb=surrealdb,
            settings=settings,
        )

        # ColBERT reranker (lazy loaded)
        self._colbert_reranker = None
        self._colbert_available = None  # None = not checked yet

        # LLM reranker (lazy loaded)
        self._llm_reranker = None
        self._llm_reranker_available = None  # None = not checked yet

    async def retrieve(
        self,
        query: str,
        context: QueryContext | None = None,
        top_k: int | None = None,
        use_reranking: bool = True,
        use_colbert_reranking: bool | None = None,
        use_llm_reranking: bool | None = None,
        use_graph_expansion: bool = True,
        use_query_rewriting: bool = True,
        use_context_gate: bool = True,
        acl_filter: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Execute full retrieval pipeline.

        Args:
            query: User query
            context: Optional query context (conversation history, user info)
            top_k: Number of results to return
            use_reranking: Whether to apply cross-encoder reranking
            use_colbert_reranking: Whether to apply ColBERT reranking (default from settings)
            use_llm_reranking: Whether to apply LLM-based reranking (default from settings)
            use_graph_expansion: Whether to use GraphRAG expansion
            use_query_rewriting: Whether to rewrite/expand query
            acl_filter: User roles for ACL filtering
            metadata_filter: Additional metadata filters

        Returns:
            RetrievalResult with ranked chunks and metadata
        """
        top_k = top_k or self.settings.retrieval_top_k
        acl_filter = acl_filter or ["*"]

        # Determine if ColBERT reranking should be used
        if use_colbert_reranking is None:
            use_colbert_reranking = self.settings.colbert_enabled

        # Determine if LLM reranking should be used
        if use_llm_reranking is None:
            use_llm_reranking = self.settings.llm_reranker_enabled

        logger.info("Starting retrieval", query=query[:100], top_k=top_k)

        # Step 1: Context awareness gate (optional)
        if use_context_gate:
            needs_retrieval = await self.context_gate.needs_retrieval(
                query,
                context.conversation_history if context else None,
            )
        else:
            needs_retrieval = True

        if not needs_retrieval:
            logger.info("Context gate: retrieval not needed")
            return RetrievalResult(
                query=query,
                chunks=[],
                total_found=0,
                retrieval_needed=False,
            )

        # Step 2: Query rewriting/expansion
        rewritten = {"queries": [query], "strategy": "none", "hyde_doc": None}
        if use_query_rewriting and self.llm:
            rewritten = await self.query_rewriter.rewrite(
                query,
                context=context.conversation_history if context else None,
            )
            logger.info(
                "Query rewritten",
                strategy=rewritten["strategy"],
                query_count=len(rewritten["queries"]),
            )

        # Step 3: Hybrid search for each query variant
        all_results = []
        seen_ids = set()

        for search_query in rewritten["queries"]:
            results = await self._hybrid_search(
                query=search_query,
                limit=top_k * 2,  # Over-fetch for fusion
                acl_filter=acl_filter,
                metadata_filter=metadata_filter,
            )

            for result in results:
                result_id = result.get("id")
                if result_id and result_id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result_id)

        # If we used HyDE, also search with the hypothetical document
        if rewritten.get("hyde_doc"):
            hyde_results = await self._hybrid_search(
                query=rewritten["hyde_doc"],
                limit=top_k,
                acl_filter=acl_filter,
                metadata_filter=metadata_filter,
            )
            for result in hyde_results:
                result_id = result.get("id")
                if result_id and result_id not in seen_ids:
                    result["source"] = "hyde"
                    all_results.append(result)
                    seen_ids.add(result_id)

        logger.info("Hybrid search complete", result_count=len(all_results))

        # Step 4: GraphRAG expansion
        if use_graph_expansion and all_results:
            all_results = await self.graph_expander.expand(
                initial_chunks=all_results,
                max_hops=2,
                expansion_limit=top_k // 2,
            )
            logger.info("Graph expansion complete", result_count=len(all_results))

        # Step 5: Cross-encoder reranking
        if use_reranking and all_results:
            all_results = await self._rerank(query, all_results, top_k * 2)
            logger.info("Cross-encoder reranking complete", result_count=len(all_results))

        # Step 6: ColBERT late interaction reranking (optional)
        colbert_used = False
        if use_colbert_reranking and all_results:
            colbert_results = await self._colbert_rerank(
                query,
                all_results,
                top_k=self.settings.colbert_rerank_top_k,
            )
            if colbert_results is not None:
                all_results = colbert_results
                colbert_used = True
                logger.info("ColBERT reranking complete", result_count=len(all_results))

        # Step 7: LLM-based reranking (optional, for semantic understanding)
        llm_reranked = False
        if use_llm_reranking and all_results and self.llm:
            llm_results = await self._llm_rerank(
                query,
                all_results,
                top_k=self.settings.llm_reranker_top_k,
            )
            if llm_results is not None:
                all_results = llm_results
                llm_reranked = True
                logger.info("LLM reranking complete", result_count=len(all_results))

        # Final sorting and selection
        if not colbert_used and not llm_reranked:
            # Sort by best available score
            all_results = sorted(
                all_results,
                key=lambda x: x.get("llm_rerank_score", x.get("colbert_score", x.get("rerank_score", x.get("score", 0)))),
                reverse=True,
            )[:top_k]

        # Convert to SourceChunk objects
        chunks = []
        for result in all_results:
            payload = result.get("payload", {})
            payload_metadata = payload.get("metadata") or {}
            if not isinstance(payload_metadata, dict):
                payload_metadata = {}

            # Preserve useful retrieval signals for downstream gating/debugging.
            # Scores can be on different scales depending on whether sparse vectors are present:
            # - dense_score: cosine similarity (0..1)
            # - rrf_score: reciprocal rank fusion (small ~0.01-0.05)
            # - score: either dense_score or rrf_score (see QdrantVectorClient)
            retrieval_signals = {
                "retrieval_score": result.get("score", 0.0),
                "retrieval_dense_score": result.get("dense_score", 0.0),
                "retrieval_rrf_score": result.get("rrf_score", 0.0),
                "retrieval_rerank_score": result.get("rerank_score", None),
                "retrieval_colbert_score": result.get("colbert_score", None),
                "retrieval_llm_rerank_score": result.get("llm_rerank_score", None),
                "retrieval_source": result.get("source", "vector_search"),
            }

            chunks.append(
                SourceChunk(
                    chunk_id=result.get("id", ""),
                    doc_id=payload.get("doc_id", ""),
                    # Support both field names: doc_title (new format) and title (ai2evolve migrated)
                    doc_title=payload.get("doc_title") or payload.get("title", ""),
                    # Support both field names: text (new format) and content (ai2evolve migrated)
                    content=payload.get("text") or payload.get("content", ""),
                    score=result.get("llm_rerank_score", result.get("colbert_score", result.get("rerank_score", result.get("score", 0)))),
                    chunk_index=payload.get("chunk_index", 0),
                    metadata={**payload_metadata, **retrieval_signals},
                )
            )

        # Calculate confidence
        if chunks:
            avg_score = sum(c.score for c in chunks[:5]) / min(5, len(chunks))
            confidence = min(max(avg_score, 0.0), 1.0)
        else:
            confidence = 0.0

        return RetrievalResult(
            query=query,
            rewritten_queries=rewritten["queries"],
            chunks=chunks,
            total_found=len(all_results),
            confidence=confidence,
            retrieval_needed=True,
            metadata={
                "rewrite_strategy": rewritten["strategy"],
                "used_hyde": bool(rewritten.get("hyde_doc")),
                "graph_expanded": use_graph_expansion,
                "reranked": use_reranking,
                "colbert_reranked": colbert_used,
                "llm_reranked": llm_reranked,
            },
        )

    async def _hybrid_search(
        self,
        query: str,
        limit: int,
        acl_filter: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform hybrid dense + sparse search."""
        # Get embeddings
        dense_vector = await self.embedding.embed_query(query)
        sparse_vector = await self.embedding.embed_sparse(query)

        # Search
        results = await self.qdrant.search_chunks_hybrid(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=limit,
            acl_filter=acl_filter,
            metadata_filter=metadata_filter,
        )

        return results

    async def _rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Rerank results using cross-encoder."""
        if not results:
            return results

        # Extract texts (support both 'text' and 'content' field names)
        texts = [r.get("payload", {}).get("text") or r.get("payload", {}).get("content", "") for r in results]

        # Rerank
        ranked = await self.embedding.rerank(query, texts, top_k=top_k)

        # Reorder results
        reranked_results = []
        for original_idx, score in ranked:
            result = results[original_idx].copy()
            result["rerank_score"] = score
            reranked_results.append(result)

        return reranked_results

    async def _get_colbert_reranker(self):
        """Lazy load ColBERT reranker."""
        if self._colbert_available is False:
            return None

        if self._colbert_reranker is None:
            try:
                from perfect_rag.retrieval.colbert_reranker import ColBERTReranker
                self._colbert_reranker = ColBERTReranker(settings=self.settings)
                await self._colbert_reranker.initialize()
                self._colbert_available = True
            except ImportError as e:
                logger.warning(
                    "ColBERT not available, skipping ColBERT reranking",
                    error=str(e),
                )
                self._colbert_available = False
                return None
            except Exception as e:
                logger.error(
                    "Failed to initialize ColBERT reranker",
                    error=str(e),
                )
                self._colbert_available = False
                return None

        return self._colbert_reranker

    async def _colbert_rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]] | None:
        """Rerank results using ColBERT late interaction.

        Returns None if ColBERT is not available.
        """
        if not results:
            return results

        reranker = await self._get_colbert_reranker()
        if reranker is None:
            return None

        # Prepare documents for ColBERT reranking
        # ColBERT expects 'content' or 'text' field - support both
        documents = []
        for r in results:
            payload = r.get("payload", {})
            doc = {
                "id": r.get("id", ""),
                "content": payload.get("text") or payload.get("content", ""),
            }
            # Preserve all original data
            doc["_original"] = r
            documents.append(doc)

        try:
            reranked = await reranker.rerank(query, documents, top_k=top_k)

            # Restore original structure with ColBERT scores
            colbert_results = []
            for doc in reranked:
                original = doc.get("_original", {})
                result = original.copy()
                result["colbert_score"] = doc.get("colbert_score", 0.0)
                result["original_rank"] = doc.get("original_rank", 0)
                colbert_results.append(result)

            return colbert_results

        except Exception as e:
            logger.error("ColBERT reranking failed", error=str(e))
            return None

    async def _get_llm_reranker(self):
        """Lazy load LLM reranker."""
        if self._llm_reranker_available is False:
            return None

        if self._llm_reranker is None:
            try:
                from perfect_rag.retrieval.llm_reranker import (
                    LLMReranker,
                    RankGPTReranker,
                    RerankStrategy,
                )

                # Determine strategy from settings
                strategy_map = {
                    "pointwise": RerankStrategy.POINTWISE,
                    "listwise": RerankStrategy.LISTWISE,
                    "pairwise": RerankStrategy.PAIRWISE,
                }
                strategy = strategy_map.get(
                    self.settings.llm_reranker_strategy,
                    RerankStrategy.POINTWISE,
                )

                # Use RankGPT if enabled, otherwise use standard LLM reranker
                if self.settings.rankgpt_enabled:
                    self._llm_reranker = RankGPTReranker(
                        llm_gateway=self.llm,
                        window_size=self.settings.rankgpt_window_size,
                        step_size=self.settings.rankgpt_step_size,
                        num_passes=self.settings.rankgpt_num_passes,
                        settings=self.settings,
                    )
                    logger.info("Initialized RankGPT reranker")
                else:
                    self._llm_reranker = LLMReranker(
                        llm_gateway=self.llm,
                        strategy=strategy,
                        max_docs_listwise=self.settings.llm_reranker_max_docs_listwise,
                        max_docs_pairwise=self.settings.llm_reranker_max_docs_pairwise,
                        settings=self.settings,
                    )
                    logger.info(
                        "Initialized LLM reranker",
                        strategy=strategy.value,
                    )

                self._llm_reranker_available = True

            except ImportError as e:
                logger.warning(
                    "LLM reranker not available",
                    error=str(e),
                )
                self._llm_reranker_available = False
                return None
            except Exception as e:
                logger.error(
                    "Failed to initialize LLM reranker",
                    error=str(e),
                )
                self._llm_reranker_available = False
                return None

        return self._llm_reranker

    async def _llm_rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]] | None:
        """Rerank results using LLM-based semantic understanding.

        Returns None if LLM reranker is not available.
        """
        if not results:
            return results

        reranker = await self._get_llm_reranker()
        if reranker is None:
            return None

        # Prepare documents for LLM reranking - support both field names
        documents = []
        for r in results:
            payload = r.get("payload", {})
            doc = {
                "id": r.get("id", ""),
                "content": payload.get("text") or payload.get("content", ""),
            }
            # Preserve all original data
            doc["_original"] = r
            documents.append(doc)

        try:
            # Handle both LLMReranker and RankGPTReranker
            from perfect_rag.retrieval.llm_reranker import LLMReranker

            if isinstance(reranker, LLMReranker):
                result = await reranker.rerank(query, documents, top_k=top_k)
                reranked = result.documents
            else:
                # RankGPTReranker returns list directly
                reranked = await reranker.rerank(query, documents, top_k=top_k)

            # Restore original structure with LLM scores
            llm_results = []
            for doc in reranked:
                original = doc.get("_original", {})
                result = original.copy()
                result["llm_rerank_score"] = doc.get("llm_rerank_score", 0.0)
                if "llm_rank" in doc:
                    result["llm_rank"] = doc["llm_rank"]
                if "pairwise_wins" in doc:
                    result["pairwise_wins"] = doc["pairwise_wins"]
                if "rankgpt_rank" in doc:
                    result["rankgpt_rank"] = doc["rankgpt_rank"]
                llm_results.append(result)

            return llm_results

        except Exception as e:
            logger.error("LLM reranking failed", error=str(e))
            return None

    async def search_entities(
        self,
        query: str,
        entity_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for entities matching query."""
        # Get query embedding
        query_vector = await self.embedding.embed_query(query)

        # Search entity vectors
        results = await self.qdrant.search_entities(
            query_vector=query_vector,
            limit=limit,
            entity_type=entity_type,
        )

        return [
            {
                "id": r.id,
                "score": r.score,
                **r.payload,
            }
            for r in results
        ]

    async def get_related_chunks(
        self,
        chunk_id: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get chunks related to a given chunk via entities."""
        # Get chunk's entities
        chunk = await self.surrealdb.get_chunk(chunk_id)
        if not chunk:
            return []

        entities = chunk.get("entities", [])
        if not entities:
            return []

        # Find other chunks mentioning same entities
        related_chunk_ids = set()
        for entity_id in entities:
            entity = await self.surrealdb.get_entity(entity_id)
            if entity:
                for other_chunk_id in entity.get("source_chunks", []):
                    if other_chunk_id != chunk_id:
                        related_chunk_ids.add(other_chunk_id)

        # Get chunk data
        related_chunks = []
        for related_id in list(related_chunk_ids)[:limit]:
            related_chunk = await self.surrealdb.get_chunk(related_id)
            if related_chunk:
                related_chunks.append(related_chunk)

        return related_chunks


# =============================================================================
# Factory Function
# =============================================================================

_pipeline: RetrievalPipeline | None = None


async def get_retrieval_pipeline(
    qdrant: QdrantVectorClient,
    surrealdb: SurrealDBClient,
    oxigraph: OxigraphClient,
    embedding_service: EmbeddingService,
    llm_gateway: Any = None,
) -> RetrievalPipeline:
    """Get or create retrieval pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RetrievalPipeline(
            qdrant=qdrant,
            surrealdb=surrealdb,
            oxigraph=oxigraph,
            embedding_service=embedding_service,
            llm_gateway=llm_gateway,
        )
    return _pipeline
