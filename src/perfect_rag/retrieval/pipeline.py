"""Complete retrieval pipeline orchestrator.

Full pipeline with:
1. Context gate (skip retrieval if not needed)
2. Query rewrite (expansion + HyDE + decomposition)
2.5. PageIndex tree search (optional)
3. Hybrid search (Dense + BM25 with phrase/proximity + RRF fusion)
4. GraphRAG expansion with entity normalization
5. Cross-encoder reranking
6. ColBERT late interaction (optional)
7. LLM reranking (optional)
8. MMR diversification
9. Confidence estimation + fallback
10. Evidence-first generation (in generation pipeline)
"""

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


class RetrievalPipeline:
    """Complete retrieval pipeline with hybrid search and GraphRAG.

    Pipeline steps:
    1. Context awareness gate (determine if retrieval needed)
    2. Query rewriting (expansion, HyDE, decomposition)
    2.5. PageIndex tree search (optional, for structured documents)
    3. Hybrid search (Dense + BM25 with phrase/proximity + RRF fusion)
    4. GraphRAG expansion with entity normalization
    5. Cross-encoder reranking
    6. ColBERT late interaction reranking (optional)
    7. LLM-based reranking (optional)
    8. MMR diversification
    9. Confidence estimation + automatic fallback
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

        # PageIndex retriever (lazy loaded)
        self._pageindex_retriever = None
        self._pageindex_available = None  # None = not checked yet

        # BM25 index (lazy loaded, per-document indexing)
        self._bm25_index = None
        self._bm25_available = None

        # Entity normalizer (lazy loaded)
        self._entity_normalizer = None
        self._entity_normalizer_available = None

        # MMR reranker (lazy loaded)
        self._mmr_reranker = None
        self._mmr_available = None

        # Confidence estimator (lazy loaded)
        self._confidence_estimator = None

        # RAG-Fusion (lazy loaded)
        self._rag_fusion = None
        self._rag_fusion_available = None

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
        use_pageindex: bool | None = None,
        use_mmr: bool | None = None,
        use_confidence_fallback: bool | None = None,
        use_entity_normalization: bool | None = None,
        doc_id: str | None = None,
        doc_metadata: dict[str, Any] | None = None,
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
            use_pageindex: Whether to use PageIndex tree search (default from settings)
            use_mmr: Whether to apply MMR diversification (default from settings)
            use_confidence_fallback: Whether to apply confidence-based fallback (default from settings)
            use_entity_normalization: Whether to normalize entities (default from settings)
            doc_id: Document ID for PageIndex
            doc_metadata: Document metadata for PageIndex decision
            acl_filter: User roles for ACL filtering
            metadata_filter: Additional metadata filters

        Returns:
            RetrievalResult with ranked chunks and metadata
        """
        top_k = top_k or self.settings.retrieval_top_k
        acl_filter = acl_filter or ["*"]

        # Determine settings from config if not specified
        if use_colbert_reranking is None:
            use_colbert_reranking = self.settings.colbert_enabled
        if use_llm_reranking is None:
            use_llm_reranking = self.settings.llm_reranker_enabled
        if use_mmr is None:
            use_mmr = self.settings.mmr_enabled
        if use_confidence_fallback is None:
            use_confidence_fallback = self.settings.confidence_fallback_enabled
        if use_entity_normalization is None:
            use_entity_normalization = self.settings.entity_normalization_enabled
        if use_pageindex is None:
            use_pageindex = self.settings.pageindex_enabled

        logger.info("Starting retrieval", query=query[:100], top_k=top_k)

        # Step 1: Context awareness gate
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

        # Detect query type for BM25 routing
        query_type = self._detect_query_type(query)
        use_phrase_matching = query_type in ["factual", "dosage", "criteria"]

        # Step 2.5: PageIndex tree search (optional)
        pageindex_ranges = None
        pageindex_used = False
        if use_pageindex and doc_id and self.llm:
            pageindex_ranges = await self._pageindex_tree_search(
                query=query,
                doc_id=doc_id,
                doc_metadata=doc_metadata,
            )
            if pageindex_ranges:
                pageindex_used = True
                pageindex_filter = self._get_pageindex_filter(pageindex_ranges)
                if pageindex_filter:
                    if metadata_filter:
                        metadata_filter = {"must": [metadata_filter, pageindex_filter]}
                    else:
                        metadata_filter = pageindex_filter
                logger.info("PageIndex tree search complete", doc_id=doc_id)

        # Step 3: Hybrid search (Dense + BM25 with phrase/proximity + RRF)
        all_results = []
        seen_ids = set()

        for search_query in rewritten["queries"]:
            results = await self._hybrid_search_enhanced(
                query=search_query,
                limit=top_k * 2,
                acl_filter=acl_filter,
                metadata_filter=metadata_filter,
                use_phrase_matching=use_phrase_matching,
            )

            for result in results:
                result_id = result.get("id")
                if result_id and result_id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result_id)

        # HyDE search
        if rewritten.get("hyde_doc"):
            hyde_results = await self._hybrid_search_enhanced(
                query=rewritten["hyde_doc"],
                limit=top_k,
                acl_filter=acl_filter,
                metadata_filter=metadata_filter,
                use_phrase_matching=False,
            )
            for result in hyde_results:
                result_id = result.get("id")
                if result_id and result_id not in seen_ids:
                    result["source"] = "hyde"
                    all_results.append(result)
                    seen_ids.add(result_id)

        logger.info("Hybrid search complete", result_count=len(all_results))

        # Step 4: GraphRAG expansion with entity normalization
        if use_graph_expansion and all_results:
            all_results = await self._expand_with_normalization(
                initial_chunks=all_results,
                max_hops=2,
                expansion_limit=top_k // 2,
                use_normalization=use_entity_normalization,
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
                query, all_results, top_k=self.settings.colbert_rerank_top_k
            )
            if colbert_results is not None:
                all_results = colbert_results
                colbert_used = True
                logger.info("ColBERT reranking complete")

        # Step 7: LLM-based reranking (optional)
        llm_reranked = False
        if use_llm_reranking and all_results and self.llm:
            llm_results = await self._llm_rerank(
                query, all_results, top_k=self.settings.llm_reranker_top_k
            )
            if llm_results is not None:
                all_results = llm_results
                llm_reranked = True
                logger.info("LLM reranking complete")

        # Step 8: MMR diversification
        mmr_used = False
        if use_mmr and all_results:
            mmr_results = await self._mmr_diversify(
                query=query,
                results=all_results,
                top_k=min(top_k, self.settings.mmr_top_k),
            )
            if mmr_results is not None:
                all_results = mmr_results
                mmr_used = True
                logger.info("MMR diversification complete")

        # Step 9: Confidence estimation + fallback
        confidence_result = None
        fallback_used = False
        if use_confidence_fallback:
            confidence_result = await self._estimate_confidence(all_results, query)

            if confidence_result and confidence_result.needs_fallback:
                logger.info(
                    "Low confidence, attempting fallback",
                    confidence=confidence_result.overall,
                    strategy=confidence_result.recommended_strategy,
                )
                fallback_results = await self._execute_fallback(
                    query=query,
                    initial_results=all_results,
                    confidence=confidence_result,
                    acl_filter=acl_filter,
                    metadata_filter=metadata_filter,
                )
                if fallback_results:
                    all_results = fallback_results["results"]
                    confidence_result = fallback_results["confidence"]
                    fallback_used = True

        # Final sorting
        if not colbert_used and not llm_reranked:
            all_results = sorted(
                all_results,
                key=lambda x: x.get("llm_rerank_score", x.get("colbert_score", x.get("rerank_score", x.get("score", 0)))),
                reverse=True,
            )[:top_k]

        # Convert to SourceChunk objects
        chunks = self._convert_to_chunks(all_results)

        # Calculate confidence if not already done
        if confidence_result is None:
            confidence = self._calculate_simple_confidence(chunks)
        else:
            confidence = confidence_result.overall

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
                "query_type": query_type,
                "phrase_matching_used": use_phrase_matching,
                "graph_expanded": use_graph_expansion,
                "entity_normalization_used": use_entity_normalization,
                "reranked": use_reranking,
                "colbert_reranked": colbert_used,
                "llm_reranked": llm_reranked,
                "mmr_diversified": mmr_used,
                "confidence_fallback_used": fallback_used,
                "pageindex_used": pageindex_used,
            },
        )

    def _detect_query_type(self, query: str) -> str:
        """Detect query type for routing decisions."""
        query_lower = query.lower()

        # Factual/dosage queries - benefit from phrase matching
        factual_patterns = [
            "cuál es la dosis", "qué dosis", "dosis de", "mg/", "mg/kg",
            "criterios de", "cuáles son los criterios", "requisitos para",
            "definición de", "qué es", "significa",
            "cuánto", "cuántos", "qué porcentaje",
        ]

        for pattern in factual_patterns:
            if pattern in query_lower:
                return "factual"

        # Comparison queries
        comparison_patterns = [
            "diferencia entre", "comparar", "vs", "versus",
            "mejor que", "peor que", "más efectivo",
        ]

        for pattern in comparison_patterns:
            if pattern in query_lower:
                return "comparison"

        # Procedural queries
        procedural_patterns = [
            "cómo se", "cómo hacer", "pasos para", "procedimiento",
            "tratamiento de", "cómo tratar",
        ]

        for pattern in procedural_patterns:
            if pattern in query_lower:
                return "procedural"

        return "exploratory"

    async def _hybrid_search_enhanced(
        self,
        query: str,
        limit: int,
        acl_filter: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        use_phrase_matching: bool = False,
    ) -> list[dict[str, Any]]:
        """Enhanced hybrid search with BM25 phrase/proximity support."""
        # Get dense embeddings
        dense_vector = await self.embedding.embed_query(query)

        # Get sparse embeddings (Qdrant's built-in)
        sparse_vector = await self.embedding.embed_sparse(query)

        # Dense search via Qdrant
        results = await self.qdrant.search_chunks_hybrid(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=limit,
            acl_filter=acl_filter,
            metadata_filter=metadata_filter,
        )

        # If phrase matching is enabled and BM25 is available, boost phrase matches
        if use_phrase_matching and self.settings.bm25_enabled:
            bm25_results = await self._bm25_search_with_phrases(
                query=query,
                limit=limit,
            )

            if bm25_results:
                # RRF fusion between dense and BM25
                results = self._rrf_fusion(
                    dense_results=results,
                    bm25_results=bm25_results,
                    k=self.settings.rag_fusion_rrf_k,
                )

        return results

    async def _bm25_search_with_phrases(
        self,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]] | None:
        """Search using BM25 with phrase matching."""
        bm25_index = await self._get_bm25_index()
        if bm25_index is None:
            return None

        try:
            results = bm25_index.search(
                query=query,
                top_k=limit,
                use_phrases=True,
            )

            # Convert to standard format
            return [
                {
                    "id": r.doc_id,
                    "score": r.score,
                    "bm25_score": r.score,
                    "phrase_matches": r.phrase_matches,
                    "source": "bm25",
                    "payload": r.metadata or {},
                }
                for r in results
            ]
        except Exception as e:
            logger.warning("BM25 search failed", error=str(e))
            return None

    async def _get_bm25_index(self):
        """Lazy load or create BM25 index."""
        if self._bm25_available is False:
            return None

        if self._bm25_index is None:
            try:
                from perfect_rag.retrieval.sparse_bm25 import BM25Index

                self._bm25_index = BM25Index(
                    k1=self.settings.bm25_k1,
                    b=self.settings.bm25_b,
                    phrase_boost=self.settings.bm25_phrase_boost,
                )
                self._bm25_available = True
                logger.info("BM25 index initialized")

                # TODO: Index existing chunks from SurrealDB
                # This would be done at startup or incrementally

            except ImportError as e:
                logger.warning("BM25 not available", error=str(e))
                self._bm25_available = False
                return None
            except Exception as e:
                logger.error("Failed to initialize BM25", error=str(e))
                self._bm25_available = False
                return None

        return self._bm25_index

    def _rrf_fusion(
        self,
        dense_results: list[dict],
        bm25_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """Reciprocal Rank Fusion for combining dense and BM25 results."""
        scores = {}
        result_data = {}

        # Dense results
        for rank, result in enumerate(dense_results, 1):
            doc_id = result.get("id")
            if doc_id:
                rrf_score = 1.0 / (k + rank)
                scores[doc_id] = scores.get(doc_id, 0) + rrf_score
                result_data[doc_id] = result

        # BM25 results
        for rank, result in enumerate(bm25_results, 1):
            doc_id = result.get("id")
            if doc_id:
                rrf_score = 1.0 / (k + rank)
                scores[doc_id] = scores.get(doc_id, 0) + rrf_score
                if doc_id not in result_data:
                    result_data[doc_id] = result

        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Build final results
        fused = []
        for doc_id in sorted_ids:
            result = result_data[doc_id].copy()
            result["rrf_score"] = scores[doc_id]
            fused.append(result)

        return fused

    async def _expand_with_normalization(
        self,
        initial_chunks: list[dict[str, Any]],
        max_hops: int,
        expansion_limit: int,
        use_normalization: bool,
    ) -> list[dict[str, Any]]:
        """GraphRAG expansion with optional entity normalization."""
        if use_normalization:
            normalizer = await self._get_entity_normalizer()
            if normalizer:
                # Normalize entities in chunks before expansion
                for chunk in initial_chunks:
                    entities = chunk.get("payload", {}).get("entities", [])
                    if entities:
                        normalized_entities = []
                        for entity in entities:
                            result = normalizer.normalize(entity)
                            normalized_entities.append(result.canonical)
                        chunk["payload"]["normalized_entities"] = normalized_entities

        # Standard graph expansion
        return await self.graph_expander.expand(
            initial_chunks=initial_chunks,
            max_hops=max_hops,
            expansion_limit=expansion_limit,
        )

    async def _get_entity_normalizer(self):
        """Lazy load entity normalizer."""
        if self._entity_normalizer_available is False:
            return None

        if self._entity_normalizer is None:
            try:
                from perfect_rag.retrieval.entity_normalization import EntityNormalizer

                self._entity_normalizer = EntityNormalizer(
                    llm_gateway=self.llm,
                    settings=self.settings,
                )
                self._entity_normalizer_available = True
                logger.info("Entity normalizer initialized")
            except ImportError as e:
                logger.warning("Entity normalization not available", error=str(e))
                self._entity_normalizer_available = False
                return None
            except Exception as e:
                logger.error("Failed to initialize entity normalizer", error=str(e))
                self._entity_normalizer_available = False
                return None

        return self._entity_normalizer

    async def _mmr_diversify(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]] | None:
        """Apply MMR diversification to results."""
        if not results:
            return results

        mmr_reranker = await self._get_mmr_reranker()
        if mmr_reranker is None:
            return None

        try:
            # Get query embedding
            query_embedding = await self.embedding.embed_query(query)

            # Get candidate embeddings
            candidate_embeddings = {}
            for r in results:
                chunk_id = r.get("id")
                content = r.get("payload", {}).get("text") or r.get("payload", {}).get("content", "")
                if chunk_id and content:
                    embedding = await self.embedding.embed_text(content)
                    candidate_embeddings[chunk_id] = embedding

            # Convert to MMR format
            candidates = [
                {
                    "id": r.get("id"),
                    "content": r.get("payload", {}).get("text") or r.get("payload", {}).get("content", ""),
                    "score": r.get("rerank_score", r.get("score", 0)),
                }
                for r in results
            ]

            # Run MMR
            mmr_results = mmr_reranker.select_diverse_sync(
                candidates=candidates,
                query_embedding=query_embedding,
                candidate_embeddings=candidate_embeddings,
                top_k=top_k,
            )

            # Restore original structure
            id_to_result = {r.get("id"): r for r in results}
            diversified = []
            for mmr_r in mmr_results:
                original = id_to_result.get(mmr_r.doc_id)
                if original:
                    result = original.copy()
                    result["mmr_score"] = mmr_r.score
                    diversified.append(result)

            return diversified

        except Exception as e:
            logger.warning("MMR diversification failed", error=str(e))
            return None

    async def _get_mmr_reranker(self):
        """Lazy load MMR reranker."""
        if self._mmr_available is False:
            return None

        if self._mmr_reranker is None:
            try:
                from perfect_rag.retrieval.mmr import MMRReranker

                self._mmr_reranker = MMRReranker(
                    lambda_param=self.settings.mmr_lambda,
                )
                self._mmr_available = True
                logger.info("MMR reranker initialized")
            except ImportError as e:
                logger.warning("MMR not available", error=str(e))
                self._mmr_available = False
                return None
            except Exception as e:
                logger.error("Failed to initialize MMR", error=str(e))
                self._mmr_available = False
                return None

        return self._mmr_reranker

    async def _estimate_confidence(self, results: list[dict], query: str):
        """Estimate confidence in retrieval results."""
        if self._confidence_estimator is None:
            try:
                from perfect_rag.retrieval.confidence import ConfidenceEstimator

                self._confidence_estimator = ConfidenceEstimator(settings=self.settings)
            except ImportError:
                return None

        # Convert to format expected by estimator
        chunks = [
            {
                "id": r.get("id"),
                "content": r.get("payload", {}).get("text") or r.get("payload", {}).get("content", ""),
                "score": r.get("rerank_score", r.get("score", 0)),
                "rerank_score": r.get("rerank_score"),
            }
            for r in results
        ]

        return self._confidence_estimator.estimate(chunks, query)

    async def _execute_fallback(
        self,
        query: str,
        initial_results: list[dict],
        confidence,
        acl_filter: list[str] | None,
        metadata_filter: dict[str, Any] | None,
    ) -> dict | None:
        """Execute fallback strategy when confidence is low."""
        if not confidence.recommended_strategy:
            return None

        strategy = confidence.recommended_strategy

        try:
            if strategy.value == "expand_top_k":
                # Retrieve more candidates
                new_results = await self._hybrid_search_enhanced(
                    query=query,
                    limit=self.settings.retrieval_top_k * 3,
                    acl_filter=acl_filter,
                    metadata_filter=None,
                    use_phrase_matching=True,
                )
            elif strategy.value == "expand_query":
                # Try expanded query
                expanded = f"{query} {' '.join(query.split()[:3])}"
                new_results = await self._hybrid_search_enhanced(
                    query=expanded,
                    limit=self.settings.retrieval_top_k * 2,
                    acl_filter=acl_filter,
                    metadata_filter=metadata_filter,
                    use_phrase_matching=False,
                )
            else:
                # Default: expand search
                new_results = await self._hybrid_search_enhanced(
                    query=query,
                    limit=self.settings.retrieval_top_k * 3,
                    acl_filter=None,
                    metadata_filter=None,
                    use_phrase_matching=False,
                )

            # Re-estimate confidence
            new_confidence = await self._estimate_confidence(new_results, query)

            if new_confidence and new_confidence.overall > confidence.overall:
                return {
                    "results": new_results,
                    "confidence": new_confidence,
                }

        except Exception as e:
            logger.warning("Fallback strategy failed", error=str(e))

        return None

    def _convert_to_chunks(self, results: list[dict]) -> list[SourceChunk]:
        """Convert results to SourceChunk objects."""
        chunks = []
        for result in results:
            payload = result.get("payload", {})
            payload_metadata = payload.get("metadata") or {}
            if not isinstance(payload_metadata, dict):
                payload_metadata = {}

            retrieval_signals = {
                "retrieval_score": result.get("score", 0.0),
                "retrieval_dense_score": result.get("dense_score", 0.0),
                "retrieval_rrf_score": result.get("rrf_score", 0.0),
                "retrieval_rerank_score": result.get("rerank_score"),
                "retrieval_colbert_score": result.get("colbert_score"),
                "retrieval_llm_rerank_score": result.get("llm_rerank_score"),
                "retrieval_mmr_score": result.get("mmr_score"),
                "retrieval_bm25_score": result.get("bm25_score"),
                "retrieval_source": result.get("source", "vector_search"),
            }

            chunks.append(
                SourceChunk(
                    chunk_id=result.get("id", ""),
                    doc_id=payload.get("doc_id", ""),
                    doc_title=payload.get("doc_title") or payload.get("title", ""),
                    content=payload.get("text") or payload.get("content", ""),
                    score=result.get("llm_rerank_score", result.get("colbert_score", result.get("rerank_score", result.get("score", 0)))),
                    chunk_index=payload.get("chunk_index", 0),
                    metadata={**payload_metadata, **retrieval_signals},
                )
            )

        return chunks

    def _calculate_simple_confidence(self, chunks: list[SourceChunk]) -> float:
        """Calculate simple confidence from chunk scores."""
        if not chunks:
            return 0.0
        avg_score = sum(c.score for c in chunks[:5]) / min(5, len(chunks))
        return min(max(avg_score, 0.0), 1.0)

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

    async def _get_pageindex_retriever(self):
        """Lazy load PageIndex retriever."""
        if self._pageindex_available is False:
            return None

        if self._pageindex_retriever is None:
            try:
                from perfect_rag.retrieval.pageindex import PageIndexRetriever

                self._pageindex_retriever = PageIndexRetriever(
                    settings=self.settings,
                    llm_gateway=self.llm,
                )
                await self._pageindex_retriever.initialize()
                self._pageindex_available = True
                logger.info("PageIndex retriever initialized")
            except ImportError as e:
                logger.warning(
                    "PageIndex not available, skipping tree-based retrieval",
                    error=str(e),
                )
                self._pageindex_available = False
                return None
            except Exception as e:
                logger.error(
                    "Failed to initialize PageIndex retriever",
                    error=str(e),
                )
                self._pageindex_available = False
                return None

        return self._pageindex_retriever

    async def _pageindex_tree_search(
        self,
        query: str,
        doc_id: str,
        doc_metadata: dict[str, Any] | None = None,
    ) -> list | None:
        """
        Execute PageIndex tree search for relevant page ranges.

        Returns None if PageIndex is not available or not applicable.
        Returns list of PageRange objects if successful.
        """
        retriever = await self._get_pageindex_retriever()
        if retriever is None:
            return None

        # Check if PageIndex should be used for this document
        doc_metadata = doc_metadata or {}
        if not retriever.should_use_pageindex(doc_metadata):
            logger.debug(
                "PageIndex not applicable for document",
                doc_id=doc_id,
                page_count=doc_metadata.get("page_count", 0),
            )
            return None

        try:
            page_ranges = await retriever.tree_search(
                query=query,
                doc_id=doc_id,
            )
            return page_ranges
        except Exception as e:
            logger.error("PageIndex tree search failed", doc_id=doc_id, error=str(e))
            return None

    def _get_pageindex_filter(
        self,
        page_ranges: list,
    ) -> dict[str, Any] | None:
        """Convert PageIndex ranges to Qdrant metadata filter."""
        if not page_ranges:
            return None

        try:
            from perfect_rag.retrieval.pageindex import PageIndexRetriever

            # Create a temporary retriever just for the filter conversion
            temp_retriever = PageIndexRetriever.__new__(PageIndexRetriever)
            return temp_retriever.get_metadata_filter(page_ranges)
        except Exception:
            # Fallback: build filter manually
            conditions = []
            for pr in page_ranges:
                conditions.append({
                    "key": "page_number",
                    "range": {
                        "gte": pr.start,
                        "lte": pr.end,
                    },
                })

            if len(conditions) == 1:
                return conditions[0]
            elif len(conditions) > 1:
                return {"should": conditions}
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
