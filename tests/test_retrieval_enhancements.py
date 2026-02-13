"""Tests for new retrieval enhancements.

Tests for:
- BM25 with phrase/proximity queries
- RAG-Fusion with intent routing
- Evidence-first generation
- MMR diversification
- Confidence estimation and fallback
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# BM25 Tests
class TestBM25Index:
    """Tests for BM25 sparse retrieval."""

    def test_bm25_tokenization(self):
        """Test tokenization with Spanish text."""
        from perfect_rag.retrieval.sparse_bm25 import BM25Index

        index = BM25Index()
        tokens = index._tokenize("¿Cuál es el tratamiento para H. pylori?")

        assert "cuál" in tokens
        assert "tratamiento" in tokens
        assert "pylori" in tokens
        # Punctuation removed
        assert "?" not in " ".join(tokens)

    def test_bm25_add_document(self):
        """Test adding documents to index."""
        from perfect_rag.retrieval.sparse_bm25 import BM25Index

        index = BM25Index()
        index.add_document("doc1", "El cáncer colorrectal es común")
        index.add_document("doc2", "El tratamiento del cáncer incluye quimioterapia")

        assert index.num_docs == 2
        assert "cáncer" in index.inverted_index
        assert len(index.inverted_index["cáncer"]) == 2

    def test_bm25_search(self):
        """Test BM25 search scoring."""
        from perfect_rag.retrieval.sparse_bm25 import BM25Index

        index = BM25Index(k1=1.5, b=0.75)
        index.add_document("doc1", "cáncer colorrectal tratamiento")
        index.add_document("doc2", "diabetes tratamiento insulina")

        results = index.search("tratamiento cáncer", top_k=5)

        assert len(results) >= 1
        # doc1 should score higher (both terms)
        assert results[0].doc_id == "doc1"
        assert results[0].score > 0

    def test_bm25_phrase_matching(self):
        """Test phrase query matching."""
        from perfect_rag.retrieval.sparse_bm25 import BM25Index

        index = BM25Index(phrase_boost=1.5)
        index.add_document("doc1", "cáncer colorrectal es una enfermedad")
        index.add_document("doc2", "el colorrectal y el cáncer son diferentes")

        results = index.search("cáncer colorrectal", top_k=5, use_phrases=True)

        # doc1 has exact phrase, should score higher
        assert results[0].doc_id == "doc1"
        assert len(results[0].phrase_matches) > 0

    def test_bm25_idf_calculation(self):
        """Test IDF calculation."""
        from perfect_rag.retrieval.sparse_bm25 import BM25Index

        index = BM25Index()
        index.add_document("doc1", "cáncer cáncer cáncer")  # Frequent in doc
        index.add_document("doc2", "diabetes")  # Rare term

        # IDF for rare term should be higher
        idf_cancer = index._idf("cáncer")
        idf_diabetes = index._idf("diabetes")

        assert idf_diabetes > idf_cancer


class TestRAGFusion:
    """Tests for RAG-Fusion retrieval."""

    def test_intent_classification(self):
        """Test query intent classification."""
        from perfect_rag.retrieval.rag_fusion import IntentClassifier, QueryIntent

        classifier = IntentClassifier()

        assert classifier.classify("¿Cuál es el tratamiento?") == QueryIntent.FACTUAL
        assert classifier.classify("Diferencia entre A y B") == QueryIntent.COMPARISON
        assert classifier.classify("Cómo se hace el procedimiento") == QueryIntent.PROCEDURAL
        assert classifier.classify("Definición de cirrosis") == QueryIntent.DEFINITIONAL

    @pytest.mark.asyncio
    async def test_query_expansion(self):
        """Test query expansion with mock LLM."""
        from perfect_rag.retrieval.rag_fusion import QueryExpander, QueryIntent

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="1. variación uno\n2. variación dos\n3. variación tres")

        expander = QueryExpander(mock_llm)
        variants = await expander.expand(
            "¿Cuál es el tratamiento?",
            num_variations=3,
            intent=QueryIntent.FACTUAL
        )

        assert len(variants) == 4  # Original + 3 variations
        assert variants[0].source == "original"
        assert all(v.intent == QueryIntent.FACTUAL for v in variants)

    def test_rrf_fusion(self):
        """Test Reciprocal Rank Fusion."""
        from perfect_rag.retrieval.rag_fusion import RAGFusion, QueryVariant, QueryIntent

        fusion = RAGFusion()

        # Mock results from two queries
        results_by_query = {
            "query1": [
                {"doc_id": "doc1", "score": 0.9},
                {"doc_id": "doc2", "score": 0.8},
                {"doc_id": "doc3", "score": 0.7},
            ],
            "query2": [
                {"doc_id": "doc2", "score": 0.95},
                {"doc_id": "doc1", "score": 0.9},
                {"doc_id": "doc4", "score": 0.6},
            ],
        }

        variants = [
            QueryVariant(query="query1", intent=QueryIntent.FACTUAL, source="original"),
            QueryVariant(query="query2", intent=QueryIntent.FACTUAL, source="expanded"),
        ]

        fused = fusion.fuse_results(results_by_query, variants, top_k=5)

        assert len(fused) <= 5
        # Documents appearing in both lists should score higher
        doc_ids = [f.doc_id for f in fused]
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids


class TestEvidenceFirst:
    """Tests for evidence-first generation."""

    @pytest.mark.asyncio
    async def test_evidence_extraction(self):
        """Test evidence extraction from chunks."""
        from perfect_rag.generation.evidence_first import EvidenceExtractor

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value='''
        {
          "evidences": [
            {"quote": "El TNM clasifica el cáncer", "doc_number": 1, "relevance": "high"}
          ],
          "contradictions": [],
          "gaps": []
        }
        ''')

        extractor = EvidenceExtractor(mock_llm)

        chunks = [
            {"id": "c1", "doc_title": "Doc1", "content": "El TNM clasifica el cáncer colorrectal"}
        ]

        evidence_set = await extractor.extract_evidence(
            "¿Qué es TNM?",
            chunks,
            max_evidences=5
        )

        assert len(evidence_set.evidences) >= 1
        assert "TNM" in evidence_set.evidences[0].evidence_text

    @pytest.mark.asyncio
    async def test_evidence_based_generation(self):
        """Test answer generation from evidence."""
        from perfect_rag.generation.evidence_first import EvidenceBasedGenerator

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="El TNM clasifica el cáncer [E1].")

        generator = EvidenceBasedGenerator(mock_llm)

        chunks = [
            {"id": "c1", "doc_title": "Guía", "content": "El sistema TNM clasifica tumores"}
        ]

        result = await generator.generate(
            "¿Qué es TNM?",
            chunks,
            max_evidences=5
        )

        assert "TNM" in result.answer
        assert "[E1]" in result.answer
        assert len(result.citations) > 0

    def test_coverage_calculation(self):
        """Test query coverage calculation."""
        from perfect_rag.generation.evidence_first import EvidenceExtractor, ExtractedEvidence

        extractor = EvidenceExtractor()

        evidences = [
            ExtractedEvidence(
                chunk_id="c1",
                doc_title="Doc",
                evidence_text="cáncer tratamiento quimioterapia",
                relevance_score=0.9
            )
        ]

        coverage = extractor._calculate_coverage(
            "tratamiento del cáncer",
            evidences
        )

        # Both terms covered
        assert coverage > 0.5


class TestMMR:
    """Tests for MMR diversification."""

    def test_mmr_selection(self):
        """Test MMR result selection."""
        from perfect_rag.retrieval.mmr import MMRReranker

        reranker = MMRReranker(lambda_param=0.7)

        candidates = [
            {"id": "d1", "content": "cáncer tratamiento", "score": 0.9},
            {"id": "d2", "content": "cáncer diagnóstico", "score": 0.85},
            {"id": "d3", "content": "diabetes insulina", "score": 0.8},
            {"id": "d4", "content": "cáncer prevención", "score": 0.75},
        ]

        results = reranker.select_diverse_sync(
            candidates=candidates,
            query_embedding=[0.1] * 10,  # Dummy
            candidate_embeddings={
                "d1": [0.9, 0.1, 0.1],
                "d2": [0.85, 0.1, 0.1],  # Similar to d1
                "d3": [0.1, 0.9, 0.1],   # Different
                "d4": [0.8, 0.1, 0.1],   # Similar to d1, d2
            },
            top_k=3,
        )

        assert len(results) == 3
        # d3 should be included due to diversity even though lower relevance
        doc_ids = [r.doc_id for r in results]
        assert "d3" in doc_ids

    def test_diversity_score(self):
        """Test diversity score calculation."""
        from perfect_rag.retrieval.mmr import ContextDiversifier

        diversifier = ContextDiversifier()

        # High similarity chunks
        similar_chunks = [
            {"content": "cáncer colorrectal tratamiento"},
            {"content": "cáncer colorrectal prevención"},
            {"content": "cáncer colorrectal diagnóstico"},
        ]

        # Diverse chunks
        diverse_chunks = [
            {"content": "cáncer tratamiento quimioterapia"},
            {"content": "diabetes insulina glucosa"},
            {"content": "hipertensión presión arterial"},
        ]

        similar_score = diversifier.compute_diversity_score(similar_chunks)
        diverse_score = diversifier.compute_diversity_score(diverse_chunks)

        assert diverse_score > similar_score


class TestConfidence:
    """Tests for confidence estimation."""

    def test_confidence_high(self):
        """Test high confidence estimation."""
        from perfect_rag.retrieval.confidence import ConfidenceEstimator, ConfidenceLevel

        estimator = ConfidenceEstimator()

        chunks = [
            {"id": "c1", "content": "respuesta exacta", "score": 0.95, "rerank_score": 0.95},
            {"id": "c2", "content": "información relevante", "score": 0.90, "rerank_score": 0.90},
            {"id": "c3", "content": "contexto adicional", "score": 0.85, "rerank_score": 0.85},
        ]

        confidence = estimator.estimate(chunks, "pregunta específica")

        assert confidence.level == ConfidenceLevel.HIGH
        assert confidence.overall > 0.8
        assert not confidence.needs_fallback

    def test_confidence_low(self):
        """Test low confidence estimation."""
        from perfect_rag.retrieval.confidence import ConfidenceEstimator, ConfidenceLevel

        estimator = ConfidenceEstimator()

        chunks = [
            {"id": "c1", "content": "información vaga", "score": 0.35},
            {"id": "c2", "content": "contenido no relacionado", "score": 0.30},
        ]

        confidence = estimator.estimate(chunks, "pregunta muy específica técnica")

        assert confidence.level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]
        assert confidence.needs_fallback
        assert confidence.recommended_strategy is not None

    def test_empty_results(self):
        """Test confidence with no results."""
        from perfect_rag.retrieval.confidence import ConfidenceEstimator, ConfidenceLevel

        estimator = ConfidenceEstimator()

        confidence = estimator.estimate([], "cualquier pregunta")

        assert confidence.level == ConfidenceLevel.VERY_LOW
        assert confidence.overall == 0.0
        assert confidence.needs_fallback

    @pytest.mark.asyncio
    async def test_fallback_execution(self):
        """Test fallback strategy execution."""
        from perfect_rag.retrieval.confidence import (
            FallbackSearchExecutor,
            FallbackStrategy,
        )

        # Mock search function
        call_count = 0

        async def mock_search(query, top_k, filters):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: poor results
                return [{"id": "c1", "content": "vago", "score": 0.3}]
            else:
                # Fallback: better results
                return [
                    {"id": "c1", "content": "información relevante", "score": 0.8},
                    {"id": "c2", "content": "más contexto", "score": 0.75},
                ]

        executor = FallbackSearchExecutor(mock_search)

        result = await executor.search_with_fallback(
            query="pregunta",
            initial_chunks=[{"id": "c1", "content": "vago", "score": 0.3}],
            min_confidence=0.5,
        )

        assert len(result.strategies_tried) > 0
        assert result.confidence.overall > 0.3  # Should improve


# Integration test
@pytest.mark.asyncio
async def test_full_pipeline_integration():
    """Test integration of all new components."""
    # This would be a more comprehensive test
    # For now, just verify imports work
    from perfect_rag.retrieval.sparse_bm25 import BM25Index
    from perfect_rag.retrieval.rag_fusion import RAGFusion, QueryIntent
    from perfect_rag.retrieval.mmr import MMRReranker
    from perfect_rag.retrieval.confidence import ConfidenceEstimator

    # Verify all modules import correctly
    assert BM25Index is not None
    assert RAGFusion is not None
    assert QueryIntent is not None
    assert MMRReranker is not None
    assert ConfidenceEstimator is not None
