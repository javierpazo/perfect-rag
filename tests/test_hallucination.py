"""Tests for hallucination detection."""

import pytest
from perfect_rag.generation.hallucination_detector import (
    HallucinationDetector,
    ContradictionDetector,
    Claim,
    VerificationResult,
)
from perfect_rag.models.query import SourceChunk


class TestClaimExtraction:
    """Tests for claim extraction."""

    def test_extract_factual_claim_with_percentage(self, hallucination_detector):
        """Test extraction of claims with percentages."""
        text = "The company grew by 50% in 2023."
        claims = hallucination_detector._extract_claims(text)

        assert len(claims) == 1
        assert claims[0].claim_type == "factual"
        assert claims[0].requires_citation is True

    def test_extract_factual_claim_with_year(self, hallucination_detector):
        """Test extraction of claims with years."""
        text = "The product was released in 2020."
        claims = hallucination_detector._extract_claims(text)

        assert len(claims) == 1
        assert claims[0].claim_type == "factual"

    def test_extract_hedged_claim(self, hallucination_detector):
        """Test extraction of hedged claims."""
        text = "The system might improve performance."
        claims = hallucination_detector._extract_claims(text)

        assert len(claims) == 1
        assert claims[0].claim_type == "hedged"
        assert claims[0].requires_citation is False

    def test_extract_opinion_claim(self, hallucination_detector):
        """Test extraction of opinion claims."""
        text = "The user interface is beautiful."
        claims = hallucination_detector._extract_claims(text)

        assert len(claims) == 1
        assert claims[0].claim_type == "opinion"
        assert claims[0].requires_citation is False

    def test_extract_multiple_claims(self, hallucination_detector):
        """Test extraction of multiple claims."""
        text = "Revenue increased by 30% in 2022. This might lead to expansion. The team is happy."
        claims = hallucination_detector._extract_claims(text)

        assert len(claims) == 3
        # First is factual (percentage + year)
        assert claims[0].claim_type == "factual"
        # Second is hedged (might)
        assert claims[1].claim_type == "hedged"
        # Third is opinion
        assert claims[2].claim_type == "opinion"

    def test_extract_entities(self, hallucination_detector):
        """Test entity extraction from claims."""
        text = "Microsoft acquired Activision Blizzard in 2023."
        claims = hallucination_detector._extract_claims(text)

        assert len(claims) == 1
        # Should extract capitalized entities
        assert "Microsoft" in claims[0].entities
        assert "Activision" in claims[0].entities or "Activision Blizzard" in claims[0].entities


class TestSupportCalculation:
    """Tests for support score calculation."""

    def test_high_support_with_entity_match(self, hallucination_detector):
        """Test high support score with entity match."""
        claim = Claim(
            text="Microsoft released Windows 11.",
            start_pos=0,
            end_pos=30,
            claim_type="factual",
            entities=["Microsoft", "Windows"],
            requires_citation=True,
        )

        chunk = SourceChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_title="Tech News",
            content="Microsoft announced the release of Windows 11 with new features.",
            score=0.9,
            chunk_index=0,
        )

        score = hallucination_detector._calculate_support(claim, chunk)
        assert score > 0.5  # Should indicate support

    def test_low_support_without_match(self, hallucination_detector):
        """Test low support score without match."""
        claim = Claim(
            text="Apple released iPhone 15.",
            start_pos=0,
            end_pos=25,
            claim_type="factual",
            entities=["Apple", "iPhone"],
            requires_citation=True,
        )

        chunk = SourceChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_title="Tech News",
            content="Google announced new Android features at the conference.",
            score=0.9,
            chunk_index=0,
        )

        score = hallucination_detector._calculate_support(claim, chunk)
        assert score < 0.5  # Should not indicate support

    def test_negation_reduces_support(self, hallucination_detector):
        """Test that negation reduces support score."""
        claim = Claim(
            text="The project was successful.",
            start_pos=0,
            end_pos=27,
            claim_type="factual",
            entities=["project"],
            requires_citation=True,
        )

        chunk_positive = SourceChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_title="Report",
            content="The project achieved all its goals and was successful.",
            score=0.9,
            chunk_index=0,
        )

        chunk_negative = SourceChunk(
            chunk_id="c2",
            doc_id="d1",
            doc_title="Report",
            content="The project was not successful and failed to meet targets.",
            score=0.9,
            chunk_index=1,
        )

        score_positive = hallucination_detector._calculate_support(claim, chunk_positive)
        score_negative = hallucination_detector._calculate_support(claim, chunk_negative)

        # Negative should be lower due to negation
        assert score_positive > score_negative


class TestVerification:
    """Tests for claim verification."""

    @pytest.mark.asyncio
    async def test_verify_grounded_claim(self, hallucination_detector):
        """Test verification of grounded claim."""
        claim = Claim(
            text="Python is a programming language.",
            start_pos=0,
            end_pos=33,
            claim_type="factual",
            entities=["Python"],
            requires_citation=True,
        )

        chunks = [
            SourceChunk(
                chunk_id="c1",
                doc_id="d1",
                doc_title="Programming Guide",
                content="Python is a high-level programming language known for its simplicity.",
                score=0.9,
                chunk_index=0,
            )
        ]

        result = await hallucination_detector._verify_claim(claim, chunks)

        assert result.is_grounded is True
        assert len(result.supporting_chunks) > 0

    @pytest.mark.asyncio
    async def test_verify_ungrounded_claim(self, hallucination_detector):
        """Test verification of ungrounded claim."""
        claim = Claim(
            text="Quantum computers will replace all classical computers by 2025.",
            start_pos=0,
            end_pos=62,
            claim_type="factual",
            entities=["Quantum"],
            requires_citation=True,
        )

        chunks = [
            SourceChunk(
                chunk_id="c1",
                doc_id="d1",
                doc_title="Tech Overview",
                content="Classical computers use binary logic gates for computation.",
                score=0.9,
                chunk_index=0,
            )
        ]

        result = await hallucination_detector._verify_claim(claim, chunks)

        assert result.is_grounded is False


class TestDetection:
    """Tests for full hallucination detection."""

    @pytest.mark.asyncio
    async def test_detect_no_hallucinations(self, hallucination_detector):
        """Test detection with no hallucinations."""
        generated_text = "Python is used for web development."
        chunks = [
            SourceChunk(
                chunk_id="c1",
                doc_id="d1",
                doc_title="Python Guide",
                content="Python is commonly used for web development with frameworks like Django.",
                score=0.9,
                chunk_index=0,
            )
        ]

        result = await hallucination_detector.detect(
            generated_text=generated_text,
            source_chunks=chunks,
            strict=False,
        )

        assert result["reject"] is False
        assert result["grounding_score"] > 0.5

    @pytest.mark.asyncio
    async def test_detect_with_hallucination(self, hallucination_detector):
        """Test detection catches hallucination."""
        generated_text = "Company XYZ grew by 500% in 2023 and became the largest in the world."
        chunks = [
            SourceChunk(
                chunk_id="c1",
                doc_id="d1",
                doc_title="Market Report",
                content="Market conditions were stable in 2023 with moderate growth.",
                score=0.9,
                chunk_index=0,
            )
        ]

        result = await hallucination_detector.detect(
            generated_text=generated_text,
            source_chunks=chunks,
            strict=True,
        )

        # Should flag ungrounded claims
        assert len(result["ungrounded_claims"]) > 0


class TestContradictionDetector:
    """Tests for contradiction detection."""

    @pytest.mark.asyncio
    async def test_detect_no_contradiction(self):
        """Test no contradiction between consistent chunks."""
        detector = ContradictionDetector()

        chunk1 = SourceChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_title="Report A",
            content="Sales increased significantly this quarter.",
            score=0.9,
            chunk_index=0,
        )

        chunk2 = SourceChunk(
            chunk_id="c2",
            doc_id="d2",
            doc_title="Report B",
            content="Revenue growth was positive in the recent period.",
            score=0.9,
            chunk_index=0,
        )

        contradictions = await detector.find_contradictions([chunk1, chunk2])
        assert len(contradictions) == 0

    @pytest.mark.asyncio
    async def test_detect_contradiction(self):
        """Test detection of contradiction between chunks."""
        detector = ContradictionDetector()

        chunk1 = SourceChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_title="Report A",
            content="The project was a success and achieved all goals.",
            score=0.9,
            chunk_index=0,
        )

        chunk2 = SourceChunk(
            chunk_id="c2",
            doc_id="d2",
            doc_title="Report B",
            content="The project was a failure and missed all targets.",
            score=0.9,
            chunk_index=0,
        )

        contradictions = await detector.find_contradictions([chunk1, chunk2])
        # Should detect success vs failure contradiction
        assert len(contradictions) > 0

    @pytest.mark.asyncio
    async def test_needs_multiple_chunks(self):
        """Test that single chunk returns no contradictions."""
        detector = ContradictionDetector()

        chunk = SourceChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_title="Report",
            content="Some content here.",
            score=0.9,
            chunk_index=0,
        )

        contradictions = await detector.find_contradictions([chunk])
        assert len(contradictions) == 0
