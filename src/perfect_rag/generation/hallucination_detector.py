"""Hallucination detection and grounding verification."""

import re
from dataclasses import dataclass
from typing import Any

import structlog

from perfect_rag.config import Settings, get_settings
from perfect_rag.models.query import SourceChunk

logger = structlog.get_logger(__name__)


@dataclass
class Claim:
    """A factual claim extracted from text."""

    text: str
    start_pos: int
    end_pos: int
    claim_type: str  # factual, opinion, hedged
    entities: list[str]
    requires_citation: bool


@dataclass
class VerificationResult:
    """Result of claim verification against sources."""

    claim: Claim
    is_grounded: bool
    supporting_chunks: list[str]  # Chunk IDs
    confidence: float
    contradiction_found: bool
    contradicting_chunks: list[str]
    explanation: str


class HallucinationDetector:
    """Detect hallucinations by verifying claims against source documents.

    Pipeline:
    1. Extract factual claims from generated text
    2. For each claim, check if it's supported by source chunks
    3. Check for contradictions between claims and sources
    4. Calculate grounding score
    """

    def __init__(
        self,
        llm_gateway: Any = None,
        settings: Settings | None = None,
    ):
        self.llm = llm_gateway
        self.settings = settings or get_settings()

        # Patterns that indicate factual claims
        self.factual_patterns = [
            r'\d+(?:\.\d+)?%',  # Percentages
            r'\$[\d,]+(?:\.\d+)?',  # Money
            r'\b(?:in|on|at)\s+\d{4}\b',  # Years
            r'\b\d+(?:,\d{3})*\b',  # Large numbers
            r'\b(?:always|never|all|none|every|no one)\b',  # Absolutes
            r'(?:is|are|was|were)\s+(?:the\s+)?(?:first|last|only|largest|smallest)',
            r'according to',
            r'studies? (?:show|indicate|suggest)',
            r'research (?:shows?|indicates?|suggests?)',
        ]

        # Hedging phrases (less likely to be hallucinations)
        self.hedging_patterns = [
            r'\b(?:may|might|could|possibly|perhaps|probably)\b',
            r'\b(?:seems?|appears?|suggests?)\b',
            r'\b(?:generally|typically|usually|often)\b',
            r'\b(?:some|many|few|certain)\b',
            r'(?:I think|I believe|in my opinion)',
        ]

    async def detect(
        self,
        generated_text: str,
        source_chunks: list[SourceChunk],
        strict: bool = False,
    ) -> dict[str, Any]:
        """Detect potential hallucinations in generated text.

        Args:
            generated_text: The LLM-generated response
            source_chunks: Source documents used for generation
            strict: If True, require all factual claims to be cited

        Returns:
            Detection result with grounding score and flagged claims
        """
        # Step 1: Extract claims
        claims = self._extract_claims(generated_text)

        # Step 2: Verify each claim
        verified_claims = []
        ungrounded_claims = []
        contradictions = []

        for claim in claims:
            result = await self._verify_claim(claim, source_chunks)
            verified_claims.append(result)

            if not result.is_grounded and claim.requires_citation:
                ungrounded_claims.append(result)

            if result.contradiction_found:
                contradictions.append(result)

        # Step 3: Calculate grounding score
        total_verifiable = sum(1 for c in claims if c.requires_citation)
        grounded_count = sum(
            1 for r in verified_claims
            if r.is_grounded and r.claim.requires_citation
        )

        grounding_score = grounded_count / total_verifiable if total_verifiable > 0 else 1.0

        # Step 4: Determine if response should be rejected
        reject = False
        reject_reason = None

        if strict and grounding_score < 0.8:
            reject = True
            reject_reason = f"Grounding score ({grounding_score:.2f}) below threshold (0.8)"
        elif contradictions:
            reject = True
            reject_reason = f"Found {len(contradictions)} contradiction(s) with source material"

        return {
            "grounding_score": grounding_score,
            "total_claims": len(claims),
            "verifiable_claims": total_verifiable,
            "grounded_claims": grounded_count,
            "ungrounded_claims": [
                {
                    "text": r.claim.text,
                    "explanation": r.explanation,
                }
                for r in ungrounded_claims
            ],
            "contradictions": [
                {
                    "claim": r.claim.text,
                    "contradicting_chunks": r.contradicting_chunks,
                    "explanation": r.explanation,
                }
                for r in contradictions
            ],
            "reject": reject,
            "reject_reason": reject_reason,
        }

    def _extract_claims(self, text: str) -> list[Claim]:
        """Extract factual claims from text."""
        claims = []

        # Split into sentences
        sentences = re.split(r'[.!?]+\s+', text)
        pos = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Determine claim type
            is_factual = any(
                re.search(pattern, sentence, re.IGNORECASE)
                for pattern in self.factual_patterns
            )
            is_hedged = any(
                re.search(pattern, sentence, re.IGNORECASE)
                for pattern in self.hedging_patterns
            )

            if is_hedged:
                claim_type = "hedged"
            elif is_factual:
                claim_type = "factual"
            else:
                claim_type = "opinion"

            # Extract entities (simple approach - capitalized words)
            entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)

            start_pos = text.find(sentence, pos)
            end_pos = start_pos + len(sentence) if start_pos >= 0 else pos + len(sentence)

            claims.append(
                Claim(
                    text=sentence,
                    start_pos=start_pos if start_pos >= 0 else pos,
                    end_pos=end_pos,
                    claim_type=claim_type,
                    entities=entities,
                    requires_citation=claim_type == "factual",
                )
            )

            pos = end_pos

        return claims

    async def _verify_claim(
        self,
        claim: Claim,
        source_chunks: list[SourceChunk],
    ) -> VerificationResult:
        """Verify a single claim against source chunks."""
        supporting_chunks = []
        contradicting_chunks = []
        max_support_score = 0.0

        # Check each source chunk
        for chunk in source_chunks:
            support_score = self._calculate_support(claim, chunk)

            if support_score > 0.5:
                supporting_chunks.append(chunk.chunk_id)
                max_support_score = max(max_support_score, support_score)
            elif support_score < -0.3:
                contradicting_chunks.append(chunk.chunk_id)

        is_grounded = len(supporting_chunks) > 0 and max_support_score > 0.5
        contradiction_found = len(contradicting_chunks) > 0

        # Generate explanation
        if is_grounded:
            explanation = f"Supported by {len(supporting_chunks)} source(s)"
        elif contradiction_found:
            explanation = f"Contradicted by {len(contradicting_chunks)} source(s)"
        else:
            explanation = "No supporting evidence found in sources"

        return VerificationResult(
            claim=claim,
            is_grounded=is_grounded,
            supporting_chunks=supporting_chunks,
            confidence=max_support_score,
            contradiction_found=contradiction_found,
            contradicting_chunks=contradicting_chunks,
            explanation=explanation,
        )

    def _calculate_support(self, claim: Claim, chunk: SourceChunk) -> float:
        """Calculate support score between claim and chunk.

        Returns:
            Score from -1 (contradiction) to 1 (strong support)
        """
        claim_lower = claim.text.lower()
        chunk_lower = chunk.content.lower()

        # Check entity overlap
        entity_matches = sum(
            1 for entity in claim.entities
            if entity.lower() in chunk_lower
        )
        entity_score = entity_matches / len(claim.entities) if claim.entities else 0

        # Check key term overlap (excluding stopwords)
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or"}
        claim_words = set(
            w for w in re.findall(r'\b\w+\b', claim_lower)
            if len(w) > 2 and w not in stopwords
        )
        chunk_words = set(re.findall(r'\b\w+\b', chunk_lower))

        word_overlap = len(claim_words.intersection(chunk_words))
        word_score = word_overlap / len(claim_words) if claim_words else 0

        # Check for negation/contradiction indicators
        negation_patterns = [
            (r'not\s+' + re.escape(w), -0.5) for w in claim_words
        ] + [
            (r'never\s+', -0.5),
            (r'false\s+', -0.5),
            (r'incorrect\s+', -0.5),
        ]

        negation_score = 0
        for pattern, score in negation_patterns:
            if re.search(pattern, chunk_lower):
                negation_score += score

        # Combine scores
        support_score = (entity_score * 0.4 + word_score * 0.6) + negation_score
        return max(-1.0, min(1.0, support_score))

    async def verify_with_llm(
        self,
        claim: str,
        source_chunks: list[SourceChunk],
    ) -> dict[str, Any]:
        """Use LLM to verify claim (more accurate but slower)."""
        if not self.llm:
            return {"verified": False, "confidence": 0.0, "reason": "No LLM available"}

        sources_text = "\n\n".join(
            f"[Source {i+1}]: {chunk.content}"
            for i, chunk in enumerate(source_chunks[:5])
        )

        prompt = f"""Verify if the following claim is supported by the source documents.

Claim: {claim}

Source Documents:
{sources_text}

Respond in JSON format:
{{
    "is_supported": true/false,
    "confidence": 0.0 to 1.0,
    "supporting_source": source number or null,
    "is_contradicted": true/false,
    "explanation": "brief explanation"
}}"""

        try:
            response = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
            )

            import json
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

        except Exception as e:
            logger.warning("LLM verification failed", error=str(e))

        return {"verified": False, "confidence": 0.0, "reason": "Verification failed"}


class ContradictionDetector:
    """Detect contradictions between multiple source chunks."""

    def __init__(self, llm_gateway: Any = None):
        self.llm = llm_gateway

    async def find_contradictions(
        self,
        chunks: list[SourceChunk],
    ) -> list[dict[str, Any]]:
        """Find contradictions between chunks.

        Returns list of contradiction pairs with explanations.
        """
        if len(chunks) < 2:
            return []

        contradictions = []

        # Pairwise comparison (limit to avoid quadratic explosion)
        chunks_to_check = chunks[:10]

        for i, chunk1 in enumerate(chunks_to_check):
            for chunk2 in chunks_to_check[i+1:]:
                is_contradiction, explanation = await self._check_contradiction(
                    chunk1, chunk2
                )
                if is_contradiction:
                    contradictions.append({
                        "chunk1_id": chunk1.chunk_id,
                        "chunk1_preview": chunk1.content[:100],
                        "chunk2_id": chunk2.chunk_id,
                        "chunk2_preview": chunk2.content[:100],
                        "explanation": explanation,
                    })

        return contradictions

    async def _check_contradiction(
        self,
        chunk1: SourceChunk,
        chunk2: SourceChunk,
    ) -> tuple[bool, str]:
        """Check if two chunks contradict each other."""
        # Simple heuristic check first
        negation_pairs = [
            ("increase", "decrease"),
            ("rise", "fall"),
            ("true", "false"),
            ("success", "failure"),
            ("positive", "negative"),
            ("yes", "no"),
            ("support", "oppose"),
        ]

        text1_lower = chunk1.content.lower()
        text2_lower = chunk2.content.lower()

        for word1, word2 in negation_pairs:
            if word1 in text1_lower and word2 in text2_lower:
                return True, f"Potential contradiction: '{word1}' vs '{word2}'"
            if word2 in text1_lower and word1 in text2_lower:
                return True, f"Potential contradiction: '{word2}' vs '{word1}'"

        # LLM verification for more nuanced contradictions
        if self.llm:
            return await self._llm_contradiction_check(chunk1, chunk2)

        return False, ""

    async def _llm_contradiction_check(
        self,
        chunk1: SourceChunk,
        chunk2: SourceChunk,
    ) -> tuple[bool, str]:
        """Use LLM to check for contradictions."""
        prompt = f"""Do these two text passages contradict each other?

Passage 1: {chunk1.content[:500]}

Passage 2: {chunk2.content[:500]}

Respond with:
- "YES" if they contradict
- "NO" if they are consistent or unrelated
Then briefly explain why.

Response:"""

        try:
            response = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150,
            )

            is_contradiction = response.strip().upper().startswith("YES")
            explanation = response.split("\n", 1)[-1].strip() if "\n" in response else ""

            return is_contradiction, explanation

        except Exception as e:
            logger.warning("LLM contradiction check failed", error=str(e))

        return False, ""
