"""Evidence-first generation for reduced hallucinations.

Two-step generation approach:
1. EVIDENCE EXTRACTION: Extract relevant facts from retrieved chunks
2. ANSWER GENERATION: Generate answer based ONLY on extracted evidence

This approach significantly reduces hallucinations by ensuring the LLM
only uses verified information from the retrieved context.

Based on research showing that explicit evidence extraction before
generation improves factual accuracy and attribution.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

import structlog

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)


@dataclass
class ExtractedEvidence:
    """A piece of evidence extracted from context."""
    chunk_id: str
    doc_title: str
    evidence_text: str
    relevance_score: float
    page_number: int | None = None
    is_contradictory: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceSet:
    """Collection of extracted evidence for a query."""
    query: str
    evidences: list[ExtractedEvidence]
    contradictions: list[ExtractedEvidence]
    coverage_score: float = 0.0
    confidence: float = 0.0


@dataclass
class EvidenceBasedAnswer:
    """Answer generated from evidence."""
    answer: str
    evidences_used: list[ExtractedEvidence]
    citations: list[dict[str, Any]]
    confidence: float
    gaps_identified: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class EvidenceExtractor:
    """Extract relevant evidence from retrieved chunks."""

    def __init__(self, llm_gateway: Any = None, settings: Settings | None = None):
        self.llm = llm_gateway
        self.settings = settings or get_settings()

    async def extract_evidence(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        max_evidences: int = 10,
    ) -> EvidenceSet:
        """Extract relevant evidence from chunks.

        Args:
            query: User query
            chunks: Retrieved chunks with 'content', 'doc_id', 'score', etc.
            max_evidences: Maximum number of evidence pieces to extract

        Returns:
            EvidenceSet with extracted evidence
        """
        if not self.llm:
            # Fallback: use chunks directly as evidence
            return self._simple_extraction(query, chunks, max_evidences)

        # Prepare context with numbered chunks
        context_parts = []
        for i, chunk in enumerate(chunks[:10]):  # Limit to top 10
            content = chunk.get("content") or chunk.get("text", "")
            title = chunk.get("doc_title") or chunk.get("title", "Unknown")
            context_parts.append(f"[{i+1}] Document: {title}\n{content[:500]}")

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""Analyze the following retrieved documents and extract evidence relevant to answering the question.

QUESTION: {query}

DOCUMENTS:
{context}

For each piece of evidence found:
1. Quote the exact relevant text (be precise, do not paraphrase)
2. Note which document [number] it comes from
3. Assess if it directly answers the question (high relevance) or provides background (medium relevance)

Also identify:
- Any CONTRADICTORY information between documents
- Information GAPS (what's missing that would help answer the question)

Respond in this JSON format:
{{
  "evidences": [
    {{
      "quote": "exact text quote",
      "doc_number": 1,
      "relevance": "high|medium|low",
      "explanation": "why this is relevant"
    }}
  ],
  "contradictions": [
    {{
      "quote1": "text from doc A",
      "doc1": 1,
      "quote2": "contradicting text from doc B",
      "doc2": 2,
      "issue": "what the contradiction is"
    }}
  ],
  "gaps": ["missing information 1", "missing information 2"]
}}"""

        try:
            response = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500,
            )

            # Parse JSON response
            import json
            # Extract JSON from response (handle markdown code blocks)
            json_match = response
            if "```json" in response:
                json_match = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_match = response.split("```")[1].split("```")[0]

            data = json.loads(json_match.strip())

            # Build evidence list
            evidences = []
            for e in data.get("evidences", [])[:max_evidences]:
                doc_idx = e.get("doc_number", 1) - 1
                if 0 <= doc_idx < len(chunks):
                    chunk = chunks[doc_idx]
                    relevance_map = {"high": 1.0, "medium": 0.6, "low": 0.3}
                    evidences.append(ExtractedEvidence(
                        chunk_id=chunk.get("chunk_id") or chunk.get("id", ""),
                        doc_title=chunk.get("doc_title") or chunk.get("title", ""),
                        evidence_text=e.get("quote", ""),
                        relevance_score=relevance_map.get(e.get("relevance", "medium"), 0.5),
                        page_number=chunk.get("page_number"),
                        metadata={"explanation": e.get("explanation", "")},
                    ))

            # Build contradictions list
            contradictions = []
            for c in data.get("contradictions", []):
                contradictions.append(ExtractedEvidence(
                    chunk_id="",
                    doc_title="contradiction",
                    evidence_text=f"{c.get('quote1', '')} VS {c.get('quote2', '')}",
                    relevance_score=0.0,
                    is_contradictory=True,
                    metadata=c,
                ))

            # Calculate coverage score
            coverage = self._calculate_coverage(query, evidences)

            return EvidenceSet(
                query=query,
                evidences=evidences,
                contradictions=contradictions,
                coverage_score=coverage,
                confidence=self._calculate_confidence(evidences, contradictions),
            )

        except Exception as e:
            logger.warning("Evidence extraction failed, using fallback", error=str(e))
            return self._simple_extraction(query, chunks, max_evidences)

    def _simple_extraction(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        max_evidences: int,
    ) -> EvidenceSet:
        """Simple fallback extraction without LLM."""
        evidences = []
        for chunk in chunks[:max_evidences]:
            evidences.append(ExtractedEvidence(
                chunk_id=chunk.get("chunk_id") or chunk.get("id", ""),
                doc_title=chunk.get("doc_title") or chunk.get("title", ""),
                evidence_text=(chunk.get("content") or chunk.get("text", ""))[:300],
                relevance_score=chunk.get("score", 0.5),
                page_number=chunk.get("page_number"),
            ))

        return EvidenceSet(
            query=query,
            evidences=evidences,
            contradictions=[],
            coverage_score=min(1.0, len(evidences) / 3),
            confidence=0.5,
        )

    def _calculate_coverage(
        self,
        query: str,
        evidences: list[ExtractedEvidence],
    ) -> float:
        """Calculate how well evidence covers the query.

        Simple heuristic based on query term coverage in evidence.
        """
        if not evidences:
            return 0.0

        query_terms = set(query.lower().split())
        if not query_terms:
            return 0.0

        covered_terms = set()
        for evidence in evidences:
            evidence_terms = set(evidence.evidence_text.lower().split())
            covered_terms.update(query_terms & evidence_terms)

        return len(covered_terms) / len(query_terms)

    def _calculate_confidence(
        self,
        evidences: list[ExtractedEvidence],
        contradictions: list[ExtractedEvidence],
    ) -> float:
        """Calculate confidence in evidence set."""
        if not evidences:
            return 0.0

        # Base confidence from relevance scores
        avg_relevance = sum(e.relevance_score for e in evidences) / len(evidences)

        # Reduce confidence for contradictions
        contradiction_penalty = len(contradictions) * 0.1

        # Boost for number of high-relevance evidences
        high_rel_count = sum(1 for e in evidences if e.relevance_score >= 0.8)
        count_boost = min(0.2, high_rel_count * 0.05)

        confidence = avg_relevance + count_boost - contradiction_penalty
        return max(0.0, min(1.0, confidence))


class EvidenceBasedGenerator:
    """Generate answers based on extracted evidence."""

    def __init__(self, llm_gateway: Any = None, settings: Settings | None = None):
        self.llm = llm_gateway
        self.settings = settings or get_settings()
        self.evidence_extractor = EvidenceExtractor(llm_gateway, settings)

    async def generate(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        max_evidences: int = 8,
        include_citations: bool = True,
    ) -> EvidenceBasedAnswer:
        """Generate answer using evidence-first approach.

        Args:
            query: User query
            chunks: Retrieved chunks
            max_evidences: Maximum evidences to use
            include_citations: Whether to include citations

        Returns:
            EvidenceBasedAnswer with generated response
        """
        # Step 1: Extract evidence
        evidence_set = await self.evidence_extractor.extract_evidence(
            query, chunks, max_evidences
        )

        if not evidence_set.evidences:
            return EvidenceBasedAnswer(
                answer="No se encontró información relevante para responder a esta pregunta.",
                evidences_used=[],
                citations=[],
                confidence=0.0,
                gaps_identified=["No se encontraron documentos relevantes"],
            )

        # Step 2: Generate answer from evidence
        if not self.llm:
            return self._simple_answer(query, evidence_set)

        # Build evidence context
        evidence_parts = []
        for i, evidence in enumerate(evidence_set.evidences[:max_evidences], 1):
            evidence_parts.append(
                f"[E{i}] {evidence.doc_title}: {evidence.evidence_text}"
            )

        evidence_context = "\n\n".join(evidence_parts)

        # Include contradictions if any
        contradiction_note = ""
        if evidence_set.contradictions:
            contradiction_note = "\n\nNOTA: Existen contradicciones en la información disponible. Indica esto en tu respuesta."

        prompt = f"""Responde a la pregunta basándote EXCLUSIVAMENTE en la evidencia proporcionada.

IMPORTANTE:
- Solo usa información que esté explícitamente en la evidencia
- Si la evidencia no es suficiente para responder completamente, indícalo
- Cita la fuente usando [E1], [E2], etc. después de cada afirmación
- No inventes información ni uses conocimiento previo

PREGUNTA: {query}

EVIDENCIA:
{evidence_context}
{contradiction_note}

RESPUESTA (con citas [E1], [E2], etc.):"""

        response = await self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
        )

        # Build citations
        citations = []
        if include_citations:
            for i, evidence in enumerate(evidence_set.evidences[:max_evidences], 1):
                citation = {
                    "id": f"E{i}",
                    "chunk_id": evidence.chunk_id,
                    "doc_title": evidence.doc_title,
                    "quote": evidence.evidence_text[:100] + "..." if len(evidence.evidence_text) > 100 else evidence.evidence_text,
                    "relevance": evidence.relevance_score,
                }
                if evidence.page_number:
                    citation["page"] = evidence.page_number
                citations.append(citation)

        # Identify gaps
        gaps = []
        if evidence_set.coverage_score < 0.5:
            gaps.append("La evidencia no cubre completamente la pregunta")
        if len(evidence_set.evidences) < 2:
            gaps.append("Poca evidencia disponible")

        return EvidenceBasedAnswer(
            answer=response.strip(),
            evidences_used=evidence_set.evidences[:max_evidences],
            citations=citations,
            confidence=evidence_set.confidence,
            gaps_identified=gaps,
            metadata={
                "coverage_score": evidence_set.coverage_score,
                "num_evidences": len(evidence_set.evidences),
                "num_contradictions": len(evidence_set.contradictions),
            },
        )

    def _simple_answer(
        self,
        query: str,
        evidence_set: EvidenceSet,
    ) -> EvidenceBasedAnswer:
        """Simple fallback answer generation."""
        # Combine evidence texts
        combined = "\n\n".join([
            f"According to {e.doc_title}: {e.evidence_text}"
            for e in evidence_set.evidences[:3]
        ])

        citations = []
        for evidence in evidence_set.evidences[:3]:
            citations.append({
                "doc_title": evidence.doc_title,
                "quote": evidence.evidence_text[:100],
            })

        return EvidenceBasedAnswer(
            answer=f"Basado en los documentos recuperados:\n\n{combined}",
            evidences_used=evidence_set.evidences[:3],
            citations=citations,
            confidence=evidence_set.confidence,
            gaps_identified=[],
        )


async def generate_with_evidence(
    query: str,
    chunks: list[dict[str, Any]],
    llm_gateway: Any = None,
    settings: Settings | None = None,
) -> EvidenceBasedAnswer:
    """Convenience function for evidence-based generation."""
    generator = EvidenceBasedGenerator(llm_gateway, settings)
    return await generator.generate(query, chunks)
