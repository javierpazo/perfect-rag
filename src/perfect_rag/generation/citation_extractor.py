"""Citation extraction and verification from generated text."""

import re
from dataclasses import dataclass
from typing import Any

import structlog

from perfect_rag.models.openai_types import Citation
from perfect_rag.models.query import SourceChunk

logger = structlog.get_logger(__name__)


@dataclass
class ExtractedCitation:
    """A citation extracted from generated text."""

    marker: str  # e.g., "[1]"
    index: int  # 0-based index
    start_pos: int  # Position in text
    end_pos: int
    context: str  # Surrounding text
    source_chunk: SourceChunk | None = None  # Matched source


class CitationExtractor:
    """Extract and verify citations from generated responses."""

    # Patterns for different citation formats
    CITATION_PATTERNS = [
        r'\[(\d+)\]',  # [1], [2], etc.
        r'\[Source\s*(\d+)\]',  # [Source 1]
        r'\(Source\s*(\d+)\)',  # (Source 1)
        r'\[Ref\s*(\d+)\]',  # [Ref 1]
        r'\(\d+\)',  # (1)
    ]

    def __init__(self):
        self.main_pattern = re.compile(r'\[(\d+)\]')
        self.all_patterns = [re.compile(p) for p in self.CITATION_PATTERNS]

    def extract_citations(
        self,
        text: str,
        source_chunks: list[SourceChunk],
    ) -> list[ExtractedCitation]:
        """Extract citation markers from text and match to sources.

        Args:
            text: Generated text containing citation markers
            source_chunks: List of source chunks in order (index 0 = [1])

        Returns:
            List of extracted citations with matched sources
        """
        citations = []

        # Find all citation markers
        for match in self.main_pattern.finditer(text):
            marker = match.group(0)
            index = int(match.group(1)) - 1  # Convert to 0-based

            # Extract surrounding context (±50 chars)
            start_ctx = max(0, match.start() - 50)
            end_ctx = min(len(text), match.end() + 50)
            context = text[start_ctx:end_ctx]

            # Match to source chunk if index is valid
            source_chunk = None
            if 0 <= index < len(source_chunks):
                source_chunk = source_chunks[index]

            citations.append(
                ExtractedCitation(
                    marker=marker,
                    index=index,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    context=context,
                    source_chunk=source_chunk,
                )
            )

        return citations

    def build_citation_objects(
        self,
        source_chunks: list[SourceChunk],
        extracted_citations: list[ExtractedCitation] | None = None,
    ) -> list[Citation]:
        """Build Citation objects for API response.

        Args:
            source_chunks: All source chunks used in context
            extracted_citations: Optional list of actually-used citations

        Returns:
            List of Citation objects
        """
        # If we have extracted citations, only include used sources
        if extracted_citations:
            used_indices = {c.index for c in extracted_citations}
        else:
            used_indices = set(range(len(source_chunks)))

        citations = []
        for i, chunk in enumerate(source_chunks):
            if i in used_indices:
                citations.append(
                    Citation(
                        source_id=chunk.doc_id,
                        source_title=chunk.doc_title,
                        chunk_index=chunk.chunk_index,
                        text_snippet=self._create_snippet(chunk.content),
                        relevance_score=chunk.score,
                    )
                )

        return citations

    def _create_snippet(self, text: str, max_length: int = 200) -> str:
        """Create a text snippet for citation display."""
        if len(text) <= max_length:
            return text

        # Try to cut at sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind(". ")
        if last_period > max_length // 2:
            return truncated[: last_period + 1]

        return truncated + "..."

    def verify_citations(
        self,
        text: str,
        source_chunks: list[SourceChunk],
        strict: bool = False,
    ) -> dict[str, Any]:
        """Verify that citations in text are accurate.

        Args:
            text: Generated text with citations
            source_chunks: Source chunks used
            strict: Whether to require all claims be cited

        Returns:
            Verification result with issues found
        """
        citations = self.extract_citations(text, source_chunks)

        issues = []
        valid_citations = []
        invalid_citations = []

        for citation in citations:
            if citation.source_chunk is None:
                # Citation references non-existent source
                issues.append({
                    "type": "invalid_reference",
                    "marker": citation.marker,
                    "context": citation.context,
                    "message": f"Citation {citation.marker} references non-existent source",
                })
                invalid_citations.append(citation)
            else:
                # Check if context relates to source content
                if self._context_matches_source(citation.context, citation.source_chunk):
                    valid_citations.append(citation)
                else:
                    issues.append({
                        "type": "mismatched_content",
                        "marker": citation.marker,
                        "context": citation.context,
                        "message": f"Citation {citation.marker} may not accurately reflect source",
                    })
                    invalid_citations.append(citation)

        # Check for uncited claims (if strict mode)
        if strict:
            uncited = self._find_uncited_claims(text, citations)
            for claim in uncited:
                issues.append({
                    "type": "uncited_claim",
                    "context": claim,
                    "message": "Factual claim without citation",
                })

        return {
            "valid": len(issues) == 0,
            "total_citations": len(citations),
            "valid_citations": len(valid_citations),
            "invalid_citations": len(invalid_citations),
            "issues": issues,
        }

    def _context_matches_source(
        self,
        context: str,
        source: SourceChunk,
    ) -> bool:
        """Check if citation context matches source content.

        This is a simplified check - production would use semantic similarity.
        """
        # Extract key terms from context (words around citation)
        context_lower = context.lower()
        source_lower = source.content.lower()

        # Find words in context (excluding common words)
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for"}
        context_words = set(
            word for word in re.findall(r'\b\w+\b', context_lower)
            if len(word) > 3 and word not in stop_words
        )

        # Check overlap with source
        matching_words = sum(1 for word in context_words if word in source_lower)

        # Require at least 30% of significant words to match
        return matching_words >= len(context_words) * 0.3 if context_words else True

    def _find_uncited_claims(
        self,
        text: str,
        citations: list[ExtractedCitation],
    ) -> list[str]:
        """Find sentences that might need citations but don't have them.

        This is a heuristic approach - detects sentences with factual patterns.
        """
        # Patterns that suggest factual claims
        factual_patterns = [
            r'[\d,]+\s*(?:percent|%)',  # Percentages
            r'in\s+\d{4}',  # Years
            r'according to',  # Attribution without citation
            r'studies show',
            r'research indicates',
            r'it was found that',
            r'the data shows',
        ]

        uncited = []
        sentences = re.split(r'[.!?]\s+', text)

        citation_positions = {c.start_pos for c in citations}

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if sentence matches factual patterns
            is_factual = any(
                re.search(pattern, sentence, re.IGNORECASE)
                for pattern in factual_patterns
            )

            if is_factual:
                # Check if there's a citation in or near this sentence
                has_citation = bool(self.main_pattern.search(sentence))
                if not has_citation:
                    uncited.append(sentence)

        return uncited

    def format_inline_citations(
        self,
        text: str,
        source_chunks: list[SourceChunk],
        format: str = "markdown",
    ) -> str:
        """Format text with enhanced inline citations.

        Args:
            text: Text with citation markers
            source_chunks: Source chunks
            format: Output format ("markdown", "html", "plain")

        Returns:
            Text with formatted citations
        """
        if format == "markdown":
            # Add hover text with source info
            for i, chunk in enumerate(source_chunks):
                marker = f"[{i + 1}]"
                replacement = f'[{i + 1}]({chunk.doc_title}: "{self._create_snippet(chunk.content, 100)}")'
                text = text.replace(marker, replacement)

        elif format == "html":
            for i, chunk in enumerate(source_chunks):
                marker = f"[{i + 1}]"
                snippet = self._create_snippet(chunk.content, 100).replace('"', '&quot;')
                replacement = f'<sup><a href="#cite-{i + 1}" title="{chunk.doc_title}: {snippet}">[{i + 1}]</a></sup>'
                text = text.replace(marker, replacement)

        return text

    def generate_bibliography(
        self,
        source_chunks: list[SourceChunk],
        format: str = "markdown",
    ) -> str:
        """Generate a bibliography/references section.

        Args:
            source_chunks: Source chunks to include
            format: Output format

        Returns:
            Formatted bibliography
        """
        if not source_chunks:
            return ""

        lines = ["", "---", "**Sources:**", ""]

        for i, chunk in enumerate(source_chunks):
            if format == "markdown":
                lines.append(
                    f"{i + 1}. **{chunk.doc_title}** - "
                    f"{self._create_snippet(chunk.content, 100)}"
                )
            elif format == "html":
                lines.append(
                    f'<p id="cite-{i + 1}">[{i + 1}] <strong>{chunk.doc_title}</strong> - '
                    f'{self._create_snippet(chunk.content, 100)}</p>'
                )
            else:
                lines.append(
                    f"[{i + 1}] {chunk.doc_title} - {self._create_snippet(chunk.content, 100)}"
                )

        return "\n".join(lines)
