"""Document chunking strategies."""

import re
from abc import ABC, abstractmethod
from typing import Any

import structlog

from perfect_rag.config import Settings, get_settings
from perfect_rag.models.chunk import Chunk

logger = structlog.get_logger(__name__)


class Chunker(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Split text into chunks.

        Args:
            text: Text content to chunk
            doc_id: Parent document ID
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        pass


class FixedSizeChunker(Chunker):
    """Fixed-size chunking with overlap."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        length_function: str = "characters",  # "characters" or "tokens"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        if not text.strip():
            return []

        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = start + self.chunk_size

            # Adjust end to avoid cutting words
            if end < len(text):
                # Find last space before end
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    Chunk(
                        id=f"{doc_id}_chunk_{chunk_idx}",
                        doc_id=doc_id,
                        content=chunk_text,
                        chunk_index=chunk_idx,
                        start_char=start,
                        end_char=end,
                        token_count=len(chunk_text.split()),
                        metadata=metadata or {},
                    )
                )
                chunk_idx += 1

            # Move start with overlap
            start = end - self.chunk_overlap
            if start <= 0:
                start = end

        return chunks


class SentenceChunker(Chunker):
    """Sentence-based chunking."""

    def __init__(
        self,
        max_sentences: int = 5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
    ):
        self.max_sentences = max_sentences
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        # Sentence splitting pattern
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        if not text.strip():
            return []

        # Split into sentences
        sentences = self.sentence_pattern.split(text)

        chunks = []
        current_sentences = []
        current_length = 0
        chunk_idx = 0
        char_offset = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_len = len(sentence)

            # Check if adding this sentence would exceed max size
            if current_length + sentence_len > self.max_chunk_size and current_sentences:
                # Save current chunk
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    Chunk(
                        id=f"{doc_id}_chunk_{chunk_idx}",
                        doc_id=doc_id,
                        content=chunk_text,
                        chunk_index=chunk_idx,
                        start_char=char_offset,
                        end_char=char_offset + len(chunk_text),
                        token_count=len(chunk_text.split()),
                        metadata=metadata or {},
                    )
                )
                chunk_idx += 1
                char_offset += len(chunk_text) + 1

                current_sentences = []
                current_length = 0

            current_sentences.append(sentence)
            current_length += sentence_len

            # Check if we've hit max sentences
            if len(current_sentences) >= self.max_sentences and current_length >= self.min_chunk_size:
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    Chunk(
                        id=f"{doc_id}_chunk_{chunk_idx}",
                        doc_id=doc_id,
                        content=chunk_text,
                        chunk_index=chunk_idx,
                        start_char=char_offset,
                        end_char=char_offset + len(chunk_text),
                        token_count=len(chunk_text.split()),
                        metadata=metadata or {},
                    )
                )
                chunk_idx += 1
                char_offset += len(chunk_text) + 1

                current_sentences = []
                current_length = 0

        # Handle remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                Chunk(
                    id=f"{doc_id}_chunk_{chunk_idx}",
                    doc_id=doc_id,
                    content=chunk_text,
                    chunk_index=chunk_idx,
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_text),
                    token_count=len(chunk_text.split()),
                    metadata=metadata or {},
                )
            )

        return chunks


class ParagraphChunker(Chunker):
    """Paragraph-based chunking."""

    def __init__(
        self,
        max_paragraphs: int = 3,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1500,
    ):
        self.max_paragraphs = max_paragraphs
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        if not text.strip():
            return []

        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_paragraphs = []
        current_length = 0
        chunk_idx = 0
        char_offset = 0

        for para in paragraphs:
            para_len = len(para)

            # If single paragraph exceeds max, use fixed-size chunking
            if para_len > self.max_chunk_size:
                # Save current chunk first
                if current_paragraphs:
                    chunk_text = "\n\n".join(current_paragraphs)
                    chunks.append(
                        Chunk(
                            id=f"{doc_id}_chunk_{chunk_idx}",
                            doc_id=doc_id,
                            content=chunk_text,
                            chunk_index=chunk_idx,
                            start_char=char_offset,
                            end_char=char_offset + len(chunk_text),
                            token_count=len(chunk_text.split()),
                            metadata=metadata or {},
                        )
                    )
                    chunk_idx += 1
                    char_offset += len(chunk_text) + 2
                    current_paragraphs = []
                    current_length = 0

                # Chunk the large paragraph
                sub_chunker = FixedSizeChunker(
                    chunk_size=self.max_chunk_size,
                    chunk_overlap=50,
                )
                sub_chunks = sub_chunker.chunk(para, doc_id, metadata)
                for sub_chunk in sub_chunks:
                    sub_chunk.id = f"{doc_id}_chunk_{chunk_idx}"
                    sub_chunk.chunk_index = chunk_idx
                    sub_chunk.start_char = char_offset + sub_chunk.start_char
                    sub_chunk.end_char = char_offset + sub_chunk.end_char
                    chunks.append(sub_chunk)
                    chunk_idx += 1

                char_offset += para_len + 2
                continue

            # Check if adding would exceed max
            if current_length + para_len > self.max_chunk_size and current_paragraphs:
                chunk_text = "\n\n".join(current_paragraphs)
                chunks.append(
                    Chunk(
                        id=f"{doc_id}_chunk_{chunk_idx}",
                        doc_id=doc_id,
                        content=chunk_text,
                        chunk_index=chunk_idx,
                        start_char=char_offset,
                        end_char=char_offset + len(chunk_text),
                        token_count=len(chunk_text.split()),
                        metadata=metadata or {},
                    )
                )
                chunk_idx += 1
                char_offset += len(chunk_text) + 2

                current_paragraphs = []
                current_length = 0

            current_paragraphs.append(para)
            current_length += para_len

            # Check max paragraphs
            if len(current_paragraphs) >= self.max_paragraphs and current_length >= self.min_chunk_size:
                chunk_text = "\n\n".join(current_paragraphs)
                chunks.append(
                    Chunk(
                        id=f"{doc_id}_chunk_{chunk_idx}",
                        doc_id=doc_id,
                        content=chunk_text,
                        chunk_index=chunk_idx,
                        start_char=char_offset,
                        end_char=char_offset + len(chunk_text),
                        token_count=len(chunk_text.split()),
                        metadata=metadata or {},
                    )
                )
                chunk_idx += 1
                char_offset += len(chunk_text) + 2

                current_paragraphs = []
                current_length = 0

        # Handle remaining
        if current_paragraphs:
            chunk_text = "\n\n".join(current_paragraphs)
            chunks.append(
                Chunk(
                    id=f"{doc_id}_chunk_{chunk_idx}",
                    doc_id=doc_id,
                    content=chunk_text,
                    chunk_index=chunk_idx,
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_text),
                    token_count=len(chunk_text.split()),
                    metadata=metadata or {},
                )
            )

        return chunks


class SemanticChunker(Chunker):
    """Semantic-aware chunking using embeddings.

    Groups sentences by semantic similarity to create coherent chunks.
    """

    def __init__(
        self,
        embedding_service: Any = None,
        similarity_threshold: float = 0.7,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
    ):
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Synchronous fallback - uses sentence chunking."""
        # For sync context, fall back to sentence chunking
        sentence_chunker = SentenceChunker(
            max_sentences=5,
            min_chunk_size=self.min_chunk_size,
            max_chunk_size=self.max_chunk_size,
        )
        return sentence_chunker.chunk(text, doc_id, metadata)

    async def chunk_async(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Async semantic chunking using embeddings."""
        if not self.embedding_service:
            # Fall back to sentence chunking
            return self.chunk(text, doc_id, metadata)

        if not text.strip():
            return []

        # Split into sentences
        sentences = self.sentence_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        # Get embeddings for all sentences
        embeddings = await self.embedding_service.embed_texts(sentences)

        # Group sentences by semantic similarity
        chunks = []
        current_group = [0]  # indices of sentences in current group
        current_length = len(sentences[0])
        chunk_idx = 0
        char_offset = 0

        import numpy as np

        for i in range(1, len(sentences)):
            # Calculate similarity with previous sentence
            sim = np.dot(embeddings[i], embeddings[i - 1])

            sentence_len = len(sentences[i])

            # Decide whether to continue or start new chunk
            if (
                sim >= self.similarity_threshold
                and current_length + sentence_len <= self.max_chunk_size
            ):
                # Continue current group
                current_group.append(i)
                current_length += sentence_len
            else:
                # Save current group if it meets minimum size
                if current_length >= self.min_chunk_size:
                    chunk_text = " ".join(sentences[j] for j in current_group)
                    chunks.append(
                        Chunk(
                            id=f"{doc_id}_chunk_{chunk_idx}",
                            doc_id=doc_id,
                            content=chunk_text,
                            chunk_index=chunk_idx,
                            start_char=char_offset,
                            end_char=char_offset + len(chunk_text),
                            token_count=len(chunk_text.split()),
                            metadata=metadata or {},
                        )
                    )
                    chunk_idx += 1
                    char_offset += len(chunk_text) + 1

                    # Start new group
                    current_group = [i]
                    current_length = sentence_len
                else:
                    # Group too small, add sentence anyway
                    current_group.append(i)
                    current_length += sentence_len

        # Handle remaining sentences
        if current_group:
            chunk_text = " ".join(sentences[j] for j in current_group)
            chunks.append(
                Chunk(
                    id=f"{doc_id}_chunk_{chunk_idx}",
                    doc_id=doc_id,
                    content=chunk_text,
                    chunk_index=chunk_idx,
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_text),
                    token_count=len(chunk_text.split()),
                    metadata=metadata or {},
                )
            )

        return chunks


class RecursiveChunker(Chunker):
    """Recursive chunking with hierarchical separators.

    Tries to split by larger units first (sections, paragraphs),
    then falls back to smaller units (sentences, words) if needed.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n## ",      # Markdown H2
            "\n### ",     # Markdown H3
            "\n\n\n",     # Multiple newlines
            "\n\n",       # Paragraph
            "\n",         # Line
            ". ",         # Sentence
            ", ",         # Clause
            " ",          # Word
        ]

    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        if not text.strip():
            return []

        chunks_text = self._recursive_split(text, self.separators)

        chunks = []
        char_offset = 0

        for idx, chunk_text in enumerate(chunks_text):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            chunks.append(
                Chunk(
                    id=f"{doc_id}_chunk_{idx}",
                    doc_id=doc_id,
                    content=chunk_text,
                    chunk_index=idx,
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_text),
                    token_count=len(chunk_text.split()),
                    metadata=metadata or {},
                )
            )
            char_offset += len(chunk_text) + 1

        return chunks

    def _recursive_split(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:
        """Recursively split text using hierarchical separators."""
        if not separators:
            # Base case: no more separators, use character split
            return self._character_split(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split by current separator
        if separator in text:
            parts = text.split(separator)
        else:
            # Separator not found, try next one
            return self._recursive_split(text, remaining_separators)

        # Process each part
        chunks = []
        for part in parts:
            if len(part) <= self.chunk_size:
                chunks.append(part)
            else:
                # Part too large, recurse
                sub_chunks = self._recursive_split(part, remaining_separators)
                chunks.extend(sub_chunks)

        # Merge small adjacent chunks
        merged = self._merge_small_chunks(chunks)
        return merged

    def _character_split(self, text: str) -> list[str]:
        """Fall back to character-based splitting."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break

            # Try to find a good break point
            for sep in [" ", "\n", ".", ","]:
                last_sep = text.rfind(sep, start, end)
                if last_sep > start:
                    end = last_sep
                    break

            chunks.append(text[start:end])
            start = end - self.chunk_overlap
            if start <= 0:
                start = end

        return chunks

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """Merge chunks that are too small."""
        merged = []
        current = ""

        for chunk in chunks:
            if len(current) + len(chunk) <= self.chunk_size:
                current += (" " if current else "") + chunk
            else:
                if current:
                    merged.append(current)
                current = chunk

        if current:
            merged.append(current)

        return merged


# =============================================================================
# Factory Function
# =============================================================================

def get_chunker(
    strategy: str = "recursive",
    settings: Settings | None = None,
    **kwargs,
) -> Chunker:
    """Get chunker based on strategy.

    Args:
        strategy: Chunking strategy ("fixed", "sentence", "paragraph", "semantic", "recursive")
        settings: Optional settings for defaults
        **kwargs: Strategy-specific parameters

    Returns:
        Configured Chunker instance
    """
    settings = settings or get_settings()

    if strategy == "fixed":
        return FixedSizeChunker(
            chunk_size=kwargs.get("chunk_size", settings.chunk_size),
            chunk_overlap=kwargs.get("chunk_overlap", settings.chunk_overlap),
        )
    elif strategy == "sentence":
        return SentenceChunker(
            max_sentences=kwargs.get("max_sentences", 5),
            min_chunk_size=kwargs.get("min_chunk_size", 100),
            max_chunk_size=kwargs.get("max_chunk_size", settings.chunk_size),
        )
    elif strategy == "paragraph":
        return ParagraphChunker(
            max_paragraphs=kwargs.get("max_paragraphs", 3),
            min_chunk_size=kwargs.get("min_chunk_size", 100),
            max_chunk_size=kwargs.get("max_chunk_size", settings.chunk_size),
        )
    elif strategy == "semantic":
        return SemanticChunker(
            embedding_service=kwargs.get("embedding_service"),
            similarity_threshold=kwargs.get("similarity_threshold", 0.7),
            min_chunk_size=kwargs.get("min_chunk_size", 100),
            max_chunk_size=kwargs.get("max_chunk_size", settings.chunk_size),
        )
    elif strategy == "recursive":
        return RecursiveChunker(
            chunk_size=kwargs.get("chunk_size", settings.chunk_size),
            chunk_overlap=kwargs.get("chunk_overlap", settings.chunk_overlap),
            separators=kwargs.get("separators"),
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
