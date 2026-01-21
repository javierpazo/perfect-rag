"""
Late Chunking implementation for improved semantic preservation.

Late chunking embeds the entire document first using long-context models,
then splits the embeddings into chunks. This preserves global context
that is lost in traditional chunk-then-embed approaches.

Reference: "Late Chunking: Contextual Chunk Embeddings Using Long-Context
Embedding Models" (arXiv:2409.04701)
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from perfect_rag.config import Settings, get_settings
from perfect_rag.models.chunk import Chunk

logger = structlog.get_logger(__name__)


@dataclass
class LateChunk:
    """A chunk created via late chunking."""

    content: str
    embedding: list[float]
    start_char: int
    end_char: int
    token_start: int
    token_end: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_chunk(self, doc_id: str, chunk_index: int) -> Chunk:
        """Convert to standard Chunk model."""
        return Chunk(
            id=f"{doc_id}_late_{chunk_index}",
            doc_id=doc_id,
            content=self.content,
            offset_start=self.start_char,
            offset_end=self.end_char,
            token_count=self.token_end - self.token_start,
            chunk_index=chunk_index,
            metadata={
                **(self.metadata or {}),
                "chunking_method": "late",
                "token_start": self.token_start,
                "token_end": self.token_end,
            },
        )


class LateChunker:
    """
    Late Chunking: Embed first, chunk later.

    Traditional chunking loses context at chunk boundaries.
    Late chunking preserves full document context in each chunk's embedding.

    Process:
    1. Embed entire document (using long-context model)
    2. Get token-level embeddings
    3. Split text into chunks
    4. Pool token embeddings for each chunk
    """

    def __init__(
        self,
        embedding_model: str = "jinaai/jina-embeddings-v3",
        max_tokens: int = 8192,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        pooling_strategy: str = "mean",  # mean, max, weighted
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.model_name = embedding_model
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pooling_strategy = pooling_strategy
        self._model = None
        self._tokenizer = None
        self._device = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Load the embedding model with token-level output support."""
        async with self._lock:
            if self._model is not None:
                return

            try:
                from transformers import AutoModel, AutoTokenizer
                import torch

                loop = asyncio.get_event_loop()

                def _load():
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name, trust_remote_code=True
                    )
                    model = AutoModel.from_pretrained(
                        self.model_name, trust_remote_code=True
                    )

                    # Determine device
                    if torch.cuda.is_available():
                        device = "cuda"
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        device = "cpu"

                    model = model.to(device)
                    model.eval()

                    return model, tokenizer, device

                self._model, self._tokenizer, self._device = await loop.run_in_executor(
                    None, _load
                )
                logger.info(
                    "Late chunking model loaded",
                    model=self.model_name,
                    device=self._device,
                )

            except ImportError as e:
                logger.error("Failed to load model - transformers not installed", error=str(e))
                raise ImportError(
                    "Late chunking requires transformers library. "
                    "Install with: pip install transformers torch"
                ) from e

    async def chunk_document(
        self,
        text: str,
        doc_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> list[LateChunk]:
        """
        Chunk a document using late chunking.

        Args:
            text: Full document text
            doc_id: Document ID for chunk ID generation
            metadata: Optional metadata to attach to chunks

        Returns:
            List of LateChunk objects with contextual embeddings
        """
        await self.initialize()

        import torch

        loop = asyncio.get_event_loop()

        def _process():
            # Tokenize full document
            encoding = self._tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_tokens,
                truncation=True,
                return_offsets_mapping=True,
            )

            input_ids = encoding["input_ids"].to(self._device)
            attention_mask = encoding["attention_mask"].to(self._device)
            offset_mapping = encoding["offset_mapping"][0].tolist()

            # Get token-level embeddings
            with torch.no_grad():
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

                # Use last hidden state for token embeddings
                # Shape: [1, seq_len, hidden_dim]
                token_embeddings = outputs.last_hidden_state[0]

            return token_embeddings.cpu().numpy(), offset_mapping

        token_embeddings, offset_mapping = await loop.run_in_executor(None, _process)

        # Find chunk boundaries based on character positions
        chunks = self._find_chunk_boundaries(text, offset_mapping)

        # Pool embeddings for each chunk
        late_chunks = []
        for chunk_info in chunks:
            start_token = chunk_info["token_start"]
            end_token = chunk_info["token_end"]

            # Get chunk embeddings
            chunk_token_embs = token_embeddings[start_token:end_token]

            if len(chunk_token_embs) == 0:
                continue

            # Pool embeddings
            if self.pooling_strategy == "mean":
                chunk_embedding = np.mean(chunk_token_embs, axis=0)
            elif self.pooling_strategy == "max":
                chunk_embedding = np.max(chunk_token_embs, axis=0)
            elif self.pooling_strategy == "weighted":
                # Weight by position (later tokens slightly more important)
                weights = np.linspace(0.8, 1.2, len(chunk_token_embs))
                weights = weights / weights.sum()
                chunk_embedding = np.average(chunk_token_embs, axis=0, weights=weights)
            else:
                chunk_embedding = np.mean(chunk_token_embs, axis=0)

            # Normalize
            norm = np.linalg.norm(chunk_embedding)
            if norm > 0:
                chunk_embedding = chunk_embedding / norm

            late_chunks.append(
                LateChunk(
                    content=chunk_info["text"],
                    embedding=chunk_embedding.tolist(),
                    start_char=chunk_info["start_char"],
                    end_char=chunk_info["end_char"],
                    token_start=start_token,
                    token_end=end_token,
                    metadata=metadata,
                )
            )

        logger.info(
            "Late chunking complete",
            doc_id=doc_id,
            num_chunks=len(late_chunks),
            pooling=self.pooling_strategy,
        )
        return late_chunks

    def _find_chunk_boundaries(
        self,
        text: str,
        offset_mapping: list[tuple[int, int]],
    ) -> list[dict[str, Any]]:
        """
        Find chunk boundaries that align with token boundaries.
        """
        chunks = []

        # Filter out special tokens (offset (0,0))
        valid_tokens = [
            (i, start, end)
            for i, (start, end) in enumerate(offset_mapping)
            if start != end
        ]

        if not valid_tokens:
            return []

        # Calculate approximate tokens per chunk
        tokens_per_chunk = self.chunk_size
        overlap_tokens = self.chunk_overlap

        i = 0
        while i < len(valid_tokens):
            # Find chunk end
            end_idx = min(i + tokens_per_chunk, len(valid_tokens))

            # Get character positions
            start_token_idx = valid_tokens[i][0]
            end_token_idx = valid_tokens[end_idx - 1][0]

            start_char = valid_tokens[i][1]
            end_char = valid_tokens[end_idx - 1][2]

            # Try to end at sentence boundary
            chunk_text = text[start_char:end_char]

            # Find last sentence end within chunk
            last_period = max(
                chunk_text.rfind(". "),
                chunk_text.rfind(".\n"),
                chunk_text.rfind("? "),
                chunk_text.rfind("! "),
            )

            if last_period > len(chunk_text) * 0.5:  # At least half the chunk
                end_char = start_char + last_period + 1
                chunk_text = text[start_char:end_char]

                # Adjust token end to match
                for j in range(end_idx - 1, i, -1):
                    if valid_tokens[j][2] <= end_char:
                        end_token_idx = valid_tokens[j][0]
                        end_idx = j + 1
                        break

            chunks.append(
                {
                    "text": chunk_text.strip(),
                    "start_char": start_char,
                    "end_char": end_char,
                    "token_start": start_token_idx,
                    "token_end": end_token_idx + 1,
                }
            )

            # Move to next chunk with overlap
            i = end_idx - overlap_tokens
            if i <= 0 or i >= len(valid_tokens):
                break
            # Prevent infinite loop - ensure forward progress
            prev_start = chunks[-1]["token_start"] if chunks else 0
            if valid_tokens[i][0] <= prev_start:
                i = end_idx

        return chunks

    async def chunk_and_convert(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[Chunk], list[list[float]]]:
        """
        Chunk document and return both Chunk objects and embeddings.

        This is useful for integration with the ingestion pipeline
        where embeddings are processed separately.

        Returns:
            Tuple of (list of Chunk objects, list of embeddings)
        """
        late_chunks = await self.chunk_document(text, doc_id, metadata)

        chunks = []
        embeddings = []

        for idx, late_chunk in enumerate(late_chunks):
            chunks.append(late_chunk.to_chunk(doc_id, idx))
            embeddings.append(late_chunk.embedding)

        return chunks, embeddings


class HybridChunker:
    """
    Hybrid chunker that combines late chunking with traditional chunking.

    Uses late chunking for long documents that benefit from global context,
    and traditional chunking for shorter documents.
    """

    def __init__(
        self,
        late_chunker: LateChunker,
        traditional_chunker: Any,  # RecursiveChunker or SemanticChunker
        late_chunk_threshold: int = 2000,  # Characters
    ):
        self.late_chunker = late_chunker
        self.traditional_chunker = traditional_chunker
        self.threshold = late_chunk_threshold

    async def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
        force_late: bool = False,
    ) -> list[Any]:
        """
        Chunk document using appropriate strategy.

        Args:
            text: Document text
            doc_id: Document ID
            metadata: Optional metadata
            force_late: Force use of late chunking regardless of length

        Returns:
            List of chunks (LateChunk or Chunk depending on strategy)
        """
        if force_late or len(text) > self.threshold:
            logger.info(
                "Using late chunking",
                doc_id=doc_id,
                text_length=len(text),
                threshold=self.threshold,
            )
            return await self.late_chunker.chunk_document(text, doc_id, metadata)
        else:
            logger.info(
                "Using traditional chunking",
                doc_id=doc_id,
                text_length=len(text),
                threshold=self.threshold,
            )
            # Use traditional chunker (synchronous)
            return self.traditional_chunker.chunk(text, doc_id, metadata)


class SimpleLateChunker:
    """
    Simplified late chunking using existing embedding service.

    Instead of token-level embeddings, uses overlapping context
    to preserve some global information. This is a lightweight
    alternative that doesn't require loading a separate model.
    """

    def __init__(
        self,
        embedding_service: Any,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        context_window: int = 200,  # Extra context to include
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.embeddings = embedding_service
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.context_window = context_window

    async def chunk_document(
        self,
        text: str,
        doc_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> list[LateChunk]:
        """
        Chunk with context-aware embeddings.

        Each chunk is embedded with surrounding context,
        but only the core text is stored.
        """
        chunks = []

        # Split into base chunks
        i = 0
        chunk_idx = 0

        while i < len(text):
            # Core chunk boundaries
            chunk_start = i
            chunk_end = min(i + self.chunk_size, len(text))

            # Find sentence boundary for clean break
            if chunk_end < len(text):
                last_period = text.rfind(".", chunk_start, chunk_end)
                if last_period > chunk_start + self.chunk_size // 2:
                    chunk_end = last_period + 1

            core_text = text[chunk_start:chunk_end]

            # Extended context for embedding
            context_start = max(0, chunk_start - self.context_window)
            context_end = min(len(text), chunk_end + self.context_window)
            context_text = text[context_start:context_end]

            # Embed with context
            embedding = await self.embeddings.embed_query(context_text)

            chunks.append(
                LateChunk(
                    content=core_text.strip(),
                    embedding=embedding,
                    start_char=chunk_start,
                    end_char=chunk_end,
                    token_start=0,  # Not tracked in simple mode
                    token_end=0,
                    metadata={
                        **(metadata or {}),
                        "chunking_method": "simple_late",
                        "context_window": self.context_window,
                    },
                )
            )

            chunk_idx += 1
            i = chunk_end - self.chunk_overlap
            if i <= chunk_start:  # Prevent infinite loop
                i = chunk_end

        logger.info(
            "Simple late chunking complete",
            doc_id=doc_id,
            num_chunks=len(chunks),
            context_window=self.context_window,
        )
        return chunks

    async def chunk_and_convert(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[Chunk], list[list[float]]]:
        """
        Chunk document and return both Chunk objects and embeddings.

        Returns:
            Tuple of (list of Chunk objects, list of embeddings)
        """
        late_chunks = await self.chunk_document(text, doc_id, metadata)

        chunks = []
        embeddings = []

        for idx, late_chunk in enumerate(late_chunks):
            chunks.append(late_chunk.to_chunk(doc_id, idx))
            embeddings.append(late_chunk.embedding)

        return chunks, embeddings


# =============================================================================
# Factory Functions
# =============================================================================


def get_late_chunker(
    strategy: str = "full",
    settings: Settings | None = None,
    embedding_service: Any = None,
    **kwargs,
) -> LateChunker | SimpleLateChunker:
    """
    Get a late chunker based on strategy.

    Args:
        strategy: "full" for LateChunker (requires transformers),
                  "simple" for SimpleLateChunker (uses existing embeddings)
        settings: Optional settings
        embedding_service: Required for "simple" strategy
        **kwargs: Additional arguments for the chunker

    Returns:
        LateChunker or SimpleLateChunker instance
    """
    settings = settings or get_settings()

    if strategy == "full":
        return LateChunker(
            embedding_model=kwargs.get(
                "embedding_model", settings.late_chunking_model
            ),
            max_tokens=kwargs.get("max_tokens", settings.late_chunking_max_tokens),
            chunk_size=kwargs.get("chunk_size", settings.chunk_size),
            chunk_overlap=kwargs.get("chunk_overlap", settings.chunk_overlap),
            pooling_strategy=kwargs.get(
                "pooling_strategy", settings.late_chunking_pooling
            ),
            settings=settings,
        )
    elif strategy == "simple":
        if embedding_service is None:
            raise ValueError("SimpleLateChunker requires an embedding_service")
        return SimpleLateChunker(
            embedding_service=embedding_service,
            chunk_size=kwargs.get("chunk_size", settings.chunk_size),
            chunk_overlap=kwargs.get("chunk_overlap", settings.chunk_overlap),
            context_window=kwargs.get(
                "context_window", settings.late_chunking_context_window
            ),
            settings=settings,
        )
    else:
        raise ValueError(f"Unknown late chunking strategy: {strategy}")


async def create_hybrid_chunker(
    traditional_chunker: Any,
    settings: Settings | None = None,
    embedding_service: Any = None,
    use_simple: bool = True,
    **kwargs,
) -> HybridChunker:
    """
    Create a hybrid chunker that combines late and traditional chunking.

    Args:
        traditional_chunker: Chunker to use for short documents
        settings: Optional settings
        embedding_service: Required if use_simple=True
        use_simple: Use SimpleLateChunker instead of full LateChunker
        **kwargs: Additional arguments

    Returns:
        HybridChunker instance
    """
    settings = settings or get_settings()

    if use_simple:
        late_chunker = get_late_chunker(
            strategy="simple",
            settings=settings,
            embedding_service=embedding_service,
            **kwargs,
        )
    else:
        late_chunker = get_late_chunker(
            strategy="full",
            settings=settings,
            **kwargs,
        )
        await late_chunker.initialize()

    return HybridChunker(
        late_chunker=late_chunker,
        traditional_chunker=traditional_chunker,
        late_chunk_threshold=kwargs.get(
            "threshold", settings.late_chunking_threshold
        ),
    )
