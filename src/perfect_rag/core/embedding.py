"""Embedding service using BGE-M3 for dense + sparse vectors."""

import asyncio
from typing import Any

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """Embedding service using BGE-M3 model.

    BGE-M3 provides:
    - Dense embeddings (1024 dimensions)
    - Sparse embeddings (lexical/BM25-style)
    - Multi-vector embeddings (ColBERT-style)

    Supports 100+ languages with excellent cross-lingual performance.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._model: SentenceTransformer | None = None
        self._reranker: Any | None = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Load embedding model (lazy loading)."""
        async with self._lock:
            if self._model is not None:
                return

            logger.info("Loading BGE-M3 embedding model...")

            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                self._load_model,
            )

            logger.info(
                "BGE-M3 model loaded",
                model=self.settings.embedding_model,
                dimension=self.settings.embedding_dimension,
            )

    def _load_model(self) -> SentenceTransformer:
        """Load the sentence transformer model."""
        model = SentenceTransformer(
            self.settings.embedding_model,
            trust_remote_code=True,
        )
        return model

    async def _ensure_model(self) -> SentenceTransformer:
        """Ensure model is loaded."""
        if self._model is None:
            await self.initialize()
        return self._model

    # =========================================================================
    # Dense Embeddings
    # =========================================================================

    async def embed_text(self, text: str) -> list[float]:
        """Generate dense embedding for a single text."""
        model = await self._ensure_model()

        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: model.encode(text, normalize_embeddings=True),
        )

        return embedding.tolist()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embeddings for multiple texts."""
        if not texts:
            return []

        model = await self._ensure_model()

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=self.settings.embedding_batch_size,
                show_progress_bar=len(texts) > 10,
            ),
        )

        return embeddings.tolist()

    async def embed_query(self, query: str) -> list[float]:
        """Generate dense embedding for a query.

        For BGE models, queries should use a specific instruction prefix
        for better retrieval performance.
        """
        model = await self._ensure_model()

        # BGE-M3 query instruction
        query_with_instruction = f"Represent this sentence for searching relevant passages: {query}"

        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: model.encode(query_with_instruction, normalize_embeddings=True),
        )

        return embedding.tolist()

    # =========================================================================
    # Sparse Embeddings (Lexical)
    # =========================================================================

    async def embed_sparse(self, text: str) -> dict[int, float]:
        """Generate sparse (lexical) embedding for text.

        Returns a dictionary mapping token IDs to weights.
        This is used for hybrid search (BM25-style retrieval).
        """
        model = await self._ensure_model()

        # Check if model supports sparse encoding
        if not hasattr(model, 'encode') or not hasattr(model[0], 'tokenizer'):
            # Fallback: create simple term frequency sparse vector
            return await self._create_simple_sparse(text)

        loop = asyncio.get_event_loop()
        sparse = await loop.run_in_executor(
            None,
            lambda: self._compute_sparse_embedding(model, text),
        )

        return sparse

    async def embed_sparse_batch(self, texts: list[str]) -> list[dict[int, float]]:
        """Generate sparse embeddings for multiple texts."""
        if not texts:
            return []

        results = []
        for text in texts:
            sparse = await self.embed_sparse(text)
            results.append(sparse)

        return results

    def _compute_sparse_embedding(
        self,
        model: SentenceTransformer,
        text: str,
    ) -> dict[int, float]:
        """Compute sparse embedding using model tokenizer."""
        try:
            # Try to use the model's tokenizer for sparse representation
            tokenizer = model.tokenizer
            tokens = tokenizer.encode(text, add_special_tokens=False)

            # Count token frequencies
            token_counts: dict[int, int] = {}
            for token_id in tokens:
                token_counts[token_id] = token_counts.get(token_id, 0) + 1

            # Apply simple TF weighting with log normalization
            sparse_vector = {}
            total_tokens = len(tokens) if tokens else 1

            for token_id, count in token_counts.items():
                # TF with sublinear scaling
                tf = 1 + np.log(count) if count > 0 else 0
                # Normalize by document length
                weight = tf / np.sqrt(total_tokens)
                if weight > 0.01:  # Filter very low weights
                    sparse_vector[int(token_id)] = float(weight)

            return sparse_vector

        except Exception as e:
            logger.warning("Sparse embedding failed, using fallback", error=str(e))
            # Fallback to simple hash-based sparse vector
            return self._simple_sparse_sync(text)

    def _simple_sparse_sync(self, text: str) -> dict[int, float]:
        """Simple fallback sparse representation using word hashing."""
        words = text.lower().split()
        word_counts: dict[str, int] = {}

        for word in words:
            # Clean word
            word = ''.join(c for c in word if c.isalnum())
            if len(word) > 2:  # Skip very short words
                word_counts[word] = word_counts.get(word, 0) + 1

        sparse_vector = {}
        total_words = len(words) if words else 1

        for word, count in word_counts.items():
            # Hash word to token ID (modulo vocab size)
            token_id = hash(word) % 100000
            # TF with sublinear scaling
            tf = 1 + np.log(count) if count > 0 else 0
            weight = tf / np.sqrt(total_words)
            if weight > 0.01:
                sparse_vector[token_id] = float(weight)

        return sparse_vector

    async def _create_simple_sparse(self, text: str) -> dict[int, float]:
        """Async wrapper for simple sparse embedding."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._simple_sparse_sync(text),
        )

    # =========================================================================
    # Combined Embeddings
    # =========================================================================

    async def embed_hybrid(
        self,
        text: str,
    ) -> tuple[list[float], dict[int, float]]:
        """Generate both dense and sparse embeddings for text."""
        dense = await self.embed_text(text)
        sparse = await self.embed_sparse(text)
        return dense, sparse

    async def embed_hybrid_batch(
        self,
        texts: list[str],
    ) -> list[tuple[list[float], dict[int, float]]]:
        """Generate both dense and sparse embeddings for multiple texts."""
        if not texts:
            return []

        # Get dense embeddings in batch
        dense_embeddings = await self.embed_texts(texts)

        # Get sparse embeddings
        sparse_embeddings = await self.embed_sparse_batch(texts)

        return list(zip(dense_embeddings, sparse_embeddings))

    # =========================================================================
    # Reranking
    # =========================================================================

    async def load_reranker(self) -> None:
        """Load cross-encoder reranker model."""
        async with self._lock:
            if self._reranker is not None:
                return

            logger.info("Loading reranker model...")

            loop = asyncio.get_event_loop()
            self._reranker = await loop.run_in_executor(
                None,
                self._load_reranker_model,
            )

            logger.info("Reranker loaded", model=self.settings.reranker_model)

    def _load_reranker_model(self) -> Any:
        """Load the cross-encoder reranker."""
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(
            self.settings.reranker_model,
            trust_remote_code=True,
        )
        return reranker

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Rerank documents for a query using cross-encoder.

        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return (None = all)

        Returns:
            List of (original_index, score) tuples, sorted by score descending
        """
        if not documents:
            return []

        if self._reranker is None:
            await self.load_reranker()

        loop = asyncio.get_event_loop()

        # Create query-document pairs
        pairs = [(query, doc) for doc in documents]

        # Get reranker scores
        scores = await loop.run_in_executor(
            None,
            lambda: self._reranker.predict(pairs),
        )

        # Create (index, score) pairs and sort by score descending
        indexed_scores = [(i, float(score)) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores

    async def rerank_with_metadata(
        self,
        query: str,
        documents: list[dict[str, Any]],
        text_key: str = "text",
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank documents with metadata, preserving all fields.

        Args:
            query: The search query
            documents: List of document dicts with text and metadata
            text_key: Key for the text field in documents
            top_k: Number of top results to return

        Returns:
            Reranked documents with added 'rerank_score' field
        """
        if not documents:
            return []

        texts = [doc.get(text_key, "") for doc in documents]
        ranked = await self.rerank(query, texts, top_k)

        results = []
        for original_idx, score in ranked:
            doc = documents[original_idx].copy()
            doc["rerank_score"] = score
            results.append(doc)

        return results

    # =========================================================================
    # Utilities
    # =========================================================================

    async def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        embeddings = await self.embed_texts([text1, text2])

        vec1 = np.array(embeddings[0])
        vec2 = np.array(embeddings[1])

        # Cosine similarity (vectors are already normalized)
        similarity = float(np.dot(vec1, vec2))
        return similarity

    async def health_check(self) -> bool:
        """Check if embedding service is operational."""
        try:
            await self._ensure_model()
            # Quick test embedding
            _ = await self.embed_text("test")
            return True
        except Exception as e:
            logger.error("Embedding health check failed", error=str(e))
            return False

    def get_info(self) -> dict[str, Any]:
        """Get embedding service information."""
        return {
            "model": self.settings.embedding_model,
            "dimension": self.settings.embedding_dimension,
            "reranker": self.settings.reranker_model,
            "batch_size": self.settings.embedding_batch_size,
            "loaded": self._model is not None,
            "reranker_loaded": self._reranker is not None,
        }


# Global service instance
_service: EmbeddingService | None = None


async def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service."""
    global _service
    if _service is None:
        _service = EmbeddingService()
        await _service.initialize()
    return _service
