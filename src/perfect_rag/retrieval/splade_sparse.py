"""SPLADE: Sparse Lexical and Expansion model for first-stage retrieval.

SPLADE learns sparse representations that combine:
- Lexical matching (like BM25)
- Learned term expansion (adds related terms)
- Contextual weighting (importance per document)

This produces high-quality sparse vectors that can be stored in Qdrant
and combined with dense vectors via RRF fusion.

Reference: "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking"
https://arxiv.org/abs/2107.05720

Available models:
- naver/splade_v3_distil (fast, good quality)
- naver/splade_v2_max (higher quality, slower)
"""

import asyncio
from typing import Any

import numpy as np
import structlog

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)

# SPLADE vocabulary size (typically 30522 for BERT tokenizer)
SPLADE_VOCAB_SIZE = 30522


class SPLADEEncoder:
    """SPLADE sparse encoder for learned sparse representations.

    SPLADE produces sparse vectors where:
    - Each dimension corresponds to a vocabulary term
    - Values represent learned importance weights
    - Related terms are automatically expanded

    This is more powerful than BM25 because:
    - Term expansion is learned from data
    - Weights are contextual (same term, different importance)
    - Can capture synonyms and related concepts
    """

    def __init__(
        self,
        model_name: str = "naver/splade_v3_distil",
        max_output_length: int = 256,
        aggregation: str = "max",  # "max" or "sum"
        settings: Settings | None = None,
    ):
        """Initialize SPLADE encoder.

        Args:
            model_name: HuggingFace model name
            max_output_length: Max sequence length for tokenizer
            aggregation: How to aggregate token scores ("max" or "sum")
            settings: Application settings
        """
        self.model_name = model_name
        self.max_output_length = max_output_length
        self.aggregation = aggregation
        self.settings = settings or get_settings()

        self._model = None
        self._tokenizer = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Load SPLADE model (lazy loading)."""
        async with self._lock:
            if self._model is not None:
                return

            logger.info("Loading SPLADE model...", model=self.model_name)

            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model)

                logger.info(
                    "SPLADE model loaded",
                    model=self.model_name,
                    vocab_size=SPLADE_VOCAB_SIZE,
                )
            except Exception as e:
                logger.error("Failed to load SPLADE model", error=str(e))
                raise

    def _load_model(self) -> None:
        """Load model and tokenizer synchronously."""
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self._model.eval()

        # Move to appropriate device
        if torch.cuda.is_available():
            self._model = self._model.cuda()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self._model = self._model.to('mps')

    async def _ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if self._model is None:
            await self.initialize()

    async def encode(self, text: str) -> dict[int, float]:
        """Encode text to sparse vector using SPLADE.

        Args:
            text: Input text

        Returns:
            Dictionary mapping token IDs to importance weights
        """
        await self._ensure_loaded()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._encode_sync, text)

    def _encode_sync(self, text: str) -> dict[int, float]:
        """Synchronous encoding."""
        import torch

        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_output_length,
            truncation=True,
            padding=True,
        )

        # Move to same device as model
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits  # (batch, seq_len, vocab_size)

        # Apply ReLU + log (SPLADE activation)
        # This creates sparse activations where only important terms have non-zero values
        sparse_activations = torch.log1p(torch.relu(logits))

        # Aggregate over sequence dimension
        if self.aggregation == "max":
            # Max pooling: take maximum activation per vocabulary term
            aggregated = sparse_activations.max(dim=1).values  # (batch, vocab_size)
        else:
            # Sum pooling: sum activations per vocabulary term
            aggregated = sparse_activations.sum(dim=1)  # (batch, vocab_size)

        # Convert to sparse dict (only non-zero values)
        activations = aggregated.squeeze(0).cpu().numpy()
        sparse_dict = {
            int(idx): float(activations[idx])
            for idx in np.where(activations > 0)[0]
        }

        return sparse_dict

    async def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 8,
    ) -> list[dict[int, float]]:
        """Encode multiple texts to sparse vectors.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing

        Returns:
            List of sparse dictionaries
        """
        await self._ensure_loaded()

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = await asyncio.gather(*[self.encode(t) for t in batch])
            results.extend(batch_results)

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get encoder statistics."""
        return {
            "model_name": self.model_name,
            "max_output_length": self.max_output_length,
            "aggregation": self.aggregation,
            "vocab_size": SPLADE_VOCAB_SIZE,
            "loaded": self._model is not None,
        }


class SPLADECachedEncoder:
    """SPLADE encoder with LRU cache for repeated queries.

    Caches the sparse representations of frequently-seen texts
    to avoid recomputing them.
    """

    def __init__(
        self,
        encoder: SPLADEEncoder,
        cache_size: int = 1000,
    ):
        self.encoder = encoder
        self.cache_size = cache_size
        self._cache: dict[str, dict[int, float]] = {}
        self._access_order: list[str] = []

    async def encode(self, text: str) -> dict[int, float]:
        """Encode with caching."""
        # Check cache
        if text in self._cache:
            return self._cache[text]

        # Encode
        result = await self.encoder.encode(text)

        # Cache result with LRU eviction
        if len(self._cache) >= self.cache_size:
            # Remove oldest
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[text] = result
        self._access_order.append(text)

        return result

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "max_size": self.cache_size,
        }


class HybridSparseRetriever:
    """Hybrid sparse retriever combining BM25 and SPLADE.

    This allows using both:
    - BM25: Fast, interpretable, good for exact term matching
    - SPLADE: Learned expansion, better for semantic matching

    Results can be combined with RRF fusion.
    """

    def __init__(
        self,
        bm25_index: Any = None,  # BM25Index from sparse_bm25.py
        splade_encoder: SPLADEEncoder | None = None,
        bm25_weight: float = 0.5,
        splade_weight: float = 0.5,
        settings: Settings | None = None,
    ):
        self.bm25_index = bm25_index
        self.splade_encoder = splade_encoder
        self.bm25_weight = bm25_weight
        self.splade_weight = splade_weight
        self.settings = settings or get_settings()

    async def search(
        self,
        query: str,
        top_k: int = 20,
        use_bm25: bool = True,
        use_splade: bool = True,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining BM25 and SPLADE.

        Args:
            query: Search query
            top_k: Number of results
            use_bm25: Whether to use BM25
            use_splade: Whether to use SPLADE

        Returns:
            List of search results
        """
        results_by_doc: dict[str, dict[str, Any]] = {}

        # BM25 search
        if use_bm25 and self.bm25_index:
            bm25_results = self.bm25_index.search(query, top_k=top_k * 2)
            for result in bm25_results:
                doc_id = result.doc_id
                if doc_id not in results_by_doc:
                    results_by_doc[doc_id] = {
                        "doc_id": doc_id,
                        "bm25_score": 0,
                        "splade_score": 0,
                        "metadata": result.metadata,
                    }
                results_by_doc[doc_id]["bm25_score"] = result.score

        # SPLADE search would require vector search in Qdrant
        # This is a placeholder - actual SPLADE search would use
        # Qdrant's sparse vector search with the SPLADE embeddings

        # Combine scores
        combined = []
        for doc_id, data in results_by_doc.items():
            combined_score = (
                self.bm25_weight * data["bm25_score"] +
                self.splade_weight * data["splade_score"]
            )
            combined.append({
                "doc_id": doc_id,
                "score": combined_score,
                "bm25_score": data["bm25_score"],
                "splade_score": data["splade_score"],
                "metadata": data["metadata"],
            })

        # Sort by combined score
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]


# Convenience function
async def get_splade_encoder(
    model_name: str = "naver/splade_v3_distil",
    settings: Settings | None = None,
) -> SPLADEEncoder:
    """Get or create SPLADE encoder."""
    encoder = SPLADEEncoder(model_name=model_name, settings=settings)
    await encoder.initialize()
    return encoder
