"""Maximal Marginal Relevance (MMR) for diverse result selection.

MMR balances relevance and diversity by iteratively selecting documents
that are both relevant to the query AND different from already selected documents.

MMR formula:
MMR = λ * Sim(d, q) - (1-λ) * max[Sim(d, d') for d' in selected]

where:
- Sim(d, q): similarity between document d and query q
- Sim(d, d'): similarity between document d and already selected document d'
- λ: trade-off parameter (0 = max diversity, 1 = max relevance)

This reduces redundancy in the retrieved context window, allowing
more diverse information to be included.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import structlog

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)


@dataclass
class MMRResult:
    """Result from MMR selection."""
    doc_id: str
    mmr_score: float
    relevance_score: float
    diversity_penalty: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


class MMRReranker:
    """MMR-based result diversification.

    Can work with:
    - Pre-computed embeddings (most efficient)
    - On-the-fly embedding computation
    - Text similarity as fallback
    """

    def __init__(
        self,
        embedding_service: Any = None,
        lambda_param: float = 0.7,
        settings: Settings | None = None,
    ):
        """Initialize MMR reranker.

        Args:
            embedding_service: Service for computing embeddings
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
            settings: Application settings
        """
        self.embedding_service = embedding_service
        self.lambda_param = lambda_param
        self.settings = settings or get_settings()

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        arr1 = np.array(vec1)
        arr2 = np.array(vec2)

        dot = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute simple text overlap similarity as fallback."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    async def compute_embedding(self, text: str) -> list[float] | None:
        """Compute embedding for a text."""
        if not self.embedding_service:
            return None

        try:
            if hasattr(self.embedding_service, 'embed_text'):
                return await self.embedding_service.embed_text(text)
            elif hasattr(self.embedding_service, 'embed_query'):
                return await self.embedding_service.embed_query(text)
        except Exception as e:
            logger.warning("Failed to compute embedding", error=str(e))

        return None

    async def select_diverse(
        self,
        candidates: list[dict[str, Any]],
        query: str,
        top_k: int = 10,
        query_embedding: list[float] | None = None,
        candidate_embeddings: dict[str, list[float]] | None = None,
        text_key: str = "content",
        id_key: str = "id",
    ) -> list[MMRResult]:
        """Select diverse candidates using MMR.

        Args:
            candidates: List of candidate documents with 'id', 'content', 'score', etc.
            query: Original query text
            top_k: Number of results to select
            query_embedding: Pre-computed query embedding (optional)
            candidate_embeddings: Pre-computed candidate embeddings (optional)
            text_key: Key for text content in candidate dict
            id_key: Key for document ID in candidate dict

        Returns:
            List of MMRResult sorted by MMR score
        """
        if not candidates:
            return []

        if len(candidates) <= top_k:
            # No need for diversification
            return [
                MMRResult(
                    doc_id=c.get(id_key, str(i)),
                    mmr_score=c.get("score", 0),
                    relevance_score=c.get("score", 0),
                    diversity_penalty=0,
                    rank=i + 1,
                    metadata=c,
                )
                for i, c in enumerate(candidates)
            ]

        # Get query embedding
        if query_embedding is None and self.embedding_service:
            query_embedding = await self.compute_embedding(query)

        # Get candidate embeddings
        embeddings: dict[str, list[float]] = {}
        if candidate_embeddings:
            embeddings = candidate_embeddings
        elif self.embedding_service:
            # Compute embeddings on demand
            for candidate in candidates:
                doc_id = candidate.get(id_key, "")
                text = candidate.get(text_key, "")
                if doc_id and text:
                    emb = await self.compute_embedding(text)
                    if emb:
                        embeddings[doc_id] = emb

        # Extract relevance scores
        relevance_scores = {}
        for candidate in candidates:
            doc_id = candidate.get(id_key, "")
            score = candidate.get("rerank_score") or candidate.get("score", 0.5)
            relevance_scores[doc_id] = score

        # MMR selection
        selected: list[MMRResult] = []
        remaining = list(candidates)

        for rank in range(top_k):
            if not remaining:
                break

            best_score = float('-inf')
            best_idx = -1
            best_diversity_penalty = 0

            for i, candidate in enumerate(remaining):
                doc_id = candidate.get(id_key, "")
                relevance = relevance_scores.get(doc_id, 0.5)

                # Compute diversity penalty
                diversity_penalty = 0.0
                if selected:
                    if embeddings and doc_id in embeddings:
                        # Use embedding similarity
                        penalties = []
                        for sel in selected:
                            sel_id = sel.doc_id
                            if sel_id in embeddings:
                                sim = self._cosine_similarity(
                                    embeddings[doc_id],
                                    embeddings[sel_id]
                                )
                                penalties.append(sim)
                        if penalties:
                            diversity_penalty = max(penalties)
                    else:
                        # Fallback to text similarity
                        candidate_text = candidate.get(text_key, "")
                        penalties = []
                        for sel in selected:
                            sel_text = next(
                                (c.get(text_key, "") for c in candidates if c.get(id_key) == sel.doc_id),
                                ""
                            )
                            if candidate_text and sel_text:
                                sim = self._text_similarity(candidate_text, sel_text)
                                penalties.append(sim)
                        if penalties:
                            diversity_penalty = max(penalties)

                # MMR score
                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * diversity_penalty

                if mmr > best_score:
                    best_score = mmr
                    best_idx = i
                    best_diversity_penalty = diversity_penalty

            if best_idx >= 0:
                best_candidate = remaining.pop(best_idx)
                doc_id = best_candidate.get(id_key, "")

                selected.append(MMRResult(
                    doc_id=doc_id,
                    mmr_score=best_score,
                    relevance_score=relevance_scores.get(doc_id, 0.5),
                    diversity_penalty=best_diversity_penalty,
                    rank=rank + 1,
                    metadata=best_candidate,
                ))

        return selected

    def select_diverse_sync(
        self,
        candidates: list[dict[str, Any]],
        query_embedding: list[float],
        candidate_embeddings: dict[str, list[float]],
        top_k: int = 10,
        id_key: str = "id",
    ) -> list[MMRResult]:
        """Synchronous MMR selection with pre-computed embeddings.

        More efficient when embeddings are already available.
        """
        if not candidates:
            return []

        if len(candidates) <= top_k:
            return [
                MMRResult(
                    doc_id=c.get(id_key, str(i)),
                    mmr_score=c.get("score", 0),
                    relevance_score=c.get("score", 0),
                    diversity_penalty=0,
                    rank=i + 1,
                    metadata=c,
                )
                for i, c in enumerate(candidates)
            ]

        # Extract relevance scores
        relevance_scores = {}
        for candidate in candidates:
            doc_id = candidate.get(id_key, "")
            score = candidate.get("rerank_score") or candidate.get("score", 0.5)
            relevance_scores[doc_id] = score

        # MMR selection
        selected: list[MMRResult] = []
        remaining = list(candidates)

        for rank in range(top_k):
            if not remaining:
                break

            best_score = float('-inf')
            best_idx = -1
            best_diversity_penalty = 0

            for i, candidate in enumerate(remaining):
                doc_id = candidate.get(id_key, "")
                relevance = relevance_scores.get(doc_id, 0.5)

                # Compute diversity penalty using embeddings
                diversity_penalty = 0.0
                if selected and doc_id in candidate_embeddings:
                    penalties = []
                    for sel in selected:
                        sel_id = sel.doc_id
                        if sel_id in candidate_embeddings:
                            sim = self._cosine_similarity(
                                candidate_embeddings[doc_id],
                                candidate_embeddings[sel_id]
                            )
                            penalties.append(sim)
                    if penalties:
                        diversity_penalty = max(penalties)

                # MMR score
                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * diversity_penalty

                if mmr > best_score:
                    best_score = mmr
                    best_idx = i
                    best_diversity_penalty = best_diversity_penalty

            if best_idx >= 0:
                best_candidate = remaining.pop(best_idx)
                doc_id = best_candidate.get(id_key, "")

                selected.append(MMRResult(
                    doc_id=doc_id,
                    mmr_score=best_score,
                    relevance_score=relevance_scores.get(doc_id, 0.5),
                    diversity_penalty=best_diversity_penalty,
                    rank=rank + 1,
                    metadata=best_candidate,
                ))

        return selected


class ContextDiversifier:
    """Diversify context window to maximize information coverage."""

    def __init__(
        self,
        embedding_service: Any = None,
        lambda_param: float = 0.6,  # Slightly more diversity for context
        settings: Settings | None = None,
    ):
        self.mmr = MMRReranker(
            embedding_service=embedding_service,
            lambda_param=lambda_param,
            settings=settings,
        )
        self.settings = settings or get_settings()

    async def diversify(
        self,
        chunks: list[dict[str, Any]],
        query: str,
        top_k: int = 10,
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """Diversify and select chunks for context.

        Args:
            chunks: Retrieved chunks
            query: User query
            top_k: Maximum chunks to select
            max_tokens: Optional token budget (approximate)

        Returns:
            Diversified list of chunks
        """
        # Apply MMR
        mmr_results = await self.mmr.select_diverse(
            candidates=chunks,
            query=query,
            top_k=top_k,
        )

        # Convert back to chunk format
        diversified = []
        total_tokens = 0

        for result in mmr_results:
            chunk = result.metadata.copy()
            chunk["mmr_score"] = result.mmr_score
            chunk["mmr_diversity_penalty"] = result.diversity_penalty
            chunk["mmr_rank"] = result.rank

            # Check token budget
            if max_tokens:
                # Rough estimate: 4 chars per token
                chunk_tokens = len(chunk.get("content", chunk.get("text", ""))) // 4
                if total_tokens + chunk_tokens > max_tokens:
                    break
                total_tokens += chunk_tokens

            diversified.append(chunk)

        logger.info(
            "Context diversified",
            original_count=len(chunks),
            final_count=len(diversified),
            total_tokens=total_tokens if max_tokens else "unlimited",
        )

        return diversified

    def compute_diversity_score(self, chunks: list[dict[str, Any]]) -> float:
        """Compute diversity score for a set of chunks.

        Higher score = more diverse.
        """
        if len(chunks) < 2:
            return 1.0

        # Compute pairwise text similarities
        similarities = []
        for i, chunk1 in enumerate(chunks):
            text1 = chunk1.get("content", chunk1.get("text", ""))
            for chunk2 in chunks[i+1:]:
                text2 = chunk2.get("content", chunk2.get("text", ""))
                sim = self.mmr._text_similarity(text1, text2)
                similarities.append(sim)

        if not similarities:
            return 1.0

        # Diversity = 1 - average similarity
        avg_similarity = sum(similarities) / len(similarities)
        return 1.0 - avg_similarity
