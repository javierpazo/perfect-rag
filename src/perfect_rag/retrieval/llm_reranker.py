"""
LLM-based reranking for improved retrieval accuracy.

Implements multiple strategies:
1. Pointwise: Score each document independently
2. Listwise: Score all documents together
3. Pairwise: Compare documents pairwise (more accurate, slower)
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class RerankStrategy(Enum):
    """LLM reranking strategies."""

    POINTWISE = "pointwise"  # Score each doc independently
    LISTWISE = "listwise"  # Score all docs together
    PAIRWISE = "pairwise"  # Compare docs pairwise


@dataclass
class RerankResult:
    """Result of LLM reranking."""

    documents: list[dict[str, Any]]
    strategy: RerankStrategy
    reasoning: list[str] = field(default_factory=list)


class LLMReranker:
    """
    LLM-based document reranker.

    Uses large language models for semantic understanding
    of query-document relevance.

    Strategies:
    - POINTWISE: Fast, parallelizable. Scores each document independently.
      Good for large document sets where relative comparison isn't critical.

    - LISTWISE: Context-aware. Scores all documents together.
      Better for smaller sets where relative ranking matters.
      Limited by LLM context window.

    - PAIRWISE: Most accurate. Compares documents in pairs.
      Best accuracy but O(n^2) complexity. Use for small, important retrievals.
    """

    def __init__(
        self,
        llm_gateway: Any,
        strategy: RerankStrategy = RerankStrategy.POINTWISE,
        max_docs_listwise: int = 10,
        max_docs_pairwise: int = 10,
        settings: Settings | None = None,
    ):
        """
        Initialize LLM reranker.

        Args:
            llm_gateway: LLM gateway for generation
            strategy: Default reranking strategy
            max_docs_listwise: Maximum documents for listwise ranking
            max_docs_pairwise: Maximum documents for pairwise comparison
            settings: Application settings
        """
        self.llm = llm_gateway
        self.strategy = strategy
        self.max_docs_listwise = max_docs_listwise
        self.max_docs_pairwise = max_docs_pairwise
        self.settings = settings or get_settings()

    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 10,
        strategy: RerankStrategy | None = None,
    ) -> RerankResult:
        """
        Rerank documents using LLM.

        Args:
            query: Search query
            documents: List of documents with 'content' field
            top_k: Number of top results to return
            strategy: Override default strategy

        Returns:
            Reranked documents with LLM scores
        """
        strategy = strategy or self.strategy

        if not documents:
            return RerankResult(documents=[], strategy=strategy)

        logger.info(
            "LLM reranking started",
            query_length=len(query),
            doc_count=len(documents),
            strategy=strategy.value,
            top_k=top_k,
        )

        if strategy == RerankStrategy.POINTWISE:
            result = await self._rerank_pointwise(query, documents, top_k)
        elif strategy == RerankStrategy.LISTWISE:
            result = await self._rerank_listwise(query, documents, top_k)
        elif strategy == RerankStrategy.PAIRWISE:
            result = await self._rerank_pairwise(query, documents, top_k)
        else:
            result = await self._rerank_pointwise(query, documents, top_k)

        logger.info(
            "LLM reranking complete",
            strategy=strategy.value,
            output_count=len(result.documents),
        )

        return result

    async def _rerank_pointwise(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int,
    ) -> RerankResult:
        """
        Score each document independently.

        Fast, parallelizable, but doesn't compare documents.
        """

        async def score_doc(doc: dict[str, Any], idx: int) -> tuple[int, float, str]:
            # Extract content - support both 'content' and 'text' keys
            content = doc.get("content") or doc.get("text", "")
            if not isinstance(content, str):
                content = str(content) if content else ""
            content = content[:2000]  # Limit content length

            prompt = f"""Rate the relevance of this document for answering the query.

Query: {query}

Document:
{content}

Score from 0-10 where:
- 10: Perfectly answers the query
- 7-9: Highly relevant, contains key information
- 4-6: Somewhat relevant
- 1-3: Marginally relevant
- 0: Not relevant

Respond with ONLY a number from 0-10."""

            try:
                response = await self.llm.generate(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0,
                )

                # Handle both string and object responses
                response_text = (
                    response if isinstance(response, str) else getattr(response, "content", str(response))
                )
                score = float(response_text.strip()) / 10.0
                score = max(0.0, min(1.0, score))
            except Exception as e:
                logger.warning(
                    "Pointwise scoring failed for document",
                    doc_idx=idx,
                    error=str(e),
                )
                score = 0.5  # Default to middle score on failure

            return (idx, score, "")

        # Score all documents in parallel
        tasks = [score_doc(doc, i) for i, doc in enumerate(documents)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        scored_docs = []
        for result in results:
            if isinstance(result, Exception):
                continue
            idx, score, reasoning = result
            doc = documents[idx].copy()
            doc["llm_rerank_score"] = score
            scored_docs.append(doc)

        # Sort by score
        scored_docs.sort(key=lambda x: x.get("llm_rerank_score", 0), reverse=True)

        return RerankResult(
            documents=scored_docs[:top_k],
            strategy=RerankStrategy.POINTWISE,
        )

    async def _rerank_listwise(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int,
    ) -> RerankResult:
        """
        Score all documents together for relative ranking.

        More context-aware but limited by context window.
        """
        # Limit documents for context window
        docs_to_rank = documents[: self.max_docs_listwise]

        # Format documents
        doc_list = "\n\n".join(
            [
                f"[{i + 1}] {(doc.get('content') or doc.get('text', ''))[:500]}..."
                for i, doc in enumerate(docs_to_rank)
            ]
        )

        prompt = f"""Rank these documents by relevance to the query.

Query: {query}

Documents:
{doc_list}

Return the document numbers in order from most to least relevant.
Format: comma-separated numbers (e.g., "3,1,5,2,4")

Ranking:"""

        try:
            response = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0,
            )

            # Handle both string and object responses
            response_text = (
                response if isinstance(response, str) else getattr(response, "content", str(response))
            )

            # Parse ranking
            ranking_str = response_text.strip()
            ranking = []
            for num in ranking_str.replace(" ", "").split(","):
                try:
                    idx = int(num) - 1  # Convert to 0-indexed
                    if 0 <= idx < len(docs_to_rank) and idx not in ranking:
                        ranking.append(idx)
                except ValueError:
                    continue

            # Add any missing documents at the end
            for i in range(len(docs_to_rank)):
                if i not in ranking:
                    ranking.append(i)

        except Exception as e:
            logger.warning("Listwise ranking failed", error=str(e))
            # Fallback to original order
            ranking = list(range(len(docs_to_rank)))

        # Reorder documents
        scored_docs = []
        for rank, idx in enumerate(ranking):
            doc = docs_to_rank[idx].copy()
            doc["llm_rerank_score"] = 1.0 - (rank / len(ranking))
            doc["llm_rank"] = rank + 1
            scored_docs.append(doc)

        return RerankResult(
            documents=scored_docs[:top_k],
            strategy=RerankStrategy.LISTWISE,
        )

    async def _rerank_pairwise(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int,
    ) -> RerankResult:
        """
        Compare documents pairwise for most accurate ranking.

        Most accurate but O(n^2) complexity.
        """
        n = min(len(documents), self.max_docs_pairwise)
        docs_to_rank = documents[:n]

        # Initialize win counts
        wins = [0] * n

        async def compare(i: int, j: int) -> int:
            """Compare two documents, return winner index."""
            doc_i_content = docs_to_rank[i].get("content") or docs_to_rank[i].get("text", "")
            doc_j_content = docs_to_rank[j].get("content") or docs_to_rank[j].get("text", "")

            if not isinstance(doc_i_content, str):
                doc_i_content = str(doc_i_content) if doc_i_content else ""
            if not isinstance(doc_j_content, str):
                doc_j_content = str(doc_j_content) if doc_j_content else ""

            doc_i_content = doc_i_content[:1000]
            doc_j_content = doc_j_content[:1000]

            prompt = f"""Which document is more relevant for answering the query?

Query: {query}

Document A:
{doc_i_content}

Document B:
{doc_j_content}

Respond with ONLY "A" or "B"."""

            try:
                response = await self.llm.generate(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0,
                )

                # Handle both string and object responses
                response_text = (
                    response if isinstance(response, str) else getattr(response, "content", str(response))
                )
                return i if "A" in response_text.upper() else j
            except Exception as e:
                logger.warning(
                    "Pairwise comparison failed",
                    doc_i=i,
                    doc_j=j,
                    error=str(e),
                )
                return i  # Default to first document

        # Compare all pairs
        tasks = []
        for i in range(n):
            for j in range(i + 1, n):
                tasks.append(compare(i, j))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count wins
        for result in results:
            if isinstance(result, Exception):
                continue
            wins[result] += 1

        # Sort by wins
        scored_docs = []
        for idx, win_count in enumerate(wins):
            doc = docs_to_rank[idx].copy()
            doc["llm_rerank_score"] = win_count / (n - 1) if n > 1 else 1.0
            doc["pairwise_wins"] = win_count
            scored_docs.append(doc)

        scored_docs.sort(key=lambda x: x.get("llm_rerank_score", 0), reverse=True)

        return RerankResult(
            documents=scored_docs[:top_k],
            strategy=RerankStrategy.PAIRWISE,
        )


class RankGPTReranker:
    """
    RankGPT-style permutation-based reranking.

    Uses sliding window approach for efficient listwise ranking.
    This approach balances accuracy and efficiency by:
    1. Ranking small windows of documents
    2. Sliding the window through the document list
    3. Making multiple passes for stability

    Reference: "Is ChatGPT Good at Search? Investigating Large Language Models as
    Re-Ranking Agent" (Sun et al., 2023)
    """

    def __init__(
        self,
        llm_gateway: Any,
        window_size: int = 5,
        step_size: int = 2,
        num_passes: int = 2,
        settings: Settings | None = None,
    ):
        """
        Initialize RankGPT reranker.

        Args:
            llm_gateway: LLM gateway for generation
            window_size: Size of the ranking window
            step_size: How much to slide the window each step
            num_passes: Number of passes through the document list
            settings: Application settings
        """
        self.llm = llm_gateway
        self.window_size = window_size
        self.step_size = step_size
        self.num_passes = num_passes
        self.settings = settings or get_settings()

    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Rerank using sliding window permutation approach.

        Args:
            query: Search query
            documents: List of documents with 'content' field
            top_k: Number of top results to return

        Returns:
            Reranked documents with scores
        """
        if len(documents) <= 1:
            return documents

        logger.info(
            "RankGPT reranking started",
            query_length=len(query),
            doc_count=len(documents),
            window_size=self.window_size,
            num_passes=self.num_passes,
        )

        # Work with a copy
        docs = [d.copy() for d in documents]

        # Multiple passes with sliding window
        for pass_num in range(self.num_passes):
            start = 0
            while start < len(docs) - 1:
                end = min(start + self.window_size, len(docs))
                window = docs[start:end]

                # Rank window
                ranked_window = await self._rank_window(query, window)
                docs[start:end] = ranked_window

                start += self.step_size

            logger.debug(
                "RankGPT pass complete",
                pass_num=pass_num + 1,
                total_passes=self.num_passes,
            )

        # Assign final scores based on position
        for i, doc in enumerate(docs):
            doc["llm_rerank_score"] = 1.0 - (i / len(docs))
            doc["rankgpt_rank"] = i + 1

        logger.info(
            "RankGPT reranking complete",
            output_count=min(top_k, len(docs)),
        )

        return docs[:top_k]

    async def _rank_window(
        self,
        query: str,
        window: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rank a small window of documents."""
        if len(window) <= 1:
            return window

        doc_list = "\n\n".join(
            [
                f"[{i + 1}] {(doc.get('content') or doc.get('text', ''))[:400]}"
                for i, doc in enumerate(window)
            ]
        )

        prompt = f"""Rank these passages by relevance to the query.

Query: {query}

Passages:
{doc_list}

Output the ranking as comma-separated numbers (e.g., "2,1,3"):"""

        try:
            response = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30,
                temperature=0,
            )

            # Handle both string and object responses
            response_text = (
                response if isinstance(response, str) else getattr(response, "content", str(response))
            )

            ranking = []
            for num in response_text.strip().split(","):
                try:
                    idx = int(num.strip()) - 1
                    if 0 <= idx < len(window) and idx not in ranking:
                        ranking.append(idx)
                except ValueError:
                    continue

            # Add missing indices
            for i in range(len(window)):
                if i not in ranking:
                    ranking.append(i)

            return [window[i] for i in ranking]

        except Exception as e:
            logger.warning("RankGPT window ranking failed", error=str(e))
            return window


class HybridLLMReranker:
    """
    Hybrid reranker combining fast cross-encoder with LLM refinement.

    Uses a two-stage approach:
    1. Fast cross-encoder reranking on all documents
    2. LLM-based refinement on top candidates

    This provides the efficiency of cross-encoders with the
    accuracy of LLM-based reranking for top results.
    """

    def __init__(
        self,
        llm_gateway: Any,
        embedding_service: Any = None,
        llm_top_k: int = 10,
        llm_strategy: RerankStrategy = RerankStrategy.LISTWISE,
        settings: Settings | None = None,
    ):
        """
        Initialize hybrid reranker.

        Args:
            llm_gateway: LLM gateway for generation
            embedding_service: Embedding service with rerank capability
            llm_top_k: Number of documents to refine with LLM
            llm_strategy: Strategy for LLM reranking
            settings: Application settings
        """
        self.llm = llm_gateway
        self.embedding = embedding_service
        self.llm_top_k = llm_top_k
        self.llm_strategy = llm_strategy
        self.settings = settings or get_settings()
        self._llm_reranker = LLMReranker(
            llm_gateway=llm_gateway,
            strategy=llm_strategy,
            settings=settings,
        )

    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Hybrid rerank using cross-encoder + LLM.

        Args:
            query: Search query
            documents: List of documents
            top_k: Number of results to return

        Returns:
            Reranked documents
        """
        if len(documents) <= 1:
            return documents

        logger.info(
            "Hybrid reranking started",
            doc_count=len(documents),
            llm_top_k=self.llm_top_k,
        )

        # Stage 1: Cross-encoder reranking (if available)
        if self.embedding is not None:
            texts = [
                doc.get("content") or doc.get("text", "") for doc in documents
            ]

            try:
                ranked = await self.embedding.rerank(query, texts, top_k=len(documents))

                # Reorder documents
                reranked_docs = []
                for original_idx, score in ranked:
                    doc = documents[original_idx].copy()
                    doc["cross_encoder_score"] = score
                    reranked_docs.append(doc)
            except Exception as e:
                logger.warning("Cross-encoder reranking failed", error=str(e))
                reranked_docs = documents
        else:
            reranked_docs = documents

        # Stage 2: LLM refinement on top candidates
        candidates = reranked_docs[: self.llm_top_k]
        result = await self._llm_reranker.rerank(
            query=query,
            documents=candidates,
            top_k=top_k,
            strategy=self.llm_strategy,
        )

        # Merge refined results with remaining documents
        remaining = reranked_docs[self.llm_top_k :]
        final_docs = result.documents + remaining

        logger.info(
            "Hybrid reranking complete",
            refined_count=len(result.documents),
            total_count=len(final_docs),
        )

        return final_docs[:top_k]


# =============================================================================
# Factory Functions
# =============================================================================

_llm_reranker: LLMReranker | None = None
_rankgpt_reranker: RankGPTReranker | None = None


async def get_llm_reranker(
    llm_gateway: Any,
    strategy: RerankStrategy = RerankStrategy.POINTWISE,
    settings: Settings | None = None,
) -> LLMReranker:
    """Get or create LLM reranker instance."""
    global _llm_reranker
    if _llm_reranker is None:
        _llm_reranker = LLMReranker(
            llm_gateway=llm_gateway,
            strategy=strategy,
            settings=settings,
        )
    return _llm_reranker


async def get_rankgpt_reranker(
    llm_gateway: Any,
    settings: Settings | None = None,
) -> RankGPTReranker:
    """Get or create RankGPT reranker instance."""
    global _rankgpt_reranker
    if _rankgpt_reranker is None:
        _rankgpt_reranker = RankGPTReranker(
            llm_gateway=llm_gateway,
            settings=settings,
        )
    return _rankgpt_reranker
