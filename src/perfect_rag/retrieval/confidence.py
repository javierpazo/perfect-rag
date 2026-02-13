"""Confidence estimation and fallback search strategies.

This module provides:
1. Confidence scoring for retrieval results
2. Automatic fallback when confidence is low
3. Multiple fallback strategies (expand search, relax filters, etc.)

Confidence is estimated based on:
- Top result reranker score
- Score distribution across results
- Query coverage in retrieved content
- Evidence extraction success rate
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import structlog

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels for retrieval results."""
    HIGH = "high"        # > 0.85 - confident in answer
    MEDIUM = "medium"    # 0.6 - 0.85 - reasonable but verify
    LOW = "low"          # 0.4 - 0.6 - uncertain, needs verification
    VERY_LOW = "very_low"  # < 0.4 - likely no good answer


class FallbackStrategy(Enum):
    """Fallback strategies when confidence is low."""
    EXPAND_TOP_K = "expand_top_k"         # Retrieve more candidates
    RELAX_FILTERS = "relax_filters"       # Remove metadata filters
    EXPAND_QUERY = "expand_query"         # Generate query variations
    ALTERNATIVE_INDEX = "alternative_index"  # Use different search method
    LOWER_THRESHOLD = "lower_threshold"   # Accept lower relevance threshold


@dataclass
class ConfidenceScore:
    """Detailed confidence scoring."""
    overall: float
    level: ConfidenceLevel
    factors: dict[str, float]
    needs_fallback: bool
    recommended_strategy: FallbackStrategy | None
    explanation: str


@dataclass
class FallbackResult:
    """Result from fallback search."""
    chunks: list[dict[str, Any]]
    confidence: ConfidenceScore
    strategies_tried: list[FallbackStrategy]
    improvement: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ConfidenceEstimator:
    """Estimate confidence in retrieval results."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

        # Thresholds
        self.high_threshold = 0.85
        self.medium_threshold = 0.6
        self.low_threshold = 0.4

    def estimate(
        self,
        chunks: list[dict[str, Any]],
        query: str,
        retrieval_metadata: dict[str, Any] | None = None,
    ) -> ConfidenceScore:
        """Estimate confidence in retrieval results.

        Args:
            chunks: Retrieved chunks with scores
            query: Original query
            retrieval_metadata: Metadata from retrieval (strategy, etc.)

        Returns:
            ConfidenceScore with detailed breakdown
        """
        if not chunks:
            return ConfidenceScore(
                overall=0.0,
                level=ConfidenceLevel.VERY_LOW,
                factors={},
                needs_fallback=True,
                recommended_strategy=FallbackStrategy.EXPAND_QUERY,
                explanation="No results retrieved",
            )

        factors = {}

        # Factor 1: Top result score (most important)
        top_score = chunks[0].get("rerank_score") or chunks[0].get("score", 0)
        factors["top_result_score"] = min(1.0, top_score)

        # Factor 2: Score distribution (gap between top and rest)
        if len(chunks) > 1:
            scores = [
                c.get("rerank_score") or c.get("score", 0)
                for c in chunks[:5]
            ]
            if scores[0] > 0:
                gap = (scores[0] - scores[-1]) / scores[0]
                factors["score_gap"] = min(1.0, gap)
            else:
                factors["score_gap"] = 0.0
        else:
            factors["score_gap"] = 0.5

        # Factor 3: Result quantity
        quantity_score = min(1.0, len(chunks) / 5)
        factors["result_quantity"] = quantity_score

        # Factor 4: Score consistency (are top results close?)
        if len(chunks) >= 3:
            top_3_scores = [
                c.get("rerank_score") or c.get("score", 0)
                for c in chunks[:3]
            ]
            avg_top_3 = sum(top_3_scores) / len(top_3_scores)
            factors["score_consistency"] = min(1.0, avg_top_3)
        else:
            factors["score_consistency"] = factors["top_result_score"]

        # Factor 5: Query term coverage in results
        query_terms = set(query.lower().split())
        if query_terms:
            covered = set()
            for chunk in chunks[:3]:
                text = (chunk.get("content") or chunk.get("text", "")).lower()
                covered.update(term for term in query_terms if term in text)
            coverage = len(covered) / len(query_terms)
            factors["query_coverage"] = coverage
        else:
            factors["query_coverage"] = 0.5

        # Weight factors
        weights = {
            "top_result_score": 0.35,
            "score_gap": 0.15,
            "result_quantity": 0.15,
            "score_consistency": 0.20,
            "query_coverage": 0.15,
        }

        overall = sum(
            factors.get(k, 0) * v
            for k, v in weights.items()
        )

        # Determine level
        if overall >= self.high_threshold:
            level = ConfidenceLevel.HIGH
        elif overall >= self.medium_threshold:
            level = ConfidenceLevel.MEDIUM
        elif overall >= self.low_threshold:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.VERY_LOW

        # Determine if fallback needed
        needs_fallback = level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]

        # Recommend strategy
        strategy = self._recommend_strategy(factors, level)

        # Generate explanation
        explanation = self._explain_confidence(factors, level)

        return ConfidenceScore(
            overall=overall,
            level=level,
            factors=factors,
            needs_fallback=needs_fallback,
            recommended_strategy=strategy,
            explanation=explanation,
        )

    def _recommend_strategy(
        self,
        factors: dict[str, float],
        level: ConfidenceLevel,
    ) -> FallbackStrategy | None:
        """Recommend fallback strategy based on weak factors."""
        if level == ConfidenceLevel.HIGH:
            return None

        # Find weakest factor
        min_factor = min(factors, key=factors.get)
        min_value = factors[min_factor]

        if min_factor == "top_result_score" and min_value < 0.5:
            return FallbackStrategy.EXPAND_QUERY
        elif min_factor == "result_quantity":
            return FallbackStrategy.EXPAND_TOP_K
        elif min_factor == "query_coverage":
            return FallbackStrategy.EXPAND_QUERY
        elif min_factor == "score_consistency":
            return FallbackStrategy.LOWER_THRESHOLD
        else:
            return FallbackStrategy.EXPAND_TOP_K

    def _explain_confidence(
        self,
        factors: dict[str, float],
        level: ConfidenceLevel,
    ) -> str:
        """Generate human-readable explanation."""
        weak_factors = [k for k, v in factors.items() if v < 0.5]

        level_descriptions = {
            ConfidenceLevel.HIGH: "High confidence in retrieval results.",
            ConfidenceLevel.MEDIUM: "Moderate confidence - results are reasonable but should be verified.",
            ConfidenceLevel.LOW: "Low confidence - results may not fully address the query.",
            ConfidenceLevel.VERY_LOW: "Very low confidence - consider rephrasing the query.",
        }

        explanation = level_descriptions[level]

        if weak_factors:
            weak_str = ", ".join(weak_factors)
            explanation += f" Weak factors: {weak_str}."

        return explanation


class FallbackSearchExecutor:
    """Execute fallback search strategies."""

    def __init__(
        self,
        search_func: Callable,
        settings: Settings | None = None,
    ):
        """Initialize fallback executor.

        Args:
            search_func: Async function to execute search
                        Signature: (query, top_k, filters, ...) -> list[dict]
        """
        self.search_func = search_func
        self.settings = settings or get_settings()
        self.confidence_estimator = ConfidenceEstimator(settings)
        self.max_fallback_attempts = 2

    async def search_with_fallback(
        self,
        query: str,
        initial_chunks: list[dict[str, Any]],
        initial_top_k: int = 10,
        filters: dict[str, Any] | None = None,
        min_confidence: float = 0.5,
    ) -> FallbackResult:
        """Execute search with automatic fallback.

        Args:
            query: User query
            initial_chunks: Initial retrieval results
            initial_top_k: Initial top_k value
            filters: Optional metadata filters
            min_confidence: Minimum acceptable confidence

        Returns:
            FallbackResult with final chunks and confidence
        """
        strategies_tried = []
        current_chunks = initial_chunks
        current_confidence = self.confidence_estimator.estimate(
            current_chunks, query
        )
        initial_confidence = current_confidence.overall

        # Check if fallback needed
        if current_confidence.overall >= min_confidence:
            return FallbackResult(
                chunks=current_chunks,
                confidence=current_confidence,
                strategies_tried=[],
                improvement=0.0,
                metadata={"fallback_triggered": False},
            )

        # Execute fallback strategies
        attempt = 0
        while (
            current_confidence.overall < min_confidence
            and attempt < self.max_fallback_attempts
        ):
            attempt += 1
            strategy = current_confidence.recommended_strategy

            if strategy is None:
                break

            strategies_tried.append(strategy)

            logger.info(
                "Executing fallback strategy",
                strategy=strategy.value,
                attempt=attempt,
                current_confidence=current_confidence.overall,
            )

            try:
                new_chunks = await self._execute_strategy(
                    strategy=strategy,
                    query=query,
                    top_k=initial_top_k,
                    filters=filters,
                    attempt=attempt,
                )

                if new_chunks:
                    new_confidence = self.confidence_estimator.estimate(
                        new_chunks, query
                    )

                    # Accept if improved
                    if new_confidence.overall > current_confidence.overall:
                        current_chunks = new_chunks
                        current_confidence = new_confidence

            except Exception as e:
                logger.warning(
                    "Fallback strategy failed",
                    strategy=strategy.value,
                    error=str(e),
                )

        improvement = current_confidence.overall - initial_confidence

        return FallbackResult(
            chunks=current_chunks,
            confidence=current_confidence,
            strategies_tried=strategies_tried,
            improvement=improvement,
            metadata={
                "fallback_triggered": True,
                "attempts": attempt,
                "initial_confidence": initial_confidence,
            },
        )

    async def _execute_strategy(
        self,
        strategy: FallbackStrategy,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
        attempt: int,
    ) -> list[dict[str, Any]]:
        """Execute a specific fallback strategy."""
        if strategy == FallbackStrategy.EXPAND_TOP_K:
            # Retrieve more candidates
            new_top_k = top_k * (2 ** attempt)
            return await self.search_func(
                query=query,
                top_k=new_top_k,
                filters=None,  # Remove filters for broader search
            )

        elif strategy == FallbackStrategy.RELAX_FILTERS:
            # Remove all filters
            return await self.search_func(
                query=query,
                top_k=top_k * 2,
                filters=None,
            )

        elif strategy == FallbackStrategy.EXPAND_QUERY:
            # Use query expansion (simplified - in practice would use LLM)
            expanded_query = f"{query} {query.split()[0]}"  # Simple expansion
            return await self.search_func(
                query=expanded_query,
                top_k=top_k * 2,
                filters=filters,
            )

        elif strategy == FallbackStrategy.LOWER_THRESHOLD:
            # Retrieve more and accept lower scores
            return await self.search_func(
                query=query,
                top_k=top_k * 3,
                filters=None,
            )

        elif strategy == FallbackStrategy.ALTERNATIVE_INDEX:
            # Try sparse-only search (would need different search function)
            return await self.search_func(
                query=query,
                top_k=top_k * 2,
                filters=None,
            )

        return []


class ConfidenceAwareRetriever:
    """Retriever with built-in confidence estimation and fallback."""

    def __init__(
        self,
        search_func: Callable,
        llm_gateway: Any = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.search_func = search_func
        self.confidence_estimator = ConfidenceEstimator(settings)
        self.fallback_executor = FallbackSearchExecutor(search_func, settings)
        self.llm = llm_gateway

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        enable_fallback: bool = True,
        min_confidence: float = 0.5,
    ) -> tuple[list[dict[str, Any]], ConfidenceScore]:
        """Retrieve with confidence estimation and optional fallback.

        Args:
            query: User query
            top_k: Number of results
            filters: Optional metadata filters
            enable_fallback: Whether to use fallback on low confidence
            min_confidence: Minimum confidence threshold

        Returns:
            Tuple of (chunks, confidence_score)
        """
        # Initial search
        initial_chunks = await self.search_func(
            query=query,
            top_k=top_k,
            filters=filters,
        )

        if not enable_fallback:
            confidence = self.confidence_estimator.estimate(
                initial_chunks, query
            )
            return initial_chunks, confidence

        # Execute with fallback
        result = await self.fallback_executor.search_with_fallback(
            query=query,
            initial_chunks=initial_chunks,
            initial_top_k=top_k,
            filters=filters,
            min_confidence=min_confidence,
        )

        return result.chunks, result.confidence

    def get_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to confidence level."""
        if score >= 0.85:
            return ConfidenceLevel.HIGH
        elif score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
