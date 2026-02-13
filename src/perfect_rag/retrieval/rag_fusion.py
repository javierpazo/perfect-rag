"""RAG-Fusion: Multi-query retrieval with intelligent routing.

RAG-Fusion improves retrieval by:
1. Generating multiple query variations
2. Routing queries by intent (factual, exploratory, comparison)
3. Executing queries in parallel
4. Fusing results with Reciprocal Rank Fusion (RRF)

Based on the RAG-Fusion paper: https://arxiv.org/abs/2401.04280
"""

import asyncio
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class QueryIntent(Enum):
    """Intent classification for query routing."""
    FACTUAL = "factual"           # Looking for specific facts/answers
    EXPLORATORY = "exploratory"   # Broad topic exploration
    COMPARISON = "comparison"     # Comparing options/entities
    PROCEDURAL = "procedural"     # How-to or step-by-step
    DEFINITIONAL = "definitional" # Looking for definitions


@dataclass
class QueryVariant:
    """A generated query variant."""
    query: str
    intent: QueryIntent
    source: str  # "original", "expanded", "hyde", "decomposed"
    score_weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    """Result from RAG-Fusion retrieval."""
    doc_id: str
    fused_score: float
    source_queries: list[str]
    intent_scores: dict[QueryIntent, float]
    rrf_score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class IntentClassifier:
    """Classify query intent for routing."""

    def __init__(self, llm_gateway: Any = None):
        self.llm = llm_gateway

        # Pattern-based classification
        self.intent_patterns = {
            QueryIntent.FACTUAL: [
                r'\b(qué|cuál|cuándo|dónde|quién|cuánto|how many|what|which|when|where|who)\b',
                r'\b(cuál es|qué es|what is|which is)\b',
            ],
            QueryIntent.EXPLORATORY: [
                r'\b(tell me about|háblame de|cuéntame sobre|explain|describe|overview)\b',
                r'\b(todo sobre|everything about|comprehensive|complete)\b',
            ],
            QueryIntent.COMPARISON: [
                r'\b(vs|versus|comparar|difference|diferencia|better|mejor|versus)\b',
                r'\b(contrast|compare|comparison)\b',
            ],
            QueryIntent.PROCEDURAL: [
                r'\b(cómo|how to|how do|pasos|steps|procedure|proceso)\b',
                r'\b(guide|guía|tutorial|instructions)\b',
            ],
            QueryIntent.DEFINITIONAL: [
                r'\b(definición|definition|define|qué significa|meaning)\b',
                r'\b(qué es|what is a|what are)\b',
            ],
        }

    def classify(self, query: str) -> QueryIntent:
        """Classify query intent.

        Uses pattern matching first, then LLM if available.
        """
        query_lower = query.lower()

        # Pattern-based classification
        scores = {intent: 0 for intent in QueryIntent}
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    scores[intent] += 1

        # Return highest scoring intent
        max_intent = max(scores, key=scores.get)
        if scores[max_intent] > 0:
            return max_intent

        # Default to factual
        return QueryIntent.FACTUAL

    async def classify_with_llm(self, query: str) -> QueryIntent:
        """Use LLM for more accurate intent classification."""
        if not self.llm:
            return self.classify(query)

        prompt = f"""Classify the intent of this query into ONE category:

Query: "{query}"

Categories:
- FACTUAL: Looking for specific facts, numbers, dates, names
- EXPLORATORY: Broad exploration of a topic, wants comprehensive overview
- COMPARISON: Comparing two or more things, looking for differences
- PROCEDURAL: How-to, step-by-step instructions
- DEFINITIONAL: Looking for a definition or explanation of a concept

Respond with ONLY the category name (uppercase)."""

        try:
            response = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=20,
            )
            intent_str = response.strip().upper()
            return QueryIntent(intent_str.lower())
        except Exception:
            return self.classify(query)


class QueryExpander:
    """Generate query variations for RAG-Fusion."""

    def __init__(self, llm_gateway: Any = None, settings: Settings | None = None):
        self.llm = llm_gateway
        self.settings = settings or get_settings()
        self.intent_classifier = IntentClassifier(llm_gateway)

    async def expand(
        self,
        query: str,
        num_variations: int = 3,
        intent: QueryIntent | None = None,
    ) -> list[QueryVariant]:
        """Generate query variations.

        Args:
            query: Original query
            num_variations: Number of variations to generate
            intent: Optional pre-classified intent

        Returns:
            List of QueryVariant objects
        """
        if not self.llm:
            return [QueryVariant(
                query=query,
                intent=intent or QueryIntent.FACTUAL,
                source="original",
                score_weight=1.0,
            )]

        if intent is None:
            intent = await self.intent_classifier.classify_with_llm(query)

        variants = [
            QueryVariant(
                query=query,
                intent=intent,
                source="original",
                score_weight=1.0,
            )
        ]

        # Generate variations based on intent
        if intent == QueryIntent.FACTUAL:
            # For factual queries, generate rephrasings
            expanded = await self._expand_factual(query, num_variations)
        elif intent == QueryIntent.EXPLORATORY:
            # For exploratory, generate sub-topics
            expanded = await self._expand_exploratory(query, num_variations)
        elif intent == QueryIntent.COMPARISON:
            # For comparison, generate specific comparison aspects
            expanded = await self._expand_comparison(query, num_variations)
        elif intent == QueryIntent.PROCEDURAL:
            # For procedural, generate step-focused queries
            expanded = await self._expand_procedural(query, num_variations)
        else:
            # Default expansion
            expanded = await self._expand_generic(query, num_variations)

        variants.extend(expanded)
        return variants

    async def _expand_factual(self, query: str, n: int) -> list[QueryVariant]:
        """Expand factual queries with rephrasings."""
        prompt = f"""Generate {n} different ways to ask this question, using different words but the same meaning:

Question: {query}

Write one variation per line, numbered 1-{n}. Be concise."""

        response = await self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=200,
        )

        variations = []
        for line in response.strip().split('\n'):
            line = line.strip()
            # Remove numbering
            if line and line[0].isdigit():
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
            if line and len(line) > 10:
                variations.append(QueryVariant(
                    query=line,
                    intent=QueryIntent.FACTUAL,
                    source="expanded",
                    score_weight=0.8,
                ))

        return variations[:n]

    async def _expand_exploratory(self, query: str, n: int) -> list[QueryVariant]:
        """Expand exploratory queries with sub-topics."""
        prompt = f"""Generate {n} specific sub-topics or aspects related to this broad topic:

Topic: {query}

Write one sub-topic per line. Focus on key aspects that would give a comprehensive understanding."""

        response = await self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=200,
        )

        variations = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and len(line) > 10:
                variations.append(QueryVariant(
                    query=line,
                    intent=QueryIntent.EXPLORATORY,
                    source="expanded",
                    score_weight=0.7,
                ))

        return variations[:n]

    async def _expand_comparison(self, query: str, n: int) -> list[QueryVariant]:
        """Expand comparison queries with specific aspects."""
        prompt = f"""Generate {n} specific comparison aspects for this comparison query:

Query: {query}

Focus on specific attributes, features, or criteria that would help compare. Write one aspect per line."""

        response = await self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=200,
        )

        variations = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and len(line) > 10:
                variations.append(QueryVariant(
                    query=line,
                    intent=QueryIntent.COMPARISON,
                    source="expanded",
                    score_weight=0.8,
                ))

        return variations[:n]

    async def _expand_procedural(self, query: str, n: int) -> list[QueryVariant]:
        """Expand procedural queries with step-focused variations."""
        prompt = f"""Generate {n} step-focused variations of this procedural query:

Query: {query}

Focus on specific steps, requirements, or key points. Write one per line."""

        response = await self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=200,
        )

        variations = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and len(line) > 10:
                variations.append(QueryVariant(
                    query=line,
                    intent=QueryIntent.PROCEDURAL,
                    source="expanded",
                    score_weight=0.8,
                ))

        return variations[:n]

    async def _expand_generic(self, query: str, n: int) -> list[QueryVariant]:
        """Generic query expansion."""
        prompt = f"""Generate {n} alternative ways to search for information about:

Topic: {query}

Write one alternative per line. Use different words and angles."""

        response = await self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=200,
        )

        variations = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and len(line) > 10:
                variations.append(QueryVariant(
                    query=line,
                    intent=QueryIntent.FACTUAL,
                    source="expanded",
                    score_weight=0.7,
                ))

        return variations[:n]


class RAGFusion:
    """RAG-Fusion: Multi-query retrieval with RRF fusion."""

    def __init__(
        self,
        llm_gateway: Any = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.llm = llm_gateway
        self.query_expander = QueryExpander(llm_gateway, settings)
        self.intent_classifier = IntentClassifier(llm_gateway)
        self.rrf_k = 60  # RRF parameter

    async def generate_queries(
        self,
        query: str,
        num_variations: int = 3,
    ) -> list[QueryVariant]:
        """Generate query variations for RAG-Fusion.

        Args:
            query: Original query
            num_variations: Number of variations per intent

        Returns:
            List of QueryVariant objects
        """
        # Classify intent
        intent = await self.intent_classifier.classify_with_llm(query)

        # Expand based on intent
        variants = await self.query_expander.expand(query, num_variations, intent)

        logger.info(
            "Generated query variants",
            original=query[:50],
            num_variants=len(variants),
            intent=intent.value,
        )

        return variants

    def fuse_results(
        self,
        results_by_query: dict[str, list[dict[str, Any]]],
        query_variants: list[QueryVariant],
        top_k: int = 20,
    ) -> list[FusionResult]:
        """Fuse results from multiple queries using RRF.

        Args:
            results_by_query: Map of query -> list of {doc_id, score, ...}
            query_variants: List of query variants with weights
            top_k: Number of results to return

        Returns:
            List of FusionResult sorted by fused score
        """
        # Build weight map
        query_weights = {v.query: v.score_weight for v in query_variants}
        query_intents = {v.query: v.intent for v in query_variants}

        # RRF fusion
        rrf_scores: dict[str, float] = {}
        intent_scores: dict[str, dict[QueryIntent, float]] = defaultdict(lambda: defaultdict(float))
        source_queries: dict[str, list[str]] = defaultdict(list)
        doc_metadata: dict[str, dict] = {}
        doc_scores: dict[str, dict[str, float]] = defaultdict(lambda: {})

        for query, results in results_by_query.items():
            weight = query_weights.get(query, 1.0)
            intent = query_intents.get(query, QueryIntent.FACTUAL)

            for rank, result in enumerate(results):
                doc_id = result.get("doc_id") or result.get("id")
                if not doc_id:
                    continue

                # RRF score
                rrf_increment = weight / (self.rrf_k + rank + 1)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_increment

                # Track intent-specific scores
                intent_scores[doc_id][intent] += rrf_increment

                # Track source queries
                if query not in source_queries[doc_id]:
                    source_queries[doc_id].append(query)

                # Store metadata
                if doc_id not in doc_metadata:
                    doc_metadata[doc_id] = result.get("metadata", {})

                # Track individual scores
                if "dense_score" in result:
                    doc_scores[doc_id]["dense"] = max(
                        doc_scores[doc_id].get("dense", 0),
                        result["dense_score"]
                    )
                if "sparse_score" in result:
                    doc_scores[doc_id]["sparse"] = max(
                        doc_scores[doc_id].get("sparse", 0),
                        result["sparse_score"]
                    )

        # Sort and create results
        sorted_docs = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for doc_id in sorted_docs[:top_k]:
            # Normalize RRF score to 0-1 range (approximate)
            max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0
            normalized_score = rrf_scores[doc_id] / max_rrf if max_rrf > 0 else 0

            results.append(FusionResult(
                doc_id=doc_id,
                fused_score=normalized_score,
                source_queries=source_queries[doc_id],
                intent_scores=dict(intent_scores[doc_id]),
                rrf_score=rrf_scores[doc_id],
                dense_score=doc_scores[doc_id].get("dense", 0),
                sparse_score=doc_scores[doc_id].get("sparse", 0),
                metadata=doc_metadata[doc_id],
            ))

        return results

    async def retrieve_with_fusion(
        self,
        query: str,
        search_func,  # Callable: (query: str, top_k: int) -> list[dict]
        top_k: int = 20,
        num_variations: int = 3,
        parallel: bool = True,
    ) -> list[FusionResult]:
        """Execute RAG-Fusion retrieval.

        Args:
            query: Original query
            search_func: Async function to search with a query
            top_k: Number of final results
            num_variations: Number of query variations
            parallel: Whether to execute queries in parallel

        Returns:
            Fused results
        """
        # Generate query variants
        variants = await self.generate_queries(query, num_variations)

        # Execute searches
        results_by_query = {}

        if parallel:
            # Execute all queries in parallel
            tasks = []
            queries = [v.query for v in variants]
            for q in queries:
                tasks.append(search_func(q, top_k * 2))

            search_results = await asyncio.gather(*tasks, return_exceptions=True)

            for q, result in zip(queries, search_results):
                if isinstance(result, Exception):
                    logger.warning("Search failed for query", query=q[:50], error=str(result))
                    results_by_query[q] = []
                else:
                    results_by_query[q] = result
        else:
            # Sequential execution
            for variant in variants:
                try:
                    results = await search_func(variant.query, top_k * 2)
                    results_by_query[variant.query] = results
                except Exception as e:
                    logger.warning("Search failed", query=variant.query[:50], error=str(e))
                    results_by_query[variant.query] = []

        # Fuse results
        fused = self.fuse_results(results_by_query, variants, top_k)

        logger.info(
            "RAG-Fusion complete",
            original_query=query[:50],
            num_variants=len(variants),
            num_results=len(fused),
        )

        return fused


def get_intent_routing_config(intent: QueryIntent) -> dict[str, Any]:
    """Get search configuration based on query intent.

    Different intents may benefit from different search parameters.
    """
    configs = {
        QueryIntent.FACTUAL: {
            "top_k": 10,
            "use_reranking": True,
            "use_expansion": False,
            "min_score": 0.5,
        },
        QueryIntent.EXPLORATORY: {
            "top_k": 20,
            "use_reranking": True,
            "use_expansion": True,
            "min_score": 0.3,
        },
        QueryIntent.COMPARISON: {
            "top_k": 15,
            "use_reranking": True,
            "use_expansion": True,
            "min_score": 0.4,
        },
        QueryIntent.PROCEDURAL: {
            "top_k": 15,
            "use_reranking": True,
            "use_expansion": False,
            "min_score": 0.4,
        },
        QueryIntent.DEFINITIONAL: {
            "top_k": 5,
            "use_reranking": True,
            "use_expansion": False,
            "min_score": 0.5,
        },
    }
    return configs.get(intent, configs[QueryIntent.FACTUAL])
