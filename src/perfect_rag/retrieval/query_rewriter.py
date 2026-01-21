"""Query rewriting and expansion for better retrieval."""

import re
from typing import Any

import structlog

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class QueryRewriter:
    """Rewrite and expand queries for better retrieval.

    Techniques:
    - Query expansion with synonyms
    - HyDE (Hypothetical Document Embeddings)
    - Multi-query generation
    - Query decomposition for complex questions
    """

    def __init__(
        self,
        llm_gateway: Any = None,
        settings: Settings | None = None,
    ):
        self.llm = llm_gateway
        self.settings = settings or get_settings()

    async def rewrite(
        self,
        query: str,
        context: list[dict[str, str]] | None = None,
        strategy: str = "auto",
    ) -> dict[str, Any]:
        """Rewrite query for better retrieval.

        Args:
            query: Original user query
            context: Optional conversation context
            strategy: Rewriting strategy ("auto", "expand", "hyde", "decompose", "multi")

        Returns:
            Dict with rewritten queries and metadata
        """
        if strategy == "auto":
            strategy = self._detect_strategy(query)

        result = {
            "original": query,
            "strategy": strategy,
            "queries": [query],  # Always include original
            "hyde_doc": None,
            "decomposed": None,
        }

        if not self.llm:
            return result

        try:
            if strategy == "expand":
                expanded = await self._expand_query(query)
                result["queries"].extend(expanded)

            elif strategy == "hyde":
                hyde_doc = await self._generate_hyde(query)
                result["hyde_doc"] = hyde_doc
                result["queries"].append(hyde_doc)

            elif strategy == "decompose":
                sub_queries = await self._decompose_query(query)
                result["decomposed"] = sub_queries
                result["queries"].extend(sub_queries)

            elif strategy == "multi":
                multi_queries = await self._generate_multi_queries(query)
                result["queries"].extend(multi_queries)

        except Exception as e:
            logger.warning("Query rewriting failed", error=str(e), strategy=strategy)

        # Deduplicate
        result["queries"] = list(dict.fromkeys(result["queries"]))

        return result

    def _detect_strategy(self, query: str) -> str:
        """Auto-detect best rewriting strategy."""
        query_lower = query.lower()

        # Complex questions with multiple parts
        if " and " in query_lower or "," in query and len(query) > 100:
            return "decompose"

        # Questions about specific facts
        if any(w in query_lower for w in ["what is", "who is", "define", "explain"]):
            return "hyde"

        # Comparison or multi-aspect questions
        if any(w in query_lower for w in ["compare", "difference", "versus", "vs"]):
            return "multi"

        # Default to expansion
        return "expand"

    async def _expand_query(self, query: str) -> list[str]:
        """Expand query with synonyms and related terms."""
        prompt = f"""Given the following search query, generate 2-3 alternative phrasings that capture the same intent but use different words or focus on different aspects.

Query: {query}

Respond with one alternative per line, without numbering or bullet points."""

        response = await self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )

        # Parse response
        alternatives = [
            line.strip()
            for line in response.strip().split("\n")
            if line.strip() and len(line.strip()) > 5
        ]

        return alternatives[:3]

    async def _generate_hyde(self, query: str) -> str:
        """Generate hypothetical document for HyDE retrieval.

        HyDE (Hypothetical Document Embeddings) generates a hypothetical
        answer/document, then uses its embedding for retrieval. This can
        improve retrieval for questions where the answer format is known.
        """
        prompt = f"""Write a short, informative passage (2-3 sentences) that would directly answer this question. Write as if you are creating an encyclopedia entry.

Question: {query}

Passage:"""

        response = await self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150,
        )

        return response.strip()

    async def _decompose_query(self, query: str) -> list[str]:
        """Decompose complex query into simpler sub-queries."""
        prompt = f"""Break down the following complex question into 2-4 simpler, self-contained questions that together address the original question.

Complex question: {query}

Respond with one question per line, without numbering."""

        response = await self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )

        sub_queries = [
            line.strip()
            for line in response.strip().split("\n")
            if line.strip() and line.strip().endswith("?")
        ]

        return sub_queries[:4]

    async def _generate_multi_queries(self, query: str) -> list[str]:
        """Generate multiple diverse queries for the same information need."""
        prompt = f"""Generate 3 different search queries that would help find information to answer this question. Each query should approach the topic from a different angle.

Question: {query}

Respond with one query per line."""

        response = await self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=200,
        )

        queries = [
            line.strip()
            for line in response.strip().split("\n")
            if line.strip() and len(line.strip()) > 5
        ]

        return queries[:3]

    async def contextualize(
        self,
        query: str,
        conversation_history: list[dict[str, str]],
    ) -> str:
        """Contextualize query using conversation history.

        Resolves pronouns and references based on previous messages.
        """
        if not conversation_history:
            return query

        # Build context summary
        recent_context = conversation_history[-4:]  # Last 2 exchanges
        context_text = "\n".join(
            f"{msg['role']}: {msg['content'][:200]}"
            for msg in recent_context
        )

        prompt = f"""Given the conversation context, rewrite the latest question to be self-contained. Replace any pronouns or references with their actual referents.

Conversation:
{context_text}

Latest question: {query}

Rewritten question (self-contained):"""

        response = await self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100,
        )

        return response.strip()


class ContextAwarenessGate:
    """Determine if retrieval is needed for a query.

    Some queries can be answered directly without retrieval:
    - Greetings
    - Simple calculations
    - General knowledge within LLM training data
    - Follow-up clarifications
    """

    def __init__(self, llm_gateway: Any = None):
        self.llm = llm_gateway

        # Patterns that don't need retrieval
        self.no_retrieval_patterns = [
            r"^(hi|hello|hey|good morning|good evening)",
            r"^(thanks|thank you|ok|okay|got it)",
            r"^what (time|day|date) is it",
            r"^(calculate|compute|what is) \d+",
            r"^(who are you|what are you|what can you do)",
        ]

    async def needs_retrieval(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> bool:
        """Determine if query needs retrieval.

        Args:
            query: User query
            conversation_history: Optional conversation context

        Returns:
            True if retrieval is needed, False otherwise
        """
        query_lower = query.lower().strip()

        # Check patterns that don't need retrieval
        for pattern in self.no_retrieval_patterns:
            if re.match(pattern, query_lower, re.IGNORECASE):
                return False

        # Very short queries often don't need retrieval
        if len(query_lower) < 10:
            return False

        # Use LLM for more nuanced decision
        if self.llm:
            return await self._llm_decision(query, conversation_history)

        # Default: assume retrieval is needed
        return True

    async def _llm_decision(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None,
    ) -> bool:
        """Use LLM to decide if retrieval is needed."""
        prompt = f"""Determine if the following query requires searching a specialized knowledge base (medical, technical, domain-specific documents) to answer accurately, or if it can be answered directly from general conversational knowledge.

IMPORTANT: If the query involves ANY of the following, respond RETRIEVAL:
- Medical terminology, diseases, treatments, classifications, or protocols
- Technical specifications, standards, or guidelines
- Domain-specific terms, acronyms, or jargon
- Questions that require precise, up-to-date, or authoritative information
- Questions about specific products, organizations, or proprietary information

Only respond DIRECT for:
- Greetings and casual conversation
- Simple math calculations
- Very basic general knowledge (like "what color is the sky")
- Clarification questions about the conversation itself

Query: {query}

Respond with only "RETRIEVAL" or "DIRECT"."""

        try:
            response = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10,
            )

            # Default to retrieval unless explicitly DIRECT
            # This is more conservative and prevents false negatives
            return "DIRECT" not in response.upper()

        except Exception:
            # Default to retrieval on error
            return True
