"""
Agentic Retrieval System for Perfect RAG.

Implements:
- Iterative retrieval with self-evaluation
- Self-RAG reflection tokens
- Corrective RAG patterns
- Multi-step reasoning
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from perfect_rag.retrieval.pipeline import RetrievalPipeline
    from perfect_rag.llm.gateway import LLMGateway

logger = structlog.get_logger(__name__)


class RetrievalDecision(Enum):
    """Self-RAG reflection decisions."""
    RETRIEVE = "retrieve"  # Need to retrieve
    NO_RETRIEVE = "no_retrieve"  # Can answer without retrieval
    CONTINUE = "continue"  # Need more retrieval
    SUFFICIENT = "sufficient"  # Have enough context


class RelevanceGrade(Enum):
    """Relevance grades for retrieved documents."""
    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    NOT_RELEVANT = "not_relevant"


class SupportGrade(Enum):
    """Support grades for generated answers."""
    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NO_SUPPORT = "no_support"


class UtilityGrade(Enum):
    """Utility grades for answers."""
    USEFUL = "useful"
    PARTIALLY_USEFUL = "partially_useful"
    NOT_USEFUL = "not_useful"


@dataclass
class RetrievalStep:
    """Record of a single retrieval step."""
    query: str
    retrieved_chunks: list[dict[str, Any]]
    relevance_scores: list[float]
    decision: RetrievalDecision
    reasoning: str = ""
    iteration: int = 0


@dataclass
class AgenticRetrievalResult:
    """Result of agentic retrieval process."""
    final_chunks: list[dict[str, Any]]
    steps: list[RetrievalStep] = field(default_factory=list)
    total_iterations: int = 0
    final_decision: RetrievalDecision = RetrievalDecision.SUFFICIENT
    confidence: float = 0.0
    web_search_used: bool = False
    query_transformations: list[str] = field(default_factory=list)


@dataclass
class GradingResult:
    """Result of grading a chunk's relevance."""
    chunk_id: str
    relevance_score: float
    relevance_grade: RelevanceGrade
    reasoning: str = ""


class AgenticRetriever:
    """
    Agentic retrieval system with iterative refinement.

    Implements the CRAG (Corrective RAG) and Self-RAG patterns:
    1. Evaluate if retrieval is needed (Self-RAG: Retrieve token)
    2. Retrieve initial documents
    3. Grade relevance of documents (Self-RAG: ISREL token)
    4. If insufficient, refine query and re-retrieve (CRAG)
    5. Repeat until sufficient or max iterations
    6. Optionally use web search for knowledge augmentation (CRAG)
    """

    def __init__(
        self,
        retrieval_pipeline: "RetrievalPipeline",
        llm_gateway: "LLMGateway",
        max_iterations: int = 3,
        relevance_threshold: float = 0.7,
        min_relevant_chunks: int = 2,
        enable_web_search: bool = False,
    ):
        """
        Initialize the agentic retriever.

        Args:
            retrieval_pipeline: The existing retrieval pipeline
            llm_gateway: LLM gateway for reflection and grading
            max_iterations: Maximum retrieval iterations
            relevance_threshold: Threshold for considering a chunk relevant
            min_relevant_chunks: Minimum relevant chunks before declaring sufficient
            enable_web_search: Enable web search for knowledge augmentation
        """
        self.retrieval = retrieval_pipeline
        self.llm = llm_gateway
        self.max_iterations = max_iterations
        self.relevance_threshold = relevance_threshold
        self.min_relevant_chunks = min_relevant_chunks
        self.enable_web_search = enable_web_search

    async def retrieve(
        self,
        query: str,
        user_id: str | None = None,
        top_k: int = 10,
        acl_filter: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> AgenticRetrievalResult:
        """
        Perform agentic retrieval with iterative refinement.

        Args:
            query: The user query
            user_id: Optional user ID for access control
            top_k: Maximum number of chunks to return
            acl_filter: Access control list filter
            metadata_filter: Metadata filter for documents

        Returns:
            AgenticRetrievalResult with final chunks and retrieval trace
        """
        result = AgenticRetrievalResult(final_chunks=[])
        all_chunks: list[dict[str, Any]] = []
        current_query = query
        seen_chunk_ids: set[str] = set()

        logger.info(
            "Starting agentic retrieval",
            query=query[:100],
            max_iterations=self.max_iterations,
        )

        # Step 1: Self-RAG - Decide if retrieval is needed
        needs_retrieval = await self._needs_retrieval(query)

        if not needs_retrieval:
            logger.info("Self-RAG: No retrieval needed for query")
            result.final_decision = RetrievalDecision.NO_RETRIEVE
            return result

        # Iterative retrieval loop
        for iteration in range(self.max_iterations):
            result.total_iterations = iteration + 1
            result.query_transformations.append(current_query)

            logger.debug(
                "Retrieval iteration",
                iteration=iteration + 1,
                query=current_query[:100],
            )

            # Step 2: Retrieve documents using existing pipeline
            retrieval_result = await self.retrieval.retrieve(
                query=current_query,
                top_k=top_k * 2,  # Over-retrieve for filtering
                use_reranking=True,
                use_graph_expansion=True,
                use_query_rewriting=False,  # We handle query refinement ourselves
                acl_filter=acl_filter,
                metadata_filter=metadata_filter,
            )

            # Convert to chunk dictionaries
            chunks = [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "content": c.content,
                    "score": c.score,
                    "doc_title": c.doc_title,
                    "chunk_index": c.chunk_index,
                    "metadata": c.metadata,
                }
                for c in retrieval_result.chunks
            ]

            # Step 3: Grade relevance of each chunk
            graded_chunks = await self._grade_relevance_batch(query, chunks)

            # Filter relevant chunks based on threshold
            relevant_chunks = [
                c for c in graded_chunks
                if c.get("relevance_score", 0) >= self.relevance_threshold
            ]

            # Create step record
            step = RetrievalStep(
                query=current_query,
                retrieved_chunks=chunks,
                relevance_scores=[c.get("relevance_score", 0) for c in graded_chunks],
                decision=RetrievalDecision.CONTINUE,
                iteration=iteration + 1,
            )

            # Deduplicate and merge with previous chunks
            for chunk in relevant_chunks:
                chunk_id = chunk["chunk_id"]
                if chunk_id not in seen_chunk_ids:
                    all_chunks.append(chunk)
                    seen_chunk_ids.add(chunk_id)

            logger.debug(
                "Grading complete",
                total_chunks=len(chunks),
                relevant_chunks=len(relevant_chunks),
                accumulated_chunks=len(all_chunks),
            )

            # Step 4: Evaluate if we have enough context
            if len(all_chunks) >= self.min_relevant_chunks:
                is_sufficient = await self._is_context_sufficient(query, all_chunks)

                if is_sufficient:
                    step.decision = RetrievalDecision.SUFFICIENT
                    step.reasoning = "Context is sufficient to answer the query"
                    result.steps.append(step)
                    logger.info(
                        "Context sufficient",
                        iteration=iteration + 1,
                        chunk_count=len(all_chunks),
                    )
                    break

            # Step 5: CRAG - Decide on correction action
            if len(relevant_chunks) == 0:
                # No relevant chunks - ambiguous case, try web search or refine
                step.decision = RetrievalDecision.CONTINUE
                step.reasoning = "No relevant chunks found, refining query"

                if self.enable_web_search and iteration == self.max_iterations - 1:
                    # Last resort: web search
                    web_chunks = await self._web_search_fallback(query)
                    for chunk in web_chunks:
                        if chunk["chunk_id"] not in seen_chunk_ids:
                            all_chunks.append(chunk)
                            seen_chunk_ids.add(chunk["chunk_id"])
                    result.web_search_used = True
                    step.reasoning = "Used web search as fallback"

            # Step 6: Refine query for next iteration
            if iteration < self.max_iterations - 1:
                current_query = await self._refine_query(
                    original_query=query,
                    current_query=current_query,
                    retrieved_chunks=chunks,
                    relevant_chunks=relevant_chunks,
                    iteration=iteration,
                )
                step.reasoning = f"Refined query to: {current_query[:100]}"

            result.steps.append(step)

        # Sort all chunks by relevance and take top_k
        all_chunks.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        result.final_chunks = all_chunks[:top_k]
        result.final_decision = RetrievalDecision.SUFFICIENT
        result.confidence = self._calculate_confidence(result.final_chunks)

        logger.info(
            "Agentic retrieval complete",
            total_iterations=result.total_iterations,
            final_chunk_count=len(result.final_chunks),
            confidence=result.confidence,
        )

        return result

    async def _needs_retrieval(self, query: str) -> bool:
        """
        Self-RAG: Decide if retrieval is needed for this query.

        Implements the [Retrieve] special token from Self-RAG.
        Some queries can be answered from model knowledge alone.
        """
        prompt = f"""Analyze this query and decide if external document retrieval is needed to provide an accurate answer.

Query: {query}

Consider:
- Is this asking about specific documents, data, or facts that might be in a knowledge base?
- Does this require up-to-date or domain-specific information?
- Is this a general knowledge question that doesn't require document lookup?
- Is this a simple greeting, math calculation, or meta question about the system?

Respond with ONLY one word: RETRIEVE or NO_RETRIEVE"""

        try:
            response = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )

            decision = "RETRIEVE" in response.upper()
            logger.debug("Retrieval decision", query=query[:50], decision=decision)
            return decision

        except Exception as e:
            logger.warning("Retrieval decision failed, defaulting to retrieve", error=str(e))
            return True

    async def _grade_relevance_batch(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Grade the relevance of each retrieved chunk.

        Implements the [ISREL] special token from Self-RAG.
        Uses LLM to evaluate if chunk helps answer the query.
        """
        if not chunks:
            return []

        # Grade chunks in parallel for speed
        async def grade_single_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
            content_preview = chunk.get("content", "")[:1500]

            prompt = f"""Grade the relevance of this document chunk for answering the query.

Query: {query}

Document chunk:
{content_preview}

Grade the relevance from 0 to 1 where:
- 1.0: Directly answers or strongly supports answering the query
- 0.7-0.9: Contains highly relevant information
- 0.4-0.6: Partially relevant, provides some useful context
- 0.1-0.3: Minimally relevant, tangentially related
- 0.0: Not relevant at all

Respond with ONLY a single decimal number between 0 and 1."""

            try:
                response = await self.llm.generate(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0,
                )

                # Parse the score
                score_text = response.strip()
                # Handle potential formatting issues
                score_text = score_text.replace(",", ".").strip()
                score = float(score_text)
                score = max(0.0, min(1.0, score))

            except (ValueError, TypeError) as e:
                logger.debug("Score parsing failed, using default", error=str(e))
                score = 0.5
            except Exception as e:
                logger.warning("Grading failed for chunk", error=str(e))
                score = 0.5

            # Determine grade category
            if score >= 0.7:
                grade = RelevanceGrade.RELEVANT
            elif score >= 0.4:
                grade = RelevanceGrade.PARTIALLY_RELEVANT
            else:
                grade = RelevanceGrade.NOT_RELEVANT

            return {
                **chunk,
                "relevance_score": score,
                "relevance_grade": grade.value,
            }

        # Run grading in parallel with concurrency limit
        semaphore = asyncio.Semaphore(5)  # Limit concurrent LLM calls

        async def grade_with_limit(chunk: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await grade_single_chunk(chunk)

        tasks = [grade_with_limit(c) for c in chunks]
        graded = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_graded = []
        for i, result in enumerate(graded):
            if isinstance(result, Exception):
                logger.warning("Chunk grading failed", error=str(result))
                valid_graded.append({**chunks[i], "relevance_score": 0.5})
            else:
                valid_graded.append(result)

        return valid_graded

    async def _is_context_sufficient(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> bool:
        """
        Evaluate if the retrieved context is sufficient to answer the query.

        This helps decide whether to continue retrieval or stop.
        """
        # Build context from top chunks
        context_parts = []
        for chunk in chunks[:5]:
            content = chunk.get("content", "")[:600]
            context_parts.append(f"[Document: {chunk.get('doc_title', 'Unknown')}]\n{content}")

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""Given this query and retrieved context, determine if the context is sufficient to provide a complete and accurate answer.

Query: {query}

Retrieved context:
{context[:3000]}

Consider:
- Does the context contain the specific information needed to answer the query?
- Are there significant gaps in the information that would require more retrieval?
- Would the answer be incomplete or potentially inaccurate with only this context?

Respond with ONLY one word: SUFFICIENT or INSUFFICIENT"""

        try:
            response = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )

            is_sufficient = "SUFFICIENT" in response.upper() and "INSUFFICIENT" not in response.upper()
            logger.debug("Sufficiency check", is_sufficient=is_sufficient)
            return is_sufficient

        except Exception as e:
            logger.warning("Sufficiency check failed", error=str(e))
            # Be conservative - assume more retrieval might help
            return len(chunks) >= self.min_relevant_chunks * 2

    async def _refine_query(
        self,
        original_query: str,
        current_query: str,
        retrieved_chunks: list[dict[str, Any]],
        relevant_chunks: list[dict[str, Any]],
        iteration: int,
    ) -> str:
        """
        Refine the query based on what was retrieved.

        Implements Corrective RAG query refinement strategy.
        """
        # Analyze what we found (or didn't find)
        if relevant_chunks:
            context_summary = "\n".join([
                f"- {c.get('content', '')[:150]}..." for c in relevant_chunks[:3]
            ])
            found_info = f"Partially relevant information found:\n{context_summary}"
        else:
            found_info = "No relevant information was found in the retrieved documents."

        # Different refinement strategies based on iteration
        if iteration == 0:
            strategy = "Try rephrasing with different keywords or more specific terms"
        else:
            strategy = "Try a broader approach or focus on related concepts"

        prompt = f"""The current search query did not retrieve sufficient information. Generate an improved search query.

Original question: {original_query}
Current search query: {current_query}

{found_info}

Strategy: {strategy}

Generate a refined search query that:
1. Targets the missing information more precisely
2. Uses alternative keywords or phrasing
3. Is specific enough to find relevant documents

Respond with ONLY the refined query, nothing else."""

        try:
            response = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3,
            )

            refined = response.strip()
            # Remove quotes if present
            refined = refined.strip('"\'')

            # Ensure we got a valid query
            if len(refined) < 3:
                return current_query

            logger.debug("Query refined", original=current_query[:50], refined=refined[:50])
            return refined

        except Exception as e:
            logger.warning("Query refinement failed", error=str(e))
            return current_query

    async def _web_search_fallback(
        self,
        query: str,
    ) -> list[dict[str, Any]]:
        """
        Fallback to web search when internal retrieval fails.

        Part of CRAG pattern for knowledge augmentation.
        """
        # This is a placeholder - actual implementation would integrate
        # with a web search API (e.g., Brave Search, SerpAPI, etc.)
        logger.info("Web search fallback triggered", query=query[:100])

        # Return empty for now - actual implementation would search the web
        # and convert results to chunk format
        return []

    def _calculate_confidence(self, chunks: list[dict[str, Any]]) -> float:
        """Calculate overall confidence based on chunk relevance scores."""
        if not chunks:
            return 0.0

        scores = [c.get("relevance_score", 0.5) for c in chunks]

        # Weighted average favoring top chunks (position-weighted)
        weights = [1.0 / (i + 1) for i in range(len(scores))]
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)

        return weighted_sum / total_weight if total_weight > 0 else 0.0


class QueryDecomposer:
    """Decompose complex queries into sub-queries for multi-hop retrieval."""

    def __init__(self, llm_gateway: "LLMGateway"):
        self.llm = llm_gateway

    async def decompose(self, query: str) -> list[str]:
        """
        Decompose a complex query into simpler sub-queries.

        This enables multi-hop retrieval where different pieces of
        information need to be gathered and combined.
        """
        prompt = f"""Analyze this question and determine if it needs to be broken down into simpler sub-questions.

Question: {query}

If the question is already simple and direct, return just the original question.
If it's complex and requires multiple pieces of information, break it into 2-4 focused sub-questions.

Rules:
- Each sub-question should be self-contained
- Sub-questions should together cover all aspects of the original question
- Don't add unnecessary sub-questions
- Keep sub-questions concise

Format: Output one question per line, no numbering or bullet points."""

        try:
            response = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0,
            )

            # Parse sub-queries
            lines = response.strip().split("\n")
            sub_queries = [
                line.strip()
                for line in lines
                if line.strip() and len(line.strip()) > 5
            ]

            # Always include original query
            if query not in sub_queries:
                sub_queries.insert(0, query)

            # Limit to 4 sub-queries
            sub_queries = sub_queries[:4]

            logger.debug(
                "Query decomposition",
                original=query[:50],
                sub_query_count=len(sub_queries),
            )

            return sub_queries

        except Exception as e:
            logger.warning("Query decomposition failed", error=str(e))
            return [query]

    async def is_complex(self, query: str) -> bool:
        """
        Determine if a query is complex enough to warrant decomposition.
        """
        # Simple heuristics first
        if len(query) < 30:
            return False

        # Check for multiple question indicators
        complex_indicators = [
            " and ",
            " or ",
            "compare",
            "difference between",
            "how does",
            "what are the",
            "explain why",
            "relationship between",
        ]

        query_lower = query.lower()
        indicator_count = sum(1 for ind in complex_indicators if ind in query_lower)

        if indicator_count >= 2:
            return True

        # For borderline cases, use LLM
        if indicator_count == 1 or len(query) > 100:
            prompt = f"""Is this question complex enough to require multiple sub-questions to answer completely?

Question: {query}

Respond with only YES or NO."""

            try:
                response = await self.llm.generate(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0,
                )
                return "YES" in response.upper()
            except Exception:
                pass

        return False


class MultiHopRetriever:
    """
    Multi-hop retrieval for questions requiring multiple reasoning steps.

    This combines query decomposition with agentic retrieval to handle
    complex questions that require gathering and synthesizing information
    from multiple sources.
    """

    def __init__(
        self,
        agentic_retriever: AgenticRetriever,
        query_decomposer: QueryDecomposer,
        max_sub_queries: int = 4,
    ):
        self.retriever = agentic_retriever
        self.decomposer = query_decomposer
        self.max_sub_queries = max_sub_queries

    async def retrieve(
        self,
        query: str,
        user_id: str | None = None,
        top_k: int = 10,
        acl_filter: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> AgenticRetrievalResult:
        """
        Perform multi-hop retrieval for complex queries.

        Steps:
        1. Check if query is complex
        2. Decompose into sub-queries if needed
        3. Retrieve for each sub-query
        4. Merge and deduplicate results
        5. Re-rank based on relevance to original query
        """
        logger.info("Starting multi-hop retrieval", query=query[:100])

        # Check complexity and decompose if needed
        is_complex = await self.decomposer.is_complex(query)

        if not is_complex:
            # Simple query - use regular agentic retrieval
            logger.debug("Query is simple, using direct retrieval")
            return await self.retriever.retrieve(
                query=query,
                user_id=user_id,
                top_k=top_k,
                acl_filter=acl_filter,
                metadata_filter=metadata_filter,
            )

        # Decompose complex query
        sub_queries = await self.decomposer.decompose(query)
        logger.info("Query decomposed", sub_query_count=len(sub_queries))

        if len(sub_queries) <= 1:
            # Decomposition didn't produce sub-queries
            return await self.retriever.retrieve(
                query=query,
                user_id=user_id,
                top_k=top_k,
                acl_filter=acl_filter,
                metadata_filter=metadata_filter,
            )

        # Retrieve for each sub-query in parallel
        retrieval_tasks = [
            self.retriever.retrieve(
                query=sub_query,
                user_id=user_id,
                top_k=top_k,
                acl_filter=acl_filter,
                metadata_filter=metadata_filter,
            )
            for sub_query in sub_queries[:self.max_sub_queries]
        ]

        all_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

        # Merge results
        all_chunks: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        merged_steps: list[RetrievalStep] = []
        total_iterations = 0
        web_search_used = False

        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                logger.warning(
                    "Sub-query retrieval failed",
                    sub_query=sub_queries[i][:50],
                    error=str(result),
                )
                continue

            # Add unique chunks
            for chunk in result.final_chunks:
                chunk_id = chunk.get("chunk_id")
                if chunk_id and chunk_id not in seen_ids:
                    # Annotate chunk with source sub-query
                    chunk["source_sub_query"] = sub_queries[i]
                    all_chunks.append(chunk)
                    seen_ids.add(chunk_id)

            # Merge steps with sub-query annotation
            for step in result.steps:
                step.reasoning = f"[Sub-query {i+1}: {sub_queries[i][:30]}...] {step.reasoning}"
                merged_steps.append(step)

            total_iterations += result.total_iterations
            web_search_used = web_search_used or result.web_search_used

        # Re-rank all chunks based on relevance to original query
        if all_chunks:
            all_chunks = await self._rerank_for_original_query(query, all_chunks)

        # Take top_k
        final_chunks = all_chunks[:top_k]

        # Calculate combined confidence
        if all_results:
            valid_results = [r for r in all_results if not isinstance(r, Exception)]
            avg_confidence = (
                sum(r.confidence for r in valid_results) / len(valid_results)
                if valid_results else 0.0
            )
        else:
            avg_confidence = 0.0

        return AgenticRetrievalResult(
            final_chunks=final_chunks,
            steps=merged_steps,
            total_iterations=total_iterations,
            final_decision=RetrievalDecision.SUFFICIENT,
            confidence=avg_confidence,
            web_search_used=web_search_used,
            query_transformations=sub_queries,
        )

    async def _rerank_for_original_query(
        self,
        original_query: str,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Re-rank merged chunks based on relevance to the original query."""
        # Use the agentic retriever's grading mechanism
        graded = await self.retriever._grade_relevance_batch(original_query, chunks)

        # Sort by the new relevance scores
        graded.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        return graded


class SelfRAGReflector:
    """
    Self-RAG reflection component for evaluating generated responses.

    Implements the remaining Self-RAG special tokens:
    - [ISSUP]: Is the response supported by the retrieved context?
    - [ISUSE]: Is the response useful for the query?
    """

    def __init__(self, llm_gateway: "LLMGateway"):
        self.llm = llm_gateway

    async def evaluate_support(
        self,
        response: str,
        context_chunks: list[dict[str, Any]],
    ) -> tuple[SupportGrade, float, str]:
        """
        Evaluate if the response is supported by the retrieved context.

        Implements [ISSUP] token - checks for hallucinations.
        """
        context = "\n\n".join([c.get("content", "")[:500] for c in context_chunks[:5]])

        prompt = f"""Evaluate if the following response is supported by the provided context.

Context:
{context}

Response:
{response}

Consider:
- Are all factual claims in the response grounded in the context?
- Does the response make unsupported extrapolations?
- Are there any hallucinated facts?

Grade the support level:
- FULLY_SUPPORTED: All claims are directly supported by the context
- PARTIALLY_SUPPORTED: Most claims are supported, minor unsupported details
- NO_SUPPORT: Major claims are not supported by the context

Respond in this format:
GRADE: [grade]
SCORE: [0.0-1.0]
REASONING: [brief explanation]"""

        try:
            result = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )

            # Parse result
            lines = result.strip().split("\n")
            grade = SupportGrade.PARTIALLY_SUPPORTED
            score = 0.5
            reasoning = ""

            for line in lines:
                if line.startswith("GRADE:"):
                    grade_str = line.replace("GRADE:", "").strip().upper()
                    if "FULLY" in grade_str:
                        grade = SupportGrade.FULLY_SUPPORTED
                    elif "NO" in grade_str:
                        grade = SupportGrade.NO_SUPPORT
                    else:
                        grade = SupportGrade.PARTIALLY_SUPPORTED
                elif line.startswith("SCORE:"):
                    try:
                        score = float(line.replace("SCORE:", "").strip())
                        score = max(0.0, min(1.0, score))
                    except ValueError:
                        pass
                elif line.startswith("REASONING:"):
                    reasoning = line.replace("REASONING:", "").strip()

            return grade, score, reasoning

        except Exception as e:
            logger.warning("Support evaluation failed", error=str(e))
            return SupportGrade.PARTIALLY_SUPPORTED, 0.5, "Evaluation failed"

    async def evaluate_utility(
        self,
        query: str,
        response: str,
    ) -> tuple[UtilityGrade, float, str]:
        """
        Evaluate if the response is useful for answering the query.

        Implements [ISUSE] token - checks answer quality.
        """
        prompt = f"""Evaluate if the following response usefully answers the query.

Query: {query}

Response: {response}

Consider:
- Does the response directly address the query?
- Is the response complete or does it leave key questions unanswered?
- Is the response clear and well-organized?

Grade the utility:
- USEFUL: Directly and completely answers the query
- PARTIALLY_USEFUL: Addresses the query but incomplete or tangential
- NOT_USEFUL: Does not meaningfully answer the query

Respond in this format:
GRADE: [grade]
SCORE: [0.0-1.0]
REASONING: [brief explanation]"""

        try:
            result = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )

            # Parse result
            lines = result.strip().split("\n")
            grade = UtilityGrade.PARTIALLY_USEFUL
            score = 0.5
            reasoning = ""

            for line in lines:
                if line.startswith("GRADE:"):
                    grade_str = line.replace("GRADE:", "").strip().upper()
                    if "NOT" in grade_str:
                        grade = UtilityGrade.NOT_USEFUL
                    elif "PARTIAL" in grade_str:
                        grade = UtilityGrade.PARTIALLY_USEFUL
                    else:
                        grade = UtilityGrade.USEFUL
                elif line.startswith("SCORE:"):
                    try:
                        score = float(line.replace("SCORE:", "").strip())
                        score = max(0.0, min(1.0, score))
                    except ValueError:
                        pass
                elif line.startswith("REASONING:"):
                    reasoning = line.replace("REASONING:", "").strip()

            return grade, score, reasoning

        except Exception as e:
            logger.warning("Utility evaluation failed", error=str(e))
            return UtilityGrade.PARTIALLY_USEFUL, 0.5, "Evaluation failed"

    async def full_reflection(
        self,
        query: str,
        response: str,
        context_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Perform full Self-RAG reflection on a generated response.
        """
        # Run both evaluations in parallel
        support_task = self.evaluate_support(response, context_chunks)
        utility_task = self.evaluate_utility(query, response)

        (support_grade, support_score, support_reasoning), \
        (utility_grade, utility_score, utility_reasoning) = await asyncio.gather(
            support_task,
            utility_task,
        )

        # Calculate overall quality score
        overall_score = (support_score * 0.6) + (utility_score * 0.4)

        # Determine if response should be regenerated
        needs_regeneration = (
            support_grade == SupportGrade.NO_SUPPORT or
            utility_grade == UtilityGrade.NOT_USEFUL or
            overall_score < 0.4
        )

        return {
            "support": {
                "grade": support_grade.value,
                "score": support_score,
                "reasoning": support_reasoning,
            },
            "utility": {
                "grade": utility_grade.value,
                "score": utility_score,
                "reasoning": utility_reasoning,
            },
            "overall_score": overall_score,
            "needs_regeneration": needs_regeneration,
        }


class CorrectiveRAGOrchestrator:
    """
    Orchestrates the full Corrective RAG workflow.

    Combines agentic retrieval with response generation and reflection
    for a complete CRAG implementation.
    """

    def __init__(
        self,
        agentic_retriever: AgenticRetriever,
        reflector: SelfRAGReflector,
        llm_gateway: "LLMGateway",
        max_regenerations: int = 2,
    ):
        self.retriever = agentic_retriever
        self.reflector = reflector
        self.llm = llm_gateway
        self.max_regenerations = max_regenerations

    async def query(
        self,
        query: str,
        user_id: str | None = None,
        top_k: int = 10,
        acl_filter: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Execute full CRAG workflow: retrieve, generate, reflect, correct.
        """
        # Step 1: Agentic retrieval
        retrieval_result = await self.retriever.retrieve(
            query=query,
            user_id=user_id,
            top_k=top_k,
            acl_filter=acl_filter,
        )

        if retrieval_result.final_decision == RetrievalDecision.NO_RETRIEVE:
            # Direct answer without retrieval
            response = await self._generate_without_context(query)
            return {
                "response": response,
                "retrieval_result": retrieval_result,
                "reflection": None,
                "regenerations": 0,
            }

        # Step 2: Generate initial response
        response = await self._generate_with_context(
            query, retrieval_result.final_chunks
        )

        # Step 3: Reflect on response
        reflection = await self.reflector.full_reflection(
            query, response, retrieval_result.final_chunks
        )

        regeneration_count = 0

        # Step 4: Corrective loop if needed
        while (
            reflection["needs_regeneration"] and
            regeneration_count < self.max_regenerations
        ):
            regeneration_count += 1
            logger.info(
                "Regenerating response",
                attempt=regeneration_count,
                support_score=reflection["support"]["score"],
                utility_score=reflection["utility"]["score"],
            )

            # Regenerate with guidance from reflection
            response = await self._regenerate_with_feedback(
                query,
                retrieval_result.final_chunks,
                reflection,
            )

            # Re-evaluate
            reflection = await self.reflector.full_reflection(
                query, response, retrieval_result.final_chunks
            )

        return {
            "response": response,
            "retrieval_result": retrieval_result,
            "reflection": reflection,
            "regenerations": regeneration_count,
        }

    async def _generate_without_context(self, query: str) -> str:
        """Generate response without retrieval context."""
        prompt = f"""Answer the following question directly based on your knowledge.

Question: {query}

Provide a clear, concise answer."""

        return await self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7,
        )

    async def _generate_with_context(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> str:
        """Generate response using retrieved context."""
        context = "\n\n---\n\n".join([
            f"[Source: {c.get('doc_title', 'Unknown')}]\n{c.get('content', '')}"
            for c in chunks[:5]
        ])

        prompt = f"""Answer the question using ONLY the provided context. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {query}

Provide a clear, well-organized answer with citations to the source documents where appropriate."""

        return await self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.3,
        )

    async def _regenerate_with_feedback(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        reflection: dict[str, Any],
    ) -> str:
        """Regenerate response incorporating reflection feedback."""
        context = "\n\n---\n\n".join([
            f"[Source: {c.get('doc_title', 'Unknown')}]\n{c.get('content', '')}"
            for c in chunks[:5]
        ])

        feedback_parts = []
        if reflection["support"]["score"] < 0.7:
            feedback_parts.append(
                f"- Ensure all claims are supported by the context. "
                f"Previous issue: {reflection['support']['reasoning']}"
            )
        if reflection["utility"]["score"] < 0.7:
            feedback_parts.append(
                f"- Make sure to directly answer the question. "
                f"Previous issue: {reflection['utility']['reasoning']}"
            )

        feedback = "\n".join(feedback_parts)

        prompt = f"""Answer the question using ONLY the provided context. Pay special attention to the feedback about previous attempt.

Context:
{context}

Question: {query}

Feedback on previous attempt:
{feedback}

Provide an improved answer that addresses the feedback while staying grounded in the context."""

        return await self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.2,  # Lower temperature for more focused regeneration
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_agentic_retriever(
    retrieval_pipeline: "RetrievalPipeline",
    llm_gateway: "LLMGateway",
    **kwargs,
) -> AgenticRetriever:
    """Create an agentic retriever with the given pipeline and LLM gateway."""
    return AgenticRetriever(
        retrieval_pipeline=retrieval_pipeline,
        llm_gateway=llm_gateway,
        **kwargs,
    )


def create_multi_hop_retriever(
    retrieval_pipeline: "RetrievalPipeline",
    llm_gateway: "LLMGateway",
    **kwargs,
) -> MultiHopRetriever:
    """Create a multi-hop retriever for complex queries."""
    agentic = AgenticRetriever(
        retrieval_pipeline=retrieval_pipeline,
        llm_gateway=llm_gateway,
        **kwargs,
    )
    decomposer = QueryDecomposer(llm_gateway)
    return MultiHopRetriever(agentic, decomposer)


def create_crag_orchestrator(
    retrieval_pipeline: "RetrievalPipeline",
    llm_gateway: "LLMGateway",
    **kwargs,
) -> CorrectiveRAGOrchestrator:
    """Create a full CRAG orchestrator."""
    agentic = AgenticRetriever(
        retrieval_pipeline=retrieval_pipeline,
        llm_gateway=llm_gateway,
        **kwargs,
    )
    reflector = SelfRAGReflector(llm_gateway)
    return CorrectiveRAGOrchestrator(
        agentic_retriever=agentic,
        reflector=reflector,
        llm_gateway=llm_gateway,
    )
