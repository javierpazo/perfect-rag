"""Generation pipeline orchestrator.

Full pipeline with:
1. Evidence extraction from chunks (evidence-first)
2. Prompt construction with verified evidence
3. LLM generation (streaming or non-streaming)
4. Citation extraction and verification
5. Response formatting
"""

from typing import Any, AsyncIterator

import structlog

from perfect_rag.config import Settings, get_settings
from perfect_rag.generation.citation_extractor import CitationExtractor
from perfect_rag.generation.prompt_builder import PromptBuilder
from perfect_rag.llm.gateway import LLMGateway
from perfect_rag.models.openai_types import Citation
from perfect_rag.models.query import GenerationResult, RetrievalResult, SourceChunk

logger = structlog.get_logger(__name__)


class GenerationPipeline:
    """Complete generation pipeline with RAG augmentation.

    Pipeline steps:
    1. Evidence extraction (evidence-first, reduces hallucinations)
    2. Prompt construction with retrieved context
    3. LLM generation (streaming or non-streaming)
    4. Citation extraction and verification
    5. Response formatting
    """

    def __init__(
        self,
        llm_gateway: LLMGateway,
        settings: Settings | None = None,
    ):
        self.llm = llm_gateway
        self.settings = settings or get_settings()
        self.prompt_builder = PromptBuilder(settings)
        self.citation_extractor = CitationExtractor()

        # Evidence-first components (lazy loaded)
        self._evidence_extractor = None
        self._evidence_generator = None

    async def generate(
        self,
        messages: list[dict[str, str]],
        retrieval_result: RetrievalResult | None = None,
        model: str | None = None,
        provider: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        prompt_mode: str = "default",
        include_citations: bool = True,
        verify_citations: bool = False,
        stream: bool = False,
        use_evidence_first: bool | None = None,
    ) -> GenerationResult | AsyncIterator[str]:
        """Generate response with RAG augmentation.

        Args:
            messages: Conversation messages
            retrieval_result: Retrieved context (if any)
            model: LLM model to use
            provider: LLM provider
            temperature: Sampling temperature
            max_tokens: Max response tokens
            prompt_mode: Prompt construction mode
            include_citations: Whether to extract citations
            verify_citations: Whether to verify citation accuracy
            stream: Whether to stream response
            use_evidence_first: Whether to use evidence-first generation (default from settings)

        Returns:
            GenerationResult or AsyncIterator of chunks if streaming
        """
        # Determine if evidence-first should be used
        if use_evidence_first is None:
            use_evidence_first = self.settings.evidence_first_enabled

        # Use evidence-first pipeline if enabled and we have chunks
        if use_evidence_first and retrieval_result and retrieval_result.chunks:
            return await self._generate_evidence_first(
                messages=messages,
                retrieval_result=retrieval_result,
                model=model,
                provider=provider,
                temperature=temperature,
                max_tokens=max_tokens,
                include_citations=include_citations,
            )

        # Standard generation
        return await self._generate_standard(
            messages=messages,
            retrieval_result=retrieval_result,
            model=model,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_mode=prompt_mode,
            include_citations=include_citations,
            verify_citations=verify_citations,
            stream=stream,
        )

    async def _generate_evidence_first(
        self,
        messages: list[dict[str, str]],
        retrieval_result: RetrievalResult,
        model: str | None,
        provider: str | None,
        temperature: float,
        max_tokens: int,
        include_citations: bool,
    ) -> GenerationResult:
        """Two-step evidence-first generation (reduces hallucinations)."""
        # Lazy load evidence components
        await self._ensure_evidence_components()

        # Extract query from messages
        query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                query = msg.get("content", "")
                break

        # Convert chunks to format expected by evidence extractor
        chunks_for_evidence = [
            {
                "id": c.chunk_id,
                "doc_title": c.doc_title,
                "content": c.content,
                "score": c.score,
            }
            for c in retrieval_result.chunks
        ]

        # Step 1: Extract evidence from chunks
        if self._evidence_extractor:
            evidence_set = await self._evidence_extractor.extract_evidence(
                query=query,
                chunks=chunks_for_evidence,
                max_evidences=self.settings.evidence_first_max_evidences,
            )

            # Check coverage
            if evidence_set.coverage < self.settings.evidence_first_min_coverage:
                logger.warning(
                    "Low evidence coverage, may have gaps",
                    coverage=evidence_set.coverage,
                )
        else:
            # Fallback: use chunks directly
            from perfect_rag.generation.evidence_first import EvidenceSet, ExtractedEvidence

            evidence_set = EvidenceSet(
                evidences=[
                    ExtractedEvidence(
                        chunk_id=c.chunk_id,
                        doc_title=c.doc_title,
                        evidence_text=c.content[:500],
                        relevance_score=c.score,
                    )
                    for c in retrieval_result.chunks[:5]
                ],
                contradictions=[],
                gaps=[],
                coverage=0.7,
            )

        # Step 2: Generate answer from evidence
        if self._evidence_generator:
            result = await self._evidence_generator.generate(
                query=query,
                chunks=chunks_for_evidence,
                max_evidences=self.settings.evidence_first_max_evidences,
            )

            response_text = result.answer
            citations = result.citations
            evidence_confidence = result.confidence
        else:
            # Fallback: standard generation with evidence prompt
            evidence_context = "\n\n".join(
                f"[E{i+1}] {e.evidence_text}"
                for i, e in enumerate(evidence_set.evidences)
            )

            enhanced_messages = messages.copy()
            if evidence_context:
                system_msg = {
                    "role": "system",
                    "content": f"Use ONLY the following verified evidence to answer. Cite sources as [E1], [E2], etc.\n\n{evidence_context}",
                }
                enhanced_messages.insert(0, system_msg)

            response_text = await self.llm.generate(
                messages=enhanced_messages,
                model=model,
                provider=provider,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            citations = []
            evidence_confidence = retrieval_result.confidence

        return GenerationResult(
            response=response_text,
            citations=citations,
            confidence=evidence_confidence,
            model=model or self.settings.default_llm_model,
            retrieval_metadata={
                **retrieval_result.metadata,
                "evidence_first": True,
                "evidence_count": len(evidence_set.evidences),
                "evidence_coverage": evidence_set.coverage,
            },
        )

    async def _generate_standard(
        self,
        messages: list[dict[str, str]],
        retrieval_result: RetrievalResult | None,
        model: str | None,
        provider: str | None,
        temperature: float,
        max_tokens: int,
        prompt_mode: str,
        include_citations: bool,
        verify_citations: bool,
        stream: bool,
    ) -> GenerationResult | AsyncIterator[str]:
        """Standard generation pipeline."""
        # Build augmented prompt if we have retrieval results
        if retrieval_result and retrieval_result.chunks:
            augmented_messages = self.prompt_builder.build_rag_prompt(
                messages=messages,
                chunks=retrieval_result.chunks,
                mode=prompt_mode,
            )
        else:
            augmented_messages = messages

        logger.info(
            "Starting generation",
            model=model or self.settings.default_llm_model,
            has_context=bool(retrieval_result and retrieval_result.chunks),
            stream=stream,
        )

        if stream:
            return self._stream_generate(
                messages=augmented_messages,
                retrieval_result=retrieval_result,
                model=model,
                provider=provider,
                temperature=temperature,
                max_tokens=max_tokens,
                include_citations=include_citations,
            )
        else:
            return await self._generate(
                messages=augmented_messages,
                retrieval_result=retrieval_result,
                model=model,
                provider=provider,
                temperature=temperature,
                max_tokens=max_tokens,
                include_citations=include_citations,
                verify_citations=verify_citations,
            )

    async def _ensure_evidence_components(self) -> None:
        """Lazy load evidence-first components."""
        if self._evidence_extractor is None:
            try:
                from perfect_rag.generation.evidence_first import (
                    EvidenceBasedGenerator,
                    EvidenceExtractor,
                )

                self._evidence_extractor = EvidenceExtractor(self.llm)
                self._evidence_generator = EvidenceBasedGenerator(self.llm)
                logger.info("Evidence-first components loaded")
            except ImportError as e:
                logger.warning("Evidence-first not available", error=str(e))
            except Exception as e:
                logger.error("Failed to load evidence components", error=str(e))

    async def _stream_generate(
        self,
        messages: list[dict[str, str]],
        retrieval_result: RetrievalResult | None,
        model: str | None,
        provider: str | None,
        temperature: float,
        max_tokens: int,
        include_citations: bool,
    ) -> AsyncIterator[str]:
        """Streaming generation.

        Yields text chunks. Citations are extracted at the end.
        """
        # Get streaming response
        stream = await self.llm.generate(
            messages=messages,
            model=model,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        # Collect full response for citation extraction
        full_response = []

        async for chunk in stream:
            full_response.append(chunk)
            yield chunk

        # After streaming complete, we could process citations
        # but for SSE, citations are typically sent in final message
        # which is handled at the API level

    async def _generate(
        self,
        messages: list[dict[str, str]],
        retrieval_result: RetrievalResult | None,
        model: str | None,
        provider: str | None,
        temperature: float,
        max_tokens: int,
        include_citations: bool,
        verify_citations: bool,
    ) -> GenerationResult:
        """Non-streaming generation."""
        # Generate response
        response_text = await self.llm.generate(
            messages=messages,
            model=model,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        # Extract citations
        citations = []
        citation_verification = None

        if include_citations and retrieval_result and retrieval_result.chunks:
            extracted = self.citation_extractor.extract_citations(
                response_text, retrieval_result.chunks
            )
            citations = self.citation_extractor.build_citation_objects(
                retrieval_result.chunks, extracted
            )

            if verify_citations:
                citation_verification = self.citation_extractor.verify_citations(
                    response_text, retrieval_result.chunks, strict=False
                )

        # Calculate confidence
        if retrieval_result:
            confidence = retrieval_result.confidence
        else:
            confidence = None

        return GenerationResult(
            response=response_text,
            citations=citations,
            confidence=confidence,
            model=model or self.settings.default_llm_model,
            retrieval_metadata=retrieval_result.metadata if retrieval_result else None,
            citation_verification=citation_verification,
        )

    async def generate_with_verification(
        self,
        messages: list[dict[str, str]],
        retrieval_result: RetrievalResult,
        model: str | None = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate with answer verification.

        Uses a two-step process:
        1. Generate initial answer
        2. Verify answer against sources
        3. Revise if needed
        """
        # Step 1: Generate initial answer
        initial_result = await self._generate(
            messages=messages,
            retrieval_result=retrieval_result,
            model=model,
            include_citations=True,
            verify_citations=True,
            **kwargs,
        )

        # Step 2: Check verification
        verification = initial_result.citation_verification
        if verification and not verification.get("valid", True):
            # There are issues - try to fix
            logger.info("Answer has citation issues, attempting revision")

            issues_summary = "\n".join(
                f"- {issue['message']}"
                for issue in verification.get("issues", [])
            )

            revision_prompt = f"""Your previous answer had some citation issues:
{issues_summary}

Please revise your answer to address these issues. Make sure all claims are properly cited with accurate references to the source materials."""

            # Add revision request to messages
            revision_messages = messages + [
                {"role": "assistant", "content": initial_result.response},
                {"role": "user", "content": revision_prompt},
            ]

            # Rebuild with context
            revision_messages = self.prompt_builder.build_rag_prompt(
                messages=revision_messages,
                chunks=retrieval_result.chunks,
                mode="strict",
            )

            # Generate revised answer
            revised_result = await self._generate(
                messages=revision_messages,
                retrieval_result=retrieval_result,
                model=model,
                include_citations=True,
                verify_citations=True,
                **kwargs,
            )

            return revised_result

        return initial_result

    async def generate_multi_step(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        model: str | None = None,
        **kwargs,
    ) -> GenerationResult:
        """Multi-step generation with planning.

        Steps:
        1. Analyze context
        2. Plan answer structure
        3. Generate answer
        4. Verify and refine
        """
        chunks = retrieval_result.chunks

        # Step 1: Analyze
        analyze_messages = self.prompt_builder.build_multi_step_prompt(
            query=query, chunks=chunks, step="analyze"
        )
        analysis = await self.llm.generate(
            messages=analyze_messages,
            model=model,
            temperature=0.3,
            max_tokens=500,
        )

        # Step 2: Plan
        plan_messages = self.prompt_builder.build_multi_step_prompt(
            query=query, chunks=chunks, step="plan"
        )
        plan_messages.append({"role": "assistant", "content": f"Analysis:\n{analysis}"})
        plan_messages.append({"role": "user", "content": "Now create a plan for the answer."})

        plan = await self.llm.generate(
            messages=plan_messages,
            model=model,
            temperature=0.3,
            max_tokens=300,
        )

        # Step 3: Generate
        generate_messages = self.prompt_builder.build_multi_step_prompt(
            query=query, chunks=chunks, step="generate"
        )
        generate_messages.append({
            "role": "assistant",
            "content": f"Analysis:\n{analysis}\n\nPlan:\n{plan}",
        })
        generate_messages.append({
            "role": "user",
            "content": "Now generate the complete answer following the plan.",
        })

        response = await self.llm.generate(
            messages=generate_messages,
            model=model,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 4096),
        )

        # Extract citations
        extracted = self.citation_extractor.extract_citations(response, chunks)
        citations = self.citation_extractor.build_citation_objects(chunks, extracted)

        return GenerationResult(
            response=response,
            citations=citations,
            confidence=retrieval_result.confidence,
            model=model or self.settings.default_llm_model,
            retrieval_metadata={
                **retrieval_result.metadata,
                "multi_step": True,
                "analysis": analysis,
                "plan": plan,
            },
        )

    def format_response_with_bibliography(
        self,
        result: GenerationResult,
        format: str = "markdown",
    ) -> str:
        """Format response with inline citations and bibliography."""
        # Get source chunks from citations
        chunks = [
            SourceChunk(
                chunk_id="",
                doc_id=c.source_id,
                doc_title=c.source_title,
                content=c.text_snippet,
                score=c.relevance_score,
                chunk_index=c.chunk_index,
            )
            for c in result.citations
        ]

        # Format with enhanced citations
        formatted = self.citation_extractor.format_inline_citations(
            result.response, chunks, format=format
        )

        # Add bibliography
        bibliography = self.citation_extractor.generate_bibliography(chunks, format=format)

        return formatted + bibliography


# =============================================================================
# Factory Function
# =============================================================================

_pipeline: GenerationPipeline | None = None


async def get_generation_pipeline(
    llm_gateway: LLMGateway,
) -> GenerationPipeline:
    """Get or create generation pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = GenerationPipeline(llm_gateway=llm_gateway)
    return _pipeline
