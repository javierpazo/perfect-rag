"""FastAPI main application with OpenAI-compatible endpoints."""

import asyncio
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import structlog
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from perfect_rag.config import get_settings
from perfect_rag.core.embedding import get_embedding_service
from perfect_rag.core.resilience import get_circuit_breaker_registry
from perfect_rag.db.oxigraph import get_oxigraph
from perfect_rag.db.qdrant import get_qdrant
from perfect_rag.db.surrealdb import get_surrealdb
from perfect_rag.generation.pipeline import GenerationPipeline
from perfect_rag.ingestion.pipeline import IngestionPipeline
from perfect_rag.llm.gateway import get_llm_gateway
from perfect_rag.models.openai_types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    ChatMessage,
    Citation,
    ModelInfo,
    ModelsResponse,
    StreamDelta,
    UsageInfo,
)
from perfect_rag.models.query import QueryContext
from perfect_rag.retrieval.pipeline import RetrievalPipeline

# Auth imports
from perfect_rag.auth import JWTAuth, User, get_current_user, get_optional_user

# Rate limiting imports
from perfect_rag.middleware.rate_limit import RateLimitMiddleware

# Feedback imports
from perfect_rag.feedback import FeedbackCollector, FeedbackLearner, LearningScheduler

# Cache imports
from perfect_rag.cache import CAGCache, CachePrewarmer, SemanticCache

# Hallucination detection
from perfect_rag.generation.hallucination_detector import HallucinationDetector, ContradictionDetector

# Multimodal imports
from perfect_rag.ingestion.multimodal import (
    ColPaliProcessor,
    CLIPEmbedder,
    MultimodalIngestionPipeline,
    create_colpali_processor,
    create_clip_embedder,
)

logger = structlog.get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Perfect RAG API...")

    # Initialize all services
    try:
        # Database connections
        surrealdb = await get_surrealdb()
        qdrant = await get_qdrant()
        oxigraph = await get_oxigraph()

        # Core services
        embedding_service = await get_embedding_service()
        llm_gateway = await get_llm_gateway()

        # Initialize pipelines
        ingestion_pipeline = IngestionPipeline(
            surrealdb=surrealdb,
            qdrant=qdrant,
            oxigraph=oxigraph,
            embedding_service=embedding_service,
            llm_gateway=llm_gateway,
        )
        await ingestion_pipeline.initialize()

        retrieval_pipeline = RetrievalPipeline(
            qdrant=qdrant,
            surrealdb=surrealdb,
            oxigraph=oxigraph,
            embedding_service=embedding_service,
            llm_gateway=llm_gateway,
        )

        generation_pipeline = GenerationPipeline(
            llm_gateway=llm_gateway,
        )

        # Initialize authentication
        jwt_auth = JWTAuth(surrealdb=surrealdb, settings=settings)

        # Initialize feedback system
        feedback_collector = FeedbackCollector(surrealdb=surrealdb)
        feedback_learner = FeedbackLearner(
            surrealdb=surrealdb,
            feedback_collector=feedback_collector,
        )
        learning_scheduler = LearningScheduler(
            learner=feedback_learner,
        )

        # Initialize caching
        cag_cache = CAGCache(
            surrealdb=surrealdb,
            embedding_service=embedding_service,
            settings=settings,
        )
        await cag_cache.initialize()

        semantic_cache = SemanticCache(
            embedding_service=embedding_service,
            settings=settings,
        )

        # Initialize cache prewarmer
        cache_prewarmer = CachePrewarmer(
            cache=cag_cache,
            surrealdb=surrealdb,
            embedding_service=embedding_service,
            settings=settings,
        )

        # Initialize hallucination detection
        hallucination_detector = HallucinationDetector(
            llm_gateway=llm_gateway,
            settings=settings,
        )
        contradiction_detector = ContradictionDetector(llm_gateway=llm_gateway)

        # Initialize multimodal pipeline
        multimodal_pipeline = None
        if settings.multimodal_enabled:
            try:
                colpali_processor = None
                clip_embedder = None

                if settings.colpali_enabled:
                    try:
                        colpali_processor = create_colpali_processor(
                            model_name=settings.colpali_model,
                            device=settings.device,
                        )
                        logger.info("ColPali processor created", model=settings.colpali_model)
                    except Exception as e:
                        logger.warning("ColPali not available", error=str(e))

                if settings.clip_enabled:
                    try:
                        clip_embedder = create_clip_embedder(
                            model_name=settings.clip_model,
                            device=settings.device,
                        )
                        logger.info("CLIP embedder created", model=settings.clip_model)
                    except Exception as e:
                        logger.warning("CLIP not available", error=str(e))

                if colpali_processor or clip_embedder:
                    multimodal_pipeline = MultimodalIngestionPipeline(
                        qdrant=qdrant,
                        surrealdb=surrealdb,
                        colpali_processor=colpali_processor,
                        clip_embedder=clip_embedder,
                        use_colpali=settings.colpali_enabled and colpali_processor is not None,
                        collection_name=settings.multimodal_collection,
                    )
                    await multimodal_pipeline.initialize()
                    logger.info("Multimodal pipeline initialized")
                else:
                    logger.warning("No multimodal models available, multimodal disabled")
            except Exception as e:
                logger.warning("Multimodal initialization failed", error=str(e))

        # Store in app state
        app.state.surrealdb = surrealdb
        app.state.qdrant = qdrant
        app.state.oxigraph = oxigraph
        app.state.embedding = embedding_service
        app.state.llm = llm_gateway
        app.state.ingestion = ingestion_pipeline
        app.state.retrieval = retrieval_pipeline
        app.state.generation = generation_pipeline

        # New services
        app.state.jwt_auth = jwt_auth
        app.state.feedback_collector = feedback_collector
        app.state.feedback_learner = feedback_learner
        app.state.learning_scheduler = learning_scheduler
        app.state.cag_cache = cag_cache
        app.state.semantic_cache = semantic_cache
        app.state.cache_prewarmer = cache_prewarmer
        app.state.hallucination_detector = hallucination_detector
        app.state.contradiction_detector = contradiction_detector
        app.state.multimodal = multimodal_pipeline

        # Start background learning scheduler
        await learning_scheduler.start()

        # Prewarm cache in background
        asyncio.create_task(cache_prewarmer.prewarm(max_entries=50))

        logger.info("All services initialized successfully")

    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        raise

    yield

    # Cleanup
    logger.info("Shutting down Perfect RAG API...")
    await app.state.learning_scheduler.stop()
    await app.state.surrealdb.disconnect()
    await app.state.qdrant.disconnect()
    await app.state.oxigraph.disconnect()


app = FastAPI(
    title="Perfect RAG API",
    description="OpenAI-compatible RAG API with GraphRAG, hybrid search, and multi-provider LLM support",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
app.add_middleware(
    RateLimitMiddleware,
    exclude_paths=["/health", "/v1/models", "/docs", "/openapi.json"],
)


# =============================================================================
# Health & Info Endpoints
# =============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    checks = {}

    try:
        checks["surrealdb"] = await app.state.surrealdb.health_check()
    except Exception:
        checks["surrealdb"] = False

    try:
        checks["qdrant"] = await app.state.qdrant.health_check()
    except Exception:
        checks["qdrant"] = False

    try:
        checks["oxigraph"] = await app.state.oxigraph.health_check()
    except Exception:
        checks["oxigraph"] = False

    try:
        checks["embedding"] = await app.state.embedding.health_check()
    except Exception:
        checks["embedding"] = False

    try:
        checks["llm"] = await app.state.llm.health_check()
    except Exception:
        checks["llm"] = {}

    all_healthy = all(
        v if isinstance(v, bool) else any(v.values())
        for v in checks.values()
    )

    return {
        "status": "healthy" if all_healthy else "degraded",
        "services": checks,
    }


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models (OpenAI-compatible)."""
    models = []

    # Add models from each provider
    llm = app.state.llm

    if "openai" in llm.available_providers:
        for model_id in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]:
            models.append(
                ModelInfo(
                    id=model_id,
                    object="model",
                    created=1700000000,
                    owned_by="openai",
                )
            )

    if "anthropic" in llm.available_providers:
        for model_id in [
            "claude-3.5-sonnet",
            "claude-3.5-haiku",
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
        ]:
            models.append(
                ModelInfo(
                    id=model_id,
                    object="model",
                    created=1700000000,
                    owned_by="anthropic",
                )
            )

    if "ollama" in llm.available_providers:
        # For Ollama, we could list local models
        models.append(
            ModelInfo(
                id="llama3.2",
                object="model",
                created=1700000000,
                owned_by="ollama",
            )
        )

    return ModelsResponse(object="list", data=models)


# =============================================================================
# OpenAI-Compatible Chat Completions
# =============================================================================


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
):
    """OpenAI-compatible chat completions endpoint with RAG.

    Supports:
    - Standard chat completions
    - RAG-enhanced responses with citations
    - Streaming (SSE)
    - Multiple LLM providers (OpenAI, Anthropic, Ollama)
    """
    start_time = time.time()
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    # Extract user from request (for ACL)
    user_roles = _extract_user_roles(http_request)

    # Determine if we should use RAG
    use_rag = request.x_use_rag if request.x_use_rag is not None else True

    try:
        if request.stream:
            return EventSourceResponse(
                _stream_completion(
                    request=request,
                    request_id=request_id,
                    use_rag=use_rag,
                    user_roles=user_roles,
                    start_time=start_time,
                ),
                media_type="text/event-stream",
            )
        else:
            return await _complete(
                request=request,
                request_id=request_id,
                use_rag=use_rag,
                user_roles=user_roles,
                start_time=start_time,
            )

    except Exception as e:
        logger.error("Chat completion failed", error=str(e), request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))


async def _complete(
    request: ChatCompletionRequest,
    request_id: str,
    use_rag: bool,
    user_roles: list[str],
    start_time: float,
) -> ChatCompletionResponse:
    """Non-streaming completion."""
    llm = app.state.llm

    # Convert messages to list of dicts
    messages = [msg.model_dump() for msg in request.messages]

    # RAG augmentation
    citations = []
    confidence = None
    rag_metadata = None

    if use_rag and messages:
        # Get the last user message for retrieval
        user_messages = [m for m in messages if m["role"] == "user"]
        if user_messages:
            query = user_messages[-1]["content"]
            retrieval_query = request.x_retrieval_query or _extract_retrieval_query(query)
            is_mcq = _looks_like_mcq(query)
            rag_min_score = request.x_rag_min_score
            if is_mcq and rag_min_score is None:
                rag_min_score = settings.rag_min_score_mcq

            use_graph = bool(request.x_rag_use_graph)
            use_query_rewriting = (
                request.x_rag_use_query_rewriting
                if request.x_rag_use_query_rewriting is not None
                else (not is_mcq)
            )
            use_context_gate = (
                request.x_rag_use_context_gate
                if request.x_rag_use_context_gate is not None
                else (not is_mcq)
            )

            rag_result = await _retrieve_and_augment(
                query=query,
                messages=messages,
                user_roles=user_roles,
                top_k=request.x_rag_top_k or settings.retrieval_top_k,
                retrieval_query=retrieval_query,
                is_mcq=is_mcq,
                rag_min_score=rag_min_score,
                use_graph=use_graph,
                use_query_rewriting=use_query_rewriting,
                use_context_gate=use_context_gate,
            )
            messages = rag_result["messages"]
            citations = rag_result["citations"]
            confidence = rag_result["confidence"]
            rag_metadata = rag_result["metadata"]

    # Generate response
    response_text = await llm.generate(
        messages=messages,
        model=request.model,
        max_tokens=request.max_tokens or 4096,
        temperature=request.temperature or 0.7,
        stream=False,
    )

    # Calculate usage (estimates)
    input_tokens = sum(len(m.get("content", "")) for m in messages) // 4
    output_tokens = len(response_text) // 4

    elapsed = time.time() - start_time

    return ChatCompletionResponse(
        id=request_id,
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        citations=citations if citations else None,
        confidence=confidence,
        x_rag_metadata=rag_metadata,
        x_latency_ms=elapsed * 1000,
    )


async def _stream_completion(
    request: ChatCompletionRequest,
    request_id: str,
    use_rag: bool,
    user_roles: list[str],
    start_time: float,
) -> AsyncIterator[dict]:
    """Streaming completion with SSE."""
    llm = app.state.llm

    # Convert messages
    messages = [msg.model_dump() for msg in request.messages]

    # RAG augmentation (before streaming)
    citations = []
    rag_metadata = None

    if use_rag and messages:
        user_messages = [m for m in messages if m["role"] == "user"]
        if user_messages:
            query = user_messages[-1]["content"]
            retrieval_query = request.x_retrieval_query or _extract_retrieval_query(query)
            is_mcq = _looks_like_mcq(query)
            rag_min_score = request.x_rag_min_score
            if is_mcq and rag_min_score is None:
                rag_min_score = settings.rag_min_score_mcq

            use_graph = bool(request.x_rag_use_graph)
            use_query_rewriting = (
                request.x_rag_use_query_rewriting
                if request.x_rag_use_query_rewriting is not None
                else (not is_mcq)
            )
            use_context_gate = (
                request.x_rag_use_context_gate
                if request.x_rag_use_context_gate is not None
                else (not is_mcq)
            )

            rag_result = await _retrieve_and_augment(
                query=query,
                messages=messages,
                user_roles=user_roles,
                top_k=request.x_rag_top_k or settings.retrieval_top_k,
                retrieval_query=retrieval_query,
                is_mcq=is_mcq,
                rag_min_score=rag_min_score,
                use_graph=use_graph,
                use_query_rewriting=use_query_rewriting,
                use_context_gate=use_context_gate,
            )
            messages = rag_result["messages"]
            citations = rag_result["citations"]
            rag_metadata = rag_result["metadata"]

    # Send initial chunk with role
    initial_chunk = ChatCompletionStreamResponse(
        id=request_id,
        object="chat.completion.chunk",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionStreamResponseChoice(
                index=0,
                delta=StreamDelta(role="assistant", content=""),
                finish_reason=None,
            )
        ],
    )
    yield {"data": initial_chunk.model_dump_json()}

    # Stream content
    try:
        stream = await llm.generate(
            messages=messages,
            model=request.model,
            max_tokens=request.max_tokens or 4096,
            temperature=request.temperature or 0.7,
            stream=True,
        )

        async for chunk_text in stream:
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                object="chat.completion.chunk",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionStreamResponseChoice(
                        index=0,
                        delta=StreamDelta(content=chunk_text),
                        finish_reason=None,
                    )
                ],
            )
            yield {"data": chunk.model_dump_json()}

    except Exception as e:
        logger.error("Streaming error", error=str(e))
        raise

    # Send final chunk with finish reason
    final_chunk = ChatCompletionStreamResponse(
        id=request_id,
        object="chat.completion.chunk",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionStreamResponseChoice(
                index=0,
                delta=StreamDelta(),
                finish_reason="stop",
            )
        ],
        citations=citations if citations else None,
        x_rag_metadata=rag_metadata,
    )
    yield {"data": final_chunk.model_dump_json()}

    # Send done signal
    yield {"data": "[DONE]"}


def _looks_like_mcq(text: str) -> bool:
    """Heuristic: detect multiple-choice prompts to avoid retrieval pollution."""
    if "opciones de respuesta" in text.lower():
        return True
    return re.search(r"(?m)^\s*[A-E]\)\s+", text) is not None


def _extract_retrieval_query(text: str) -> str:
    """Extract a cleaner retrieval query from a full MCQ prompt."""
    if not _looks_like_mcq(text):
        return text.strip()

    lowered = text.lower()
    split_at = lowered.find("opciones de respuesta")
    if split_at != -1:
        head = text[:split_at]
    else:
        m = re.search(r"(?m)^\s*A\)\s+", text)
        head = text[: m.start()] if m else text

    head = re.sub(r"(?i)^\s*pregunta\s*(m[eé]dica)?\s*:\s*", "", head).strip()
    return head or text.strip()


def _best_dense_score(chunks: list[Any]) -> float:
    """Best-effort extraction of top dense similarity from retrieved chunks."""
    best = 0.0
    for chunk in chunks:
        metadata = getattr(chunk, "metadata", None) or {}
        val = metadata.get("retrieval_dense_score")
        if isinstance(val, (int, float)) and val > best:
            best = float(val)
    return best


async def _retrieve_and_augment(
    query: str,
    messages: list[dict[str, str]],
    user_roles: list[str],
    top_k: int = 10,
    *,
    retrieval_query: str | None = None,
    is_mcq: bool = False,
    rag_min_score: float | None = None,
    use_graph: bool = True,
    use_query_rewriting: bool = True,
    use_context_gate: bool = True,
) -> dict[str, Any]:
    """Retrieve relevant context and augment messages.

    Uses the full retrieval pipeline with:
    - Query rewriting (HyDE, expansion, decomposition)
    - Hybrid search (dense + sparse)
    - GraphRAG expansion
    - Reranking
    """
    retrieval = app.state.retrieval
    generation = app.state.generation

    # Build query context
    context = QueryContext(
        conversation_history=messages,
        user_roles=user_roles,
    )

    retrieval_query = (retrieval_query or query).strip()

    # Execute full retrieval pipeline
    retrieval_result = await retrieval.retrieve(
        query=retrieval_query,
        context=context,
        top_k=top_k,
        use_reranking=True,
        use_graph_expansion=use_graph,
        use_query_rewriting=use_query_rewriting,
        use_context_gate=use_context_gate,
        acl_filter=user_roles,
    )

    # Check if retrieval was needed/successful
    if not retrieval_result.retrieval_needed or not retrieval_result.chunks:
        return {
            "messages": messages,
            "citations": [],
            "confidence": None,
            "metadata": {"sources_found": 0, "retrieval_needed": retrieval_result.retrieval_needed},
        }

    # Quality gate: for MCQ prompts, only inject context if the retrieval signal is strong.
    if is_mcq and rag_min_score is not None:
        top_dense = _best_dense_score(retrieval_result.chunks)
        if top_dense < rag_min_score:
            return {
                "messages": messages,
                "citations": [],
                "confidence": None,
                "metadata": {
                    "sources_found": retrieval_result.total_found,
                    "retrieval_needed": retrieval_result.retrieval_needed,
                    "gated": True,
                    "gate": {
                        "min_dense_score": rag_min_score,
                        "top_dense_score": top_dense,
                        "reason": "low_retrieval_signal",
                    },
                    "retrieval_query": retrieval_query,
                    **retrieval_result.metadata,
                },
            }

    # Build augmented messages using prompt builder
    augmented_messages = generation.prompt_builder.build_rag_prompt(
        messages=messages,
        chunks=retrieval_result.chunks,
        mode="default",
        task="mcq" if is_mcq else None,
    )

    # Build citations
    citations = [
        Citation(
            source_id=chunk.doc_id,
            source_title=chunk.doc_title,
            chunk_index=chunk.chunk_index,
            text_snippet=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
            relevance_score=chunk.score,
        )
        for chunk in retrieval_result.chunks
    ]

    return {
        "messages": augmented_messages,
        "citations": citations,
        "confidence": retrieval_result.confidence,
        "metadata": {
            "sources_found": retrieval_result.total_found,
            "sources_used": len(citations),
            "avg_relevance": retrieval_result.confidence,
            "retrieval_query": retrieval_query,
            "top_dense_score": _best_dense_score(retrieval_result.chunks),
            **retrieval_result.metadata,
        },
    }


def _extract_user_roles(request: Request) -> list[str]:
    """Extract user roles from request headers."""
    # Check for authorization header or custom roles header
    roles_header = request.headers.get("X-User-Roles", "")
    if roles_header:
        return [r.strip() for r in roles_header.split(",") if r.strip()]

    # Default to public access
    return ["*"]


# =============================================================================
# Document Management Endpoints
# =============================================================================


class DocumentUploadRequest(BaseModel):
    """Request body for URL-based document upload."""
    url: str | None = None
    title: str | None = None
    acl: list[str] | None = None
    tags: list[str] | None = None
    extract_graph: bool = True


@app.post("/v1/documents")
async def upload_document(
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
    title: str | None = Form(None),
    acl: str | None = Form(None),  # Comma-separated roles
    tags: str | None = Form(None),  # Comma-separated tags
    extract_graph: bool = Form(True),
    background_tasks: BackgroundTasks = None,
):
    """Upload a document for ingestion.

    Supports:
    - File upload (multipart form)
    - URL ingestion (form field)

    The document will be chunked, embedded, and optionally have
    entities/relations extracted for the knowledge graph.
    """
    ingestion = app.state.ingestion

    if not file and not url:
        raise HTTPException(
            status_code=400,
            detail="Either file or url must be provided",
        )

    # Parse ACL and tags
    acl_list = [r.strip() for r in acl.split(",")] if acl else ["*"]
    tags_list = [t.strip() for t in tags.split(",")] if tags else []

    metadata = {
        "title": title,
        "tags": tags_list,
        "acl": acl_list,
    }

    try:
        if file:
            # Save uploaded file temporarily
            suffix = Path(file.filename).suffix if file.filename else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            # Ingest from temp file
            result = await ingestion.ingest(
                source=tmp_path,
                metadata=metadata,
                acl=acl_list,
                extract_graph=extract_graph,
            )

            # Clean up temp file in background
            background_tasks.add_task(Path(tmp_path).unlink, missing_ok=True)

        else:
            # URL ingestion - fetch and ingest
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()

            # Determine file type from content-type or URL
            content_type = response.headers.get("content-type", "")
            if "html" in content_type:
                suffix = ".html"
            elif "pdf" in content_type:
                suffix = ".pdf"
            elif "json" in content_type:
                suffix = ".json"
            else:
                suffix = Path(url).suffix or ".txt"

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            metadata["source"] = url

            result = await ingestion.ingest(
                source=tmp_path,
                metadata=metadata,
                acl=acl_list,
                extract_graph=extract_graph,
            )

            background_tasks.add_task(Path(tmp_path).unlink, missing_ok=True)

        return result

    except Exception as e:
        logger.error("Document ingestion failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/documents")
async def list_documents(
    limit: int = 100,
    offset: int = 0,
):
    """List all documents."""
    surrealdb = app.state.surrealdb
    docs = await surrealdb.list_documents(limit=limit, offset=offset)
    return {"documents": docs, "total": len(docs)}


@app.get("/v1/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get document details."""
    surrealdb = app.state.surrealdb
    doc = await surrealdb.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.delete("/v1/documents/{doc_id}")
async def delete_document(doc_id: str, background_tasks: BackgroundTasks):
    """Delete a document and its chunks."""
    surrealdb = app.state.surrealdb
    qdrant = app.state.qdrant

    # Delete from SurrealDB
    await surrealdb.delete_document(doc_id)

    # Delete chunks from Qdrant in background
    background_tasks.add_task(qdrant.delete_chunks_by_doc, doc_id)

    return {"status": "deleted", "doc_id": doc_id}


# =============================================================================
# Multimodal Document Endpoints
# =============================================================================


class MultimodalSearchRequest(BaseModel):
    """Multimodal search request body."""
    query: str
    top_k: int = 10
    include_images: bool = True


class MultimodalSearchResult(BaseModel):
    """Single multimodal search result."""
    chunk_id: str
    doc_id: str
    score: float
    page_number: int | None = None
    text_preview: str = ""
    content: str = ""
    images: list[dict[str, Any]] = []


class MultimodalSearchResponse(BaseModel):
    """Multimodal search response."""
    results: list[MultimodalSearchResult]
    total: int
    query: str


@app.post("/v1/multimodal/documents")
async def upload_multimodal_document(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    acl: str | None = Form(None),  # Comma-separated roles
    tags: str | None = Form(None),  # Comma-separated tags
    background_tasks: BackgroundTasks = None,
):
    """Upload a multimodal document (PDF with images, image files).

    Supports:
    - PDF files (processed page-by-page with visual embeddings)
    - Image files (PNG, JPG, JPEG, WebP, GIF)

    Uses ColPali for vision-language retrieval when available,
    falls back to CLIP for image embedding.
    """
    multimodal = app.state.multimodal

    if not multimodal:
        raise HTTPException(
            status_code=503,
            detail="Multimodal processing is not available. Enable multimodal_enabled in settings.",
        )

    # Check file type
    suffix = Path(file.filename).suffix.lower() if file.filename else ""
    supported_formats = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif"}

    if suffix not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {suffix}. Supported formats: {', '.join(supported_formats)}",
        )

    # Check file size
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)

    if file_size_mb > settings.max_image_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {file_size_mb:.1f}MB. Maximum: {settings.max_image_size_mb}MB",
        )

    # Parse ACL and tags
    acl_list = [r.strip() for r in acl.split(",")] if acl else ["*"]
    tags_list = [t.strip() for t in tags.split(",")] if tags else []

    metadata = {
        "title": title or file.filename,
        "tags": tags_list,
        "content_type": "multimodal",
        "file_format": suffix,
        "file_size_mb": file_size_mb,
    }

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Ingest with multimodal pipeline
        result = await multimodal.ingest(
            file_path=tmp_path,
            metadata=metadata,
            acl=acl_list,
        )

        # Clean up temp file in background
        background_tasks.add_task(Path(tmp_path).unlink, missing_ok=True)

        return result

    except Exception as e:
        logger.error("Multimodal document ingestion failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/multimodal/search", response_model=MultimodalSearchResponse)
async def search_multimodal(
    request: MultimodalSearchRequest,
    http_request: Request,
):
    """Search multimodal documents using vision-language retrieval.

    Searches across documents indexed with ColPali or CLIP embeddings,
    enabling semantic search across PDFs, images, and documents with
    visual content.
    """
    multimodal = app.state.multimodal

    if not multimodal:
        raise HTTPException(
            status_code=503,
            detail="Multimodal processing is not available. Enable multimodal_enabled in settings.",
        )

    user_roles = _extract_user_roles(http_request)

    try:
        results = await multimodal.search(
            query=request.query,
            top_k=request.top_k,
            acl_filter=user_roles,
        )

        # Format results
        formatted_results = []
        for r in results:
            # Optionally exclude image data to reduce response size
            images = r.get("images", []) if request.include_images else []

            formatted_results.append(MultimodalSearchResult(
                chunk_id=r["chunk_id"],
                doc_id=r.get("doc_id", ""),
                score=r.get("score", 0.0),
                page_number=r.get("page_number"),
                text_preview=r.get("text_preview", ""),
                content=r.get("content", ""),
                images=images,
            ))

        return MultimodalSearchResponse(
            results=formatted_results,
            total=len(formatted_results),
            query=request.query,
        )

    except Exception as e:
        logger.error("Multimodal search failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/multimodal/documents/{doc_id}")
async def get_multimodal_document(doc_id: str):
    """Get details of a multimodal document including all pages/images."""
    surrealdb = app.state.surrealdb

    # Get document metadata
    doc = await surrealdb.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Get all chunks for this document
    chunks = await surrealdb.get_chunks_by_doc_id(doc_id)

    return {
        "document": doc,
        "chunks": chunks,
        "total_chunks": len(chunks),
    }


@app.get("/v1/multimodal/status")
async def get_multimodal_status():
    """Get multimodal processing status and capabilities."""
    multimodal = app.state.multimodal

    return {
        "enabled": multimodal is not None,
        "colpali_enabled": settings.colpali_enabled,
        "colpali_model": settings.colpali_model if settings.colpali_enabled else None,
        "clip_enabled": settings.clip_enabled,
        "clip_model": settings.clip_model if settings.clip_enabled else None,
        "collection_name": settings.multimodal_collection,
        "ocr_enabled": settings.ocr_enabled,
        "max_image_size_mb": settings.max_image_size_mb,
        "supported_formats": [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif"],
    }


# =============================================================================
# Search Endpoints
# =============================================================================


class SearchRequest(BaseModel):
    """Search request body."""
    query: str
    top_k: int = 10
    use_reranking: bool = True
    use_graph_expansion: bool = False
    use_query_rewriting: bool = False
    metadata_filter: dict[str, Any] | None = None


@app.post("/v1/search")
async def search(
    request: SearchRequest,
    http_request: Request,
):
    """Direct search endpoint (non-chat).

    Returns ranked chunks matching the query with full retrieval pipeline.
    """
    retrieval = app.state.retrieval
    user_roles = _extract_user_roles(http_request)

    # Execute retrieval pipeline
    result = await retrieval.retrieve(
        query=request.query,
        top_k=request.top_k,
        use_reranking=request.use_reranking,
        use_graph_expansion=request.use_graph_expansion,
        use_query_rewriting=request.use_query_rewriting,
        acl_filter=user_roles,
        metadata_filter=request.metadata_filter,
    )

    # Convert to response format
    results = [
        {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "doc_title": chunk.doc_title,
            "content": chunk.content,
            "score": chunk.score,
            "chunk_index": chunk.chunk_index,
            "metadata": chunk.metadata,
        }
        for chunk in result.chunks
    ]

    return {
        "results": results,
        "total": result.total_found,
        "confidence": result.confidence,
        "rewritten_queries": result.rewritten_queries,
        "metadata": result.metadata,
    }


# =============================================================================
# Stats & Admin Endpoints
# =============================================================================


@app.get("/v1/stats")
async def get_stats():
    """Get usage statistics."""
    llm = app.state.llm
    return llm.get_usage_stats()


@app.get("/v1/stats/embedding")
async def get_embedding_stats():
    """Get embedding service info."""
    embedding = app.state.embedding
    return embedding.get_info()


# =============================================================================
# Graph Endpoints
# =============================================================================


@app.get("/v1/graph/entity/{entity_id}")
async def get_entity_neighborhood(entity_id: str, hops: int = 1):
    """Get entity and its neighborhood from knowledge graph."""
    oxigraph = app.state.oxigraph
    neighborhood = await oxigraph.get_entity_neighborhood(entity_id, max_hops=hops)
    return neighborhood


@app.get("/v1/graph/path")
async def find_path(start_id: str, end_id: str, max_length: int = 3):
    """Find paths between two entities."""
    oxigraph = app.state.oxigraph
    paths = await oxigraph.find_paths(start_id, end_id, max_length=max_length)
    return {"paths": paths}


@app.get("/v1/graph/entities")
async def list_entities_by_type(entity_type: str, limit: int = 100):
    """List entities by type."""
    oxigraph = app.state.oxigraph
    entities = await oxigraph.get_entities_by_type(entity_type, limit=limit)
    return {"entities": entities}


# =============================================================================
# Authentication Endpoints
# =============================================================================


class LoginRequest(BaseModel):
    """Login request body."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class APIKeyRequest(BaseModel):
    """API key creation request."""
    name: str
    scopes: list[str] | None = None
    expires_days: int | None = None


@app.post("/v1/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Authenticate and get access token."""
    jwt_auth = app.state.jwt_auth

    # Verify credentials (simplified - in production use proper password hashing)
    user = await jwt_auth.authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Generate token
    token = jwt_auth.create_access_token(user)

    return TokenResponse(
        access_token=token,
        expires_in=settings.access_token_expire_minutes * 60,
    )


@app.post("/v1/auth/api-keys")
async def create_api_key(
    request: APIKeyRequest,
    current_user: User = Depends(get_current_user),
):
    """Create a new API key for the authenticated user."""
    jwt_auth = app.state.jwt_auth

    api_key, key_id = await jwt_auth.create_api_key(
        user_id=current_user.id,
        name=request.name,
        scopes=request.scopes,
        expires_days=request.expires_days,
    )

    return {
        "api_key": api_key,  # Only shown once
        "key_id": key_id,
        "name": request.name,
    }


@app.delete("/v1/auth/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user),
):
    """Revoke an API key."""
    jwt_auth = app.state.jwt_auth
    await jwt_auth.revoke_api_key(key_id)
    return {"status": "revoked", "key_id": key_id}


@app.get("/v1/auth/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return current_user.model_dump(exclude={"hashed_password"})


# =============================================================================
# Feedback Endpoints
# =============================================================================


class FeedbackRequest(BaseModel):
    """Feedback submission request."""
    query_id: str
    feedback_type: str  # thumbs_up, thumbs_down, citation_click, correction
    chunk_id: str | None = None
    comment: str | None = None
    correction: str | None = None


@app.post("/v1/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    http_request: Request,
    current_user: User | None = Depends(get_optional_user),
):
    """Submit feedback for a query response."""
    collector = app.state.feedback_collector

    user_id = current_user.id if current_user else None

    result = await collector.submit_feedback(
        query_id=request.query_id,
        feedback_type=request.feedback_type,
        user_id=user_id,
        chunk_id=request.chunk_id,
        comment=request.comment,
        correction=request.correction,
    )

    return result


@app.get("/v1/feedback/stats")
async def get_feedback_stats(
    current_user: User = Depends(get_current_user),
):
    """Get feedback statistics (requires authentication)."""
    collector = app.state.feedback_collector
    return await collector.get_feedback_stats()


@app.post("/v1/feedback/learn")
async def trigger_learning(
    current_user: User = Depends(get_current_user),
):
    """Trigger a learning cycle from feedback (admin only)."""
    if "admin" not in current_user.roles:
        raise HTTPException(status_code=403, detail="Admin role required")

    learner = app.state.feedback_learner
    result = await learner.learn_from_feedback()
    return result


# =============================================================================
# Cache Endpoints
# =============================================================================


@app.get("/v1/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    cag_cache = app.state.cag_cache
    semantic_cache = app.state.semantic_cache

    return {
        "cag_cache": await cag_cache.get_stats(),
        "semantic_cache": await semantic_cache.get_stats(),
    }


@app.post("/v1/cache/prewarm")
async def prewarm_cache(
    strategies: list[str] | None = None,
    max_entries: int = 100,
    current_user: User = Depends(get_current_user),
):
    """Trigger cache prewarming (requires authentication)."""
    prewarmer = app.state.cache_prewarmer
    result = await prewarmer.prewarm(strategies=strategies, max_entries=max_entries)
    return result


@app.delete("/v1/cache/clear")
async def clear_cache(
    current_user: User = Depends(get_current_user),
):
    """Clear all cache entries (admin only)."""
    if "admin" not in current_user.roles:
        raise HTTPException(status_code=403, detail="Admin role required")

    cag_cache = app.state.cag_cache
    semantic_cache = app.state.semantic_cache

    cag_cleared = await cag_cache.get_stats()  # Get count before clear
    await semantic_cache.clear()

    return {
        "status": "cleared",
        "cag_entries_cleared": cag_cleared.get("entries", 0),
    }


# =============================================================================
# Embeddings Endpoint
# =============================================================================


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""
    input: str | list[str]
    model: str = "BAAI/bge-m3"
    encoding_format: str = "float"


class EmbeddingData(BaseModel):
    """Single embedding result."""
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response."""
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: dict[str, int]


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings for text (OpenAI-compatible).

    Supports both single strings and lists of strings.
    """
    embedding_service = app.state.embedding

    # Normalize input to list
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input

    # Generate embeddings
    embeddings = await embedding_service.embed_documents(texts)

    # Build response
    data = [
        EmbeddingData(
            embedding=emb,
            index=i,
        )
        for i, emb in enumerate(embeddings)
    ]

    # Estimate tokens
    total_tokens = sum(len(text) // 4 for text in texts)

    return EmbeddingResponse(
        data=data,
        model=settings.embedding_model,
        usage={
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    )


# =============================================================================
# Resilience Endpoints
# =============================================================================


@app.get("/v1/resilience/circuit-breakers")
async def get_circuit_breaker_stats(
    current_user: User = Depends(get_current_user),
):
    """Get circuit breaker statistics."""
    registry = get_circuit_breaker_registry()
    return registry.get_all_stats()


@app.post("/v1/resilience/circuit-breakers/reset")
async def reset_circuit_breakers(
    current_user: User = Depends(get_current_user),
):
    """Reset all circuit breakers (admin only)."""
    if "admin" not in current_user.roles:
        raise HTTPException(status_code=403, detail="Admin role required")

    registry = get_circuit_breaker_registry()
    await registry.reset_all()
    return {"status": "reset"}


# =============================================================================
# Hallucination Detection Endpoints
# =============================================================================


class HallucinationCheckRequest(BaseModel):
    """Request to check for hallucinations in generated text."""
    generated_text: str
    source_chunk_ids: list[str]
    strict: bool = False


@app.post("/v1/verify/hallucination")
async def check_hallucinations(
    request: HallucinationCheckRequest,
    current_user: User | None = Depends(get_optional_user),
):
    """Check generated text for hallucinations against source chunks."""
    detector = app.state.hallucination_detector
    surrealdb = app.state.surrealdb

    # Fetch source chunks
    from perfect_rag.models.query import SourceChunk

    chunks_data = await surrealdb.get_chunks_by_ids(request.source_chunk_ids)
    source_chunks = [
        SourceChunk(
            chunk_id=c["id"],
            doc_id=c.get("doc_id", ""),
            doc_title=c.get("metadata", {}).get("title", ""),
            content=c.get("content", ""),
            score=1.0,
            chunk_index=c.get("chunk_index", 0),
        )
        for c in chunks_data
    ]

    result = await detector.detect(
        generated_text=request.generated_text,
        source_chunks=source_chunks,
        strict=request.strict,
    )

    return result


# =============================================================================
# Run with Uvicorn
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "perfect_rag.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=1,
    )
