"""Configuration management using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # Database Connections
    # =========================================================================
    surrealdb_url: str = Field(
        default="ws://localhost:8529",
        description="SurrealDB WebSocket URL",
    )
    surrealdb_user: str = Field(default="root")
    surrealdb_pass: str = Field(default="root")
    surrealdb_namespace: str = Field(default="rag")
    surrealdb_database: str = Field(default="main")

    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant HTTP URL",
    )
    qdrant_api_key: str | None = Field(default=None)
    qdrant_chunks_collection: str = Field(
        default="chunks",
        description="Qdrant collection name for chunks (can use 'ai2evolve_migrated' for migrated data)",
    )
    qdrant_entities_collection: str = Field(
        default="entities",
        description="Qdrant collection name for entities",
    )

    oxigraph_url: str = Field(
        default="http://localhost:7878",
        description="Oxigraph SPARQL endpoint URL",
    )

    # =========================================================================
    # Graph Store Settings
    # =========================================================================
    use_oxigraph: bool = Field(
        default=False,
        description="Use Oxigraph for SPARQL reasoning (optional, SurrealDB is primary)",
    )
    graph_store_type: str = Field(
        default="surrealdb",
        description="Primary graph store: 'surrealdb' (default) or 'hybrid' (SurrealDB + Oxigraph)",
    )

    # =========================================================================
    # LLM Providers
    # =========================================================================
    openai_api_key: str | None = Field(default=None)
    openai_org_id: str | None = Field(default=None)
    openai_base_url: str | None = Field(default=None)

    anthropic_api_key: str | None = Field(default=None)

    ollama_url: str | None = Field(default=None)

    default_llm_provider: Literal["openai", "anthropic", "ollama"] = Field(
        default="openai",
        description="Default LLM provider to use",
    )
    default_llm_model: str = Field(
        default="gpt-4o-mini",
        description="Default model for generation",
    )

    # =========================================================================
    # Model Settings
    # =========================================================================
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="Sentence transformer model for embeddings",
    )
    embedding_dimension: int = Field(
        default=1024,
        description="Embedding vector dimension (1024 for BGE-M3)",
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation",
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Cross-encoder model for reranking",
    )
    device: Literal["cpu", "cuda", "mps", "auto"] = Field(
        default="auto",
        description="Device for ML models",
    )

    # =========================================================================
    # RAG Settings
    # =========================================================================
    chunk_size: int = Field(
        default=512,
        ge=100,
        le=8000,
        description="Target chunk size in tokens",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap between chunks in tokens",
    )
    top_k_retrieval: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of chunks to retrieve before reranking",
    )
    top_k_rerank: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of chunks to keep after reranking",
    )
    max_context_tokens: int = Field(
        default=8000,
        ge=1000,
        le=128000,
        description="Maximum tokens in context window",
    )

    # Retrieval quality gate (useful for multiple-choice prompts)
    rag_min_score_mcq: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum dense similarity to inject context for multiple-choice prompts",
    )

    # GraphRAG settings
    graph_max_hops: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum hops for graph expansion",
    )
    graph_min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for graph relations",
    )

    # =========================================================================
    # Late Chunking Settings
    # =========================================================================
    late_chunking_enabled: bool = Field(
        default=False,
        description="Enable late chunking for improved semantic preservation",
    )
    late_chunking_model: str = Field(
        default="jinaai/jina-embeddings-v3",
        description="Model for late chunking (requires long context support)",
    )
    late_chunking_max_tokens: int = Field(
        default=8192,
        ge=512,
        le=32768,
        description="Maximum tokens for late chunking model",
    )
    late_chunking_pooling: str = Field(
        default="mean",
        description="Pooling strategy for late chunking (mean, max, weighted)",
    )
    late_chunking_threshold: int = Field(
        default=2000,
        ge=500,
        le=50000,
        description="Character threshold for using late chunking (hybrid mode)",
    )
    late_chunking_context_window: int = Field(
        default=200,
        ge=50,
        le=1000,
        description="Context window for simple late chunking",
    )
    late_chunking_strategy: str = Field(
        default="simple",
        description="Late chunking strategy: 'full' (transformers) or 'simple' (existing embeddings)",
    )

    # =========================================================================
    # Multimodal RAG Settings
    # =========================================================================
    multimodal_enabled: bool = Field(
        default=True,
        description="Enable multimodal document processing",
    )
    colpali_model: str = Field(
        default="vidore/colpali-v1.2",
        description="ColPali model for vision-language retrieval",
    )
    colpali_enabled: bool = Field(
        default=True,
        description="Use ColPali for multimodal documents (requires GPU)",
    )
    clip_model: str = Field(
        default="openai/clip-vit-large-patch14",
        description="CLIP model for image/text embedding",
    )
    clip_enabled: bool = Field(
        default=True,
        description="Enable CLIP for image embedding fallback",
    )
    multimodal_collection: str = Field(
        default="multimodal_chunks",
        description="Qdrant collection name for multimodal chunks",
    )
    ocr_enabled: bool = Field(
        default=True,
        description="Enable OCR text extraction for images",
    )
    pdf_dpi: int = Field(
        default=144,
        ge=72,
        le=300,
        description="DPI for PDF page rendering",
    )
    max_image_size_mb: float = Field(
        default=10.0,
        ge=1.0,
        le=100.0,
        description="Maximum image size in MB",
    )
    image_embedding_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for image embedding in combined image-text embeddings",
    )

    # =========================================================================
    # ColBERT Settings
    # =========================================================================
    colbert_enabled: bool = Field(
        default=True,
        description="Enable ColBERT late interaction reranking",
    )
    colbert_model: str = Field(
        default="colbert-ir/colbertv2.0",
        description="ColBERT model for late interaction reranking",
    )
    colbert_rerank_top_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of documents to rerank with ColBERT from initial retrieval",
    )
    colbert_index_path: str | None = Field(
        default=None,
        description="Path for storing ColBERT indexes (default: ~/.cache/colbert_indexes)",
    )
    colbert_use_as_primary: bool = Field(
        default=False,
        description="Use ColBERT as primary retriever instead of just reranker",
    )

    # =========================================================================
    # LLM Reranker Settings
    # =========================================================================
    llm_reranker_enabled: bool = Field(
        default=False,
        description="Enable LLM-based reranking for improved accuracy",
    )
    llm_reranker_strategy: Literal["pointwise", "listwise", "pairwise"] = Field(
        default="pointwise",
        description="LLM reranking strategy (pointwise: fast, listwise: balanced, pairwise: accurate)",
    )
    llm_reranker_top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of documents to rerank with LLM",
    )
    llm_reranker_max_docs_listwise: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum documents for listwise ranking (limited by context window)",
    )
    llm_reranker_max_docs_pairwise: int = Field(
        default=10,
        ge=1,
        le=15,
        description="Maximum documents for pairwise comparison (O(n^2) complexity)",
    )
    rankgpt_enabled: bool = Field(
        default=False,
        description="Use RankGPT sliding window approach instead of basic LLM reranking",
    )
    rankgpt_window_size: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Window size for RankGPT sliding window ranking",
    )
    rankgpt_step_size: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Step size for RankGPT sliding window",
    )
    rankgpt_num_passes: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Number of passes through document list for RankGPT",
    )
    hybrid_reranker_enabled: bool = Field(
        default=False,
        description="Use hybrid cross-encoder + LLM reranking",
    )
    hybrid_reranker_llm_top_k: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Number of top candidates to refine with LLM in hybrid mode",
    )

    # =========================================================================
    # Cache-Augmented Generation (CAG)
    # =========================================================================
    cag_max_entries: int = Field(
        default=1000,
        ge=100,
        le=100000,
        description="Maximum CAG cache entries",
    )
    cag_similarity_threshold: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Minimum similarity for cache hit",
    )
    cag_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=720,
        description="Cache entry TTL in hours",
    )
    cag_max_context_tokens: int = Field(
        default=4000,
        ge=500,
        le=32000,
        description="Maximum tokens from cached context",
    )

    # Semantic cache settings
    semantic_cache_enabled: bool = Field(
        default=True,
        description="Enable semantic query cache",
    )
    semantic_cache_exact_threshold: float = Field(
        default=0.98,
        ge=0.9,
        le=1.0,
        description="Similarity for exact match cache hit",
    )
    semantic_cache_similar_threshold: float = Field(
        default=0.85,
        ge=0.7,
        le=0.97,
        description="Similarity for adapted cache hit",
    )

    # =========================================================================
    # Budget & Limits
    # =========================================================================
    monthly_budget_usd: float = Field(
        default=100.0,
        ge=0,
        description="Monthly budget limit in USD",
    )
    max_tokens_per_request: int = Field(
        default=4096,
        ge=100,
        le=128000,
        description="Maximum tokens per request",
    )
    rate_limit_requests_per_minute: int = Field(
        default=60,
        ge=1,
        description="Rate limit per user per minute",
    )
    rate_limit_rpm: int = Field(
        default=60,
        ge=1,
        description="Rate limit requests per minute",
    )
    rate_limit_tpm: int = Field(
        default=100000,
        ge=1000,
        description="Rate limit tokens per minute",
    )
    rate_limit_global_rpm: int = Field(
        default=1000,
        ge=10,
        description="Global rate limit requests per minute",
    )

    # =========================================================================
    # Security
    # =========================================================================
    secret_key: str = Field(
        default="change-this-to-a-random-secret-key-in-production",
        min_length=32,
        description="Secret key for JWT signing",
    )
    jwt_secret_key: str = Field(
        default="change-this-to-a-random-secret-key-in-production",
        min_length=32,
        description="JWT secret key (alias for secret_key)",
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm",
    )
    jwt_expire_minutes: int = Field(
        default=30,
        ge=5,
        description="JWT token expiration in minutes",
    )
    access_token_expire_minutes: int = Field(
        default=30,
        ge=5,
        description="Access token expiration in minutes",
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="Header name for API key authentication",
    )

    # Default admin credentials (for development)
    admin_username: str | None = Field(
        default="admin",
        description="Default admin username",
    )
    admin_password: str | None = Field(
        default="admin",
        description="Default admin password (change in production!)",
    )

    # =========================================================================
    # Logging
    # =========================================================================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
    )
    log_format: Literal["json", "console"] = Field(
        default="json",
        description="Log output format",
    )

    # =========================================================================
    # Server
    # =========================================================================
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=4, ge=1, le=32)
    debug: bool = Field(default=False)

    # =========================================================================
    # Validators
    # =========================================================================
    @field_validator("device")
    @classmethod
    def resolve_device(cls, v: str) -> str:
        """Resolve 'auto' device to actual device."""
        if v == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
                elif torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"
        return v

    @property
    def has_openai(self) -> bool:
        """Check if OpenAI is configured."""
        return bool(self.openai_api_key and self.openai_api_key.strip())

    @property
    def has_anthropic(self) -> bool:
        """Check if Anthropic is configured."""
        return bool(self.anthropic_api_key and self.anthropic_api_key.strip())

    @property
    def has_ollama(self) -> bool:
        """Check if Ollama is configured."""
        return bool(self.ollama_url and self.ollama_url.strip())

    @property
    def available_providers(self) -> list[str]:
        """List of available LLM providers."""
        providers = []
        if self.has_openai:
            providers.append("openai")
        if self.has_anthropic:
            providers.append("anthropic")
        if self.has_ollama:
            providers.append("ollama")
        return providers

    @property
    def retrieval_top_k(self) -> int:
        """Alias for top_k_retrieval for compatibility."""
        return self.top_k_retrieval

    @property
    def api_host(self) -> str:
        """Alias for host."""
        return self.host

    @property
    def api_port(self) -> int:
        """Alias for port."""
        return self.port


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
