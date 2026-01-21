"""Retrieval pipeline components."""

from perfect_rag.retrieval.pipeline import RetrievalPipeline, get_retrieval_pipeline
from perfect_rag.retrieval.query_rewriter import QueryRewriter
from perfect_rag.retrieval.graphrag import GraphRAGExpander
from perfect_rag.retrieval.colbert_reranker import (
    ColBERTReranker,
    ColBERTIndexManager,
    get_colbert_reranker,
    get_colbert_index_manager,
)
from perfect_rag.retrieval.llm_reranker import (
    LLMReranker,
    RankGPTReranker,
    HybridLLMReranker,
    RerankStrategy,
    RerankResult,
    get_llm_reranker,
    get_rankgpt_reranker,
)
from perfect_rag.retrieval.agentic_retrieval import (
    AgenticRetriever,
    AgenticRetrievalResult,
    RetrievalDecision,
    RelevanceGrade,
    RetrievalStep,
    QueryDecomposer,
    MultiHopRetriever,
    SelfRAGReflector,
    CorrectiveRAGOrchestrator,
    SupportGrade,
    UtilityGrade,
    create_agentic_retriever,
    create_multi_hop_retriever,
    create_crag_orchestrator,
)
from perfect_rag.retrieval.temporal import (
    TemporalQueryParser,
    TemporalFilter,
    TemporalRanker,
    TemporalRetrievalEnhancer,
    TemporalExpression,
    TemporalParseResult,
)
from perfect_rag.retrieval.federated import (
    FederatedRetriever,
    SourceRegistry,
    ResultAggregator,
    SourceRouter,
    SourceConfig,
    SourceHealth,
    SourceStatus,
    FederatedResult,
)

__all__ = [
    # Core pipeline
    "RetrievalPipeline",
    "get_retrieval_pipeline",
    "QueryRewriter",
    "GraphRAGExpander",
    # ColBERT late interaction
    "ColBERTReranker",
    "ColBERTIndexManager",
    "get_colbert_reranker",
    "get_colbert_index_manager",
    # LLM-based reranking
    "LLMReranker",
    "RankGPTReranker",
    "HybridLLMReranker",
    "RerankStrategy",
    "RerankResult",
    "get_llm_reranker",
    "get_rankgpt_reranker",
    # Agentic retrieval
    "AgenticRetriever",
    "AgenticRetrievalResult",
    "RetrievalDecision",
    "RelevanceGrade",
    "RetrievalStep",
    "QueryDecomposer",
    "MultiHopRetriever",
    # Self-RAG reflection
    "SelfRAGReflector",
    "SupportGrade",
    "UtilityGrade",
    # Corrective RAG
    "CorrectiveRAGOrchestrator",
    # Factory functions
    "create_agentic_retriever",
    "create_multi_hop_retriever",
    "create_crag_orchestrator",
    # Temporal reasoning
    "TemporalQueryParser",
    "TemporalFilter",
    "TemporalRanker",
    "TemporalRetrievalEnhancer",
    "TemporalExpression",
    "TemporalParseResult",
    # Federated RAG
    "FederatedRetriever",
    "SourceRegistry",
    "ResultAggregator",
    "SourceRouter",
    "SourceConfig",
    "SourceHealth",
    "SourceStatus",
    "FederatedResult",
]
