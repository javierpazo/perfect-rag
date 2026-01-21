"""Perfect RAG - Production-ready RAG system with GraphRAG and OpenAI-compatible API."""

__version__ = "0.1.0"

# Evaluation metrics
from perfect_rag.evaluation import RAGASResult, RAGASEvaluator, RetrievalMetrics

__all__ = [
    "__version__",
    "RAGASResult",
    "RAGASEvaluator",
    "RetrievalMetrics",
]
