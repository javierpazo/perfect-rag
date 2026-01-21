"""Document ingestion pipeline."""

from perfect_rag.ingestion.loaders import DocumentLoader, load_document
from perfect_rag.ingestion.chunker import (
    Chunker,
    SemanticChunker,
    FixedSizeChunker,
    SentenceChunker,
    ParagraphChunker,
    RecursiveChunker,
    get_chunker,
)
from perfect_rag.ingestion.late_chunker import (
    LateChunk,
    LateChunker,
    SimpleLateChunker,
    HybridChunker,
    get_late_chunker,
    create_hybrid_chunker,
)
from perfect_rag.ingestion.multimodal import (
    MultimodalChunk,
    ColPaliProcessor,
    CLIPEmbedder,
    MultimodalLoader,
    MultimodalIngestionPipeline,
    create_colpali_processor,
    create_clip_embedder,
)
from perfect_rag.ingestion.pipeline import IngestionPipeline

__all__ = [
    # Loaders
    "DocumentLoader",
    "load_document",
    # Traditional chunkers
    "Chunker",
    "SemanticChunker",
    "FixedSizeChunker",
    "SentenceChunker",
    "ParagraphChunker",
    "RecursiveChunker",
    "get_chunker",
    # Late chunking
    "LateChunk",
    "LateChunker",
    "SimpleLateChunker",
    "HybridChunker",
    "get_late_chunker",
    "create_hybrid_chunker",
    # Multimodal
    "MultimodalChunk",
    "ColPaliProcessor",
    "CLIPEmbedder",
    "MultimodalLoader",
    "MultimodalIngestionPipeline",
    "create_colpali_processor",
    "create_clip_embedder",
    # Pipeline
    "IngestionPipeline",
]
