"""Document ingestion pipeline orchestrator."""

import asyncio
from pathlib import Path
from typing import Any, BinaryIO

import structlog

from perfect_rag.config import Settings, get_settings
from perfect_rag.core.embedding import EmbeddingService
from perfect_rag.db.oxigraph import OxigraphClient
from perfect_rag.db.qdrant import QdrantVectorClient
from perfect_rag.db.surrealdb import SurrealDBClient
from perfect_rag.ingestion.chunker import Chunker, RecursiveChunker, SemanticChunker
from perfect_rag.ingestion.extractor import (
    EntityExtractor,
    GraphBuilder,
    RelationExtractor,
)
from perfect_rag.ingestion.late_chunker import (
    LateChunker,
    SimpleLateChunker,
    HybridChunker,
    LateChunk,
    get_late_chunker,
    create_hybrid_chunker,
)
from perfect_rag.ingestion.loaders import DocumentLoader, load_document
from perfect_rag.models.chunk import Chunk
from perfect_rag.models.document import Document, DocumentStatus
from perfect_rag.models.entity import Entity
from perfect_rag.models.relation import Relation

logger = structlog.get_logger(__name__)


class IngestionPipeline:
    """Complete document ingestion pipeline.

    Pipeline steps:
    1. Load document (various formats)
    2. Chunk text (recursive/semantic)
    3. Generate embeddings (dense + sparse)
    4. Extract entities (NER + LLM)
    5. Extract relations (patterns + LLM)
    6. Store in databases (SurrealDB, Qdrant, Oxigraph)
    """

    def __init__(
        self,
        surrealdb: SurrealDBClient,
        qdrant: QdrantVectorClient,
        oxigraph: OxigraphClient,
        embedding_service: EmbeddingService,
        llm_gateway: Any = None,
        settings: Settings | None = None,
    ):
        self.surrealdb = surrealdb
        self.qdrant = qdrant
        self.oxigraph = oxigraph
        self.embedding = embedding_service
        self.llm_gateway = llm_gateway
        self.settings = settings or get_settings()

        # Initialize chunkers
        self.chunker: Chunker = RecursiveChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        # Initialize late chunker if enabled
        self.late_chunker: SimpleLateChunker | LateChunker | None = None
        self.hybrid_chunker: HybridChunker | None = None
        self._late_chunking_initialized = False

        # Initialize extractors
        self.entity_extractor = EntityExtractor(
            settings=self.settings,
            use_spacy=True,
            use_llm=bool(llm_gateway),
            llm_gateway=llm_gateway,
        )
        self.relation_extractor = RelationExtractor(
            settings=self.settings,
            use_patterns=True,
            use_llm=bool(llm_gateway),
            llm_gateway=llm_gateway,
        )
        self.graph_builder = GraphBuilder(
            entity_extractor=self.entity_extractor,
            relation_extractor=self.relation_extractor,
        )

    async def initialize(self) -> None:
        """Initialize pipeline components."""
        await self.entity_extractor.initialize()

        # Initialize late chunking if enabled
        if self.settings.late_chunking_enabled and not self._late_chunking_initialized:
            await self._initialize_late_chunking()

        logger.info("Ingestion pipeline initialized")

    async def _initialize_late_chunking(self) -> None:
        """Initialize late chunking components."""
        try:
            if self.settings.late_chunking_strategy == "simple":
                # Use SimpleLateChunker with existing embedding service
                self.late_chunker = SimpleLateChunker(
                    embedding_service=self.embedding,
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                    context_window=self.settings.late_chunking_context_window,
                    settings=self.settings,
                )
            else:
                # Use full LateChunker with transformers
                self.late_chunker = LateChunker(
                    embedding_model=self.settings.late_chunking_model,
                    max_tokens=self.settings.late_chunking_max_tokens,
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                    pooling_strategy=self.settings.late_chunking_pooling,
                    settings=self.settings,
                )
                await self.late_chunker.initialize()

            # Create hybrid chunker
            self.hybrid_chunker = HybridChunker(
                late_chunker=self.late_chunker,
                traditional_chunker=self.chunker,
                late_chunk_threshold=self.settings.late_chunking_threshold,
            )

            self._late_chunking_initialized = True
            logger.info(
                "Late chunking initialized",
                strategy=self.settings.late_chunking_strategy,
                threshold=self.settings.late_chunking_threshold,
            )
        except Exception as e:
            logger.warning(
                "Failed to initialize late chunking, falling back to traditional",
                error=str(e),
            )
            self.late_chunker = None
            self.hybrid_chunker = None

    async def ingest(
        self,
        source: str | Path | BinaryIO,
        metadata: dict[str, Any] | None = None,
        acl: list[str] | None = None,
        extract_graph: bool = True,
        chunking_strategy: str = "recursive",
    ) -> dict[str, Any]:
        """Ingest a document through the full pipeline.

        Args:
            source: File path, URL, or file-like object
            metadata: Optional metadata for the document
            acl: Access control list (roles that can access)
            extract_graph: Whether to extract entities/relations
            chunking_strategy: Chunking strategy to use

        Returns:
            Ingestion result with document ID, chunk count, entity count
        """
        logger.info("Starting document ingestion", source=str(source))

        # Step 1: Load document
        doc = await load_document(source, metadata)
        doc.acl = acl or ["*"]
        doc.status = DocumentStatus.PROCESSING

        # Store document metadata
        await self.surrealdb.create_document(doc)
        logger.info("Document loaded", doc_id=doc.id, title=doc.metadata.title)

        try:
            # Step 2: Chunk document
            chunks = await self._chunk_document(doc, chunking_strategy)
            logger.info("Document chunked", doc_id=doc.id, chunk_count=len(chunks))

            # Step 3: Generate embeddings
            chunks_with_embeddings = await self._embed_chunks(chunks)
            logger.info("Embeddings generated", doc_id=doc.id)

            # Step 4: Store chunks in SurrealDB and Qdrant
            await self._store_chunks(doc, chunks_with_embeddings)
            logger.info("Chunks stored", doc_id=doc.id)

            # Step 5: Extract and store graph (optional)
            entities = []
            relations = []
            if extract_graph:
                entities, relations = await self._extract_graph(doc, chunks)
                await self._store_graph(entities, relations)
                logger.info(
                    "Graph extracted",
                    doc_id=doc.id,
                    entities=len(entities),
                    relations=len(relations),
                )

            # Update document status
            doc.status = DocumentStatus.INDEXED
            doc.chunk_count = len(chunks)
            doc.entity_count = len(entities)
            await self.surrealdb.update_document(doc)

            result = {
                "doc_id": doc.id,
                "status": "success",
                "chunks": len(chunks),
                "entities": len(entities),
                "relations": len(relations),
            }
            logger.info("Document ingestion complete", **result)
            return result

        except Exception as e:
            # Mark document as failed
            doc.status = DocumentStatus.FAILED
            await self.surrealdb.update_document(doc)
            logger.error("Document ingestion failed", doc_id=doc.id, error=str(e))
            raise

    async def _chunk_document(
        self,
        doc: Document,
        strategy: str = "recursive",
    ) -> list[Chunk]:
        """Chunk document content."""
        metadata = {
            "doc_title": doc.metadata.title,
            "doc_source": doc.metadata.source,
        }

        # Late chunking strategies
        if strategy == "late" and self.late_chunker is not None:
            # Use late chunking directly
            chunks, embeddings = await self.late_chunker.chunk_and_convert(
                text=doc.content,
                doc_id=doc.id,
                metadata=metadata,
            )
            # Store embeddings for later use (avoid re-embedding)
            self._late_chunk_embeddings = {
                chunk.id: emb for chunk, emb in zip(chunks, embeddings)
            }
            return chunks

        if strategy == "hybrid" and self.hybrid_chunker is not None:
            # Use hybrid chunking (auto-selects based on document length)
            result = await self.hybrid_chunker.chunk(
                text=doc.content,
                doc_id=doc.id,
                metadata=metadata,
            )
            # Handle both LateChunk and Chunk results
            if result and isinstance(result[0], LateChunk):
                chunks = []
                self._late_chunk_embeddings = {}
                for idx, late_chunk in enumerate(result):
                    chunk = late_chunk.to_chunk(doc.id, idx)
                    chunks.append(chunk)
                    self._late_chunk_embeddings[chunk.id] = late_chunk.embedding
                return chunks
            return result

        if strategy == "semantic" and isinstance(self.chunker, SemanticChunker):
            # Use async semantic chunking
            return await self.chunker.chunk_async(
                text=doc.content,
                doc_id=doc.id,
                metadata=metadata,
            )
        else:
            # Sync chunking
            return self.chunker.chunk(
                text=doc.content,
                doc_id=doc.id,
                metadata=metadata,
            )

    async def _embed_chunks(
        self,
        chunks: list[Chunk],
    ) -> list[dict[str, Any]]:
        """Generate embeddings for chunks.

        If late chunking was used, embeddings are already computed.
        Otherwise, generate them using the embedding service.
        """
        results = []

        # Check if we have pre-computed embeddings from late chunking
        late_embeddings = getattr(self, "_late_chunk_embeddings", {})

        # Separate chunks that need embedding from those that don't
        chunks_to_embed = []
        chunks_with_late_emb = []

        for chunk in chunks:
            if chunk.id in late_embeddings:
                chunks_with_late_emb.append(chunk)
            else:
                chunks_to_embed.append(chunk)

        # Generate embeddings for chunks that need them
        if chunks_to_embed:
            texts = [chunk.content for chunk in chunks_to_embed]
            embeddings = await self.embedding.embed_hybrid_batch(texts)

            for chunk, (dense, sparse) in zip(chunks_to_embed, embeddings):
                results.append({
                    "chunk": chunk,
                    "dense_vector": dense,
                    "sparse_vector": sparse,
                })

        # Use pre-computed embeddings for late-chunked content
        for chunk in chunks_with_late_emb:
            dense = late_embeddings[chunk.id]
            # Generate sparse embedding for hybrid search
            sparse = await self.embedding.embed_sparse(chunk.content)
            results.append({
                "chunk": chunk,
                "dense_vector": dense,
                "sparse_vector": sparse,
            })

        # Clear the temporary storage
        if late_embeddings:
            self._late_chunk_embeddings = {}

        # Sort results to maintain chunk order
        chunk_id_to_idx = {chunk.id: idx for idx, chunk in enumerate(chunks)}
        results.sort(key=lambda x: chunk_id_to_idx.get(x["chunk"].id, 0))

        return results

    async def _store_chunks(
        self,
        doc: Document,
        chunks_with_embeddings: list[dict[str, Any]],
    ) -> None:
        """Store chunks in SurrealDB and Qdrant."""
        # Prepare batch for Qdrant
        qdrant_points = []

        for item in chunks_with_embeddings:
            chunk: Chunk = item["chunk"]
            dense = item["dense_vector"]
            sparse = item["sparse_vector"]

            # Store in SurrealDB
            await self.surrealdb.create_chunk(chunk)

            # Prepare Qdrant point
            qdrant_points.append({
                "id": chunk.id,
                "dense_vector": dense,
                "sparse_vector": sparse,
                "payload": {
                    "doc_id": doc.id,
                    "doc_title": doc.metadata.title,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.content,
                    "acl": doc.acl,
                    "metadata": chunk.metadata,
                },
            })

        # Batch upsert to Qdrant
        await self.qdrant.upsert_chunks_batch(qdrant_points)

    async def _extract_graph(
        self,
        doc: Document,
        chunks: list[Chunk],
    ) -> tuple[list[Entity], list[Relation]]:
        """Extract entities and relations from chunks."""
        chunk_dicts = [{"id": c.id, "content": c.content} for c in chunks]
        return await self.graph_builder.build_from_chunks(chunk_dicts, doc.id)

    async def _store_graph(
        self,
        entities: list[Entity],
        relations: list[Relation],
    ) -> None:
        """Store entities and relations in SurrealDB and Oxigraph."""
        # Store entities
        for entity in entities:
            await self.surrealdb.create_entity(entity)

        # Store relations in SurrealDB (graph edges)
        for relation in relations:
            await self.surrealdb.create_relation(relation)

        # Sync to RDF store (Oxigraph)
        await self.oxigraph.sync_entities_batch(entities)
        await self.oxigraph.sync_relations_batch(relations)

    async def ingest_batch(
        self,
        sources: list[str | Path | BinaryIO],
        metadata: dict[str, Any] | None = None,
        acl: list[str] | None = None,
        concurrency: int = 3,
    ) -> list[dict[str, Any]]:
        """Ingest multiple documents concurrently.

        Args:
            sources: List of file paths or file objects
            metadata: Optional shared metadata
            acl: Optional shared ACL
            concurrency: Max concurrent ingestions

        Returns:
            List of ingestion results
        """
        semaphore = asyncio.Semaphore(concurrency)
        results = []

        async def process_one(source):
            async with semaphore:
                try:
                    return await self.ingest(source, metadata, acl)
                except Exception as e:
                    return {
                        "source": str(source),
                        "status": "failed",
                        "error": str(e),
                    }

        tasks = [process_one(source) for source in sources]
        results = await asyncio.gather(*tasks)

        return results

    async def reindex_document(self, doc_id: str) -> dict[str, Any]:
        """Re-index an existing document.

        Useful for updating embeddings or re-extracting graph.
        """
        # Get document
        doc_data = await self.surrealdb.get_document(doc_id)
        if not doc_data:
            raise ValueError(f"Document not found: {doc_id}")

        # Delete existing chunks from Qdrant
        await self.qdrant.delete_chunks_by_doc(doc_id)

        # Re-process
        doc = Document(**doc_data)
        chunks = await self._chunk_document(doc)
        chunks_with_embeddings = await self._embed_chunks(chunks)
        await self._store_chunks(doc, chunks_with_embeddings)

        # Re-extract graph
        entities, relations = await self._extract_graph(doc, chunks)
        await self._store_graph(entities, relations)

        # Update document
        doc.chunk_count = len(chunks)
        doc.entity_count = len(entities)
        await self.surrealdb.update_document(doc)

        return {
            "doc_id": doc_id,
            "status": "reindexed",
            "chunks": len(chunks),
            "entities": len(entities),
            "relations": len(relations),
        }


# =============================================================================
# Factory Function
# =============================================================================

_pipeline: IngestionPipeline | None = None


async def get_ingestion_pipeline(
    surrealdb: SurrealDBClient,
    qdrant: QdrantVectorClient,
    oxigraph: OxigraphClient,
    embedding_service: EmbeddingService,
    llm_gateway: Any = None,
) -> IngestionPipeline:
    """Get or create ingestion pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline(
            surrealdb=surrealdb,
            qdrant=qdrant,
            oxigraph=oxigraph,
            embedding_service=embedding_service,
            llm_gateway=llm_gateway,
        )
        await _pipeline.initialize()
    return _pipeline
