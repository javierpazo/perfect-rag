"""
Multimodal RAG support for Perfect RAG.

Implements:
1. ColPali - Vision-language retrieval for documents as images
2. Image embedding with CLIP
3. OCR fallback for text extraction
4. Unified multimodal indexing
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO
import base64
import structlog

logger = structlog.get_logger()


@dataclass
class MultimodalChunk:
    """A chunk that may contain text, images, or both."""
    chunk_id: str
    content: str  # Text content
    images: list[dict[str, Any]] = field(default_factory=list)  # Image data
    embedding: list[float] = None  # Combined embedding
    image_embeddings: list[list[float]] = field(default_factory=list)
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ColPaliProcessor:
    """
    ColPali: Treat documents as images for vision-language retrieval.

    Instead of OCR + chunking, ColPali:
    1. Renders document pages as images
    2. Uses vision-language model to create embeddings
    3. Retrieves based on visual + textual understanding

    This captures layout, tables, figures that OCR misses.
    """

    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.2",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    async def initialize(self) -> None:
        """Load ColPali model."""
        if self._model is not None:
            return

        try:
            from colpali_engine.models import ColPali
            from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
            import torch

            loop = asyncio.get_event_loop()

            def _load():
                model = ColPali.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                ).to(self.device).eval()

                processor = BaseVisualRetrieverProcessor.from_pretrained(self.model_name)
                return model, processor

            self._model, self._processor = await loop.run_in_executor(None, _load)
            logger.info("ColPali model loaded", model=self.model_name)

        except ImportError:
            logger.warning("ColPali not installed. Install: pip install colpali-engine")
            raise ImportError("Install colpali-engine: pip install colpali-engine")

    async def process_pdf(
        self,
        pdf_path: str | Path,
        doc_id: str,
    ) -> list[MultimodalChunk]:
        """
        Process PDF document using ColPali.

        Args:
            pdf_path: Path to PDF file
            doc_id: Document ID for chunk references

        Returns:
            List of multimodal chunks (one per page)
        """
        await self.initialize()

        from pdf2image import convert_from_path
        import torch

        loop = asyncio.get_event_loop()

        # Convert PDF to images
        def _convert():
            return convert_from_path(str(pdf_path), dpi=144)

        images = await loop.run_in_executor(None, _convert)

        chunks = []
        for page_num, image in enumerate(images):
            # Process with ColPali
            def _embed():
                with torch.no_grad():
                    batch = self._processor.process_images([image]).to(self.device)
                    embeddings = self._model(**batch)
                    # ColPali returns multi-vector embeddings
                    return embeddings.cpu().numpy()

            embeddings = await loop.run_in_executor(None, _embed)

            # Also extract text via OCR for hybrid search
            text = await self._extract_text_ocr(image)

            # Encode image as base64 for storage
            import io
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode()

            chunk = MultimodalChunk(
                chunk_id=f"{doc_id}_page_{page_num + 1}",
                content=text,
                images=[{
                    "data": image_b64,
                    "format": "png",
                    "page": page_num + 1,
                }],
                embedding=embeddings.mean(axis=1).flatten().tolist(),  # Pooled embedding
                image_embeddings=[embeddings[0].tolist()],  # Full multi-vector
                page_number=page_num + 1,
                metadata={"source": str(pdf_path)},
            )
            chunks.append(chunk)

        logger.info("PDF processed with ColPali", pages=len(chunks))
        return chunks

    async def _extract_text_ocr(self, image) -> str:
        """Extract text from image using OCR."""
        try:
            import pytesseract
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None,
                lambda: pytesseract.image_to_string(image)
            )
            return text
        except Exception as e:
            logger.warning("OCR failed", error=str(e))
            return ""

    async def embed_query(self, query: str) -> list[float]:
        """Embed a text query for retrieval."""
        await self.initialize()

        import torch
        loop = asyncio.get_event_loop()

        def _embed():
            with torch.no_grad():
                batch = self._processor.process_queries([query]).to(self.device)
                embeddings = self._model(**batch)
                return embeddings.cpu().numpy()

        embeddings = await loop.run_in_executor(None, _embed)
        return embeddings.mean(axis=1).flatten().tolist()


class CLIPEmbedder:
    """
    CLIP-based image and text embedding.

    Simpler alternative to ColPali, works with:
    - Standalone images
    - Image-text pairs
    - Unified embedding space
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    async def initialize(self) -> None:
        """Load CLIP model."""
        if self._model is not None:
            return

        try:
            from transformers import CLIPModel, CLIPProcessor
            import torch

            loop = asyncio.get_event_loop()

            def _load():
                model = CLIPModel.from_pretrained(self.model_name)
                processor = CLIPProcessor.from_pretrained(self.model_name)

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device).eval()

                return model, processor

            self._model, self._processor = await loop.run_in_executor(None, _load)
            logger.info("CLIP model loaded", model=self.model_name)

        except ImportError:
            logger.warning("Transformers not available for CLIP")
            raise

    async def embed_image(self, image) -> list[float]:
        """Embed an image."""
        await self.initialize()

        import torch
        loop = asyncio.get_event_loop()

        def _embed():
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                embeddings = self._model.get_image_features(**inputs)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            return embeddings.cpu().numpy().flatten().tolist()

        return await loop.run_in_executor(None, _embed)

    async def embed_text(self, text: str) -> list[float]:
        """Embed text."""
        await self.initialize()

        import torch
        loop = asyncio.get_event_loop()

        def _embed():
            inputs = self._processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                embeddings = self._model.get_text_features(**inputs)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            return embeddings.cpu().numpy().flatten().tolist()

        return await loop.run_in_executor(None, _embed)

    async def embed_image_text_pair(
        self,
        image,
        text: str,
        alpha: float = 0.5,
    ) -> list[float]:
        """
        Create combined embedding for image-text pair.

        Args:
            image: PIL Image
            text: Associated text
            alpha: Weight for image embedding (1-alpha for text)
        """
        img_emb = await self.embed_image(image)
        txt_emb = await self.embed_text(text)

        import numpy as np
        combined = np.array(img_emb) * alpha + np.array(txt_emb) * (1 - alpha)
        combined = combined / np.linalg.norm(combined)

        return combined.tolist()


class MultimodalLoader:
    """
    Unified multimodal document loader.

    Handles:
    - PDFs with images/figures
    - Image files
    - Documents with embedded images
    """

    def __init__(
        self,
        colpali_processor: ColPaliProcessor | None = None,
        clip_embedder: CLIPEmbedder | None = None,
        use_colpali: bool = True,
    ):
        self.colpali = colpali_processor
        self.clip = clip_embedder
        self.use_colpali = use_colpali and colpali_processor is not None

    async def load(
        self,
        file_path: str | Path,
        doc_id: str,
    ) -> list[MultimodalChunk]:
        """
        Load a multimodal document.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return await self._load_pdf(path, doc_id)
        elif suffix in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
            return await self._load_image(path, doc_id)
        else:
            # Fallback to text loading
            logger.warning("Unsupported format for multimodal", format=suffix)
            return []

    async def _load_pdf(
        self,
        path: Path,
        doc_id: str,
    ) -> list[MultimodalChunk]:
        """Load PDF with multimodal processing."""
        if self.use_colpali and self.colpali:
            return await self.colpali.process_pdf(path, doc_id)
        else:
            # Fallback: extract images and process with CLIP
            return await self._load_pdf_with_clip(path, doc_id)

    async def _load_pdf_with_clip(
        self,
        path: Path,
        doc_id: str,
    ) -> list[MultimodalChunk]:
        """Fallback PDF processing with CLIP."""
        from pdf2image import convert_from_path
        import io

        loop = asyncio.get_event_loop()
        images = await loop.run_in_executor(
            None,
            lambda: convert_from_path(str(path), dpi=144)
        )

        chunks = []
        for i, image in enumerate(images):
            # Extract text
            try:
                import pytesseract
                text = await loop.run_in_executor(
                    None,
                    lambda img=image: pytesseract.image_to_string(img)
                )
            except Exception:
                text = ""

            # Embed with CLIP if available
            if self.clip:
                if text:
                    embedding = await self.clip.embed_image_text_pair(image, text[:500])
                else:
                    embedding = await self.clip.embed_image(image)
            else:
                embedding = []

            # Encode image
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode()

            chunks.append(MultimodalChunk(
                chunk_id=f"{doc_id}_page_{i + 1}",
                content=text,
                images=[{"data": image_b64, "format": "png", "page": i + 1}],
                embedding=embedding,
                page_number=i + 1,
                metadata={"source": str(path)},
            ))

        return chunks

    async def _load_image(
        self,
        path: Path,
        doc_id: str,
    ) -> list[MultimodalChunk]:
        """Load standalone image."""
        from PIL import Image
        import io

        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(None, lambda: Image.open(path))

        # Try OCR
        try:
            import pytesseract
            text = await loop.run_in_executor(
                None,
                lambda: pytesseract.image_to_string(image)
            )
        except Exception:
            text = ""

        # Embed
        if self.clip:
            embedding = await self.clip.embed_image(image)
        else:
            embedding = []

        # Encode
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode()

        return [MultimodalChunk(
            chunk_id=f"{doc_id}_image",
            content=text,
            images=[{"data": image_b64, "format": "png"}],
            embedding=embedding,
            metadata={"source": str(path)},
        )]


class MultimodalIngestionPipeline:
    """
    Pipeline for ingesting multimodal documents.

    Integrates with the main RAG system to:
    1. Process multimodal documents (PDFs with images, standalone images)
    2. Generate embeddings using ColPali or CLIP
    3. Store in vector database with image metadata
    4. Enable multimodal retrieval
    """

    def __init__(
        self,
        qdrant,
        surrealdb,
        colpali_processor: ColPaliProcessor | None = None,
        clip_embedder: CLIPEmbedder | None = None,
        use_colpali: bool = True,
        collection_name: str = "multimodal_chunks",
    ):
        self.qdrant = qdrant
        self.surrealdb = surrealdb
        self.loader = MultimodalLoader(
            colpali_processor=colpali_processor,
            clip_embedder=clip_embedder,
            use_colpali=use_colpali,
        )
        self.collection_name = collection_name
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the multimodal collection in Qdrant."""
        if self._initialized:
            return

        # Determine embedding dimension based on model
        if self.loader.use_colpali and self.loader.colpali:
            # ColPali typically uses 128-dimensional embeddings
            dimension = 128
        elif self.loader.clip:
            # CLIP ViT-L/14 uses 768-dimensional embeddings
            dimension = 768
        else:
            # Default dimension
            dimension = 768

        # Create collection if needed
        try:
            await self.qdrant.create_collection(
                collection_name=self.collection_name,
                dimension=dimension,
                distance="Cosine",
            )
            logger.info("Multimodal collection created", collection=self.collection_name)
        except Exception as e:
            logger.debug("Collection may already exist", error=str(e))

        self._initialized = True

    async def ingest(
        self,
        file_path: str | Path,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        acl: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Ingest a multimodal document.

        Args:
            file_path: Path to the document
            doc_id: Optional document ID (generated if not provided)
            metadata: Additional metadata to store
            acl: Access control list

        Returns:
            Ingestion result with document ID and chunk count
        """
        await self.initialize()

        import uuid
        path = Path(file_path)

        # Generate document ID if not provided
        if not doc_id:
            doc_id = f"mm_{uuid.uuid4().hex[:12]}"

        # Load and process document
        chunks = await self.loader.load(path, doc_id)

        if not chunks:
            logger.warning("No chunks extracted from document", path=str(path))
            return {
                "doc_id": doc_id,
                "chunks_created": 0,
                "status": "no_content",
            }

        # Prepare metadata
        base_metadata = {
            "doc_id": doc_id,
            "source": str(path),
            "filename": path.name,
            "multimodal": True,
            "acl": acl or ["*"],
            **(metadata or {}),
        }

        # Store document metadata in SurrealDB
        await self.surrealdb.create_document(
            doc_id=doc_id,
            title=base_metadata.get("title", path.stem),
            content_type="multimodal",
            metadata=base_metadata,
        )

        # Store chunks in Qdrant
        points = []
        for chunk in chunks:
            if not chunk.embedding:
                logger.warning("Chunk has no embedding", chunk_id=chunk.chunk_id)
                continue

            point_metadata = {
                **base_metadata,
                "chunk_id": chunk.chunk_id,
                "page_number": chunk.page_number,
                "has_text": bool(chunk.content),
                "has_images": bool(chunk.images),
                "text_preview": chunk.content[:200] if chunk.content else "",
            }

            # Store image data separately (don't include in vector DB metadata)
            # Images will be stored in SurrealDB
            points.append({
                "id": chunk.chunk_id,
                "vector": chunk.embedding,
                "payload": point_metadata,
            })

            # Store full chunk data in SurrealDB
            await self.surrealdb.create_chunk(
                chunk_id=chunk.chunk_id,
                doc_id=doc_id,
                content=chunk.content,
                metadata={
                    "images": chunk.images,
                    "page_number": chunk.page_number,
                    "image_embeddings": chunk.image_embeddings,
                },
            )

        # Batch upsert to Qdrant
        if points:
            await self.qdrant.upsert_points(
                collection_name=self.collection_name,
                points=points,
            )

        logger.info(
            "Multimodal document ingested",
            doc_id=doc_id,
            chunks=len(chunks),
            points_stored=len(points),
        )

        return {
            "doc_id": doc_id,
            "chunks_created": len(chunks),
            "points_stored": len(points),
            "status": "success",
        }

    async def search(
        self,
        query: str,
        top_k: int = 10,
        acl_filter: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search multimodal collection.

        Args:
            query: Text query
            top_k: Number of results
            acl_filter: Access control filter

        Returns:
            List of matching chunks with metadata
        """
        await self.initialize()

        # Generate query embedding
        if self.loader.use_colpali and self.loader.colpali:
            query_embedding = await self.loader.colpali.embed_query(query)
        elif self.loader.clip:
            query_embedding = await self.loader.clip.embed_text(query)
        else:
            raise ValueError("No embedding model available for search")

        # Build filter
        search_filter = None
        if acl_filter and "*" not in acl_filter:
            search_filter = {
                "should": [
                    {"key": "acl", "match": {"any": acl_filter}},
                    {"key": "acl", "match": {"value": "*"}},
                ]
            }

        # Search Qdrant
        results = await self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=search_filter,
        )

        # Enrich with full chunk data from SurrealDB
        enriched = []
        for result in results:
            chunk_data = await self.surrealdb.get_chunk(result["id"])
            enriched.append({
                "chunk_id": result["id"],
                "score": result["score"],
                "doc_id": result["payload"].get("doc_id"),
                "page_number": result["payload"].get("page_number"),
                "text_preview": result["payload"].get("text_preview"),
                "content": chunk_data.get("content") if chunk_data else "",
                "images": chunk_data.get("metadata", {}).get("images", []) if chunk_data else [],
            })

        return enriched


# Factory functions for creating processors
def create_colpali_processor(
    model_name: str = "vidore/colpali-v1.2",
    device: str = "auto",
) -> ColPaliProcessor:
    """Create a ColPali processor instance."""
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return ColPaliProcessor(model_name=model_name, device=device)


def create_clip_embedder(
    model_name: str = "openai/clip-vit-large-patch14",
    device: str = "auto",
) -> CLIPEmbedder:
    """Create a CLIP embedder instance."""
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return CLIPEmbedder(model_name=model_name, device=device)
