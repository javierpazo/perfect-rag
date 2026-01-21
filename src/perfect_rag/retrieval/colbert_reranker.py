"""ColBERT late interaction reranker for improved retrieval accuracy."""

import asyncio
from typing import Any
from pathlib import Path

import structlog

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class ColBERTReranker:
    """
    ColBERT-based reranker using late interaction for improved accuracy.

    Late interaction computes fine-grained similarity between query and document
    tokens, achieving 15-30% better accuracy than dense embeddings alone.

    ColBERT (Contextualized Late Interaction over BERT) works by:
    1. Encoding query and document tokens independently
    2. Computing MaxSim (maximum similarity) between each query token
       and all document tokens
    3. Summing these similarities for the final score

    This allows for more nuanced matching than single-vector approaches
    while remaining efficient through pre-computation of document embeddings.
    """

    def __init__(
        self,
        model_name: str | None = None,
        index_path: str | None = None,
        device: str | None = None,
        n_gpu: int = 1,
        settings: Settings | None = None,
    ):
        """
        Initialize ColBERT reranker.

        Args:
            model_name: ColBERT model name (default from settings)
            index_path: Path for storing ColBERT indexes
            device: Device to run on ("cuda", "cpu", "mps")
            n_gpu: Number of GPUs to use
            settings: Application settings
        """
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.colbert_model
        self.index_path = index_path or str(
            Path.home() / ".cache" / "colbert_index"
        )
        self.device = device or self.settings.device
        self.n_gpu = n_gpu
        self._model = None
        self._index = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Lazy load the ColBERT model."""
        async with self._lock:
            if self._model is not None:
                return

            try:
                from ragatouille import RAGPretrainedModel

                logger.info("Loading ColBERT model", model=self.model_name)

                # Load in thread pool to not block event loop
                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(
                    None,
                    lambda: RAGPretrainedModel.from_pretrained(self.model_name)
                )
                logger.info("ColBERT model loaded successfully", model=self.model_name)

            except ImportError:
                logger.warning(
                    "RAGatouille not installed, ColBERT disabled. "
                    "Install with: pip install ragatouille"
                )
                raise ImportError(
                    "RAGatouille is required for ColBERT. "
                    "Install with: pip install ragatouille"
                )

    @property
    def is_initialized(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank documents using ColBERT late interaction.

        Args:
            query: The search query
            documents: List of documents with 'content' or 'text' and 'id' fields
            top_k: Number of top results to return (default: all)

        Returns:
            Reranked documents with ColBERT scores added
        """
        await self.initialize()

        if not documents:
            return []

        top_k = top_k or len(documents)

        # Extract texts for reranking - support both 'content' and 'text' keys
        texts = []
        for doc in documents:
            text = doc.get("content") or doc.get("text", "")
            if isinstance(text, str):
                texts.append(text)
            else:
                texts.append(str(text) if text else "")

        doc_ids = [doc.get("id", str(i)) for i, doc in enumerate(documents)]

        logger.debug(
            "Reranking documents with ColBERT",
            query_length=len(query),
            doc_count=len(documents),
            top_k=top_k,
        )

        # Run reranking in thread pool
        loop = asyncio.get_event_loop()

        def _rerank():
            # RAGatouille rerank API
            results = self._model.rerank(
                query=query,
                documents=texts,
                k=min(top_k, len(texts)),
            )
            return results

        try:
            results = await loop.run_in_executor(None, _rerank)
        except Exception as e:
            logger.error("ColBERT reranking failed", error=str(e))
            # Return original documents on failure
            return documents[:top_k]

        # Map results back to original documents
        reranked = []
        for result in results:
            idx = result.get("result_index", 0)
            if idx < len(documents):
                doc = documents[idx].copy()
                doc["colbert_score"] = result.get("score", 0.0)
                doc["original_rank"] = idx
                reranked.append(doc)

        logger.debug(
            "ColBERT reranking complete",
            input_count=len(documents),
            output_count=len(reranked),
        )

        return reranked[:top_k]

    async def create_index(
        self,
        documents: list[dict[str, Any]],
        index_name: str = "perfect_rag",
        max_document_length: int = 512,
        split_documents: bool = False,
    ) -> str:
        """
        Create a ColBERT index for fast retrieval.

        Creating an index pre-computes document embeddings, enabling
        very fast retrieval at query time. This is useful when you have
        a static document collection.

        Args:
            documents: List of documents with 'content' and 'id' fields
            index_name: Name for the index
            max_document_length: Maximum tokens per document
            split_documents: Whether to split long documents

        Returns:
            Path to the created index
        """
        await self.initialize()

        texts = []
        for doc in documents:
            text = doc.get("content") or doc.get("text", "")
            texts.append(str(text) if text else "")

        doc_ids = [doc.get("id", str(i)) for i, doc in enumerate(documents)]
        metadatas = [
            {k: v for k, v in doc.items() if k not in ("content", "text")}
            for doc in documents
        ]

        logger.info(
            "Creating ColBERT index",
            index_name=index_name,
            doc_count=len(documents),
        )

        loop = asyncio.get_event_loop()

        def _index():
            return self._model.index(
                collection=texts,
                document_ids=doc_ids,
                document_metadatas=metadatas,
                index_name=index_name,
                max_document_length=max_document_length,
                split_documents=split_documents,
            )

        try:
            index_path = await loop.run_in_executor(None, _index)
            self._index = index_path

            logger.info(
                "ColBERT index created",
                path=str(index_path),
                docs=len(documents),
            )
            return str(index_path)

        except Exception as e:
            logger.error("ColBERT index creation failed", error=str(e))
            raise

    async def search(
        self,
        query: str,
        top_k: int = 10,
        index_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search using ColBERT index (primary retrieval mode).

        This method uses a pre-built ColBERT index for fast retrieval.
        Use this when you want ColBERT as the primary retriever instead
        of just a reranker.

        Args:
            query: Search query
            top_k: Number of results
            index_name: Optional index name to load

        Returns:
            List of search results with scores
        """
        await self.initialize()

        # Load specific index if provided
        if index_name:
            from ragatouille import RAGPretrainedModel

            logger.info("Loading ColBERT index", index_name=index_name)
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: RAGPretrainedModel.from_index(index_name)
            )

        loop = asyncio.get_event_loop()

        def _search():
            return self._model.search(query=query, k=top_k)

        try:
            results = await loop.run_in_executor(None, _search)
        except Exception as e:
            logger.error("ColBERT search failed", error=str(e))
            return []

        # Format results
        formatted_results = []
        for i, r in enumerate(results):
            formatted_results.append({
                "id": r.get("document_id", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0.0),
                "colbert_score": r.get("score", 0.0),
                "rank": i,
                **r.get("document_metadata", {}),
            })

        return formatted_results

    async def add_to_index(
        self,
        documents: list[dict[str, Any]],
        index_name: str | None = None,
    ) -> None:
        """
        Add documents to an existing ColBERT index.

        Args:
            documents: Documents to add
            index_name: Index to update
        """
        await self.initialize()

        texts = []
        for doc in documents:
            text = doc.get("content") or doc.get("text", "")
            texts.append(str(text) if text else "")

        doc_ids = [doc.get("id", str(i)) for i, doc in enumerate(documents)]
        metadatas = [
            {k: v for k, v in doc.items() if k not in ("content", "text")}
            for doc in documents
        ]

        loop = asyncio.get_event_loop()

        def _add():
            return self._model.add_to_index(
                new_collection=texts,
                new_document_ids=doc_ids,
                new_document_metadatas=metadatas,
            )

        await loop.run_in_executor(None, _add)
        logger.info("Added documents to ColBERT index", count=len(documents))

    async def delete_from_index(
        self,
        document_ids: list[str],
    ) -> None:
        """
        Delete documents from the ColBERT index.

        Args:
            document_ids: IDs of documents to delete
        """
        await self.initialize()

        loop = asyncio.get_event_loop()

        def _delete():
            return self._model.delete_from_index(document_ids=document_ids)

        await loop.run_in_executor(None, _delete)
        logger.info("Deleted documents from ColBERT index", count=len(document_ids))

    def get_info(self) -> dict[str, Any]:
        """Get information about the ColBERT reranker."""
        return {
            "model": self.model_name,
            "device": self.device,
            "index_path": self.index_path,
            "initialized": self.is_initialized,
            "has_index": self._index is not None,
        }


class ColBERTIndexManager:
    """
    Manage ColBERT indexes for the RAG system.

    This class provides a higher-level interface for managing multiple
    ColBERT indexes organized by collection name.
    """

    def __init__(
        self,
        base_path: str | None = None,
        settings: Settings | None = None,
    ):
        """
        Initialize the index manager.

        Args:
            base_path: Base directory for storing indexes
            settings: Application settings
        """
        self.settings = settings or get_settings()
        self.base_path = Path(
            base_path or Path.home() / ".cache" / "colbert_indexes"
        )
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._reranker = ColBERTReranker(settings=settings)
        self._indexes: dict[str, str] = {}  # collection_name -> index_path

    async def initialize(self) -> None:
        """Initialize the underlying ColBERT model."""
        await self._reranker.initialize()

    async def add_documents(
        self,
        documents: list[dict[str, Any]],
        collection_name: str = "default",
    ) -> str:
        """
        Add documents to a ColBERT collection.

        Args:
            documents: Documents with 'content' and 'id' fields
            collection_name: Name of the collection

        Returns:
            Path to the created/updated index
        """
        index_path = self.base_path / collection_name

        result = await self._reranker.create_index(
            documents=documents,
            index_name=str(index_path),
        )

        self._indexes[collection_name] = result
        return result

    async def search(
        self,
        query: str,
        collection_name: str = "default",
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search within a specific collection.

        Args:
            query: Search query
            collection_name: Collection to search
            top_k: Number of results

        Returns:
            Search results
        """
        index_path = self._indexes.get(collection_name)
        if not index_path:
            index_path = str(self.base_path / collection_name)

        return await self._reranker.search(
            query=query,
            top_k=top_k,
            index_name=index_path,
        )

    async def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank documents using ColBERT.

        This is the primary interface for using ColBERT as a reranker
        in the retrieval pipeline.

        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of results to return

        Returns:
            Reranked documents with ColBERT scores
        """
        return await self._reranker.rerank(query, documents, top_k)

    def list_collections(self) -> list[str]:
        """List available collections."""
        collections = []
        if self.base_path.exists():
            for path in self.base_path.iterdir():
                if path.is_dir():
                    collections.append(path.name)
        return collections

    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection and its index.

        Args:
            collection_name: Collection to delete

        Returns:
            True if deleted, False if not found
        """
        import shutil

        index_path = self.base_path / collection_name
        if index_path.exists():
            shutil.rmtree(index_path)
            self._indexes.pop(collection_name, None)
            logger.info("Deleted ColBERT collection", name=collection_name)
            return True
        return False

    def get_info(self) -> dict[str, Any]:
        """Get information about the index manager."""
        return {
            "base_path": str(self.base_path),
            "collections": self.list_collections(),
            "reranker_info": self._reranker.get_info(),
        }


# =============================================================================
# Factory Functions
# =============================================================================

_reranker: ColBERTReranker | None = None
_index_manager: ColBERTIndexManager | None = None


async def get_colbert_reranker(settings: Settings | None = None) -> ColBERTReranker:
    """Get or create the global ColBERT reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = ColBERTReranker(settings=settings)
    return _reranker


async def get_colbert_index_manager(
    settings: Settings | None = None,
) -> ColBERTIndexManager:
    """Get or create the global ColBERT index manager instance."""
    global _index_manager
    if _index_manager is None:
        _index_manager = ColBERTIndexManager(settings=settings)
    return _index_manager
