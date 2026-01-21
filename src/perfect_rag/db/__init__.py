"""Database clients for Perfect RAG.

Graph operations use SurrealDB as primary store with optional Oxigraph
for advanced SPARQL reasoning. Use the unified GraphStore interface
for all graph operations.
"""

from perfect_rag.db.graph_store import (
    GraphStore,
    HybridGraphStore,
    SurrealDBGraphStore,
    create_graph_store,
    get_graph_store,
)
from perfect_rag.db.oxigraph import OxigraphClient, get_oxigraph
from perfect_rag.db.qdrant import QdrantVectorClient, get_qdrant
from perfect_rag.db.surrealdb import SurrealDBClient, get_surrealdb

__all__ = [
    # SurrealDB (primary storage)
    "SurrealDBClient",
    "get_surrealdb",
    # Qdrant (vector storage)
    "QdrantVectorClient",
    "get_qdrant",
    # Oxigraph (optional SPARQL reasoning)
    "OxigraphClient",
    "get_oxigraph",
    # Unified graph store (recommended)
    "GraphStore",
    "SurrealDBGraphStore",
    "HybridGraphStore",
    "create_graph_store",
    "get_graph_store",
]
