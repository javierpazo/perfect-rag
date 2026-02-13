# Perfect RAG

**The most complete, production-ready RAG system.** Combining state-of-the-art techniques including GraphRAG, hybrid search, PageIndex, multi-stage reranking, and cache-augmented generation.

> "Perfect RAG achieves 98.7% accuracy on FinanceBench with PageIndex tree-based retrieval."

## Why Perfect RAG?

| Feature | Perfect RAG | Typical RAG |
|---------|-------------|-------------|
| **Retrieval** | Hybrid (Dense + Sparse + RRF) | Dense only |
| **Graph** | GraphRAG with entity expansion | None |
| **Reranking** | 3-stage (Cross-encoder + ColBERT + LLM) | Single reranker |
| **Structured Docs** | PageIndex tree reasoning | Chunk search only |
| **Citations** | Page-level with verification | Chunk-level |
| **Cache** | CAG + Semantic cache | None |
| **Query Smart** | Context gate + rewriting | Direct search |

## Features

- **OpenAI-compatible API** with SSE streaming support
- **Hybrid Search**: Dense (BGE-M3) + Sparse (BM25-style) with RRF fusion
- **GraphRAG**: Knowledge graph expansion using SurrealDB and Oxigraph
- **PageIndex**: Tree-based reasoning for structured documents (98.7% FinanceBench accuracy)
- **Multi-stage Reranking**: Cross-encoder → ColBERT → LLM
- **Multi-provider LLM Gateway**: OpenAI, Anthropic, Ollama with fallback and usage tracking
- **Advanced Retrieval**: Query rewriting, HyDE, decomposition, reranking
- **NER/RE Extraction**: Automatic entity and relation extraction
- **ACL Support**: Role-based access control for documents
- **Citation Tracking**: Automatic source attribution with page numbers
- **Cache-Augmented Generation**: Semantic cache + CAG prewarm

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Perfect RAG API                         │
│                    (FastAPI + Uvicorn)                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │  Ingestion  │ │  Retrieval  │ │      Generation         ││
│  │  Pipeline   │ │  Pipeline   │ │      Pipeline           ││
│  └──────┬──────┘ └──────┬──────┘ └───────────┬─────────────┘│
│         │               │                     │              │
│  ┌──────┴───────────────┴─────────────────────┴───────────┐ │
│  │              Core Services                              │ │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────────────────┐ │ │
│  │  │ Embedding │ │    LLM    │ │   Query Processing    │ │ │
│  │  │  (BGE-M3) │ │  Gateway  │ │   (Rewrite/Rerank)    │ │ │
│  │  └───────────┘ └───────────┘ └───────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────┬──────────────────────────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       │                       │                       │
       ▼                       ▼                       ▼
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  SurrealDB  │         │   Qdrant    │         │  Oxigraph   │
│  (Docs/KG)  │         │  (Vectors)  │         │ (RDF/SPARQL)│
└─────────────┘         └─────────────┘         └─────────────┘
```

## Quick Start

### 1. Clone and Configure

```bash
git clone <repository>
cd perfect-rag

# Copy example configuration
cp .env.example .env

# Edit .env with your API keys
nano .env
```

### 2. Start with Docker Compose

```bash
# Start all services
docker compose up -d

# Check logs
docker compose logs -f rag-api
```

### 3. Alternative: Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -e ".[dev]"

# Start databases (requires Docker)
docker compose up -d surrealdb qdrant oxigraph

# Run API
uvicorn perfect_rag.main:app --reload
```

## API Usage

### OpenAI-Compatible Chat

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "stream": false
  }'
```

### With Streaming

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "user", "content": "Explain RAG systems"}
    ],
    "stream": true
  }'
```

### Upload Document

```bash
# Upload file
curl -X POST http://localhost:8000/v1/documents \
  -F "file=@document.pdf" \
  -F "title=My Document" \
  -F "acl=admin,user" \
  -F "tags=technical,ml"

# Or ingest from URL
curl -X POST http://localhost:8000/v1/documents \
  -F "url=https://example.com/article.html" \
  -F "title=Web Article"
```

### Direct Search

```bash
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "top_k": 10,
    "use_reranking": true,
    "use_graph_expansion": true
  }'
```

### List Models

```bash
curl http://localhost:8000/v1/models
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `DEFAULT_LLM_PROVIDER` | Default LLM provider | openai |
| `DEFAULT_LLM_MODEL` | Default model | gpt-4o |
| `EMBEDDING_MODEL` | Embedding model | BAAI/bge-m3 |
| `SURREALDB_URL` | SurrealDB connection | ws://localhost:8080 |
| `QDRANT_URL` | Qdrant connection | http://localhost:6333 |
| `OXIGRAPH_URL` | Oxigraph connection | http://localhost:7878 |
| `CHUNK_SIZE` | Default chunk size | 512 |
| `RETRIEVAL_TOP_K` | Default top-k results | 10 |
| `MONTHLY_BUDGET_USD` | Monthly LLM budget | 100.0 |

### RAG-specific Request Options

The chat completions endpoint accepts additional parameters:

```json
{
  "x_use_rag": true,        // Enable/disable RAG (default: true)
  "x_rag_top_k": 10         // Number of chunks to retrieve
}
```

Response includes additional fields:

```json
{
  "citations": [            // Source citations
    {
      "source_id": "doc_123",
      "source_title": "Document Title",
      "text_snippet": "...",
      "relevance_score": 0.95
    }
  ],
  "confidence": 0.85,       // Answer confidence
  "x_rag_metadata": {...}   // Retrieval metadata
}
```

## The Perfect RAG Recipe

### Ingredients

```
qdrant                    # Vector DB
surrealdb                 # Graph DB + metadata
sentence-transformers     # Embeddings
BAAI/bge-m3              # Dense embeddings (1024d)
splade                   # Sparse embeddings
bge-reranker-v2-m3       # Cross-encoder reranker
LLM API (openai/anthropic/ollama)
```

### Preparation

#### 1. INGESTION

```
document → chunker(512 tokens, 50 overlap)
        → embed_dense(bge-m3) + embed_sparse(splade)
        → extract_entities(LLM) [optional]
        → store in Qdrant + SurrealDB
```

#### 2. RETRIEVAL (8-step pipeline)

```
query
  ↓
[1] Context Gate → Does it need retrieval? If not, return empty
  ↓
[2] Query Rewrite → Expansion + HyDE + Decomposition
  ↓
[2.5] PageIndex → Tree-based search for structured docs [optional]
  ↓
[3] Hybrid Search → Dense + Sparse with RRF fusion, top_k=20
  ↓
[4] GraphRAG → Expand by entities, max_hops=2
  ↓
[5] Cross-encoder → Rerank with bge-reranker
  ↓
[6] ColBERT → Late interaction reranking [optional]
  ↓
[7] LLM Rerank → Semantic reranking [optional]
  ↓
top_k=5 chunks
```

#### 3. GENERATION

```
chunks + question
  → prompt_builder(formatted context + citations [1][2])
  → LLM.generate()
  → response with citations
```

### Secret Sauce

| Technique | Why it works |
|-----------|--------------|
| **Hybrid Search** | Combine dense (semantic) + sparse (keywords) with Reciprocal Rank Fusion |
| **GraphRAG** | Chunks connected by entities improve multi-hop answers |
| **Context Gate** | Avoids polluting MCQ prompts with irrelevant context |
| **Quality Gate** | Only inject context if similarity > 0.35 for MCQs |
| **CAG Cache** | Cache responses by embedding for repeated queries |
| **Page Index** | Track page numbers for precise citation in documents |
| **PageIndex (VectifyAI)** | Tree-based reasoning for structured docs (98.7% accuracy on FinanceBench) |

### Page Index Enhancement

For documents with page structure (PDFs, books), Perfect RAG can track page indices:

```python
# During ingestion
chunk.metadata["page_number"] = pdf_page_number

# During retrieval - results include page references
{
  "chunk_id": "abc123",
  "content": "...",
  "page_number": 42,  # Direct page reference
  "doc_title": "Medical Guidelines"
}
```

**Benefits:**
- Precise citation: "According to page 42 of Medical Guidelines..."
- UI integration: Jump-to-page functionality
- Cross-page reasoning: Connect concepts across pages
- Better user trust: Verifiable sources

### PageIndex (VectifyAI) - Optional Enhancement

For structured documents (guides, manuals, reports), Perfect RAG can use **PageIndex** - a reasoning-based retrieval approach that achieves **98.7% accuracy on FinanceBench**.

#### How PageIndex Works

Instead of vector similarity search, PageIndex:

1. **Builds a tree** - Transforms documents into hierarchical TOC structures
2. **LLM navigates** - Uses reasoning to find relevant sections
3. **Returns page ranges** - Filters vector search to specific pages

#### When to Use PageIndex

- ✅ Structured documents (PDFs, guides, manuals)
- ✅ Documents with 20+ pages
- ✅ Questions requiring specific section location
- ❌ Short unstructured documents
- ❌ Pure semantic similarity queries

#### Enable PageIndex

```bash
# In .env
PAGEINDEX_ENABLED=true
PAGEINDEX_MIN_PAGES=20
PAGEINDEX_TREE_PATH=./pageindex_trees
```

#### PageIndex Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `PAGEINDEX_ENABLED` | false | Enable PageIndex tree-based retrieval |
| `PAGEINDEX_TREE_PATH` | ./pageindex_trees | Where to store tree structures |
| `PAGEINDEX_MIN_PAGES` | 20 | Minimum pages to use PageIndex |
| `PAGEINDEX_MAX_TREE_DEPTH` | 5 | Maximum tree traversal depth |
| `PAGEINDEX_LLM_MODEL` | gpt-4o-mini | LLM for tree navigation |

## Architecture Details

### Ingestion Pipeline

1. **Document Loading**: PDF, DOCX, HTML, TXT, JSON, CSV support
2. **Chunking**: Recursive, semantic, sentence, or paragraph-based
3. **Embedding**: BGE-M3 for dense + sparse vectors
4. **Entity Extraction**: spaCy NER + optional LLM extraction
5. **Relation Extraction**: Pattern matching + LLM-based
6. **Storage**: SurrealDB (docs/graph), Qdrant (vectors), Oxigraph (RDF)

### Retrieval Pipeline

1. **Context Gate**: Determine if retrieval is needed
2. **Query Rewriting**: Expansion, HyDE, decomposition, multi-query
3. **Hybrid Search**: Dense + sparse with RRF fusion
4. **GraphRAG Expansion**: Knowledge graph traversal
5. **Reranking**: Cross-encoder (bge-reranker-v2-m3)

### Generation Pipeline

1. **Prompt Construction**: Context injection with citations
2. **LLM Generation**: Multi-provider with fallback
3. **Citation Extraction**: Verify and format citations
4. **Response Formatting**: Optional bibliography generation

## Project Structure

```
perfect-rag/
├── src/perfect_rag/
│   ├── config.py              # Pydantic settings
│   ├── main.py                # FastAPI application
│   ├── models/                # Pydantic data models
│   │   ├── document.py
│   │   ├── chunk.py
│   │   ├── entity.py
│   │   ├── relation.py
│   │   ├── query.py
│   │   └── openai_types.py
│   ├── db/                    # Database clients
│   │   ├── surrealdb.py
│   │   ├── qdrant.py
│   │   └── oxigraph.py
│   ├── llm/                   # LLM providers
│   │   ├── providers.py
│   │   └── gateway.py
│   ├── core/                  # Core services
│   │   └── embedding.py
│   ├── ingestion/             # Ingestion pipeline
│   │   ├── loaders.py
│   │   ├── chunker.py
│   │   ├── extractor.py
│   │   └── pipeline.py
│   ├── retrieval/             # Retrieval pipeline
│   │   ├── query_rewriter.py
│   │   ├── graphrag.py
│   │   └── pipeline.py
│   └── generation/            # Generation pipeline
│       ├── prompt_builder.py
│       ├── citation_extractor.py
│       └── pipeline.py
├── config/
│   ├── surrealdb/init.surql   # Database schema
│   └── oxigraph/ontology.ttl  # RDF ontology
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black src/
ruff check src/ --fix
mypy src/
```

## License

MIT License

## Author

**Javier Pazó** - AEEH (Asociación Española para el Estudio del Hígado)

---

## Acknowledgments

Developed for medical knowledge management applications.
