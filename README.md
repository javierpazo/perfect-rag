# Perfect RAG

A production-ready RAG (Retrieval-Augmented Generation) system with GraphRAG, hybrid search, and multi-provider LLM support.

## Features

- **OpenAI-compatible API** with SSE streaming support
- **Hybrid Search**: Dense (BGE-M3) + Sparse (BM25-style) with RRF fusion
- **GraphRAG**: Knowledge graph expansion using SurrealDB and Oxigraph
- **Multi-provider LLM Gateway**: OpenAI, Anthropic, Ollama with fallback and usage tracking
- **Advanced Retrieval**: Query rewriting, HyDE, decomposition, reranking
- **NER/RE Extraction**: Automatic entity and relation extraction
- **ACL Support**: Role-based access control for documents
- **Citation Tracking**: Automatic source attribution in responses

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

**Javier Pazó** - AEEH (Asociación Española de Enfermedades Hepáticas)

---

## Acknowledgments

Developed for medical knowledge management applications.
