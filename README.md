# Perfect RAG

**A production-grade RAG system with state-of-the-art retrieval techniques.** Combining GraphRAG, hybrid search, multi-stage reranking, and cache-augmented generation.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Benchmarked Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Top Score (with reranking)** | 0.998 | Cross-encoder reranking on medical Q&A |
| **Top Score (baseline)** | 0.805 | Dense vector search only |
| **Improvement** | +24% | From reranking pipeline |
| **Latency P50** | 70ms | Full pipeline with reranking |
| **Latency P95** | 92ms | Full pipeline with reranking |
| **Success Rate** | 100% | 210 queries tested |

> See [eval/results/](eval/results/) for full benchmark data and reproduction scripts.

## Why Perfect RAG?

| Feature | Perfect RAG | Typical RAG |
|---------|-------------|-------------|
| **Retrieval** | Hybrid (Dense + Sparse + RRF) | Dense only |
| **Graph** | GraphRAG with entity expansion | None |
| **Reranking** | 3-stage (Cross-encoder + ColBERT + LLM) | Single reranker or none |
| **Citations** | Page-level with verification | Chunk-level or none |
| **Cache** | CAG + Semantic cache | None |
| **Query Smart** | Context gate + rewriting | Direct search |

## Features

- **OpenAI-compatible API** - Drop-in replacement for OpenAI chat completions
- **Hybrid Search** - Dense (BGE-M3) + Sparse (TF-IDF style) with RRF fusion
- **GraphRAG** - Knowledge graph expansion using SurrealDB
- **Multi-stage Reranking** - Cross-encoder → ColBERT → LLM (optional)
- **Multi-provider LLM** - OpenAI, Anthropic, Ollama with fallback
- **Query Rewriting** - Expansion, HyDE, decomposition
- **Citation Tracking** - Automatic source attribution with page numbers
- **Semantic Cache** - Cache responses by embedding similarity

## Quick Start

### 1. Clone and Configure

```bash
git clone https://github.com/javierpazo/perfect-rag.git
cd perfect-rag

cp .env.example .env
# Edit .env with your API keys
```

### 2. Start Services

```bash
# Development (with persistent storage)
docker compose --profile dev up -d

# Production (see docker-compose.prod.yml)
docker compose --profile prod up -d
```

### 3. Run API

```bash
# Local development
pip install -e ".[dev]"
uvicorn perfect_rag.main:app --reload
```

## API Usage

### OpenAI-Compatible Chat

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "What is machine learning?"}],
    "stream": false
  }'
```

### Upload Document

```bash
curl -X POST http://localhost:8000/v1/documents \
  -F "file=@document.pdf" \
  -F "title=My Document"
```

### Search

```bash
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "top_k": 10,
    "use_reranking": true
  }'
```

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
└──────────────────────┬───────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  SurrealDB  │ │   Qdrant    │ │  Oxigraph   │
│  (Docs/KG)  │ │  (Vectors)  │ │ (RDF/SPARQL)│
└─────────────┘ └─────────────┘ └─────────────┘
```

## The Perfect RAG Recipe

### Retrieval Pipeline (7 steps)

```
query
  ↓
[1] Context Gate → Skip retrieval if not needed
  ↓
[2] Query Rewrite → Expansion + HyDE + Decomposition
  ↓
[3] Hybrid Search → Dense + Sparse with RRF fusion
  ↓
[4] GraphRAG → Expand by entities (optional)
  ↓
[5] Cross-encoder → Rerank with bge-reranker
  ↓
[6] ColBERT → Late interaction (optional)
  ↓
[7] LLM Rerank → Semantic reranking (optional)
  ↓
top_k chunks
```

### Secret Sauce

| Technique | Impact | Why it works |
|-----------|--------|--------------|
| **Cross-encoder Reranking** | +24% top score | Semantic relevance scoring |
| **Hybrid Search** | +recall | Combines semantic + keyword matching |
| **GraphRAG** | +multi-hop | Entity-connected context |
| **Context Gate** | -latency | Skips retrieval when not needed |
| **Semantic Cache** | -cost | Reuses similar query responses |

## Evaluation

### Run Benchmarks

```bash
# Full benchmark suite
python eval/run_all.py

# Specific benchmark
python eval/benchmarks/retrieval_quality.py
python eval/benchmarks/ablation_study.py
python eval/benchmarks/latency_cost.py
```

### Ablation Studies

| Configuration | Top Score | Latency P50 |
|--------------|-----------|-------------|
| Full pipeline | 0.998 | 70ms |
| - Cross-encoder | 0.805 | 20ms |
| - GraphRAG | 0.998 | 72ms |
| - Query rewrite | 0.998 | 68ms |

See [eval/ablations/](eval/ablations/) for detailed results.

### Metrics Tracked

- **Retrieval Quality**: Top-k relevance scores, MRR, NDCG
- **Latency**: P50, P95, P99 by pipeline stage
- **Cost**: Tokens per query, $/1k queries
- **Citation Quality**: Groundedness, attribution accuracy

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `DEFAULT_LLM_PROVIDER` | LLM provider | openai |
| `EMBEDDING_MODEL` | Embedding model | BAAI/bge-m3 |
| `SURREALDB_URL` | SurrealDB connection | ws://localhost:8529 |
| `QDRANT_URL` | Qdrant connection | http://localhost:6333 |

### Optional Features

| Feature | Setting | Default |
|---------|---------|---------|
| ColBERT reranking | `COLBERT_ENABLED` | false |
| LLM reranking | `LLM_RERANKER_ENABLED` | false |
| GraphRAG | `GRAPH_MAX_HOPS` | 2 |
| Semantic cache | `SEMANTIC_CACHE_ENABLED` | true |

## Project Structure

```
perfect-rag/
├── src/perfect_rag/
│   ├── config.py              # Pydantic settings
│   ├── main.py                # FastAPI application
│   ├── models/                # Data models
│   ├── db/                    # Database clients
│   ├── llm/                   # LLM providers
│   ├── core/                  # Embedding, resilience
│   ├── ingestion/             # Document processing
│   ├── retrieval/             # Retrieval pipeline
│   └── generation/            # Response generation
├── eval/                      # Evaluation scripts & results
│   ├── benchmarks/            # Benchmark scripts
│   ├── ablations/             # Ablation studies
│   └── results/               # Benchmark results
├── docker-compose.yml         # Development setup
├── docker-compose.prod.yml    # Production setup
└── pyproject.toml
```

## Production Deployment

### Requirements

- Persistent volumes for SurrealDB and Qdrant
- Rate limiting (Redis-based for distributed)
- Health checks and metrics endpoint
- Proper secrets management

### Docker Compose Production

```bash
docker compose -f docker-compose.prod.yml up -d
```

See [docs/deployment.md](docs/deployment.md) for detailed production setup.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black src/ && ruff check src/ --fix
```

## License

MIT License

## Author

**Javier Pazó** - AEEH (Spanish Association for the Study of the Liver)

---

## Acknowledgments

- BGE-M3 embeddings by BAAI
- GraphRAG concepts from Microsoft Research
- PageIndex concept from VectifyAI (98.7% FinanceBench accuracy - their benchmark, not ours)
