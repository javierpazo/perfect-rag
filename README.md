# Perfect RAG

**A production-grade RAG system with state-of-the-art retrieval techniques.** Combining GraphRAG, hybrid search, multi-stage reranking, and cache-augmented generation.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Benchmarked Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Reranker Score (with reranking)** | 0.998 | Cross-encoder relevance for top result |
| **Reranker Score (baseline)** | 0.805 | Dense vector similarity only |
| **Latency P50** | 70ms | Full pipeline with reranking |
| **Latency P95** | 92ms | Full pipeline with reranking |
| **Success Rate** | 100% | 210 queries tested |

> **Important**: "Reranker Score" measures the cross-encoder's confidence in result relevance, NOT retrieval accuracy against ground truth labels. For production evaluation, use ground truth metrics (Recall@k, nDCG@k, MRR) with domain-specific test sets.

> **Note on Sparse**: The current BM25 implementation is Python-based for prototyping. For production workloads requiring maximum recall, consider integrating Elasticsearch/Tantivy for true inverted index with phrase/proximity queries.

> See [eval/results/](eval/results/) for full benchmark data.

## Why Perfect RAG?

| Feature | Perfect RAG | Typical RAG |
|---------|-------------|-------------|
| **Retrieval** | Hybrid (Dense + BM25 + RRF) | Dense only |
| **Sparse** | BM25 with phrase/proximity queries | TF-IDF or none |
| **Graph** | GraphRAG with entity expansion | None |
| **Reranking** | 3-stage (Cross-encoder + ColBERT + LLM) | Single reranker or none |
| **Multi-query** | RAG-Fusion with intent routing | Single query |
| **Generation** | Evidence-first (2-step) | Direct generation |
| **Diversity** | MMR for context diversification | None |
| **Citations** | Page-level with verification | Chunk-level or none |
| **Cache** | CAG + Semantic cache | None |
| **Fallback** | Confidence-based automatic fallback | None |
| **Query Smart** | Context gate + rewriting | Direct search |

## Features

- **OpenAI-compatible API** - Drop-in replacement for OpenAI chat completions
- **Hybrid Search** - Dense (BGE-M3) + BM25 with phrase/proximity queries + RRF fusion
- **GraphRAG** - Knowledge graph expansion using SurrealDB
- **Multi-stage Reranking** - Cross-encoder → ColBERT → LLM (optional)
- **RAG-Fusion** - Multi-query retrieval with intent routing
- **Evidence-First Generation** - 2-step generation to reduce hallucinations
- **MMR Diversification** - Maximal Marginal Relevance for diverse context
- **Confidence Estimation** - Automatic fallback on low confidence
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

### Ingredients

```
qdrant                    # Vector DB
surrealdb                 # Graph DB + metadata
sentence-transformers     # Embeddings framework
BAAI/bge-m3              # Dense embeddings (1024d)
BAAI/bge-reranker-v2-m3  # Cross-encoder reranker
LLM API (openai/anthropic/ollama)
```

### Preparation

#### 1. INGESTION

```
document → chunker(512 tokens, 50 overlap)
        → embed_dense(bge-m3)
        → extract_entities(LLM) [optional]
        → store in Qdrant (vectors) + SurrealDB (metadata + graph)
```

#### 2. RETRIEVAL (10-step pipeline)

```
query
  ↓
[1] Context Gate → Skip retrieval if not needed
  ↓
[2] Query Rewrite → Expansion + HyDE + Decomposition
  ↓
[2.5] RAG-Fusion → Multi-query with intent routing (optional)
  ↓
[3] Hybrid Search → Dense + BM25 with phrase/proximity + RRF fusion
  ↓
[4] GraphRAG → Expand by entities (optional)
  ↓
[5] Cross-encoder → Rerank with bge-reranker
  ↓
[6] ColBERT → Late interaction (optional)
  ↓
[7] LLM Rerank → Semantic reranking (optional)
  ↓
[8] MMR Diversification → Reduce redundancy in context
  ↓
[9] Confidence Estimation → Score result quality
  ↓
[10] Evidence-First Generation → 2-step answer generation
  ↓
answer with citations
```

#### 3. GENERATION

```
chunks + question
  → evidence_extractor() → verified facts only
  → prompt_builder(formatted context + citations [1][2])
  → LLM.generate()
  → response with page-level citations
```

### Secret Sauce

| Technique | Impact | Why it works |
|-----------|--------|--------------|
| **Cross-encoder Reranking** | +24% relevance score | Semantic relevance scoring |
| **Evidence-First Generation** | -hallucinations | LLM only uses verified evidence |
| **RAG-Fusion** | +recall | Multiple query angles, RRF fusion |
| **MMR Diversification** | +coverage | Diverse context, less redundancy |
| **Confidence Fallback** | +reliability | Auto-retry on low confidence |
| **GraphRAG** | +multi-hop | Entity-connected context |
| **Context Gate** | -latency | Skips retrieval when not needed |
| **Semantic Cache** | -cost | Reuses similar query responses |

### What Makes This "Top-Tier" for Domain-Specific RAG

1. **Evidence-First by Default**: Generation is constrained to verified facts from retrieved chunks
2. **Multi-Stage Reranking**: Cross-encoder → ColBERT → LLM provides semantic refinement
3. **Automatic Quality Gates**: Confidence estimation triggers fallback when results are uncertain
4. **Diversity in Context**: MMR prevents redundant information from dominating the context window

## Evaluation

### Run Benchmarks

```bash
# IR metrics (requires ground truth)
python eval/metrics_ir.py --dataset eval/datasets/medqa_small

# Q&A metrics (EM/F1/attribution)
python eval/metrics_qa.py --dataset eval/datasets/medqa_small

# Load test (requires k6)
k6 run eval/load_test.js

# Comprehensive benchmark (reranker scores)
python benchmarks/benchmark_comprehensive.py --iterations 3
```

### Ablation Studies

| Configuration | Reranker Score | Latency P50 |
|--------------|----------------|-------------|
| Full pipeline | 0.998 | 70ms |
| - Cross-encoder | 0.805 | 20ms |
| - GraphRAG | 0.998 | 72ms |
| - Query rewrite | 0.998 | 68ms |

See [eval/ablations/](eval/ablations/) for detailed results.

### Metrics

**Current benchmark measures:**
- **Reranker Score**: Cross-encoder confidence (0.0-1.0) - NOT retrieval accuracy

**Ground truth metrics (recommended):**
- **Recall@k**: Fraction of relevant docs in top-k
- **nDCG@k**: Normalized discounted cumulative gain
- **MRR@k**: Mean reciprocal rank
- **EM/F1**: Answer quality vs reference

> **Limitation**: Current results show reranker confidence, not retrieval accuracy against ground truth. See [eval/README.md](eval/README.md) to add ground truth evaluation.

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

### Core Features (Enabled by Default)

| Feature | Setting | Default | Description |
|---------|---------|---------|-------------|
| BM25 sparse retrieval | `BM25_ENABLED` | true | Real BM25 with phrase/proximity |
| MMR diversification | `MMR_ENABLED` | true | Diverse context selection |
| Evidence-first generation | `EVIDENCE_FIRST_ENABLED` | true | 2-step generation reduces hallucinations |
| Confidence fallback | `CONFIDENCE_FALLBACK_ENABLED` | true | Auto-retry on low confidence |
| Entity normalization | `ENTITY_NORMALIZATION_ENABLED` | true | Canonical forms for entities |
| Semantic cache | `SEMANTIC_CACHE_ENABLED` | true | Cache similar queries |
| GraphRAG | `GRAPH_MAX_HOPS` | 2 | Knowledge graph expansion |

### Optional Features

| Feature | Setting | Default |
|---------|---------|---------|
| ColBERT reranking | `COLBERT_ENABLED` | false |
| LLM reranking | `LLM_RERANKER_ENABLED` | false |
| PageIndex (tree search) | `PAGEINDEX_ENABLED` | false |
| SPLADE sparse | `SPLADE_ENABLED` | false |

### Retrieval Modules

| Module | Description | File |
|--------|-------------|------|
| **BM25 Index** | Real BM25 with phrase/proximity queries | `retrieval/sparse_bm25.py` |
| **SPLADE** | Learned sparse representations | `retrieval/splade_sparse.py` |
| **RAG-Fusion** | Multi-query with intent routing | `retrieval/rag_fusion.py` |
| **MMR** | Maximal Marginal Relevance diversification | `retrieval/mmr.py` |
| **Confidence** | Estimation + automatic fallback | `retrieval/confidence.py` |
| **Evidence-First** | 2-step generation (extract evidence → answer) | `generation/evidence_first.py` |
| **Entity Normalizer** | Canonical forms + deduplication | `retrieval/entity_normalization.py` |

### Integration Status

| Feature | Status | Notes |
|---------|--------|-------|
| BM25 in main pipeline | ✅ Integrated | Active in `_hybrid_search_enhanced()` |
| Phrase/proximity routing | ✅ Integrated | Query type detection in `retrieve()` |
| Evidence-first generation | ✅ Integrated | Default in generation pipeline |
| MMR diversification | ✅ Integrated | Active by default |
| Confidence + fallback | ✅ Integrated | Auto-triggers on low scores |
| Entity normalization | ✅ Integrated | Used in GraphRAG expansion |

> **Note**: All core modules are now integrated and active by default. Optional features (ColBERT, LLM reranking, PageIndex) can be enabled via settings.

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
- Reranking best practices from Cohere and BAAI
