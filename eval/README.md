# Perfect RAG Evaluation

This directory contains reproducible benchmarks and ablation studies for Perfect RAG.

## Quick Start

```bash
# Run all benchmarks
python eval/run_all.py

# Run specific benchmark
python eval/benchmarks/retrieval_quality.py --config eval/configs/full_pipeline.yaml
```

## Benchmarked Configurations

| Config | Description |
|--------|-------------|
| `baseline` | Dense vector search only (no reranking) |
| `cross_encoder` | Dense + Cross-encoder reranking |
| `full_pipeline` | Dense + GraphRAG + Cross-encoder |
| `colbert` | Full pipeline + ColBERT late interaction |
| `llm_rerank` | Full pipeline + LLM reranking |

## Metrics Explained

### Retrieval Quality

| Metric | Definition | Range |
|--------|------------|-------|
| **Top Score** | Reranker relevance score for top-1 result | 0.0 - 1.0 |
| **Avg Score** | Mean relevance score for top-k results | 0.0 - 1.0 |
| **MRR@k** | Mean Reciprocal Rank at k | 0.0 - 1.0 |
| **Recall@k** | Fraction of relevant docs in top-k | 0.0 - 1.0 |

**Note on "Top Score"**: This is the cross-encoder (BGE-reranker-v2-m3) relevance score.
- Score > 0.9: Highly relevant, direct answer to query
- Score 0.5-0.9: Relevant, contains useful information
- Score < 0.5: Low relevance, may be tangentially related

### Latency

| Metric | Definition |
|--------|------------|
| **P50** | Median latency (50th percentile) |
| **P95** | 95th percentile latency |
| **P99** | 99th percentile latency |

### Cost

| Metric | Definition |
|--------|------------|
| **Tokens/query** | Average tokens consumed per query |
| **$/1k queries** | Estimated cost for 1000 queries |

## Benchmark Dataset

### Medical Q&A (AEEH Guidelines)

| Property | Value |
|----------|-------|
| **Queries** | 10 |
| **Iterations** | 3 per query |
| **Total runs** | 30 per configuration |
| **Corpus** | Medical guidelines (Gastroenterology) |
| **Corpus size** | ~500 documents |
| **Document type** | PDFs with structured sections |
| **Query types** | Factual, diagnostic, therapeutic |

### Query Examples

```json
[
  {"id": "q1", "query": "clasificación TNM del cáncer colorrectal", "type": "factual"},
  {"id": "q2", "query": "tratamiento de primera línea para Helicobacter pylori", "type": "therapeutic"},
  {"id": "q3", "query": "algoritmo diagnóstico de enfermedad inflamatoria intestinal", "type": "diagnostic"}
]
```

## Reproducibility

### Environment

| Requirement | Value |
|-------------|-------|
| Python | 3.11+ |
| PyTorch | 2.0+ |
| CUDA | 12.x (optional) |
| Embedding model | BAAI/bge-m3 |
| Reranker model | BAAI/bge-reranker-v2-m3 |

### Seeds

All benchmarks use fixed seeds for reproducibility:
- NumPy seed: 42
- Random seed: 42
- Torch seed: 42

### Hardware

Benchmarks run on:
- CPU: Varies (report your specs)
- GPU: Optional (CUDA for faster inference)
- RAM: 16GB+ recommended

### Cold vs Warm Cache

- **Cold**: First query (includes model loading)
- **Warm**: Subsequent queries (models in memory)

All reported metrics are **warm cache** (excludes first query).

## Results Summary

### Main Results (Medical Q&A)

| Configuration | Top Score | P50 (ms) | P95 (ms) | Success |
|--------------|-----------|----------|----------|---------|
| baseline | 0.805 | 20 | 44 | 100% |
| cross_encoder | 0.998 | 74 | 84 | 100% |
| full_pipeline | 0.998 | 72 | 75 | 100% |
| colbert | 0.998 | 124 | 165 | 100% |
| llm_rerank | 0.998 | 70 | 92 | 100% |

### Key Findings

1. **Cross-encoder reranking improves top score by 24%** (0.805 → 0.998)
2. **Latency overhead is acceptable** (+50-54ms for cross-encoder)
3. **ColBERT adds latency without quality improvement** on this dataset
4. **GraphRAG has minimal latency impact** with comparable quality

### Limitations

- Small query set (10 queries)
- Single domain (medical)
- No external baselines (BM25, SPLADE, etc.)
- No human evaluation of answer quality

## Ablation Studies

See [ablations/](ablations/) for detailed ablation results:

- [Without Cross-encoder](ablations/no_rerank.md)
- [Without GraphRAG](ablations/no_graph.md)
- [Without Query Rewrite](ablations/no_rewrite.md)
- [Without HyDE](ablations/no_hyde.md)

## Running Your Own Benchmarks

1. Prepare your corpus in `eval/data/`
2. Create query file in `eval/data/queries.json`
3. Run benchmark:

```bash
python eval/benchmarks/retrieval_quality.py \
  --queries eval/data/queries.json \
  --output eval/results/my_results.json \
  --iterations 5
```

## Contributing Benchmarks

To add a new benchmark:

1. Create script in `eval/benchmarks/`
2. Add config in `eval/configs/`
3. Document in this README
4. Submit results to `eval/results/`
