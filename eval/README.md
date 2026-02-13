# Perfect RAG Evaluation

This directory contains reproducible benchmarks and ablation studies for Perfect RAG.

## Quick Start

```bash
# Run evaluation suite
python eval/run_all.py

# Run comprehensive benchmark (all configurations)
python benchmarks/benchmark_comprehensive.py --iterations 3

# Run ground truth benchmark (requires labeled data)
python eval/ground_truth_benchmark.py --queries eval/data/queries_with_gt.json
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

### Reranker Score (Primary Metric in Current Benchmarks)

| Metric | Definition | Range |
|--------|------------|-------|
| **Reranker Score** | Cross-encoder relevance score for top result | 0.0 - 1.0 |

**Important**: This measures the reranker's confidence, NOT retrieval accuracy against ground truth.

- Score > 0.9: Reranker is very confident the document is relevant
- Score 0.5-0.9: Reranker sees moderate relevance
- Score < 0.5: Reranker thinks document is not relevant

### Ground Truth Metrics (Recommended)

| Metric | Definition | Range |
|--------|------------|-------|
| **Hit@k** | Fraction of queries where relevant doc appears in top-k | 0.0 - 1.0 |
| **MRR** | Mean Reciprocal Rank of first relevant doc | 0.0 - 1.0 |
| **Recall@k** | Fraction of relevant docs found in top-k | 0.0 - 1.0 |

**To use these metrics**, you need queries with ground truth labels:

```json
{
  "queries": [
    {
      "id": "q1",
      "query": "What is TNM classification?",
      "relevant_doc_ids": ["doc_123", "doc_456"]
    }
  ]
}
```

### Latency Metrics

| Metric | Definition |
|--------|------------|
| **P50** | Median latency (50th percentile) |
| **P95** | 95th percentile latency |

## Current Benchmark Dataset

### Medical Q&A (AEEH Guidelines)

| Property | Value |
|----------|-------|
| **Queries** | 10 |
| **Iterations** | 3 per query |
| **Total runs** | 30 per configuration |
| **Corpus** | Medical guidelines (Gastroenterology) |
| **Document type** | PDFs with structured sections |
| **Ground truth** | Not available (reranker scores only) |

### Limitations

1. **No ground truth labels** - We measure reranker confidence, not actual accuracy
2. **Small query set** - Only 10 queries
3. **Single domain** - Medical guidelines only
4. **No external baselines** - No comparison with BM25, SPLADE, etc.

## Results Summary

### Reranker Scores (Current Benchmark)

| Configuration | Reranker Score | P50 (ms) | P95 (ms) | Success |
|--------------|----------------|----------|----------|---------|
| baseline | 0.805 | 20 | 44 | 100% |
| cross_encoder | 0.998 | 74 | 84 | 100% |
| full_pipeline | 0.998 | 72 | 75 | 100% |
| colbert | 0.998 | 124 | 165 | 100% |
| llm_rerank | 0.998 | 70 | 92 | 100% |

**Interpretation**: The cross-encoder is more confident about top results when used for reranking. This suggests improved ranking but does NOT prove better retrieval accuracy.

### Key Findings

1. **Reranker confidence increases with cross-encoder** (0.805 → 0.998)
2. **Latency overhead is ~50ms** for cross-encoder reranking
3. **ColBERT adds latency without improving reranker confidence**
4. **GraphRAG has minimal latency impact**

## Ablation Studies

See [ablations/README.md](ablations/README.md) for detailed ablation results.

## Running Your Own Benchmarks

### 1. Prepare Ground Truth (Recommended)

Create `eval/data/queries_with_gt.json`:

```json
{
  "queries": [
    {
      "id": "q1",
      "query": "What is TNM staging?",
      "relevant_doc_ids": ["doc_abc", "doc_def"],
      "relevant_titles": ["TNM Classification Guide"]
    }
  ]
}
```

### 2. Run Benchmark

```bash
# With ground truth (recommended)
python eval/ground_truth_benchmark.py --queries eval/data/queries_with_gt.json

# Without ground truth (reranker scores only)
python benchmarks/benchmark_comprehensive.py --iterations 3
```

## File Structure

```
eval/
├── README.md                    # This file
├── run_all.py                   # Run all benchmarks
├── ground_truth_benchmark.py    # Benchmark with ground truth metrics
├── ablations/
│   └── README.md                # Ablation study results
├── data/
│   └── queries.json             # Queries without ground truth
└── results/
    └── benchmark_results.json   # Benchmark results
```

## Adding New Benchmarks

1. Create script in `eval/` or `benchmarks/`
2. Use consistent metric definitions
3. Include ground truth if possible
4. Document methodology in this README
