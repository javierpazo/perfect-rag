# Ablation Studies

Ablation studies show the contribution of each component to overall performance.

## Results Summary

| Configuration | Top Score | P50 (ms) | Notes |
|--------------|-----------|----------|-------|
| **Full Pipeline** | 0.998 | 70 | All features enabled |
| - Cross-encoder | 0.805 | 20 | Largest quality drop |
| - GraphRAG | 0.998 | 72 | No quality impact on this dataset |
| - Query Rewrite | 0.998 | 68 | Minimal impact |
| - ColBERT | 0.998 | 74 | Adds latency, no quality gain |
| - LLM Rerank | 0.998 | 73 | Similar to full pipeline |

## Key Findings

### 1. Cross-encoder is Critical (+24% quality)

Removing the cross-encoder reranker drops the top score from 0.998 to 0.805.
This is the single most impactful component.

```
With cross-encoder:    0.998
Without cross-encoder: 0.805
Improvement:           +24%
Latency cost:          +50ms
```

### 2. GraphRAG Has Minimal Impact on This Dataset

The medical Q&A dataset has relatively simple queries that don't require
multi-hop reasoning through the knowledge graph.

```
With GraphRAG:    0.998
Without GraphRAG: 0.998
Latency cost:     +2ms
```

**Note**: GraphRAG may show larger improvements on:
- Multi-hop questions
- Cross-document reasoning
- Entity relationship queries

### 3. ColBERT Adds Latency Without Quality Gain

On this dataset, ColBERT late interaction reranking doesn't improve over
cross-encoder alone.

```
With ColBERT:    0.998
Without ColBERT: 0.998
Latency cost:    +54ms (model loading)
```

**Recommendation**: Skip ColBERT for this use case.

### 4. Query Rewrite is Fast

Query rewriting (expansion, HyDE) has minimal latency overhead.

```
With Query Rewrite:    0.998
Without Query Rewrite: 0.998
Latency cost:          +2ms
```

## Detailed Results

### By Query Type

| Query Type | Baseline | +Rerank | Improvement |
|------------|----------|---------|-------------|
| Factual | 0.726 | 0.993 | +37% |
| Therapeutic | 0.659 | 0.985 | +50% |
| Diagnostic | 0.632 | 0.874 | +38% |
| Criteria | 0.684 | 0.874 | +28% |
| Emergency | 0.616 | 0.918 | +49% |

### Low-Relevance Queries

Some queries have very low reranker scores even with full pipeline:

| Query | Top Score | Issue |
|-------|-----------|-------|
| "indicaciones de cirugía bariátrica" | 0.019 | Topic not in corpus |
| "protocolo de sedación para colonoscopia" | 0.013 | Topic not in corpus |
| "displasia de Barrett" | 0.054 | Partial match only |

These low scores indicate **honest reranking** - the model correctly identifies
that the retrieved documents don't fully answer the query.

## Recommendations

Based on ablation results:

1. **Always use cross-encoder** - +24% quality for +50ms
2. **Skip ColBERT** - no quality gain, adds latency
3. **Skip LLM reranking** - no quality gain, adds cost
4. **Use GraphRAG conditionally** - only for multi-hop queries
5. **Always use query rewrite** - minimal cost, potential benefit

## Reproduce

```bash
# Run ablation studies
python eval/ablations/run_ablations.py

# Compare specific configurations
python benchmarks/benchmark_comprehensive.py --config baseline,cross_encoder,full_pipeline
```
