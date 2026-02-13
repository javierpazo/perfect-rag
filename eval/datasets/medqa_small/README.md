# medqa_small Dataset

A small manually-annotated dataset for retrieval evaluation.

## Files

- `queries.jsonl` - Queries with reference answers
- `qrels.tsv` - Relevance judgments (TREC format)

## Statistics

| Metric | Value |
|--------|-------|
| Queries | 5 |
| Avg relevant docs per query | 2 |
| Categories | factual, therapeutic, diagnostic, preventive |

## Query Format

```json
{
  "qid": "q1",
  "query": "clasificación TNM del cáncer colorrectal",
  "answer_ref": "Reference answer for evaluation...",
  "category": "factual"
}
```

## Qrels Format

TREC standard format:
```
qid    iter    doc_id    relevance
q1     0       doc_123   2
```

Relevance levels:
- 0: Not relevant
- 1: Partially relevant
- 2: Highly relevant

## Usage

```bash
# Run IR metrics evaluation
python eval/metrics_ir.py --dataset eval/datasets/medqa_small
```

## Limitations

- Small size (5 queries) - suitable for quick validation, not comprehensive evaluation
- Manual annotations - may contain errors
- Single domain (gastroenterology) - results may not generalize
