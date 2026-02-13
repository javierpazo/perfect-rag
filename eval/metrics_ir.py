#!/usr/bin/env python3
"""
Standard IR Metrics Evaluation

Uses ranx for computing standard retrieval metrics:
- Recall@k
- nDCG@k
- MRR@k
- Hit@k

Input: qrels.tsv (ground truth) + run.tsv (system output)
Output: metrics.json

Usage:
    python eval/metrics_ir.py --dataset eval/datasets/medqa_small --run eval/runs/baseline.tsv
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class RetrievalResult:
    """Single retrieval result."""
    qid: str
    doc_id: str
    rank: int
    score: float


def load_qrels(filepath: Path) -> dict[str, dict[str, int]]:
    """Load qrels in TREC format.

    Returns: {qid: {doc_id: relevance}}
    """
    qrels = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 4:
                qid, _, doc_id, rel = parts[:4]
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][doc_id] = int(rel)
    return qrels


def load_run(filepath: Path) -> dict[str, list[tuple[str, int, float]]]:
    """Load run in TREC format.

    Returns: {qid: [(doc_id, rank, score), ...]}
    """
    run = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 6:
                qid, _, doc_id, rank, score, _ = parts[:6]
                if qid not in run:
                    run[qid] = []
                run[qid].append((doc_id, int(rank), float(score)))
    return run


def save_run(results: dict[str, list[tuple[str, int, float]]], filepath: Path):
    """Save run in TREC format."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# TREC run file\n")
        for qid, docs in results.items():
            for doc_id, rank, score in docs:
                f.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score:.4f}\tperfect-rag\n")


def compute_metrics(
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[tuple[str, int, float]]],
    k_values: list[int] = None,
) -> dict[str, float]:
    """Compute IR metrics.

    Simple implementation without external dependencies.
    For production, use ranx or pytrec_eval.
    """
    k_values = k_values or [1, 3, 5, 10, 20]

    metrics = {}

    # For each k value
    for k in k_values:
        recalls = []
        ndcgs = []
        hits = []
        mrrs = []

        for qid, relevant_docs in qrels.items():
            if qid not in run:
                continue

            # Get top-k results
            top_k = sorted(run[qid], key=lambda x: x[1])[:k]
            retrieved_ids = [doc_id for doc_id, _, _ in top_k]

            # Binary relevance for recall/hit
            binary_rel = {doc_id: 1 if rel > 0 else 0 for doc_id, rel in relevant_docs.items()}
            total_relevant = sum(binary_rel.values())

            # Recall@k
            if total_relevant > 0:
                found = sum(binary_rel.get(doc_id, 0) for doc_id in retrieved_ids)
                recalls.append(found / total_relevant)
            else:
                recalls.append(0.0)

            # Hit@k (any relevant in top-k?)
            hit = 1.0 if any(binary_rel.get(doc_id, 0) > 0 for doc_id in retrieved_ids) else 0.0
            hits.append(hit)

            # nDCG@k
            dcg = 0.0
            for i, doc_id in enumerate(retrieved_ids):
                rel = binary_rel.get(doc_id, 0)
                dcg += rel / (i + 1)  # Simplified DCG

            # Ideal DCG
            ideal_rels = sorted(binary_rel.values(), reverse=True)[:k]
            idcg = sum(rel / (i + 1) for i, rel in enumerate(ideal_rels))

            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)

            # MRR@k
            mrr = 0.0
            for i, doc_id in enumerate(retrieved_ids):
                if binary_rel.get(doc_id, 0) > 0:
                    mrr = 1.0 / (i + 1)
                    break
            mrrs.append(mrr)

        # Aggregate
        if recalls:
            metrics[f"recall@{k}"] = sum(recalls) / len(recalls)
            metrics[f"ndcg@{k}"] = sum(ndcgs) / len(ndcgs)
            metrics[f"hit@{k}"] = sum(hits) / len(hits)
            metrics[f"mrr@{k}"] = sum(mrrs) / len(mrrs)

    return metrics


def compute_qa_metrics(
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[tuple[str, int, float]]],
    citations: dict[str, list[str]] = None,
) -> dict[str, float]:
    """Compute Q&A specific metrics.

    - Attribution: % of queries where at least 1 citation is in qrels
    """
    if not citations:
        return {"attribution": 0.0}

    attribution_scores = []
    for qid, cited_docs in citations.items():
        if qid not in qrels:
            continue

        relevant_docs = {doc_id for doc_id, rel in qrels[qid].items() if rel > 0}
        cited_set = set(cited_docs)

        # Attribution: at least one citation is relevant
        attr = 1.0 if cited_set & relevant_docs else 0.0
        attribution_scores.append(attr)

    return {
        "attribution": sum(attribution_scores) / len(attribution_scores) if attribution_scores else 0.0
    }


async def run_retrieval_and_save(
    dataset_path: Path,
    output_path: Path,
    use_reranking: bool = True,
):
    """Run retrieval and save results in TREC format."""
    import asyncio

    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer, CrossEncoder

    # Load queries
    queries = []
    with open(dataset_path / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))

    # Initialize
    print("Loading models...")
    client = QdrantClient(url="http://localhost:6333")
    model = SentenceTransformer("BAAI/bge-m3")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3") if use_reranking else None

    # Run retrieval
    results = {}
    for q in queries:
        qid = q["qid"]
        query = q["query"]

        # Embed
        embedding = model.encode(query, normalize_embeddings=True).tolist()

        # Search
        search_results = client.query_points(
            collection_name="ai2evolve_migrated",
            query=embedding,
            using="dense",
            limit=20,
            with_payload=True,
        )

        candidates = []
        for point in search_results.points:
            payload = point.payload or {}
            doc_id = payload.get("doc_title") or payload.get("title", str(point.id))
            candidates.append((doc_id, point.score, payload.get("text", "")[:500]))

        # Rerank
        if reranker and candidates:
            texts = [c[2] for c in candidates]
            scores = reranker.predict([(query, t) for t in texts])
            candidates = [(c[0], s, c[2]) for c, s in zip(candidates, scores)]
            candidates.sort(key=lambda x: x[1], reverse=True)

        # Store
        results[qid] = [
            (doc_id, rank + 1, score)
            for rank, (doc_id, score, _) in enumerate(candidates[:20])
        ]

        print(f"  {qid}: retrieved {len(results[qid])} docs")

    # Save
    save_run(results, output_path)
    print(f"Saved run to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="IR Metrics Evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory (contains queries.jsonl and qrels.tsv)",
    )
    parser.add_argument(
        "--run",
        type=str,
        help="Path to run file (TREC format). If not provided, runs retrieval.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for metrics JSON",
    )
    parser.add_argument(
        "--no-reranking",
        action="store_true",
        help="Disable reranking when running retrieval",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    # Load qrels
    qrels_path = dataset_path / "qrels.tsv"
    if not qrels_path.exists():
        print(f"Error: qrels not found at {qrels_path}")
        return 1

    qrels = load_qrels(qrels_path)
    print(f"Loaded {len(qrels)} queries with relevance judgments")

    # Get or create run
    if args.run:
        run_path = Path(args.run)
    else:
        # Run retrieval
        run_name = "no_rerank" if args.no_reranking else "with_rerank"
        run_path = dataset_path / f"run_{run_name}.tsv"

        if not run_path.exists():
            print("Running retrieval...")
            import asyncio
            asyncio.run(run_retrieval_and_save(
                dataset_path,
                run_path,
                use_reranking=not args.no_reranking,
            ))

    run = load_run(run_path)
    print(f"Loaded run with {len(run)} queries")

    # Compute metrics
    metrics = compute_metrics(qrels, run)

    # Print results
    print("\n" + "=" * 50)
    print("IR METRICS")
    print("=" * 50)

    print(f"\n{'Metric':<15} {'Value':<10}")
    print("-" * 30)
    for metric, value in sorted(metrics.items()):
        print(f"{metric:<15} {value:<10.4f}")

    # Save
    output_path = Path(args.output) if args.output else dataset_path / "metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
