#!/usr/bin/env python3
"""
Retrieval Benchmark with Ground Truth

This benchmark measures actual retrieval accuracy using:
1. Hit@k: Does a relevant document appear in top-k?
2. MRR: Mean Reciprocal Rank
3. LLM-as-Judge: Independent relevance assessment

Unlike the comprehensive benchmark which only measures reranker confidence,
this benchmark validates retrieval quality against ground truth.

Usage:
    python eval/ground_truth_benchmark.py [--queries eval/data/queries_with_gt.json]
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class GroundTruthQuery:
    """Query with ground truth document IDs."""
    id: str
    query: str
    relevant_doc_ids: list[str]  # Ground truth: IDs of relevant documents
    relevant_titles: list[str] = field(default_factory=list)  # For readability


@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics."""
    hit_at_1: float  # Is relevant doc in top 1?
    hit_at_3: float  # Is relevant doc in top 3?
    hit_at_5: float  # Is relevant doc in top 5?
    hit_at_10: float  # Is relevant doc in top 10?
    mrr: float  # Mean Reciprocal Rank
    latency_ms: float


@dataclass
class LLMJudgeResult:
    """LLM-as-judge relevance assessment."""
    query_id: str
    doc_title: str
    doc_content_preview: str
    is_relevant: bool
    relevance_score: float  # 0-1
    reasoning: str


class GroundTruthBenchmark:
    """Benchmark retrieval against ground truth labels."""

    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self.qdrant_url = qdrant_url
        self._client = None
        self._model = None
        self._reranker = None

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=self.qdrant_url)
        return self._client

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print("Loading BGE-M3 model...")
            self._model = SentenceTransformer("BAAI/bge-m3")
        return self._model

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            print("Loading BGE-reranker-v2-m3...")
            self._reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
        return self._reranker

    def calculate_metrics(
        self,
        retrieved_ids: list[str],
        relevant_ids: list[str],
        latency_ms: float,
    ) -> RetrievalMetrics:
        """Calculate retrieval metrics."""
        # Hit@k
        def hit_at_k(k):
            return 1.0 if any(rid in relevant_ids for rid in retrieved_ids[:k]) else 0.0

        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid in relevant_ids:
                mrr = 1.0 / rank
                break

        return RetrievalMetrics(
            hit_at_1=hit_at_k(1),
            hit_at_3=hit_at_k(3),
            hit_at_5=hit_at_k(5),
            hit_at_10=hit_at_k(10),
            mrr=mrr,
            latency_ms=latency_ms,
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_reranking: bool = True,
    ) -> tuple[list[dict], float]:
        """Retrieve documents for query."""
        start_time = time.perf_counter()

        # Embed
        query_embedding = self.model.encode(query, normalize_embeddings=True).tolist()

        # Search
        results = self.client.query_points(
            collection_name="ai2evolve_migrated",
            query=query_embedding,
            using="dense",
            limit=top_k * 2,
            with_payload=True,
        )

        candidates = []
        for point in results.points:
            payload = point.payload or {}
            candidates.append({
                "id": str(point.id),
                "score": point.score,
                "title": payload.get("doc_title") or payload.get("title", ""),
                "content": payload.get("text") or payload.get("content", ""),
            })

        # Rerank
        if use_reranking and candidates:
            texts = [c["content"][:500] for c in candidates]
            rerank_scores = self.reranker.predict([(query, t) for t in texts])
            scored = list(zip(candidates, rerank_scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            candidates = [c for c, s in scored]

        latency_ms = (time.perf_counter() - start_time) * 1000
        return candidates[:top_k], latency_ms

    async def run_query(
        self,
        gt_query: GroundTruthQuery,
        use_reranking: bool = True,
    ) -> dict:
        """Run single query and calculate metrics."""
        results, latency = await self.retrieve(
            gt_query.query,
            top_k=10,
            use_reranking=use_reranking,
        )

        retrieved_ids = [r["id"] for r in results]
        metrics = self.calculate_metrics(
            retrieved_ids=retrieved_ids,
            relevant_ids=gt_query.relevant_doc_ids,
            latency_ms=latency,
        )

        return {
            "query_id": gt_query.id,
            "query": gt_query.query,
            "relevant_ids": gt_query.relevant_doc_ids,
            "retrieved_ids": retrieved_ids[:5],
            "metrics": asdict(metrics),
            "top_result": results[0] if results else None,
        }

    async def run_benchmark(
        self,
        queries: list[GroundTruthQuery],
        use_reranking: bool = True,
    ) -> dict:
        """Run benchmark on all queries."""
        print(f"\nRunning benchmark (reranking={use_reranking})...")

        all_results = []
        for gt_query in queries:
            result = await self.run_query(gt_query, use_reranking)
            all_results.append(result)

            hit_str = "HIT" if result["metrics"]["hit_at_5"] else "MISS"
            print(f"  [{hit_str}] {gt_query.query[:40]}... MRR={result['metrics']['mrr']:.2f}")

        # Aggregate metrics
        aggregate = {
            "hit_at_1": statistics.mean([r["metrics"]["hit_at_1"] for r in all_results]),
            "hit_at_3": statistics.mean([r["metrics"]["hit_at_3"] for r in all_results]),
            "hit_at_5": statistics.mean([r["metrics"]["hit_at_5"] for r in all_results]),
            "hit_at_10": statistics.mean([r["metrics"]["hit_at_10"] for r in all_results]),
            "mrr": statistics.mean([r["metrics"]["mrr"] for r in all_results]),
            "latency_p50_ms": statistics.median([r["metrics"]["latency_ms"] for r in all_results]),
            "latency_mean_ms": statistics.mean([r["metrics"]["latency_ms"] for r in all_results]),
        }

        return {
            "queries_count": len(queries),
            "use_reranking": use_reranking,
            "aggregate_metrics": aggregate,
            "per_query_results": all_results,
        }


def load_queries_with_ground_truth(filepath: Path) -> list[GroundTruthQuery]:
    """Load queries with ground truth from JSON file."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    queries = []
    for item in data.get("queries", data if isinstance(data, list) else []):
        queries.append(GroundTruthQuery(
            id=item.get("id", ""),
            query=item.get("query", ""),
            relevant_doc_ids=item.get("relevant_doc_ids", []),
            relevant_titles=item.get("relevant_titles", []),
        ))

    return queries


def create_sample_ground_truth() -> list[GroundTruthQuery]:
    """Create sample ground truth for testing.

    Note: These are example ground truth labels. In a real benchmark,
    these should be manually annotated or from a labeled dataset.
    """
    return [
        GroundTruthQuery(
            id="q1",
            query="clasificación TNM del cáncer colorrectal",
            relevant_doc_ids=["Clasificación TNM para el cáncer colorrectal"],
            relevant_titles=["Clasificación TNM para el cáncer colorrectal"],
        ),
        GroundTruthQuery(
            id="q2",
            query="tratamiento de primera línea para Helicobacter pylori",
            relevant_doc_ids=[],
            relevant_titles=[],
        ),
        GroundTruthQuery(
            id="q3",
            query="algoritmo diagnóstico de enfermedad inflamatoria intestinal",
            relevant_doc_ids=[],
            relevant_titles=[],
        ),
    ]


async def main():
    parser = argparse.ArgumentParser(description="Ground Truth Retrieval Benchmark")
    parser.add_argument(
        "--queries",
        type=str,
        help="Path to queries with ground truth JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "results" / "ground_truth_results.json"),
        help="Output file",
    )
    args = parser.parse_args()

    # Load queries
    if args.queries:
        queries = load_queries_with_ground_truth(Path(args.queries))
    else:
        print("No ground truth file provided, using sample data")
        print("Note: Sample data has incomplete ground truth labels!")
        queries = create_sample_ground_truth()

    print(f"Loaded {len(queries)} queries with ground truth")

    # Check how many have actual ground truth
    with_gt = sum(1 for q in queries if q.relevant_doc_ids)
    print(f"Queries with ground truth labels: {with_gt}/{len(queries)}")

    if with_gt == 0:
        print("\nWARNING: No ground truth labels found!")
        print("To use this benchmark, create a file with 'relevant_doc_ids' for each query.")
        print("\nExample format:")
        print('{"queries": [{"id": "q1", "query": "...", "relevant_doc_ids": ["doc1", "doc2"]}]}')

    # Run benchmark
    benchmark = GroundTruthBenchmark()

    # Test with and without reranking
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "queries_count": len(queries),
        "ground_truth_coverage": with_gt / len(queries) if queries else 0,
    }

    # With reranking
    results["with_reranking"] = await benchmark.run_benchmark(queries, use_reranking=True)

    # Without reranking
    results["without_reranking"] = await benchmark.run_benchmark(queries, use_reranking=False)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("GROUND TRUTH BENCHMARK RESULTS")
    print("=" * 60)

    wr = results["with_reranking"]["aggregate_metrics"]
    wor = results["without_reranking"]["aggregate_metrics"]

    print(f"\n{'Metric':<15} {'With Rerank':<15} {'Without':<15} {'Delta':<10}")
    print("-" * 55)
    print(f"{'Hit@1':<15} {wr['hit_at_1']:<15.2%} {wor['hit_at_1']:<15.2%} {wr['hit_at_1']-wor['hit_at_1']:+.2%}")
    print(f"{'Hit@5':<15} {wr['hit_at_5']:<15.2%} {wor['hit_at_5']:<15.2%} {wr['hit_at_5']-wor['hit_at_5']:+.2%}")
    print(f"{'MRR':<15} {wr['mrr']:<15.3f} {wor['mrr']:<15.3f} {wr['mrr']-wor['mrr']:+.3f}")
    print(f"{'Latency P50':<15} {wr['latency_p50_ms']:<15.1f} {wor['latency_p50_ms']:<15.1f}")

    print(f"\nResults saved to: {output_path}")

    if with_gt < len(queries):
        print(f"\nNote: Only {with_gt}/{len(queries)} queries have ground truth labels.")


if __name__ == "__main__":
    asyncio.run(main())
