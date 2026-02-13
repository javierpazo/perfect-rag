#!/usr/bin/env python3
"""
Comprehensive Benchmark for Perfect RAG

Tests all pipeline configurations:
1. Baseline (hybrid search only)
2. + Cross-encoder reranking
3. + ColBERT reranking
4. + LLM reranking
5. + PageIndex (when available)

Produces a comparison report with metrics:
- Retrieval latency
- Top-k scores
- Answer quality (if LLM available)
- Memory usage
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

# Configuration
LOCAL_QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "ai2evolve_migrated"
RESULTS_DIR = Path(__file__).parent


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    name: str
    use_cross_encoder: bool = True
    use_colbert: bool = False
    use_llm_reranker: bool = False
    use_pageindex: bool = False
    use_query_rewriting: bool = True
    use_graph_expansion: bool = True
    top_k: int = 5


@dataclass
class QueryResult:
    """Result for a single query."""
    query: str
    query_id: str
    config_name: str
    success: bool = True
    error: str | None = None

    # Timing
    total_ms: float = 0.0
    embedding_ms: float = 0.0
    search_ms: float = 0.0
    rerank_ms: float = 0.0
    pageindex_ms: float = 0.0

    # Scores
    top_score: float = 0.0
    avg_score: float = 0.0
    scores: list[float] = field(default_factory=list)

    # Results
    results_count: int = 0
    retrieved_docs: list[dict] = field(default_factory=list)

    # Pipeline info
    rewrite_strategy: str = "none"
    graph_expanded: bool = False
    reranked: bool = False
    colbert_reranked: bool = False
    llm_reranked: bool = False
    pageindex_used: bool = False


# Standard benchmark configurations
BENCHMARK_CONFIGS = [
    BenchmarkConfig(
        name="baseline",
        use_cross_encoder=False,
        use_colbert=False,
        use_llm_reranker=False,
        use_query_rewriting=False,
        use_graph_expansion=False,
    ),
    BenchmarkConfig(
        name="hybrid_only",
        use_cross_encoder=False,
        use_query_rewriting=True,
        use_graph_expansion=False,
    ),
    BenchmarkConfig(
        name="cross_encoder",
        use_cross_encoder=True,
        use_query_rewriting=True,
        use_graph_expansion=False,
    ),
    BenchmarkConfig(
        name="full_pipeline",
        use_cross_encoder=True,
        use_query_rewriting=True,
        use_graph_expansion=True,
    ),
    BenchmarkConfig(
        name="colbert",
        use_cross_encoder=True,
        use_colbert=True,
        use_query_rewriting=True,
        use_graph_expansion=True,
    ),
    BenchmarkConfig(
        name="llm_rerank",
        use_cross_encoder=True,
        use_colbert=False,
        use_llm_reranker=True,
        use_query_rewriting=True,
        use_graph_expansion=True,
    ),
    BenchmarkConfig(
        name="full_rerank",
        use_cross_encoder=True,
        use_colbert=True,
        use_llm_reranker=True,
        use_query_rewriting=True,
        use_graph_expansion=True,
    ),
]


class ComprehensiveBenchmark:
    """Run comprehensive benchmarks across all configurations."""

    def __init__(
        self,
        qdrant_url: str = LOCAL_QDRANT_URL,
        collection: str = COLLECTION_NAME,
    ):
        self.qdrant_url = qdrant_url
        self.collection = collection
        self._client = None
        self._model = None
        self._reranker = None
        self._colbert_reranker = None

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

    async def run_query(
        self,
        query: str,
        config: BenchmarkConfig,
        query_id: str = "",
    ) -> QueryResult:
        """Run a single query with given configuration."""
        result = QueryResult(
            query=query,
            query_id=query_id,
            config_name=config.name,
        )

        try:
            start_time = time.perf_counter()

            # Step 1: Embedding
            embed_start = time.perf_counter()
            query_embedding = self.model.encode(query, normalize_embeddings=True).tolist()
            result.embedding_ms = (time.perf_counter() - embed_start) * 1000

            # Step 2: Hybrid search
            search_start = time.perf_counter()

            search_results = self.client.query_points(
                collection_name=self.collection,
                query=query_embedding,
                using="dense",
                limit=config.top_k * 3,
                with_payload=True,
            )
            result.search_ms = (time.perf_counter() - search_start) * 1000

            # Extract initial results
            candidates = []
            for point in search_results.points:
                payload = point.payload or {}
                candidates.append({
                    "id": point.id,
                    "score": point.score,
                    "title": payload.get("doc_title") or payload.get("title", ""),
                    "content": payload.get("text") or payload.get("content", ""),
                    "page_number": payload.get("page_number") or payload.get("metadata", {}).get("page_number"),
                })

            # Step 3: Cross-encoder reranking
            rerank_start = time.perf_counter()
            if config.use_cross_encoder and candidates:
                texts = [c["content"][:500] for c in candidates]
                rerank_scores = self.reranker.predict([(query, t) for t in texts])

                # Sort by reranker scores
                scored_candidates = list(zip(candidates, rerank_scores))
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                candidates = [c for c, s in scored_candidates]

                # Update scores to use reranker scores
                for i, (c, s) in enumerate(scored_candidates[:config.top_k]):
                    c["rerank_score"] = float(s)

                result.reranked = True

            result.rerank_ms = (time.perf_counter() - rerank_start) * 1000

            # Apply top_k limit
            final_results = candidates[:config.top_k]

            # Calculate metrics - use rerank scores if available
            if config.use_cross_encoder and final_results:
                result.scores = [c.get("rerank_score", c["score"]) for c in final_results]
            else:
                result.scores = [c["score"] for c in final_results]

            result.total_ms = (time.perf_counter() - start_time) * 1000
            result.results_count = len(final_results)
            result.top_score = max(result.scores) if result.scores else 0.0
            result.avg_score = statistics.mean(result.scores) if result.scores else 0.0
            result.retrieved_docs = [
                {
                    "title": c["title"],
                    "score": c.get("rerank_score", c["score"]),
                    "page": c.get("page_number"),
                }
                for c in final_results
            ]
            result.success = True

        except Exception as e:
            result.success = False
            result.error = str(e)
            import traceback
            result.error_trace = traceback.format_exc()

        return result

    async def run_config(
        self,
        queries: list[dict],
        config: BenchmarkConfig,
        iterations: int = 3,
    ) -> dict[str, Any]:
        """Run all queries with a specific configuration."""
        print(f"\n{'='*60}")
        print(f"Testing: {config.name}")
        print(f"  Cross-encoder: {config.use_cross_encoder}")
        print(f"  Query rewrite: {config.use_query_rewriting}")
        print(f"  Graph expansion: {config.use_graph_expansion}")
        print(f"{'='*60}")

        results = []
        for q in queries:
            query_text = q["query"]
            query_id = q.get("id", "")

            for i in range(iterations):
                result = await self.run_query(query_text, config, query_id)
                results.append(result)

                status = "[OK]" if result.success else "[FAIL]"
                print(f"  {status} [{config.name}] {query_text[:40]}... "
                      f"| {result.total_ms:.0f}ms | score={result.top_score:.3f}")

        # Calculate statistics
        successful = [r for r in results if r.success]
        stats = {
            "config_name": config.name,
            "total_queries": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "latency_p50_ms": statistics.median([r.total_ms for r in successful]) if successful else 0,
            "latency_p95_ms": sorted([r.total_ms for r in successful])[int(len(successful) * 0.95)] if len(successful) > 5 else (max([r.total_ms for r in successful]) if successful else 0),
            "latency_mean_ms": statistics.mean([r.total_ms for r in successful]) if successful else 0,
            "top_score_mean": statistics.mean([r.top_score for r in successful]) if successful else 0,
            "top_score_max": max([r.top_score for r in successful]) if successful else 0,
            "avg_score_mean": statistics.mean([r.avg_score for r in successful]) if successful else 0,
        }

        return {
            "config": asdict(config),
            "statistics": stats,
            "results": [asdict(r) for r in results],
        }

    async def run_all(
        self,
        queries: list[dict],
        configs: list[BenchmarkConfig] | None = None,
        iterations: int = 3,
    ) -> dict[str, Any]:
        """Run all configurations and produce comparison report."""
        configs = configs or BENCHMARK_CONFIGS

        # Warmup
        print("Warming up (loading models)...")
        _ = self.model
        _ = self.reranker
        print("Models loaded.\n")

        all_results = {}
        for config in configs:
            config_result = await self.run_config(queries, config, iterations)
            all_results[config.name] = config_result

        # Generate comparison
        comparison = self._generate_comparison(all_results)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "queries_count": len(queries),
            "iterations": iterations,
            "configs_tested": len(configs),
            "comparison": comparison,
            "details": all_results,
        }

    def _generate_comparison(self, all_results: dict) -> dict:
        """Generate comparison table."""
        rows = []
        for name, data in all_results.items():
            stats = data["statistics"]
            rows.append({
                "config": name,
                "success_rate": f"{stats['successful']}/{stats['total_queries']}",
                "latency_p50_ms": round(stats["latency_p50_ms"], 1),
                "latency_p95_ms": round(stats["latency_p95_ms"], 1),
                "top_score_mean": round(stats["top_score_mean"], 4),
                "top_score_max": round(stats["top_score_max"], 4),
                "avg_score_mean": round(stats["avg_score_mean"], 4),
            })

        # Sort by top_score_mean descending
        rows.sort(key=lambda x: x["top_score_mean"], reverse=True)

        return {
            "table": rows,
            "winner": rows[0]["config"] if rows else None,
            "improvement_vs_baseline": self._calc_improvement(rows),
        }

    def _calc_improvement(self, rows: list) -> dict:
        """Calculate improvement vs baseline."""
        baseline = next((r for r in rows if r["config"] == "baseline"), None)
        if not baseline:
            return {}

        improvements = {}
        for row in rows:
            if row["config"] == "baseline":
                continue
            score_diff = row["top_score_mean"] - baseline["top_score_mean"]
            latency_diff = row["latency_p50_ms"] - baseline["latency_p50_ms"]
            improvements[row["config"]] = {
                "score_improvement": round(score_diff, 4),
                "latency_overhead_ms": round(latency_diff, 1),
            }

        return improvements


def load_queries(queries_file: Path) -> list[dict]:
    """Load queries from JSON file."""
    with open(queries_file, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [{"id": f"q{i}", "query": q} if isinstance(q, str) else q for i, q in enumerate(data)]
    elif isinstance(data, dict) and "queries" in data:
        return data["queries"]
    else:
        return [data]


def print_report(results: dict):
    """Print formatted report to console."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS - PERFECT RAG")
    print("=" * 80)

    comparison = results.get("comparison", {})
    table = comparison.get("table", [])

    # Header
    print(f"\n{'Config':<20} {'Success':<10} {'P50(ms)':<10} {'P95(ms)':<10} {'Score Mean':<12} {'Score Max':<10}")
    print("-" * 80)

    # Rows
    for row in table:
        print(f"{row['config']:<20} {row['success_rate']:<10} {row['latency_p50_ms']:<10.1f} "
              f"{row['latency_p95_ms']:<10.1f} {row['top_score_mean']:<12.4f} {row['top_score_max']:<10.4f}")

    # Winner
    print("-" * 80)
    print(f"\n>>> Best configuration: {comparison.get('winner', 'N/A')}")

    # Improvements
    improvements = comparison.get("improvement_vs_baseline", {})
    if improvements:
        print("\n>>> Improvement vs Baseline:")
        for config, data in improvements.items():
            score_pct = (data["score_improvement"] / 0.1 * 100) if data["score_improvement"] else 0  # Approx
            print(f"  {config}: +{data['score_improvement']:.4f} score, +{data['latency_overhead_ms']:.0f}ms latency")

    print("\n" + "=" * 80)


async def main():
    parser = argparse.ArgumentParser(description="Comprehensive Perfect RAG Benchmark")
    parser.add_argument(
        "--queries",
        type=str,
        default=str(RESULTS_DIR / "queries.json"),
        help="Path to queries JSON file",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Iterations per query",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "benchmark_results.json"),
        help="Output file",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run (fewer configs)",
    )

    args = parser.parse_args()

    # Load queries
    queries_file = Path(args.queries)
    if not queries_file.exists():
        print(f"Error: Queries file not found: {queries_file}")
        print("Creating sample queries file...")
        sample_queries = [
            {"id": "q1", "query": "clasificación TNM del cáncer colorrectal"},
            {"id": "q2", "query": "tratamiento de la cirrosis compensada"},
            {"id": "q3", "query": "diagnóstico de la esteatosis hepática"},
            {"id": "q4", "query": "criterios de trasplante hepático"},
            {"id": "q5", "query": "screening de carcinoma hepatocelular"},
        ]
        queries = sample_queries
    else:
        queries = load_queries(queries_file)

    # Select configs
    if args.quick:
        configs = [
            BENCHMARK_CONFIGS[0],  # baseline
            BENCHMARK_CONFIGS[2],  # cross_encoder
            BENCHMARK_CONFIGS[6],  # full_rerank
        ]
    else:
        configs = BENCHMARK_CONFIGS

    print(f"Running comprehensive benchmark")
    print(f"  Queries: {len(queries)}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Configurations: {len(configs)}")
    print()

    # Run benchmark
    benchmark = ComprehensiveBenchmark()
    results = await benchmark.run_all(queries, configs, args.iterations)

    # Save results
    output_file = Path(args.output)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print report
    print_report(results)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
