#!/usr/bin/env python3
"""
Perfect RAG Evaluation Suite

Runs all benchmarks and generates comprehensive reports.

Usage:
    python eval/run_all.py [--quick] [--output-dir eval/results]

Options:
    --quick         Run quick benchmarks (fewer iterations)
    --output-dir    Directory for results (default: eval/results)
    --config        Specific config to run (default: all)
"""

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

EVAL_DIR = Path(__file__).parent
BENCHMARKS_DIR = EVAL_DIR / "benchmarks"
RESULTS_DIR = EVAL_DIR / "results"


def run_benchmark(script: Path, args: list[str]) -> dict:
    """Run a benchmark script and return results."""
    cmd = [sys.executable, str(script)] + args
    result = subprocess.run(cmd, capture_output=True, text=True)

    return {
        "script": str(script),
        "args": args,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Perfect RAG evaluation suite")
    parser.add_argument("--quick", action="store_true", help="Quick run (fewer iterations)")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--config", type=str, default="all", help="Config to run")
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Perfect RAG Evaluation Suite")
    print("=" * 60)
    print(f"Started: {datetime.utcnow().isoformat()}")
    print(f"Output: {output_dir}")
    print()

    iterations = 2 if args.quick else 3

    # Run comprehensive benchmark
    print("\n[1/3] Running retrieval quality benchmark...")
    benchmark_script = Path(__file__).parent.parent / "benchmarks" / "benchmark_comprehensive.py"

    if benchmark_script.exists():
        result = run_benchmark(
            benchmark_script,
            ["--iterations", str(iterations), "--output", str(output_dir / "benchmark_results.json")],
        )
        if result["success"]:
            print("  [OK] Retrieval quality benchmark completed")
        else:
            print(f"  [FAIL] {result['stderr'][:200]}")
    else:
        print("  [SKIP] benchmark_comprehensive.py not found")

    # Copy results to eval/results
    benchmark_results = Path(__file__).parent.parent / "benchmarks" / "benchmark_results.json"
    if benchmark_results.exists():
        import shutil
        shutil.copy(benchmark_results, output_dir / "benchmark_results.json")
        print(f"  Results copied to {output_dir / 'benchmark_results.json'}")

    # Generate summary
    print("\n[2/3] Generating summary...")
    results_file = output_dir / "benchmark_results.json"
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)

        comparison = data.get("comparison", {})
        table = comparison.get("table", [])

        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"{'Config':<20} {'P50(ms)':<10} {'Top Score':<12} {'Success':<10}")
        print("-" * 60)
        for row in table:
            print(f"{row['config']:<20} {row['latency_p50_ms']:<10.1f} {row['top_score_max']:<12.4f} {row['success_rate']:<10}")

        print("-" * 60)
        print(f"Best: {comparison.get('winner', 'N/A')}")

    # Save run metadata
    print("\n[3/3] Saving run metadata...")
    metadata = {
        "timestamp": datetime.utcnow().isoformat(),
        "iterations": iterations,
        "quick_mode": args.quick,
        "config": args.config,
        "python_version": sys.version,
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
