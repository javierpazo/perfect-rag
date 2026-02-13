#!/usr/bin/env python3
"""
End-to-End Q&A Metrics

Computes:
- EM (Exact Match)
- F1 (Token-level F1)
- Attribution (% of citations that are in ground truth)

Usage:
    python eval/metrics_qa.py --dataset eval/datasets/medqa_small --results eval/results/qa_results.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class QAResult:
    """Q&A evaluation result."""
    qid: str
    query: str
    predicted_answer: str
    reference_answer: str
    predicted_citations: list[str]
    relevant_doc_ids: list[str]


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def compute_em(predicted: str, reference: str) -> float:
    """Compute Exact Match score."""
    pred_norm = normalize_text(predicted)
    ref_norm = normalize_text(reference)
    return 1.0 if pred_norm == ref_norm else 0.0


def compute_f1(predicted: str, reference: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = set(normalize_text(predicted).split())
    ref_tokens = set(normalize_text(reference).split())

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = pred_tokens & ref_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_attribution(
    predicted_citations: list[str],
    relevant_doc_ids: list[str],
) -> float:
    """Compute attribution score.

    Returns 1.0 if at least one predicted citation is in ground truth.
    """
    if not predicted_citations or not relevant_doc_ids:
        return 0.0

    # Normalize for comparison
    pred_set = {c.lower().strip() for c in predicted_citations}
    rel_set = {r.lower().strip() for r in relevant_doc_ids}

    # Check overlap
    return 1.0 if pred_set & rel_set else 0.0


def evaluate_qa(results: list[QAResult]) -> dict:
    """Evaluate Q&A results."""
    em_scores = []
    f1_scores = []
    attr_scores = []

    for result in results:
        # EM and F1
        em = compute_em(result.predicted_answer, result.reference_answer)
        f1 = compute_f1(result.predicted_answer, result.reference_answer)

        em_scores.append(em)
        f1_scores.append(f1)

        # Attribution
        attr = compute_attribution(result.predicted_citations, result.relevant_doc_ids)
        attr_scores.append(attr)

    return {
        "exact_match": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "attribution": sum(attr_scores) / len(attr_scores) if attr_scores else 0.0,
        "queries_count": len(results),
    }


def run_qa_evaluation(
    dataset_path: Path,
    output_path: Path,
):
    """Run Q&A evaluation with LLM."""
    import asyncio

    # Load queries with references
    queries = []
    with open(dataset_path / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))

    # Load qrels for attribution
    qrels = {}
    with open(dataset_path / "qrels.tsv", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4 and not line.startswith("#"):
                qid, _, doc_id, rel = parts[:4]
                if qid not in qrels:
                    qrels[qid] = []
                if int(rel) > 0:
                    qrels[qid].append(doc_id)

    # Check if LLM is available
    try:
        from perfect_rag.llm.gateway import LLMGateway
        from perfect_rag.config import get_settings
        settings = get_settings()
        llm = LLMGateway(settings=settings)
    except Exception as e:
        print(f"Warning: LLM not available ({e})")
        print("Using mock results for demonstration.")
        llm = None

    results = []
    for q in queries:
        qid = q["qid"]
        query = q["query"]
        ref_answer = q["answer_ref"]
        relevant_docs = qrels.get(qid, [])

        if llm:
            # Generate answer
            response = asyncio.run(llm.generate(
                messages=[{"role": "user", "content": query}],
                max_tokens=500,
            ))

            if hasattr(response, 'content'):
                pred_answer = response.content
                citations = []  # TODO: Extract citations from RAG context
            else:
                pred_answer = str(response)
                citations = []
        else:
            # Mock results
            pred_answer = f"[Mock answer for: {query}]"
            citations = []

        results.append(QAResult(
            qid=qid,
            query=query,
            predicted_answer=pred_answer,
            reference_answer=ref_answer,
            predicted_citations=citations,
            relevant_doc_ids=relevant_docs,
        ))

        print(f"  {qid}: EM={compute_em(pred_answer, ref_answer):.2f}, F1={compute_f1(pred_answer, ref_answer):.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Q&A Metrics Evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--results",
        type=str,
        help="Path to existing results JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for metrics JSON",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    # Load or run evaluation
    if args.results:
        with open(args.results, encoding="utf-8") as f:
            data = json.load(f)
        results = [QAResult(**r) for r in data.get("results", [])]
    else:
        print("Running Q&A evaluation...")
        results = run_qa_evaluation(dataset_path, args.output)

    # Compute metrics
    metrics = evaluate_qa(results)

    # Print results
    print("\n" + "=" * 50)
    print("Q&A METRICS")
    print("=" * 50)
    print(f"\nExact Match: {metrics['exact_match']:.2%}")
    print(f"F1 Score:   {metrics['f1']:.4f}")
    print(f"Attribution: {metrics['attribution']:.2%}")

    # Save
    output_path = Path(args.output) if args.output else dataset_path / "qa_metrics.json"
    output_data = {
        "metrics": metrics,
        "per_query": [
            {
                "qid": r.qid,
                "em": compute_em(r.predicted_answer, r.reference_answer),
                "f1": compute_f1(r.predicted_answer, r.reference_answer),
                "attribution": compute_attribution(r.predicted_citations, r.relevant_doc_ids),
            }
            for r in results
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()
