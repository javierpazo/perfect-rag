"""
RAGAS (Retrieval Augmented Generation Assessment) evaluation metrics.

Implements the core metrics for RAG evaluation:
- Context Precision: Are retrieved contexts relevant?
- Context Recall: Are all relevant contexts retrieved?
- Faithfulness: Is the answer grounded in context?
- Answer Relevancy: Does the answer address the question?
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class RAGASResult:
    """Results from RAGAS evaluation."""
    context_precision: float = 0.0
    context_recall: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0

    # Detailed breakdowns
    precision_per_context: list[float] = field(default_factory=list)
    faithfulness_claims: list[dict[str, Any]] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Harmonic mean of all metrics."""
        metrics = [
            self.context_precision,
            self.context_recall,
            self.faithfulness,
            self.answer_relevancy,
        ]
        metrics = [m for m in metrics if m > 0]
        if not metrics:
            return 0.0
        return len(metrics) / sum(1/m for m in metrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "overall_score": self.overall_score,
            "precision_per_context": self.precision_per_context,
            "faithfulness_claims": self.faithfulness_claims,
        }


class RAGASEvaluator:
    """
    Evaluate RAG systems using RAGAS metrics.

    Can use either:
    1. LLM-based evaluation (more accurate, slower)
    2. Embedding-based evaluation (faster, good approximation)
    """

    def __init__(
        self,
        llm_gateway=None,
        embedding_service=None,
        use_llm: bool = True,
    ):
        self.llm = llm_gateway
        self.embeddings = embedding_service
        self.use_llm = use_llm and llm_gateway is not None

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> RAGASResult:
        """
        Evaluate a single RAG response.

        Args:
            question: The user's question
            answer: The generated answer
            contexts: Retrieved context passages
            ground_truth: Optional ground truth answer for recall

        Returns:
            RAGASResult with all metrics
        """
        result = RAGASResult()

        # Run evaluations in parallel
        tasks = [
            self._context_precision(question, contexts),
            self._faithfulness(answer, contexts),
            self._answer_relevancy(question, answer),
        ]

        if ground_truth:
            tasks.append(self._context_recall(contexts, ground_truth))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Context precision
        if not isinstance(results[0], Exception):
            precision_result = results[0]
            if isinstance(precision_result, tuple):
                result.context_precision = precision_result[0]
                result.precision_per_context = precision_result[1]
            else:
                result.context_precision = precision_result
        else:
            logger.warning("Context precision evaluation failed", error=str(results[0]))

        # Faithfulness
        if not isinstance(results[1], Exception):
            faithfulness_result = results[1]
            if isinstance(faithfulness_result, tuple):
                result.faithfulness = faithfulness_result[0]
                result.faithfulness_claims = faithfulness_result[1]
            else:
                result.faithfulness = faithfulness_result
        else:
            logger.warning("Faithfulness evaluation failed", error=str(results[1]))

        # Answer relevancy
        if not isinstance(results[2], Exception):
            result.answer_relevancy = results[2]
        else:
            logger.warning("Answer relevancy evaluation failed", error=str(results[2]))

        # Context recall (if ground truth provided)
        if ground_truth and len(results) > 3:
            if not isinstance(results[3], Exception):
                result.context_recall = results[3]
            else:
                logger.warning("Context recall evaluation failed", error=str(results[3]))

        return result

    async def _context_precision(
        self,
        question: str,
        contexts: list[str],
    ) -> tuple[float, list[float]]:
        """
        Measure if retrieved contexts are relevant to the question.

        Uses: Proportion of contexts that contain relevant information.
        """
        if not contexts:
            return 0.0, []

        if self.use_llm:
            return await self._context_precision_llm(question, contexts)
        else:
            return await self._context_precision_embedding(question, contexts)

    async def _context_precision_llm(
        self,
        question: str,
        contexts: list[str],
    ) -> tuple[float, list[float]]:
        """LLM-based context precision."""
        relevant_count = 0
        scores = []

        for i, ctx in enumerate(contexts):
            eval_prompt = f"""Evaluate if the following context is relevant to answering the question.

Question: {question}

Context {i+1}: {ctx[:2000]}

Is this context relevant to answering the question? Respond with ONLY "relevant" or "not relevant"."""

            try:
                response = await self.llm.generate(
                    messages=[{"role": "user", "content": eval_prompt}],
                    max_tokens=10,
                    temperature=0,
                )

                response_lower = response.lower().strip()
                if "not relevant" in response_lower:
                    scores.append(0.0)
                elif "relevant" in response_lower:
                    relevant_count += 1
                    scores.append(1.0)
                else:
                    # Unclear response, default to 0.5
                    scores.append(0.5)
                    relevant_count += 0.5
            except Exception as e:
                logger.warning("Failed to evaluate context precision", context_idx=i, error=str(e))
                scores.append(0.5)
                relevant_count += 0.5

        return relevant_count / len(contexts), scores

    async def _context_precision_embedding(
        self,
        question: str,
        contexts: list[str],
    ) -> tuple[float, list[float]]:
        """Embedding-based context precision (faster)."""
        if not self.embeddings:
            return 0.5, [0.5] * len(contexts)

        q_emb = await self.embeddings.embed_query(question)
        q_emb = np.array(q_emb)

        scores = []
        for ctx in contexts:
            c_emb = await self.embeddings.embed_text(ctx[:512])
            c_emb = np.array(c_emb)
            similarity = np.dot(q_emb, c_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb) + 1e-10)
            scores.append(float(similarity))

        # Contexts with similarity > 0.7 are considered relevant
        threshold = 0.7
        relevant_count = sum(1 for s in scores if s > threshold)
        return relevant_count / len(contexts), scores

    async def _context_recall(
        self,
        contexts: list[str],
        ground_truth: str,
    ) -> float:
        """
        Measure if contexts contain info needed for ground truth answer.

        Extracts claims from ground truth and checks if contexts support them.
        """
        if not contexts or not ground_truth:
            return 0.0

        if self.use_llm:
            # Extract claims from ground truth
            claims_prompt = f"""Extract the key factual claims from this answer. List each claim on a new line, starting with a dash (-).

Answer: {ground_truth}

Claims:"""

            response = await self.llm.generate(
                messages=[{"role": "user", "content": claims_prompt}],
                max_tokens=500,
                temperature=0,
            )

            claims = [c.strip().lstrip('-').strip() for c in response.split("\n") if c.strip() and len(c.strip()) > 5]

            if not claims:
                return 1.0

            # Check if each claim is supported by any context
            combined_context = "\n\n".join(contexts)
            supported = 0

            for claim in claims:
                check_prompt = f"""Is this claim supported by the context? Answer only YES or NO.

Claim: {claim}

Context: {combined_context[:4000]}

Supported:"""

                response = await self.llm.generate(
                    messages=[{"role": "user", "content": check_prompt}],
                    max_tokens=5,
                    temperature=0,
                )

                if "yes" in response.lower():
                    supported += 1

            return supported / len(claims)
        else:
            # Embedding-based approximation
            if not self.embeddings:
                # Fallback: simple word overlap
                gt_words = set(ground_truth.lower().split())
                context_words = set(" ".join(contexts).lower().split())
                if not gt_words:
                    return 1.0
                return len(gt_words & context_words) / len(gt_words)

            gt_emb = await self.embeddings.embed_text(ground_truth)
            gt_emb = np.array(gt_emb)

            max_similarity = 0
            for ctx in contexts:
                c_emb = await self.embeddings.embed_text(ctx[:512])
                c_emb = np.array(c_emb)
                similarity = np.dot(gt_emb, c_emb) / (np.linalg.norm(gt_emb) * np.linalg.norm(c_emb) + 1e-10)
                max_similarity = max(max_similarity, float(similarity))

            return max_similarity

    async def _faithfulness(
        self,
        answer: str,
        contexts: list[str],
    ) -> tuple[float, list[dict[str, Any]]]:
        """
        Measure if answer is grounded in the provided contexts.

        Extracts claims from answer and verifies each against contexts.
        """
        if not answer or not contexts:
            return 0.0, []

        if self.use_llm:
            # Extract claims from answer
            claims_prompt = f"""Extract all factual claims from this answer. List each claim on a new line, starting with a dash (-).
Only include factual statements, not opinions or hedged statements.

Answer: {answer}

Claims:"""

            response = await self.llm.generate(
                messages=[{"role": "user", "content": claims_prompt}],
                max_tokens=500,
                temperature=0,
            )

            claims = [c.strip().lstrip('-').strip() for c in response.split("\n") if c.strip() and len(c.strip()) > 10]

            if not claims:
                return 1.0, []  # No claims to verify

            combined_context = "\n\n".join(contexts)
            supported = 0
            claims_detail = []

            for claim in claims:
                verify_prompt = f"""Is this claim supported by the context? Answer only YES or NO.

Claim: {claim}

Context: {combined_context[:4000]}

Supported:"""

                response = await self.llm.generate(
                    messages=[{"role": "user", "content": verify_prompt}],
                    max_tokens=5,
                    temperature=0,
                )

                is_supported = "yes" in response.lower()
                if is_supported:
                    supported += 1

                claims_detail.append({
                    "claim": claim,
                    "supported": is_supported,
                })

            return supported / len(claims), claims_detail
        else:
            # Embedding-based approximation
            if not self.embeddings:
                # Fallback: simple word overlap
                answer_words = set(answer.lower().split())
                context_words = set(" ".join(contexts).lower().split())
                if not answer_words:
                    return 1.0, []
                overlap = len(answer_words & context_words) / len(answer_words)
                return overlap, []

            a_emb = await self.embeddings.embed_text(answer)
            a_emb = np.array(a_emb)
            combined = " ".join(contexts)
            c_emb = await self.embeddings.embed_text(combined[:1000])
            c_emb = np.array(c_emb)

            similarity = np.dot(a_emb, c_emb) / (np.linalg.norm(a_emb) * np.linalg.norm(c_emb) + 1e-10)
            return float(np.clip(similarity, 0, 1)), []

    async def _answer_relevancy(
        self,
        question: str,
        answer: str,
    ) -> float:
        """
        Measure if answer addresses the question.

        Generates questions from answer and compares to original.
        """
        if not question or not answer:
            return 0.0

        if self.use_llm:
            # Generate question from answer
            gen_prompt = f"""Given this answer, what question was likely asked? Generate only the question, nothing else.

Answer: {answer}

Question:"""

            response = await self.llm.generate(
                messages=[{"role": "user", "content": gen_prompt}],
                max_tokens=100,
                temperature=0,
            )

            generated_q = response.strip()

            # Compare original and generated questions
            if self.embeddings:
                q1_emb = await self.embeddings.embed_query(question)
                q2_emb = await self.embeddings.embed_query(generated_q)
                q1_emb = np.array(q1_emb)
                q2_emb = np.array(q2_emb)
                similarity = np.dot(q1_emb, q2_emb) / (np.linalg.norm(q1_emb) * np.linalg.norm(q2_emb) + 1e-10)
                return float(np.clip(similarity, 0, 1))
            else:
                # Fallback: word overlap (Jaccard similarity)
                q1_words = set(question.lower().split())
                q2_words = set(generated_q.lower().split())
                if not q1_words or not q2_words:
                    return 0.0
                intersection = len(q1_words & q2_words)
                union = len(q1_words | q2_words)
                return intersection / union if union > 0 else 0.0
        else:
            # Direct embedding comparison
            if not self.embeddings:
                return 0.5  # Default when no evaluation method available

            q_emb = await self.embeddings.embed_query(question)
            a_emb = await self.embeddings.embed_text(answer)
            q_emb = np.array(q_emb)
            a_emb = np.array(a_emb)
            similarity = np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb) + 1e-10)
            return float(np.clip(similarity, 0, 1))

    async def evaluate_batch(
        self,
        samples: list[dict[str, Any]],
        concurrency: int = 5,
    ) -> dict[str, Any]:
        """
        Evaluate multiple samples and return aggregate metrics.

        Args:
            samples: List of dicts with 'question', 'answer', 'contexts', optional 'ground_truth'
            concurrency: Maximum number of concurrent evaluations

        Returns:
            Aggregate metrics across all samples
        """
        results = []
        semaphore = asyncio.Semaphore(concurrency)

        async def evaluate_with_semaphore(sample: dict[str, Any]) -> RAGASResult:
            async with semaphore:
                return await self.evaluate(
                    question=sample["question"],
                    answer=sample["answer"],
                    contexts=sample["contexts"],
                    ground_truth=sample.get("ground_truth"),
                )

        tasks = [evaluate_with_semaphore(sample) for sample in samples]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, RAGASResult)]

        if not valid_results:
            return {
                "context_precision": 0.0,
                "context_recall": 0.0,
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "overall_score": 0.0,
                "num_samples": 0,
                "num_errors": len(results),
            }

        # Aggregate metrics
        context_precisions = [r.context_precision for r in valid_results]
        context_recalls = [r.context_recall for r in valid_results if r.context_recall > 0]
        faithfulness_scores = [r.faithfulness for r in valid_results]
        relevancy_scores = [r.answer_relevancy for r in valid_results]
        overall_scores = [r.overall_score for r in valid_results]

        return {
            "context_precision": float(np.mean(context_precisions)) if context_precisions else 0.0,
            "context_precision_std": float(np.std(context_precisions)) if context_precisions else 0.0,
            "context_recall": float(np.mean(context_recalls)) if context_recalls else 0.0,
            "context_recall_std": float(np.std(context_recalls)) if context_recalls else 0.0,
            "faithfulness": float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0,
            "faithfulness_std": float(np.std(faithfulness_scores)) if faithfulness_scores else 0.0,
            "answer_relevancy": float(np.mean(relevancy_scores)) if relevancy_scores else 0.0,
            "answer_relevancy_std": float(np.std(relevancy_scores)) if relevancy_scores else 0.0,
            "overall_score": float(np.mean(overall_scores)) if overall_scores else 0.0,
            "overall_score_std": float(np.std(overall_scores)) if overall_scores else 0.0,
            "num_samples": len(valid_results),
            "num_errors": len(results) - len(valid_results),
            "individual_results": [r.to_dict() for r in valid_results],
        }


class RetrievalMetrics:
    """Traditional IR metrics for retrieval evaluation."""

    @staticmethod
    def recall_at_k(
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int,
    ) -> float:
        """Recall@K: proportion of relevant docs in top-k."""
        if not relevant_ids:
            return 0.0
        retrieved_set = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        return len(retrieved_set & relevant_set) / len(relevant_set)

    @staticmethod
    def precision_at_k(
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int,
    ) -> float:
        """Precision@K: proportion of top-k that are relevant."""
        if k == 0:
            return 0.0
        retrieved_set = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        return len(retrieved_set & relevant_set) / k

    @staticmethod
    def f1_at_k(
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int,
    ) -> float:
        """F1@K: harmonic mean of precision and recall at K."""
        precision = RetrievalMetrics.precision_at_k(retrieved_ids, relevant_ids, k)
        recall = RetrievalMetrics.recall_at_k(retrieved_ids, relevant_ids, k)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def mrr(
        retrieved_ids: list[str],
        relevant_ids: list[str],
    ) -> float:
        """Mean Reciprocal Rank."""
        relevant_set = set(relevant_ids)
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def average_precision(
        retrieved_ids: list[str],
        relevant_ids: list[str],
    ) -> float:
        """Average Precision (AP): area under precision-recall curve."""
        if not relevant_ids:
            return 0.0

        relevant_set = set(relevant_ids)
        num_relevant = 0
        precision_sum = 0.0

        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                precision_sum += precision_at_i

        return precision_sum / len(relevant_ids) if relevant_ids else 0.0

    @staticmethod
    def ndcg_at_k(
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int,
        relevance_scores: dict[str, float] | None = None,
    ) -> float:
        """Normalized Discounted Cumulative Gain at K.

        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            k: Number of top results to consider
            relevance_scores: Optional dict mapping doc_id to relevance score (default: binary)
        """
        import math

        if relevance_scores is None:
            # Binary relevance: 1 if relevant, 0 otherwise
            relevance_scores = {doc_id: 1.0 for doc_id in relevant_ids}

        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            rel = relevance_scores.get(doc_id, 0.0)
            # Using log2(i + 2) because i is 0-indexed, and we want log2(rank + 1)
            dcg += rel / math.log2(i + 2)

        # Ideal DCG (sort by relevance)
        sorted_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(sorted_relevances))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def hit_rate_at_k(
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int,
    ) -> float:
        """Hit Rate@K: 1 if at least one relevant doc is in top-k, else 0."""
        retrieved_set = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        return 1.0 if retrieved_set & relevant_set else 0.0

    @classmethod
    def evaluate_retrieval(
        cls,
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k_values: list[int] | None = None,
    ) -> dict[str, float]:
        """Compute all retrieval metrics.

        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            k_values: K values to evaluate (default: [1, 3, 5, 10])

        Returns:
            Dictionary with all metrics
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        results = {
            "mrr": cls.mrr(retrieved_ids, relevant_ids),
            "average_precision": cls.average_precision(retrieved_ids, relevant_ids),
        }

        for k in k_values:
            results[f"recall@{k}"] = cls.recall_at_k(retrieved_ids, relevant_ids, k)
            results[f"precision@{k}"] = cls.precision_at_k(retrieved_ids, relevant_ids, k)
            results[f"f1@{k}"] = cls.f1_at_k(retrieved_ids, relevant_ids, k)
            results[f"ndcg@{k}"] = cls.ndcg_at_k(retrieved_ids, relevant_ids, k)
            results[f"hit_rate@{k}"] = cls.hit_rate_at_k(retrieved_ids, relevant_ids, k)

        return results

    @classmethod
    def evaluate_retrieval_batch(
        cls,
        queries: list[dict[str, Any]],
        k_values: list[int] | None = None,
    ) -> dict[str, float]:
        """Compute average retrieval metrics over multiple queries.

        Args:
            queries: List of dicts with 'retrieved_ids' and 'relevant_ids'
            k_values: K values to evaluate

        Returns:
            Dictionary with averaged metrics
        """
        if not queries:
            return {}

        all_results = []
        for query in queries:
            result = cls.evaluate_retrieval(
                retrieved_ids=query["retrieved_ids"],
                relevant_ids=query["relevant_ids"],
                k_values=k_values,
            )
            all_results.append(result)

        # Average all metrics
        averaged = {}
        for key in all_results[0].keys():
            values = [r[key] for r in all_results]
            averaged[key] = float(np.mean(values))
            averaged[f"{key}_std"] = float(np.std(values))

        averaged["num_queries"] = len(queries)
        return averaged
