"""Feedback-driven learning and weight adjustment."""

import asyncio
from datetime import datetime, timedelta
from typing import Any

import structlog

from perfect_rag.config import Settings, get_settings
from perfect_rag.db.surrealdb import SurrealDBClient
from perfect_rag.feedback.collector import FeedbackCollector
from perfect_rag.models.query import FeedbackType

logger = structlog.get_logger(__name__)


class FeedbackLearner:
    """Learn from user feedback to improve retrieval and generation.

    Learning mechanisms:
    1. Retrieval weight adjustment (dense vs sparse vs graph)
    2. Context awareness gate threshold tuning
    3. Cache promotion based on useful queries
    4. Chunk quality scoring
    5. Query pattern analysis
    """

    def __init__(
        self,
        surrealdb: SurrealDBClient,
        feedback_collector: FeedbackCollector,
        settings: Settings | None = None,
    ):
        self.surrealdb = surrealdb
        self.feedback = feedback_collector
        self.settings = settings or get_settings()

        # Current retrieval weights
        self._dense_weight = 0.5
        self._sparse_weight = 0.3
        self._graph_weight = 0.2

        # Context gate threshold
        self._gate_threshold = 0.7

        # Chunk quality scores (chunk_id -> score)
        self._chunk_scores: dict[str, float] = {}

        # Learning rates
        self._weight_lr = 0.01
        self._gate_lr = 0.005

    @property
    def retrieval_weights(self) -> dict[str, float]:
        """Current retrieval weights."""
        return {
            "dense": self._dense_weight,
            "sparse": self._sparse_weight,
            "graph": self._graph_weight,
        }

    @property
    def gate_threshold(self) -> float:
        """Current context awareness gate threshold."""
        return self._gate_threshold

    async def process_feedback_batch(
        self,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """Process a batch of feedback and update weights.

        Args:
            since: Process feedback since this time (default: last 24h)

        Returns:
            Summary of adjustments made
        """
        if since is None:
            since = datetime.utcnow() - timedelta(hours=24)

        # Get feedback stats
        stats = await self.feedback.get_feedback_stats(start_date=since)

        adjustments = {
            "feedback_processed": stats["total"],
            "positive_ratio": stats["positive_ratio"],
            "weight_adjustments": {},
            "gate_adjustment": 0,
        }

        if stats["total"] < 10:
            logger.info("Not enough feedback to learn from", count=stats["total"])
            return adjustments

        # Analyze feedback patterns
        await self._analyze_retrieval_performance(since, adjustments)
        await self._analyze_gate_performance(since, adjustments)
        await self._update_chunk_scores(since)

        # Persist learned weights
        await self._save_weights()

        logger.info(
            "Feedback learning complete",
            feedback_count=stats["total"],
            adjustments=adjustments,
        )

        return adjustments

    async def _analyze_retrieval_performance(
        self,
        since: datetime,
        adjustments: dict[str, Any],
    ) -> None:
        """Analyze retrieval performance from feedback."""
        # Get queries with their feedback and retrieval metadata
        result = await self.surrealdb.client.query(
            """
            SELECT
                q.id,
                q.dense_weight,
                q.sparse_weight,
                q.graph_weight,
                f.feedback_type
            FROM query_log AS q
            JOIN feedback AS f ON f.query_id = q.id
            WHERE q.timestamp >= $since
            """,
            {"since": since.isoformat()},
        )

        if not result or not result[0].get("result"):
            return

        # Analyze which weight configurations led to positive feedback
        positive_configs = []
        negative_configs = []

        for row in result[0]["result"]:
            config = (
                row.get("dense_weight", 0.5),
                row.get("sparse_weight", 0.3),
                row.get("graph_weight", 0.2),
            )

            if row.get("feedback_type") in ["thumbs_up", "citation_click"]:
                positive_configs.append(config)
            elif row.get("feedback_type") == "thumbs_down":
                negative_configs.append(config)

        # Calculate average good and bad weights
        if positive_configs:
            avg_good_dense = sum(c[0] for c in positive_configs) / len(positive_configs)
            avg_good_sparse = sum(c[1] for c in positive_configs) / len(positive_configs)
            avg_good_graph = sum(c[2] for c in positive_configs) / len(positive_configs)

            # Adjust weights toward positive configurations
            self._dense_weight += self._weight_lr * (avg_good_dense - self._dense_weight)
            self._sparse_weight += self._weight_lr * (avg_good_sparse - self._sparse_weight)
            self._graph_weight += self._weight_lr * (avg_good_graph - self._graph_weight)

            # Normalize weights
            total = self._dense_weight + self._sparse_weight + self._graph_weight
            self._dense_weight /= total
            self._sparse_weight /= total
            self._graph_weight /= total

            adjustments["weight_adjustments"] = {
                "dense": self._dense_weight,
                "sparse": self._sparse_weight,
                "graph": self._graph_weight,
            }

    async def _analyze_gate_performance(
        self,
        since: datetime,
        adjustments: dict[str, Any],
    ) -> None:
        """Analyze context awareness gate performance."""
        # Get queries where retrieval was skipped vs performed
        result = await self.surrealdb.client.query(
            """
            SELECT
                q.retrieval_needed,
                f.feedback_type,
                count() as count
            FROM query_log AS q
            JOIN feedback AS f ON f.query_id = q.id
            WHERE q.timestamp >= $since
            GROUP BY q.retrieval_needed, f.feedback_type
            """,
            {"since": since.isoformat()},
        )

        if not result or not result[0].get("result"):
            return

        # Calculate false positive/negative rates
        skipped_positive = 0  # Skipped retrieval, got positive feedback (good)
        skipped_negative = 0  # Skipped retrieval, got negative feedback (bad)
        retrieved_positive = 0  # Did retrieval, got positive feedback (good)
        retrieved_negative = 0  # Did retrieval, got negative feedback (need better retrieval)

        for row in result[0]["result"]:
            retrieval_needed = row.get("retrieval_needed", True)
            feedback = row.get("feedback_type")
            count = row.get("count", 0)

            if not retrieval_needed:
                if feedback in ["thumbs_up", "citation_click"]:
                    skipped_positive += count
                elif feedback == "thumbs_down":
                    skipped_negative += count
            else:
                if feedback in ["thumbs_up", "citation_click"]:
                    retrieved_positive += count
                elif feedback == "thumbs_down":
                    retrieved_negative += count

        # Adjust gate threshold
        # If skipping retrieval leads to negative feedback, lower threshold (retrieve more)
        # If retrieval often leads to negative feedback, raise threshold (retrieve less)

        if skipped_positive + skipped_negative > 0:
            skip_success_rate = skipped_positive / (skipped_positive + skipped_negative)

            if skip_success_rate < 0.5:
                # Gate is too aggressive, lower threshold
                self._gate_threshold -= self._gate_lr
                adjustments["gate_adjustment"] = -self._gate_lr
            elif skip_success_rate > 0.8:
                # Gate is too conservative, raise threshold
                self._gate_threshold += self._gate_lr
                adjustments["gate_adjustment"] = self._gate_lr

        # Clamp threshold
        self._gate_threshold = max(0.3, min(0.95, self._gate_threshold))

    async def _update_chunk_scores(self, since: datetime) -> None:
        """Update chunk quality scores based on feedback."""
        # Get chunks that were cited in positive/negative feedback
        result = await self.surrealdb.client.query(
            """
            SELECT
                q.retrieved_chunk_ids,
                f.feedback_type,
                f.cited_chunks
            FROM query_log AS q
            JOIN feedback AS f ON f.query_id = q.id
            WHERE q.timestamp >= $since
            """,
            {"since": since.isoformat()},
        )

        if not result or not result[0].get("result"):
            return

        for row in result[0]["result"]:
            feedback = row.get("feedback_type")
            retrieved = row.get("retrieved_chunk_ids", []) or []
            cited = row.get("cited_chunks", []) or []

            # Positive feedback boosts cited chunks
            if feedback in ["thumbs_up", "citation_click"]:
                for chunk_id in cited:
                    self._chunk_scores[chunk_id] = self._chunk_scores.get(chunk_id, 1.0) + 0.1

            # Negative feedback penalizes retrieved but not cited chunks
            elif feedback == "thumbs_down":
                for chunk_id in retrieved:
                    if chunk_id not in cited:
                        self._chunk_scores[chunk_id] = max(
                            0.1,
                            self._chunk_scores.get(chunk_id, 1.0) - 0.05
                        )

    async def _save_weights(self) -> None:
        """Persist learned weights to database."""
        weights_data = {
            "dense_weight": self._dense_weight,
            "sparse_weight": self._sparse_weight,
            "graph_weight": self._graph_weight,
            "gate_threshold": self._gate_threshold,
            "chunk_scores": dict(list(self._chunk_scores.items())[:1000]),  # Top 1000
            "updated_at": datetime.utcnow().isoformat(),
        }

        await self.surrealdb.client.query(
            """
            UPSERT learning_weights:current CONTENT $data
            """,
            {"data": weights_data},
        )

    async def load_weights(self) -> bool:
        """Load persisted weights from database."""
        result = await self.surrealdb.client.query(
            "SELECT * FROM learning_weights:current"
        )

        if result and result[0].get("result"):
            data = result[0]["result"][0]
            self._dense_weight = data.get("dense_weight", 0.5)
            self._sparse_weight = data.get("sparse_weight", 0.3)
            self._graph_weight = data.get("graph_weight", 0.2)
            self._gate_threshold = data.get("gate_threshold", 0.7)
            self._chunk_scores = data.get("chunk_scores", {})

            logger.info(
                "Loaded learned weights",
                dense=self._dense_weight,
                sparse=self._sparse_weight,
                graph=self._graph_weight,
                gate=self._gate_threshold,
            )
            return True

        return False

    def get_chunk_boost(self, chunk_id: str) -> float:
        """Get quality boost factor for a chunk."""
        return self._chunk_scores.get(chunk_id, 1.0)

    async def get_hot_chunks(self, limit: int = 100) -> list[str]:
        """Get chunk IDs with highest quality scores.

        Useful for cache warming.
        """
        sorted_chunks = sorted(
            self._chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [chunk_id for chunk_id, _ in sorted_chunks[:limit]]

    async def suggest_improvements(self) -> list[dict[str, Any]]:
        """Suggest improvements based on feedback patterns."""
        suggestions = []

        # Get queries with negative feedback
        bad_queries = await self.feedback.get_queries_needing_improvement()

        if bad_queries:
            suggestions.append({
                "type": "review_queries",
                "message": f"{len(bad_queries)} queries received multiple negative ratings",
                "queries": bad_queries[:5],
            })

        # Check weight balance
        if self._graph_weight < 0.1:
            suggestions.append({
                "type": "graph_underused",
                "message": "Graph retrieval weight is very low. Consider enriching knowledge graph.",
                "current_weight": self._graph_weight,
            })

        # Check gate performance
        if self._gate_threshold > 0.9:
            suggestions.append({
                "type": "gate_conservative",
                "message": "Context gate rarely skips retrieval. Consider lowering threshold.",
                "current_threshold": self._gate_threshold,
            })

        return suggestions


# =============================================================================
# Scheduled Learning Jobs
# =============================================================================


class LearningScheduler:
    """Schedule periodic learning jobs."""

    def __init__(
        self,
        learner: FeedbackLearner,
        settings: Settings | None = None,
    ):
        self.learner = learner
        self.settings = settings or get_settings()
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the learning scheduler."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Learning scheduler started")

    async def stop(self) -> None:
        """Stop the learning scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Learning scheduler stopped")

    async def _run_loop(self) -> None:
        """Main learning loop."""
        while self._running:
            try:
                # Run learning every hour
                await asyncio.sleep(3600)

                if self._running:
                    logger.info("Running scheduled learning job")
                    await self.learner.process_feedback_batch()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Learning job failed", error=str(e))
                await asyncio.sleep(60)  # Wait before retry
