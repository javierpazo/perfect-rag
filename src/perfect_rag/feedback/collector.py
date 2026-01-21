"""Feedback collection and storage."""

from datetime import datetime
from typing import Any

import structlog

from perfect_rag.config import Settings, get_settings
from perfect_rag.db.surrealdb import SurrealDBClient
from perfect_rag.models.query import Feedback, FeedbackType

logger = structlog.get_logger(__name__)


class FeedbackCollector:
    """Collect and store user feedback on RAG responses.

    Feedback types:
    - THUMBS_UP: Positive rating
    - THUMBS_DOWN: Negative rating
    - CORRECTION: User provides correct answer
    - CITATION_CLICK: User clicked on a citation (implicit positive)
    - FOLLOW_UP: User asked follow-up question
    """

    def __init__(
        self,
        surrealdb: SurrealDBClient,
        settings: Settings | None = None,
    ):
        self.surrealdb = surrealdb
        self.settings = settings or get_settings()

        # In-memory cache for quick stats
        self._feedback_cache: dict[str, list[Feedback]] = {}
        self._query_feedback_count: dict[str, int] = {}

    async def submit_feedback(
        self,
        query_id: str,
        feedback_type: FeedbackType,
        user_id: str | None = None,
        text: str | None = None,
        corrected_answer: str | None = None,
        cited_chunks: list[str] | None = None,
    ) -> Feedback:
        """Submit feedback for a query.

        Args:
            query_id: ID of the query being rated
            feedback_type: Type of feedback
            user_id: Optional user ID
            text: Optional feedback text
            corrected_answer: User-provided correct answer (for corrections)
            cited_chunks: Chunks the user found useful

        Returns:
            Created Feedback object
        """
        feedback = Feedback(
            query_id=query_id,
            feedback_type=feedback_type,
            text=text,
            corrected_answer=corrected_answer,
            cited_chunks=cited_chunks,
            timestamp=datetime.utcnow(),
        )

        # Store in database
        await self._store_feedback(feedback, user_id)

        # Update cache
        if query_id not in self._feedback_cache:
            self._feedback_cache[query_id] = []
        self._feedback_cache[query_id].append(feedback)
        self._query_feedback_count[query_id] = self._query_feedback_count.get(query_id, 0) + 1

        logger.info(
            "Feedback submitted",
            query_id=query_id,
            feedback_type=feedback_type.value,
            user_id=user_id,
        )

        return feedback

    async def _store_feedback(self, feedback: Feedback, user_id: str | None) -> None:
        """Store feedback in SurrealDB."""
        feedback_data = {
            "query_id": feedback.query_id,
            "feedback_type": feedback.feedback_type.value,
            "text": feedback.text,
            "corrected_answer": feedback.corrected_answer,
            "cited_chunks": feedback.cited_chunks,
            "timestamp": feedback.timestamp.isoformat(),
            "user_id": user_id,
        }

        await self.surrealdb.client.query(
            "CREATE feedback CONTENT $data",
            {"data": feedback_data},
        )

    async def get_feedback_for_query(self, query_id: str) -> list[Feedback]:
        """Get all feedback for a query."""
        # Check cache first
        if query_id in self._feedback_cache:
            return self._feedback_cache[query_id]

        # Query database
        result = await self.surrealdb.client.query(
            "SELECT * FROM feedback WHERE query_id = $query_id",
            {"query_id": query_id},
        )

        feedbacks = []
        if result and result[0].get("result"):
            for row in result[0]["result"]:
                feedbacks.append(
                    Feedback(
                        query_id=row["query_id"],
                        feedback_type=FeedbackType(row["feedback_type"]),
                        text=row.get("text"),
                        corrected_answer=row.get("corrected_answer"),
                        cited_chunks=row.get("cited_chunks"),
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                    )
                )

        self._feedback_cache[query_id] = feedbacks
        return feedbacks

    async def get_feedback_stats(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """Get feedback statistics.

        Returns:
            Stats including total counts, positive/negative ratio, etc.
        """
        # Build query
        query = "SELECT feedback_type, count() as count FROM feedback"
        params: dict[str, Any] = {}

        conditions = []
        if start_date:
            conditions.append("timestamp >= $start_date")
            params["start_date"] = start_date.isoformat()
        if end_date:
            conditions.append("timestamp <= $end_date")
            params["end_date"] = end_date.isoformat()

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " GROUP BY feedback_type"

        result = await self.surrealdb.client.query(query, params)

        # Process results
        stats = {
            "total": 0,
            "by_type": {},
            "positive_ratio": 0.0,
        }

        positive = 0
        negative = 0

        if result and result[0].get("result"):
            for row in result[0]["result"]:
                feedback_type = row["feedback_type"]
                count = row["count"]
                stats["by_type"][feedback_type] = count
                stats["total"] += count

                if feedback_type in ["thumbs_up", "citation_click"]:
                    positive += count
                elif feedback_type == "thumbs_down":
                    negative += count

        if positive + negative > 0:
            stats["positive_ratio"] = positive / (positive + negative)

        return stats

    async def get_queries_needing_improvement(
        self,
        min_negative_feedback: int = 2,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get queries that received negative feedback.

        Useful for identifying areas to improve.
        """
        result = await self.surrealdb.client.query(
            """
            SELECT query_id, count() as negative_count
            FROM feedback
            WHERE feedback_type = 'thumbs_down'
            GROUP BY query_id
            HAVING count() >= $min_count
            ORDER BY negative_count DESC
            LIMIT $limit
            """,
            {"min_count": min_negative_feedback, "limit": limit},
        )

        queries = []
        if result and result[0].get("result"):
            for row in result[0]["result"]:
                # Get query details
                query_data = await self.surrealdb.client.query(
                    "SELECT * FROM query_log WHERE id = $id",
                    {"id": row["query_id"]},
                )

                query_info = {
                    "query_id": row["query_id"],
                    "negative_count": row["negative_count"],
                }

                if query_data and query_data[0].get("result"):
                    query_info["query"] = query_data[0]["result"][0].get("query")
                    query_info["response"] = query_data[0]["result"][0].get("response")

                queries.append(query_info)

        return queries

    async def get_useful_chunks(
        self,
        min_citations: int = 3,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get chunks that users found most useful (via citations clicks).

        Useful for cache warming.
        """
        result = await self.surrealdb.client.query(
            """
            SELECT cited_chunks, count() as citation_count
            FROM feedback
            WHERE feedback_type = 'citation_click' AND cited_chunks IS NOT NULL
            GROUP BY cited_chunks
            HAVING count() >= $min_count
            ORDER BY citation_count DESC
            LIMIT $limit
            """,
            {"min_count": min_citations, "limit": limit},
        )

        if result and result[0].get("result"):
            return result[0]["result"]
        return []

    async def record_implicit_feedback(
        self,
        query_id: str,
        event_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record implicit feedback events.

        Events like:
        - citation_click: User clicked a citation
        - dwell_time: Time spent reading response
        - copy_text: User copied response text
        - follow_up: User asked a follow-up question
        """
        if event_type == "citation_click":
            chunk_id = metadata.get("chunk_id") if metadata else None
            await self.submit_feedback(
                query_id=query_id,
                feedback_type=FeedbackType.CITATION_CLICK,
                cited_chunks=[chunk_id] if chunk_id else None,
            )
        elif event_type == "follow_up":
            await self.submit_feedback(
                query_id=query_id,
                feedback_type=FeedbackType.FOLLOW_UP,
                text=metadata.get("follow_up_query") if metadata else None,
            )
        else:
            # Store as generic event
            event_data = {
                "query_id": query_id,
                "event_type": event_type,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat(),
            }
            await self.surrealdb.client.query(
                "CREATE feedback_event CONTENT $data",
                {"data": event_data},
            )
