"""Personalization engine for user-adapted retrieval.

Provides user profiling, preference tracking, and personalized reranking
for RAG systems.
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class InterestCategory(str, Enum):
    """Categories for user interests."""

    TECHNICAL = "technical"
    BUSINESS = "business"
    SCIENTIFIC = "scientific"
    LEGAL = "legal"
    MEDICAL = "medical"
    EDUCATIONAL = "educational"
    NEWS = "news"
    CREATIVE = "creative"
    OTHER = "other"


class InteractionType(str, Enum):
    """Types of user interactions for tracking."""

    QUERY = "query"
    CLICK = "click"
    BOOKMARK = "bookmark"
    FEEDBACK_POSITIVE = "feedback_positive"
    FEEDBACK_NEGATIVE = "feedback_negative"
    DWELL_TIME = "dwell_time"
    COPY = "copy"
    SHARE = "share"


@dataclass
class UserInteraction:
    """Record of a user interaction."""

    interaction_type: InteractionType
    timestamp: datetime
    query: str | None = None
    chunk_ids: list[str] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    dwell_time_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "interaction_type": self.interaction_type.value,
            "timestamp": self.timestamp.isoformat(),
            "query": self.query,
            "chunk_ids": self.chunk_ids,
            "document_ids": self.document_ids,
            "topics": self.topics,
            "dwell_time_seconds": self.dwell_time_seconds,
            "metadata": self.metadata,
        }


class UserProfile(BaseModel):
    """User profile with preferences and history."""

    user_id: str = Field(..., description="Unique user identifier")

    # Explicit preferences
    preferred_categories: list[str] = Field(default_factory=list)
    preferred_topics: list[str] = Field(default_factory=list)
    preferred_sources: list[str] = Field(default_factory=list)
    blocked_sources: list[str] = Field(default_factory=list)
    language_preference: str = Field(default="en")
    detail_level: str = Field(
        default="medium",
        description="Preferred detail level: brief, medium, detailed",
    )

    # Learned preferences (from interactions)
    topic_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Topic -> weight mapping learned from interactions",
    )
    source_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Source -> weight mapping learned from interactions",
    )
    category_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Category -> weight mapping learned from interactions",
    )

    # History
    recent_queries: list[str] = Field(
        default_factory=list,
        description="Recent queries (last 100)",
    )
    recent_topics: list[str] = Field(
        default_factory=list,
        description="Recent topics of interest",
    )
    interaction_count: int = Field(default=0)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_active_at: datetime | None = None

    # Settings
    personalization_enabled: bool = Field(default=True)
    history_retention_days: int = Field(default=90)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

    def get_top_topics(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N topics by weight."""
        sorted_topics = sorted(
            self.topic_weights.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_topics[:n]

    def get_top_sources(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N sources by weight."""
        sorted_sources = sorted(
            self.source_weights.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_sources[:n]


class ProfileStore:
    """In-memory profile storage with persistence hooks."""

    def __init__(self, surrealdb_client: Any = None):
        self._profiles: dict[str, UserProfile] = {}
        self._interactions: dict[str, list[UserInteraction]] = {}
        self._lock = asyncio.Lock()
        self.surrealdb = surrealdb_client

    async def get_profile(self, user_id: str) -> UserProfile | None:
        """Get user profile."""
        async with self._lock:
            if user_id in self._profiles:
                return self._profiles[user_id]

            # Try to load from database
            if self.surrealdb:
                try:
                    result = await self.surrealdb.client.query(
                        "SELECT * FROM user_profile WHERE user_id = $user_id",
                        {"user_id": user_id},
                    )
                    if result and result[0].get("result"):
                        data = result[0]["result"][0]
                        profile = UserProfile(**data)
                        self._profiles[user_id] = profile
                        return profile
                except Exception as e:
                    logger.warning("Failed to load profile from DB", error=str(e))

            return None

    async def save_profile(self, profile: UserProfile) -> None:
        """Save user profile."""
        async with self._lock:
            profile.updated_at = datetime.utcnow()
            self._profiles[profile.user_id] = profile

            # Persist to database
            if self.surrealdb:
                try:
                    await self.surrealdb.client.query(
                        """
                        UPSERT user_profile SET
                            user_id = $user_id,
                            preferred_categories = $preferred_categories,
                            preferred_topics = $preferred_topics,
                            preferred_sources = $preferred_sources,
                            blocked_sources = $blocked_sources,
                            language_preference = $language_preference,
                            detail_level = $detail_level,
                            topic_weights = $topic_weights,
                            source_weights = $source_weights,
                            category_weights = $category_weights,
                            recent_queries = $recent_queries,
                            recent_topics = $recent_topics,
                            interaction_count = $interaction_count,
                            created_at = $created_at,
                            updated_at = $updated_at,
                            last_active_at = $last_active_at,
                            personalization_enabled = $personalization_enabled
                        WHERE user_id = $user_id
                        """,
                        profile.model_dump(),
                    )
                except Exception as e:
                    logger.warning("Failed to save profile to DB", error=str(e))

    async def create_profile(self, user_id: str, **kwargs) -> UserProfile:
        """Create a new user profile."""
        profile = UserProfile(user_id=user_id, **kwargs)
        await self.save_profile(profile)
        return profile

    async def delete_profile(self, user_id: str) -> bool:
        """Delete user profile and all interactions."""
        async with self._lock:
            if user_id in self._profiles:
                del self._profiles[user_id]
            if user_id in self._interactions:
                del self._interactions[user_id]

            if self.surrealdb:
                try:
                    await self.surrealdb.client.query(
                        "DELETE FROM user_profile WHERE user_id = $user_id",
                        {"user_id": user_id},
                    )
                    await self.surrealdb.client.query(
                        "DELETE FROM user_interaction WHERE user_id = $user_id",
                        {"user_id": user_id},
                    )
                except Exception as e:
                    logger.warning("Failed to delete profile from DB", error=str(e))
                    return False

            return True

    async def add_interaction(
        self,
        user_id: str,
        interaction: UserInteraction,
    ) -> None:
        """Add an interaction for a user."""
        async with self._lock:
            if user_id not in self._interactions:
                self._interactions[user_id] = []
            self._interactions[user_id].append(interaction)

            # Keep only recent interactions
            max_interactions = 1000
            if len(self._interactions[user_id]) > max_interactions:
                self._interactions[user_id] = self._interactions[user_id][-max_interactions:]

    async def get_interactions(
        self,
        user_id: str,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[UserInteraction]:
        """Get user interactions."""
        async with self._lock:
            interactions = self._interactions.get(user_id, [])
            if since:
                interactions = [i for i in interactions if i.timestamp >= since]
            return interactions[-limit:]


class PreferenceTracker:
    """Track and learn user preferences from interactions.

    Uses exponential moving average to weight recent interactions more heavily.
    """

    def __init__(
        self,
        profile_store: ProfileStore,
        settings: Settings | None = None,
    ):
        self.store = profile_store
        self.settings = settings or get_settings()

        # Learning parameters
        self.learning_rate = 0.1  # How fast to adapt
        self.decay_rate = 0.01  # Daily decay for old preferences
        self.positive_weight = 1.0  # Weight for positive signals
        self.negative_weight = 0.5  # Weight for negative signals
        self.click_weight = 0.3  # Weight for click signals
        self.dwell_weight = 0.5  # Weight for dwell time signals

    async def track_interaction(
        self,
        user_id: str,
        interaction: UserInteraction,
    ) -> None:
        """Track a user interaction and update preferences.

        Args:
            user_id: User ID
            interaction: Interaction to track
        """
        # Store the interaction
        await self.store.add_interaction(user_id, interaction)

        # Get or create profile
        profile = await self.store.get_profile(user_id)
        if not profile:
            profile = await self.store.create_profile(user_id)

        if not profile.personalization_enabled:
            return

        # Update profile based on interaction type
        await self._update_profile_from_interaction(profile, interaction)

        # Save updated profile
        profile.interaction_count += 1
        profile.last_active_at = datetime.utcnow()
        await self.store.save_profile(profile)

        logger.debug(
            "Tracked interaction",
            user_id=user_id,
            interaction_type=interaction.interaction_type.value,
        )

    async def _update_profile_from_interaction(
        self,
        profile: UserProfile,
        interaction: UserInteraction,
    ) -> None:
        """Update profile weights based on interaction."""
        # Determine signal strength
        if interaction.interaction_type == InteractionType.FEEDBACK_POSITIVE:
            signal = self.positive_weight
        elif interaction.interaction_type == InteractionType.FEEDBACK_NEGATIVE:
            signal = -self.negative_weight
        elif interaction.interaction_type == InteractionType.CLICK:
            signal = self.click_weight
        elif interaction.interaction_type == InteractionType.DWELL_TIME:
            # Scale by dwell time (longer = more interested)
            dwell = interaction.dwell_time_seconds or 0
            signal = min(self.dwell_weight * (dwell / 60), 1.0)  # Cap at 1 minute
        elif interaction.interaction_type == InteractionType.BOOKMARK:
            signal = self.positive_weight
        elif interaction.interaction_type == InteractionType.QUERY:
            signal = self.click_weight * 0.5  # Queries indicate mild interest
        else:
            signal = 0.1

        # Update topic weights
        for topic in interaction.topics:
            current = profile.topic_weights.get(topic, 0.5)
            updated = current + self.learning_rate * signal * (1 - current if signal > 0 else current)
            profile.topic_weights[topic] = max(0.0, min(1.0, updated))

        # Update recent queries
        if interaction.query:
            profile.recent_queries.append(interaction.query)
            profile.recent_queries = profile.recent_queries[-100:]  # Keep last 100

        # Update recent topics
        profile.recent_topics.extend(interaction.topics)
        profile.recent_topics = profile.recent_topics[-50:]  # Keep last 50

    async def apply_decay(self, user_id: str) -> None:
        """Apply time decay to preferences."""
        profile = await self.store.get_profile(user_id)
        if not profile:
            return

        # Calculate days since last decay
        if profile.last_active_at:
            days_inactive = (datetime.utcnow() - profile.last_active_at).days
            decay_factor = (1 - self.decay_rate) ** days_inactive

            # Apply decay to topic weights
            for topic in profile.topic_weights:
                profile.topic_weights[topic] *= decay_factor

            # Apply decay to source weights
            for source in profile.source_weights:
                profile.source_weights[source] *= decay_factor

        await self.store.save_profile(profile)

    async def get_interest_vector(
        self,
        user_id: str,
        embedding_service: Any = None,
    ) -> list[float] | None:
        """Get a vector representation of user interests.

        Args:
            user_id: User ID
            embedding_service: Embedding service for vectorization

        Returns:
            Interest vector or None
        """
        profile = await self.store.get_profile(user_id)
        if not profile or not embedding_service:
            return None

        # Build interest description from top topics
        top_topics = profile.get_top_topics(20)
        if not top_topics:
            return None

        # Create a weighted interest description
        interest_parts = []
        for topic, weight in top_topics:
            if weight > 0.3:  # Only significant interests
                interest_parts.append(topic)

        if not interest_parts:
            return None

        interest_text = " ".join(interest_parts[:10])  # Top 10 topics
        try:
            vector = await embedding_service.embed_text(interest_text)
            return vector
        except Exception as e:
            logger.warning("Failed to create interest vector", error=str(e))
            return None


class PersonalizedReranker:
    """Rerank search results based on user preferences.

    Applies personalization boosting to retrieved chunks based on:
    - Topic matching with user interests
    - Source preferences
    - Historical click patterns
    """

    def __init__(
        self,
        profile_store: ProfileStore,
        settings: Settings | None = None,
    ):
        self.store = profile_store
        self.settings = settings or get_settings()

        # Reranking parameters
        self.personalization_weight = 0.3  # How much personalization affects ranking

    async def rerank(
        self,
        user_id: str,
        chunks: list[dict[str, Any]],
        personalization_strength: float | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank chunks based on user preferences.

        Args:
            user_id: User ID
            chunks: List of retrieved chunks
            personalization_strength: Override default personalization weight (0-1)

        Returns:
            Reranked chunks
        """
        profile = await self.store.get_profile(user_id)
        if not profile or not profile.personalization_enabled:
            return chunks

        weight = personalization_strength or self.personalization_weight

        reranked = []
        for chunk in chunks:
            chunk_copy = chunk.copy()
            original_score = chunk_copy.get("score", 1.0)

            # Calculate personalization boost
            boost = await self._calculate_personalization_boost(chunk, profile)

            # Apply boost
            final_score = original_score * (1 - weight) + original_score * weight * boost
            chunk_copy["score"] = final_score
            chunk_copy["personalization_boost"] = boost
            reranked.append(chunk_copy)

        # Sort by new score
        reranked.sort(key=lambda x: x.get("score", 0), reverse=True)

        logger.debug(
            "Personalized reranking complete",
            user_id=user_id,
            chunks_count=len(chunks),
        )

        return reranked

    async def _calculate_personalization_boost(
        self,
        chunk: dict[str, Any],
        profile: UserProfile,
    ) -> float:
        """Calculate personalization boost for a chunk."""
        boosts = []
        payload = chunk.get("payload", {})
        metadata = payload.get("metadata", {})

        # Topic boost
        chunk_topics = metadata.get("topics", [])
        if not chunk_topics:
            # Try to extract from content
            content = payload.get("text", "")
            # Simple keyword matching (could use NER in production)
            chunk_topics = self._extract_topics(content)

        topic_boost = 0.0
        for topic in chunk_topics:
            topic_lower = topic.lower()
            if topic_lower in profile.topic_weights:
                topic_boost = max(topic_boost, profile.topic_weights[topic_lower])
        if topic_boost > 0:
            boosts.append(topic_boost)

        # Source boost
        source = metadata.get("source", "")
        if source:
            if source in profile.blocked_sources:
                return 0.1  # Heavily penalize blocked sources
            if source in profile.preferred_sources:
                boosts.append(1.5)
            elif source in profile.source_weights:
                boosts.append(profile.source_weights[source])

        # Category boost
        category = metadata.get("category", "")
        if category:
            if category in profile.preferred_categories:
                boosts.append(1.3)
            elif category in profile.category_weights:
                boosts.append(profile.category_weights[category])

        # Calculate final boost
        if not boosts:
            return 1.0  # Neutral boost

        # Average the boosts
        return sum(boosts) / len(boosts)

    def _extract_topics(self, content: str, max_topics: int = 5) -> list[str]:
        """Simple topic extraction from content."""
        # This is a simplified version - in production, use NER or topic modeling
        words = content.lower().split()
        # Filter to meaningful words (could use stopwords list)
        meaningful = [w for w in words if len(w) > 4 and w.isalpha()]
        # Count frequencies
        from collections import Counter
        word_counts = Counter(meaningful)
        return [word for word, _ in word_counts.most_common(max_topics)]


class PersonalizationEngine:
    """Main personalization engine combining all components.

    Provides a unified interface for:
    - User profile management
    - Preference tracking
    - Personalized retrieval
    """

    def __init__(
        self,
        surrealdb_client: Any = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.store = ProfileStore(surrealdb_client)
        self.tracker = PreferenceTracker(self.store, settings)
        self.reranker = PersonalizedReranker(self.store, settings)

    async def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile."""
        profile = await self.store.get_profile(user_id)
        if not profile:
            profile = await self.store.create_profile(user_id)
        return profile

    async def update_preferences(
        self,
        user_id: str,
        preferred_topics: list[str] | None = None,
        preferred_sources: list[str] | None = None,
        blocked_sources: list[str] | None = None,
        preferred_categories: list[str] | None = None,
        language_preference: str | None = None,
        detail_level: str | None = None,
        personalization_enabled: bool | None = None,
    ) -> UserProfile:
        """Update user preferences explicitly."""
        profile = await self.get_or_create_profile(user_id)

        if preferred_topics is not None:
            profile.preferred_topics = preferred_topics
            # Also initialize topic weights
            for topic in preferred_topics:
                if topic not in profile.topic_weights:
                    profile.topic_weights[topic] = 0.8

        if preferred_sources is not None:
            profile.preferred_sources = preferred_sources

        if blocked_sources is not None:
            profile.blocked_sources = blocked_sources

        if preferred_categories is not None:
            profile.preferred_categories = preferred_categories

        if language_preference is not None:
            profile.language_preference = language_preference

        if detail_level is not None:
            profile.detail_level = detail_level

        if personalization_enabled is not None:
            profile.personalization_enabled = personalization_enabled

        await self.store.save_profile(profile)
        return profile

    async def track_query(
        self,
        user_id: str,
        query: str,
        retrieved_chunk_ids: list[str] | None = None,
        topics: list[str] | None = None,
    ) -> None:
        """Track a query interaction."""
        interaction = UserInteraction(
            interaction_type=InteractionType.QUERY,
            timestamp=datetime.utcnow(),
            query=query,
            chunk_ids=retrieved_chunk_ids or [],
            topics=topics or [],
        )
        await self.tracker.track_interaction(user_id, interaction)

    async def track_click(
        self,
        user_id: str,
        chunk_id: str,
        document_id: str | None = None,
        topics: list[str] | None = None,
    ) -> None:
        """Track a click interaction."""
        interaction = UserInteraction(
            interaction_type=InteractionType.CLICK,
            timestamp=datetime.utcnow(),
            chunk_ids=[chunk_id],
            document_ids=[document_id] if document_id else [],
            topics=topics or [],
        )
        await self.tracker.track_interaction(user_id, interaction)

    async def track_feedback(
        self,
        user_id: str,
        is_positive: bool,
        chunk_ids: list[str] | None = None,
        query: str | None = None,
        topics: list[str] | None = None,
    ) -> None:
        """Track feedback interaction."""
        interaction = UserInteraction(
            interaction_type=(
                InteractionType.FEEDBACK_POSITIVE if is_positive
                else InteractionType.FEEDBACK_NEGATIVE
            ),
            timestamp=datetime.utcnow(),
            query=query,
            chunk_ids=chunk_ids or [],
            topics=topics or [],
        )
        await self.tracker.track_interaction(user_id, interaction)

    async def track_dwell_time(
        self,
        user_id: str,
        chunk_id: str,
        dwell_seconds: float,
        topics: list[str] | None = None,
    ) -> None:
        """Track dwell time on a chunk."""
        interaction = UserInteraction(
            interaction_type=InteractionType.DWELL_TIME,
            timestamp=datetime.utcnow(),
            chunk_ids=[chunk_id],
            topics=topics or [],
            dwell_time_seconds=dwell_seconds,
        )
        await self.tracker.track_interaction(user_id, interaction)

    async def personalize_results(
        self,
        user_id: str,
        chunks: list[dict[str, Any]],
        strength: float | None = None,
    ) -> list[dict[str, Any]]:
        """Personalize search results for a user.

        Args:
            user_id: User ID
            chunks: Retrieved chunks
            strength: Personalization strength (0-1)

        Returns:
            Personalized chunks
        """
        return await self.reranker.rerank(user_id, chunks, strength)

    async def get_user_context(self, user_id: str) -> dict[str, Any]:
        """Get user context for prompt enhancement.

        Returns context that can be added to LLM prompts for
        more personalized responses.
        """
        profile = await self.store.get_profile(user_id)
        if not profile or not profile.personalization_enabled:
            return {}

        context = {
            "preferred_topics": profile.preferred_topics,
            "top_interests": [t for t, _ in profile.get_top_topics(5)],
            "detail_level": profile.detail_level,
            "language": profile.language_preference,
            "recent_queries": profile.recent_queries[-5:],
        }

        return context

    async def delete_user_data(self, user_id: str) -> bool:
        """Delete all user data (GDPR right to erasure)."""
        return await self.store.delete_profile(user_id)

    async def export_user_data(self, user_id: str) -> dict[str, Any]:
        """Export all user data (GDPR data portability)."""
        profile = await self.store.get_profile(user_id)
        interactions = await self.store.get_interactions(user_id, limit=10000)

        return {
            "profile": profile.to_dict() if profile else None,
            "interactions": [i.to_dict() for i in interactions],
            "exported_at": datetime.utcnow().isoformat(),
        }


# Module-level singleton
_personalization_engine: PersonalizationEngine | None = None


async def get_personalization_engine(
    surrealdb_client: Any = None,
) -> PersonalizationEngine:
    """Get or create personalization engine."""
    global _personalization_engine
    if _personalization_engine is None:
        _personalization_engine = PersonalizationEngine(surrealdb_client)
    return _personalization_engine
