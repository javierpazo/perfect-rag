"""Temporal reasoning for time-aware retrieval.

Provides temporal query parsing, filtering, and ranking for time-sensitive
document retrieval in RAG systems.
"""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import structlog
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class TemporalOperator(str, Enum):
    """Temporal operators for filtering."""

    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    BETWEEN = "between"
    EXACT = "exact"
    AROUND = "around"  # Within a range


@dataclass
class TemporalExpression:
    """Parsed temporal expression from a query."""

    operator: TemporalOperator
    start_date: datetime | None = None
    end_date: datetime | None = None
    reference_date: datetime | None = None
    is_relative: bool = False
    original_text: str = ""
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operator": self.operator.value,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "reference_date": self.reference_date.isoformat() if self.reference_date else None,
            "is_relative": self.is_relative,
            "original_text": self.original_text,
            "confidence": self.confidence,
        }


@dataclass
class TemporalParseResult:
    """Result of temporal query parsing."""

    has_temporal_constraint: bool = False
    expressions: list[TemporalExpression] = field(default_factory=list)
    cleaned_query: str = ""
    recency_preference: float = 0.0  # 0 = no preference, 1 = strong recency preference

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "has_temporal_constraint": self.has_temporal_constraint,
            "expressions": [e.to_dict() for e in self.expressions],
            "cleaned_query": self.cleaned_query,
            "recency_preference": self.recency_preference,
        }


class TemporalQueryParser:
    """Parse temporal expressions from natural language queries.

    Supports:
    - Relative time: "last week", "yesterday", "3 months ago"
    - Absolute time: "in 2023", "before March", "after January 15"
    - Date ranges: "between 2020 and 2022", "from May to July"
    - Recency indicators: "recent", "latest", "current"
    """

    # Relative time patterns
    RELATIVE_PATTERNS = [
        # Last N time_units
        (r"\blast\s+(\d+)\s+(day|week|month|year|hour|minute)s?\b", "last_n"),
        (r"\bpast\s+(\d+)\s+(day|week|month|year|hour|minute)s?\b", "last_n"),
        # Last time_unit
        (r"\blast\s+(day|week|month|year|quarter)\b", "last_unit"),
        (r"\bprevious\s+(day|week|month|year|quarter)\b", "last_unit"),
        # Ago patterns
        (r"\b(\d+)\s+(day|week|month|year|hour|minute)s?\s+ago\b", "n_ago"),
        # Simple relative
        (r"\byesterday\b", "yesterday"),
        (r"\btoday\b", "today"),
        (r"\bthis\s+(week|month|year|quarter)\b", "this_unit"),
        # Recency
        (r"\b(recent|recently|latest|current|newest)\b", "recency"),
    ]

    # Absolute time patterns
    ABSOLUTE_PATTERNS = [
        # Year patterns
        (r"\bin\s+(\d{4})\b", "in_year"),
        (r"\b(from|since)\s+(\d{4})\b", "from_year"),
        (r"\b(before|prior to)\s+(\d{4})\b", "before_year"),
        (r"\b(after|since)\s+(\d{4})\b", "after_year"),
        # Month patterns
        (r"\bin\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{4}))?\b", "in_month"),
        (r"\b(before|prior to)\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{4}))?\b", "before_month"),
        (r"\b(after)\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{4}))?\b", "after_month"),
        # Date ranges
        (r"\bbetween\s+(\d{4})\s+and\s+(\d{4})\b", "between_years"),
        (r"\bfrom\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+to\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{4}))?\b", "month_range"),
        # Specific date
        (r"\bon\s+(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b", "specific_date"),
        (r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b", "month_day"),
    ]

    MONTH_MAP = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._compiled_relative = [
            (re.compile(p, re.IGNORECASE), t) for p, t in self.RELATIVE_PATTERNS
        ]
        self._compiled_absolute = [
            (re.compile(p, re.IGNORECASE), t) for p, t in self.ABSOLUTE_PATTERNS
        ]

    async def parse(
        self,
        query: str,
        reference_time: datetime | None = None,
    ) -> TemporalParseResult:
        """Parse temporal expressions from query.

        Args:
            query: User query
            reference_time: Reference time for relative expressions (default: now)

        Returns:
            TemporalParseResult with parsed expressions
        """
        reference_time = reference_time or datetime.utcnow()
        expressions: list[TemporalExpression] = []
        recency_preference = 0.0
        cleaned_query = query

        # Parse relative patterns
        for pattern, pattern_type in self._compiled_relative:
            for match in pattern.finditer(query):
                expr = self._parse_relative_match(
                    match, pattern_type, reference_time
                )
                if expr:
                    expressions.append(expr)
                    cleaned_query = cleaned_query.replace(match.group(0), "").strip()
                    if pattern_type == "recency":
                        recency_preference = max(recency_preference, 0.8)

        # Parse absolute patterns
        for pattern, pattern_type in self._compiled_absolute:
            for match in pattern.finditer(query):
                expr = self._parse_absolute_match(
                    match, pattern_type, reference_time
                )
                if expr:
                    expressions.append(expr)
                    cleaned_query = cleaned_query.replace(match.group(0), "").strip()

        # Clean up extra whitespace
        cleaned_query = " ".join(cleaned_query.split())

        result = TemporalParseResult(
            has_temporal_constraint=len(expressions) > 0,
            expressions=expressions,
            cleaned_query=cleaned_query,
            recency_preference=recency_preference,
        )

        logger.debug(
            "Parsed temporal query",
            query=query[:100],
            expressions_count=len(expressions),
            has_temporal=result.has_temporal_constraint,
        )

        return result

    def _parse_relative_match(
        self,
        match: re.Match,
        pattern_type: str,
        reference_time: datetime,
    ) -> TemporalExpression | None:
        """Parse a relative time match."""
        try:
            if pattern_type == "last_n":
                n = int(match.group(1))
                unit = match.group(2).lower()
                delta = self._get_timedelta(n, unit)
                start_date = reference_time - delta
                return TemporalExpression(
                    operator=TemporalOperator.BETWEEN,
                    start_date=start_date,
                    end_date=reference_time,
                    reference_date=reference_time,
                    is_relative=True,
                    original_text=match.group(0),
                )

            elif pattern_type == "last_unit":
                unit = match.group(1).lower()
                start_date, end_date = self._get_last_unit_range(unit, reference_time)
                return TemporalExpression(
                    operator=TemporalOperator.BETWEEN,
                    start_date=start_date,
                    end_date=end_date,
                    reference_date=reference_time,
                    is_relative=True,
                    original_text=match.group(0),
                )

            elif pattern_type == "n_ago":
                n = int(match.group(1))
                unit = match.group(2).lower()
                delta = self._get_timedelta(n, unit)
                target_date = reference_time - delta
                return TemporalExpression(
                    operator=TemporalOperator.AROUND,
                    start_date=target_date - timedelta(days=1),
                    end_date=target_date + timedelta(days=1),
                    reference_date=target_date,
                    is_relative=True,
                    original_text=match.group(0),
                )

            elif pattern_type == "yesterday":
                yesterday = reference_time - timedelta(days=1)
                return TemporalExpression(
                    operator=TemporalOperator.EXACT,
                    start_date=yesterday.replace(hour=0, minute=0, second=0),
                    end_date=yesterday.replace(hour=23, minute=59, second=59),
                    reference_date=yesterday,
                    is_relative=True,
                    original_text=match.group(0),
                )

            elif pattern_type == "today":
                return TemporalExpression(
                    operator=TemporalOperator.EXACT,
                    start_date=reference_time.replace(hour=0, minute=0, second=0),
                    end_date=reference_time,
                    reference_date=reference_time,
                    is_relative=True,
                    original_text=match.group(0),
                )

            elif pattern_type == "this_unit":
                unit = match.group(1).lower()
                start_date, end_date = self._get_this_unit_range(unit, reference_time)
                return TemporalExpression(
                    operator=TemporalOperator.DURING,
                    start_date=start_date,
                    end_date=end_date,
                    reference_date=reference_time,
                    is_relative=True,
                    original_text=match.group(0),
                )

            elif pattern_type == "recency":
                # Recency indicator - boost recent documents
                return TemporalExpression(
                    operator=TemporalOperator.AFTER,
                    start_date=reference_time - timedelta(days=30),
                    reference_date=reference_time,
                    is_relative=True,
                    original_text=match.group(0),
                    confidence=0.7,  # Lower confidence - it's a preference, not hard filter
                )

        except (ValueError, IndexError) as e:
            logger.warning("Failed to parse relative time", error=str(e))
            return None

        return None

    def _parse_absolute_match(
        self,
        match: re.Match,
        pattern_type: str,
        reference_time: datetime,
    ) -> TemporalExpression | None:
        """Parse an absolute time match."""
        try:
            if pattern_type == "in_year":
                year = int(match.group(1))
                return TemporalExpression(
                    operator=TemporalOperator.DURING,
                    start_date=datetime(year, 1, 1),
                    end_date=datetime(year, 12, 31, 23, 59, 59),
                    is_relative=False,
                    original_text=match.group(0),
                )

            elif pattern_type == "from_year":
                year = int(match.group(2))
                return TemporalExpression(
                    operator=TemporalOperator.AFTER,
                    start_date=datetime(year, 1, 1),
                    is_relative=False,
                    original_text=match.group(0),
                )

            elif pattern_type == "before_year":
                year = int(match.group(2))
                return TemporalExpression(
                    operator=TemporalOperator.BEFORE,
                    end_date=datetime(year, 1, 1),
                    is_relative=False,
                    original_text=match.group(0),
                )

            elif pattern_type == "after_year":
                year = int(match.group(2))
                return TemporalExpression(
                    operator=TemporalOperator.AFTER,
                    start_date=datetime(year, 12, 31),
                    is_relative=False,
                    original_text=match.group(0),
                )

            elif pattern_type == "in_month":
                month_name = match.group(1).lower()
                month = self.MONTH_MAP[month_name]
                year = int(match.group(2)) if match.group(2) else reference_time.year
                start_date, end_date = self._get_month_range(year, month)
                return TemporalExpression(
                    operator=TemporalOperator.DURING,
                    start_date=start_date,
                    end_date=end_date,
                    is_relative=False,
                    original_text=match.group(0),
                )

            elif pattern_type == "before_month":
                month_name = match.group(2).lower()
                month = self.MONTH_MAP[month_name]
                year = int(match.group(3)) if match.group(3) else reference_time.year
                return TemporalExpression(
                    operator=TemporalOperator.BEFORE,
                    end_date=datetime(year, month, 1),
                    is_relative=False,
                    original_text=match.group(0),
                )

            elif pattern_type == "after_month":
                month_name = match.group(2).lower()
                month = self.MONTH_MAP[month_name]
                year = int(match.group(3)) if match.group(3) else reference_time.year
                _, end_date = self._get_month_range(year, month)
                return TemporalExpression(
                    operator=TemporalOperator.AFTER,
                    start_date=end_date,
                    is_relative=False,
                    original_text=match.group(0),
                )

            elif pattern_type == "between_years":
                year1 = int(match.group(1))
                year2 = int(match.group(2))
                return TemporalExpression(
                    operator=TemporalOperator.BETWEEN,
                    start_date=datetime(min(year1, year2), 1, 1),
                    end_date=datetime(max(year1, year2), 12, 31, 23, 59, 59),
                    is_relative=False,
                    original_text=match.group(0),
                )

            elif pattern_type == "month_range":
                month1_name = match.group(1).lower()
                month2_name = match.group(2).lower()
                month1 = self.MONTH_MAP[month1_name]
                month2 = self.MONTH_MAP[month2_name]
                year = int(match.group(3)) if match.group(3) else reference_time.year
                start_date, _ = self._get_month_range(year, month1)
                _, end_date = self._get_month_range(year, month2)
                return TemporalExpression(
                    operator=TemporalOperator.BETWEEN,
                    start_date=start_date,
                    end_date=end_date,
                    is_relative=False,
                    original_text=match.group(0),
                )

            elif pattern_type == "specific_date":
                day = int(match.group(1))
                month = int(match.group(2))
                year = int(match.group(3))
                if year < 100:
                    year += 2000 if year < 50 else 1900
                target_date = datetime(year, month, day)
                return TemporalExpression(
                    operator=TemporalOperator.EXACT,
                    start_date=target_date.replace(hour=0, minute=0, second=0),
                    end_date=target_date.replace(hour=23, minute=59, second=59),
                    reference_date=target_date,
                    is_relative=False,
                    original_text=match.group(0),
                )

            elif pattern_type == "month_day":
                month_name = match.group(1).lower()
                month = self.MONTH_MAP[month_name]
                day = int(match.group(2))
                year = int(match.group(3)) if match.group(3) else reference_time.year
                target_date = datetime(year, month, day)
                return TemporalExpression(
                    operator=TemporalOperator.EXACT,
                    start_date=target_date.replace(hour=0, minute=0, second=0),
                    end_date=target_date.replace(hour=23, minute=59, second=59),
                    reference_date=target_date,
                    is_relative=False,
                    original_text=match.group(0),
                )

        except (ValueError, IndexError) as e:
            logger.warning("Failed to parse absolute time", error=str(e))
            return None

        return None

    def _get_timedelta(self, n: int, unit: str) -> timedelta:
        """Get timedelta for n units."""
        if unit in ("minute", "minutes"):
            return timedelta(minutes=n)
        elif unit in ("hour", "hours"):
            return timedelta(hours=n)
        elif unit in ("day", "days"):
            return timedelta(days=n)
        elif unit in ("week", "weeks"):
            return timedelta(weeks=n)
        elif unit in ("month", "months"):
            return timedelta(days=n * 30)  # Approximate
        elif unit in ("year", "years"):
            return timedelta(days=n * 365)  # Approximate
        return timedelta(days=n)

    def _get_last_unit_range(
        self,
        unit: str,
        reference_time: datetime,
    ) -> tuple[datetime, datetime]:
        """Get date range for 'last week/month/year'."""
        if unit == "day":
            start = reference_time - timedelta(days=1)
            return start.replace(hour=0, minute=0, second=0), start.replace(hour=23, minute=59, second=59)
        elif unit == "week":
            start = reference_time - timedelta(weeks=1)
            start = start - timedelta(days=start.weekday())  # Start of that week
            end = start + timedelta(days=6)
            return start.replace(hour=0, minute=0, second=0), end.replace(hour=23, minute=59, second=59)
        elif unit == "month":
            last_month = reference_time - relativedelta(months=1)
            start = last_month.replace(day=1, hour=0, minute=0, second=0)
            end = (start + relativedelta(months=1)) - timedelta(seconds=1)
            return start, end
        elif unit == "year":
            last_year = reference_time.year - 1
            return datetime(last_year, 1, 1), datetime(last_year, 12, 31, 23, 59, 59)
        elif unit == "quarter":
            # Get last quarter
            current_quarter = (reference_time.month - 1) // 3
            if current_quarter == 0:
                start = datetime(reference_time.year - 1, 10, 1)
                end = datetime(reference_time.year - 1, 12, 31, 23, 59, 59)
            else:
                start_month = (current_quarter - 1) * 3 + 1
                end_month = start_month + 2
                start = datetime(reference_time.year, start_month, 1)
                end = datetime(reference_time.year, end_month, 28, 23, 59, 59)  # Approximate
            return start, end
        return reference_time - timedelta(days=7), reference_time

    def _get_this_unit_range(
        self,
        unit: str,
        reference_time: datetime,
    ) -> tuple[datetime, datetime]:
        """Get date range for 'this week/month/year'."""
        if unit == "week":
            start = reference_time - timedelta(days=reference_time.weekday())
            return start.replace(hour=0, minute=0, second=0), reference_time
        elif unit == "month":
            start = reference_time.replace(day=1, hour=0, minute=0, second=0)
            return start, reference_time
        elif unit == "year":
            start = reference_time.replace(month=1, day=1, hour=0, minute=0, second=0)
            return start, reference_time
        elif unit == "quarter":
            quarter = (reference_time.month - 1) // 3
            start_month = quarter * 3 + 1
            start = datetime(reference_time.year, start_month, 1)
            return start, reference_time
        return reference_time.replace(hour=0, minute=0, second=0), reference_time

    def _get_month_range(
        self,
        year: int,
        month: int,
    ) -> tuple[datetime, datetime]:
        """Get date range for a specific month."""
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
        else:
            end = datetime(year, month + 1, 1) - timedelta(seconds=1)
        return start, end


@dataclass
class TemporalChunk:
    """Chunk with temporal metadata."""

    chunk_id: str
    content: str
    score: float
    created_at: datetime | None = None
    modified_at: datetime | None = None
    document_date: datetime | None = None
    content_dates: list[datetime] = field(default_factory=list)
    temporal_relevance: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class TemporalFilter:
    """Filter chunks by temporal relevance.

    Filters documents based on:
    - Document creation/modification date
    - Dates mentioned in content
    - Temporal expressions from query
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.parser = TemporalQueryParser(settings)

    async def filter(
        self,
        chunks: list[dict[str, Any]],
        temporal_result: TemporalParseResult,
        strict: bool = False,
    ) -> list[dict[str, Any]]:
        """Filter chunks by temporal constraints.

        Args:
            chunks: List of chunk dictionaries
            temporal_result: Parsed temporal expressions
            strict: If True, exclude non-matching chunks; if False, just lower their scores

        Returns:
            Filtered/scored chunks
        """
        if not temporal_result.has_temporal_constraint:
            return chunks

        filtered_chunks = []

        for chunk in chunks:
            temporal_score = await self._calculate_temporal_score(
                chunk, temporal_result
            )

            if strict and temporal_score < 0.3:
                continue

            # Adjust the chunk's score
            chunk_copy = chunk.copy()
            original_score = chunk_copy.get("score", 1.0)
            chunk_copy["temporal_score"] = temporal_score
            chunk_copy["score"] = original_score * (0.5 + 0.5 * temporal_score)
            filtered_chunks.append(chunk_copy)

        # Sort by adjusted score
        filtered_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)

        logger.debug(
            "Temporal filtering complete",
            input_count=len(chunks),
            output_count=len(filtered_chunks),
        )

        return filtered_chunks

    async def _calculate_temporal_score(
        self,
        chunk: dict[str, Any],
        temporal_result: TemporalParseResult,
    ) -> float:
        """Calculate temporal relevance score for a chunk."""
        # Get chunk dates
        payload = chunk.get("payload", {})
        metadata = payload.get("metadata", {})

        # Try different date fields
        chunk_date = None
        for date_field in ["created_at", "modified_at", "document_date", "date", "timestamp"]:
            date_str = metadata.get(date_field)
            if date_str:
                try:
                    if isinstance(date_str, datetime):
                        chunk_date = date_str
                    else:
                        chunk_date = date_parser.parse(str(date_str))
                    break
                except (ValueError, TypeError):
                    continue

        if not chunk_date:
            # No date found, return neutral score
            return 0.5

        # Check against each temporal expression
        max_score = 0.0
        for expr in temporal_result.expressions:
            score = self._score_against_expression(chunk_date, expr)
            max_score = max(max_score, score)

        return max_score

    def _score_against_expression(
        self,
        chunk_date: datetime,
        expr: TemporalExpression,
    ) -> float:
        """Score a chunk date against a temporal expression."""
        if expr.operator == TemporalOperator.BEFORE:
            if expr.end_date and chunk_date < expr.end_date:
                return 1.0
            return 0.0

        elif expr.operator == TemporalOperator.AFTER:
            if expr.start_date and chunk_date > expr.start_date:
                return 1.0
            return 0.0

        elif expr.operator in (TemporalOperator.DURING, TemporalOperator.BETWEEN):
            if expr.start_date and expr.end_date:
                if expr.start_date <= chunk_date <= expr.end_date:
                    return 1.0
                # Partial score for close dates
                if chunk_date < expr.start_date:
                    days_off = (expr.start_date - chunk_date).days
                elif chunk_date > expr.end_date:
                    days_off = (chunk_date - expr.end_date).days
                else:
                    days_off = 0
                return max(0, 1.0 - (days_off / 365))
            return 0.5

        elif expr.operator == TemporalOperator.EXACT:
            if expr.start_date and expr.end_date:
                if expr.start_date <= chunk_date <= expr.end_date:
                    return 1.0
            return 0.0

        elif expr.operator == TemporalOperator.AROUND:
            if expr.reference_date:
                days_diff = abs((chunk_date - expr.reference_date).days)
                # Score decreases with distance
                return max(0, 1.0 - (days_diff / 30))
            return 0.5

        return 0.5


class TemporalRanker:
    """Rank documents by temporal relevance.

    Applies temporal boosting to search results based on:
    - Recency preference (for 'latest', 'recent' queries)
    - Temporal match (for specific date queries)
    - Document freshness
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    async def rank(
        self,
        chunks: list[dict[str, Any]],
        temporal_result: TemporalParseResult | None = None,
        recency_weight: float = 0.2,
        reference_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Rank chunks with temporal boosting.

        Args:
            chunks: List of chunk dictionaries
            temporal_result: Optional parsed temporal expressions
            recency_weight: Weight for recency boosting (0-1)
            reference_time: Reference time for recency calculations

        Returns:
            Reranked chunks
        """
        reference_time = reference_time or datetime.utcnow()

        # Adjust recency weight based on query
        if temporal_result and temporal_result.recency_preference > 0:
            recency_weight = max(recency_weight, temporal_result.recency_preference)

        ranked_chunks = []
        for chunk in chunks:
            chunk_copy = chunk.copy()
            original_score = chunk_copy.get("score", 1.0)

            # Calculate recency boost
            recency_boost = await self._calculate_recency_boost(
                chunk, reference_time
            )

            # Calculate temporal match boost (if temporal constraint exists)
            temporal_boost = 1.0
            if temporal_result and temporal_result.has_temporal_constraint:
                temporal_boost = chunk_copy.get("temporal_score", 0.5)

            # Combine scores
            final_score = (
                original_score * (1 - recency_weight)
                + original_score * recency_weight * recency_boost * temporal_boost
            )

            chunk_copy["score"] = final_score
            chunk_copy["recency_boost"] = recency_boost
            chunk_copy["temporal_boost"] = temporal_boost
            ranked_chunks.append(chunk_copy)

        # Sort by final score
        ranked_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)

        return ranked_chunks

    async def _calculate_recency_boost(
        self,
        chunk: dict[str, Any],
        reference_time: datetime,
    ) -> float:
        """Calculate recency boost for a chunk."""
        # Get chunk date
        payload = chunk.get("payload", {})
        metadata = payload.get("metadata", {})

        chunk_date = None
        for date_field in ["created_at", "modified_at", "document_date", "date", "timestamp"]:
            date_str = metadata.get(date_field)
            if date_str:
                try:
                    if isinstance(date_str, datetime):
                        chunk_date = date_str
                    else:
                        chunk_date = date_parser.parse(str(date_str))
                    break
                except (ValueError, TypeError):
                    continue

        if not chunk_date:
            return 0.5  # Neutral boost

        # Calculate age in days
        age_days = (reference_time - chunk_date).days

        # Exponential decay
        # Documents less than a day old get full boost
        # 30-day-old documents get ~0.37 boost
        # 90-day-old documents get ~0.05 boost
        decay_rate = 0.033  # Controls how fast the boost decays
        boost = max(0, 1.0 * (2.718281828 ** (-decay_rate * age_days)))

        return boost

    async def boost_recent(
        self,
        chunks: list[dict[str, Any]],
        days_threshold: int = 30,
        boost_factor: float = 1.5,
    ) -> list[dict[str, Any]]:
        """Apply simple recency boost to recent documents.

        Args:
            chunks: List of chunk dictionaries
            days_threshold: Documents newer than this get boosted
            boost_factor: Multiplier for recent documents

        Returns:
            Boosted chunks
        """
        reference_time = datetime.utcnow()
        threshold_date = reference_time - timedelta(days=days_threshold)

        boosted_chunks = []
        for chunk in chunks:
            chunk_copy = chunk.copy()

            # Get chunk date
            payload = chunk_copy.get("payload", {})
            metadata = payload.get("metadata", {})

            chunk_date = None
            for date_field in ["created_at", "modified_at", "document_date", "date"]:
                date_str = metadata.get(date_field)
                if date_str:
                    try:
                        if isinstance(date_str, datetime):
                            chunk_date = date_str
                        else:
                            chunk_date = date_parser.parse(str(date_str))
                        break
                    except (ValueError, TypeError):
                        continue

            if chunk_date and chunk_date >= threshold_date:
                chunk_copy["score"] = chunk_copy.get("score", 1.0) * boost_factor
                chunk_copy["recent_boost"] = True

            boosted_chunks.append(chunk_copy)

        # Re-sort by score
        boosted_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)

        return boosted_chunks


class TemporalRetrievalEnhancer:
    """Combines temporal parsing, filtering, and ranking for retrieval.

    Provides a unified interface for temporal-aware retrieval.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.parser = TemporalQueryParser(settings)
        self.filter = TemporalFilter(settings)
        self.ranker = TemporalRanker(settings)

    async def enhance_retrieval(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        strict_temporal_filter: bool = False,
        recency_weight: float = 0.2,
    ) -> tuple[list[dict[str, Any]], TemporalParseResult]:
        """Enhance retrieval results with temporal awareness.

        Args:
            query: User query
            chunks: Retrieved chunks
            strict_temporal_filter: If True, remove non-matching chunks
            recency_weight: Weight for recency in ranking

        Returns:
            Tuple of (enhanced chunks, temporal parse result)
        """
        # Parse temporal expressions
        temporal_result = await self.parser.parse(query)

        if not temporal_result.has_temporal_constraint:
            # No temporal constraint, just apply light recency boost
            enhanced = await self.ranker.rank(
                chunks,
                recency_weight=recency_weight * 0.5,  # Reduced weight
            )
            return enhanced, temporal_result

        # Apply temporal filter
        filtered = await self.filter.filter(
            chunks,
            temporal_result,
            strict=strict_temporal_filter,
        )

        # Apply temporal ranking
        ranked = await self.ranker.rank(
            filtered,
            temporal_result=temporal_result,
            recency_weight=recency_weight,
        )

        logger.info(
            "Temporal retrieval enhancement complete",
            query_has_temporal=temporal_result.has_temporal_constraint,
            expressions_count=len(temporal_result.expressions),
            input_chunks=len(chunks),
            output_chunks=len(ranked),
        )

        return ranked, temporal_result


# Module-level singleton
_temporal_enhancer: TemporalRetrievalEnhancer | None = None


async def get_temporal_enhancer() -> TemporalRetrievalEnhancer:
    """Get or create temporal retrieval enhancer."""
    global _temporal_enhancer
    if _temporal_enhancer is None:
        _temporal_enhancer = TemporalRetrievalEnhancer()
    return _temporal_enhancer
