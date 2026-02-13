"""Real BM25 sparse retrieval with phrase and proximity queries.

This module implements proper BM25 scoring (not just TF-IDF) with support for:
- BM25 scoring with k1 and b parameters
- Phrase queries (exact phrase matching)
- Proximity queries (terms within N words)
- Document length normalization

Based on the BM25 ranking function:
score(D, Q) = sum(IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl)))

where:
- IDF(qi) = inverse document frequency of query term qi
- f(qi, D) = frequency of qi in document D
- |D| = document length
- avgdl = average document length in corpus
- k1, b = tuning parameters (typically k1=1.2-2.0, b=0.75)
"""

import asyncio
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)


@dataclass
class BM25Document:
    """Document for BM25 indexing."""
    doc_id: str
    text: str
    tokens: list[str] = field(default_factory=list)
    token_positions: dict[str, list[int]] = field(default_factory=dict)
    doc_length: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PhraseMatch:
    """A phrase match in a document."""
    doc_id: str
    start_pos: int
    end_pos: int
    phrase: str
    score_boost: float = 1.0


@dataclass
class BM25Result:
    """BM25 search result."""
    doc_id: str
    score: float
    phrase_matches: list[PhraseMatch] = field(default_factory=list)
    proximity_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BM25Index:
    """BM25 index with phrase and proximity support.

    This is a proper BM25 implementation with:
    - Inverse document frequency (IDF) calculation
    - Document length normalization
    - Phrase query support (exact phrase matching)
    - Proximity query support (terms within N positions)

    The index can be built incrementally and supports incremental updates.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        phrase_boost: float = 1.5,
        proximity_window: int = 10,
        settings: Settings | None = None,
    ):
        """Initialize BM25 index.

        Args:
            k1: BM25 term frequency saturation parameter (default: 1.5)
            b: BM25 document length normalization (default: 0.75)
            phrase_boost: Score boost for phrase matches (default: 1.5)
            proximity_window: Max distance for proximity scoring (default: 10)
            settings: Application settings
        """
        self.k1 = k1
        self.b = b
        self.phrase_boost = phrase_boost
        self.proximity_window = proximity_window
        self.settings = settings or get_settings()

        # Index structures
        self.documents: dict[str, BM25Document] = {}
        self.inverted_index: dict[str, set[str]] = defaultdict(set)  # token -> doc_ids
        self.doc_freqs: dict[str, int] = defaultdict(int)  # token -> document frequency
        self.term_freqs: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))  # doc_id -> token -> freq
        self.doc_lengths: dict[str, int] = {}
        self.avgdl: float = 0.0
        self.num_docs: int = 0
        self._lock = asyncio.Lock()

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for indexing.

        Lowercase, remove punctuation, split on whitespace.
        Preserves position information for phrase/proximity queries.
        """
        # Lowercase
        text = text.lower()
        # Keep alphanumeric and spaces, replace other with space
        text = re.sub(r'[^a-záéíóúñü0-9\s]', ' ', text)
        # Split and filter empty
        tokens = [t for t in text.split() if t and len(t) > 1]
        return tokens

    def _tokenize_with_positions(self, text: str) -> tuple[list[str], dict[str, list[int]]]:
        """Tokenize text and return position mapping.

        Returns:
            Tuple of (tokens list, token -> positions dict)
        """
        tokens = self._tokenize(text)
        token_positions = defaultdict(list)
        for pos, token in enumerate(tokens):
            token_positions[token].append(pos)
        return tokens, dict(token_positions)

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a document to the index.

        Args:
            doc_id: Unique document identifier
            text: Document text content
            metadata: Optional metadata to store with document
        """
        # Remove existing document if present
        if doc_id in self.documents:
            self.remove_document(doc_id)

        # Tokenize
        tokens, token_positions = self._tokenize_with_positions(text)

        # Create document
        doc = BM25Document(
            doc_id=doc_id,
            text=text.lower(),
            tokens=tokens,
            token_positions=token_positions,
            doc_length=len(tokens),
            metadata=metadata or {},
        )

        # Update index structures
        self.documents[doc_id] = doc
        self.doc_lengths[doc_id] = len(tokens)

        # Update inverted index and term frequencies
        token_counts = Counter(tokens)
        for token, count in token_counts.items():
            self.inverted_index[token].add(doc_id)
            self.doc_freqs[token] = len(self.inverted_index[token])
            self.term_freqs[doc_id][token] = count

        # Update statistics
        self.num_docs = len(self.documents)
        if self.num_docs > 0:
            self.avgdl = sum(self.doc_lengths.values()) / self.num_docs

    def remove_document(self, doc_id: str) -> None:
        """Remove a document from the index."""
        if doc_id not in self.documents:
            return

        # Get document tokens
        doc = self.documents[doc_id]
        tokens = set(doc.tokens)

        # Update inverted index
        for token in tokens:
            self.inverted_index[token].discard(doc_id)
            if not self.inverted_index[token]:
                del self.inverted_index[token]
                if token in self.doc_freqs:
                    del self.doc_freqs[token]
            else:
                self.doc_freqs[token] = len(self.inverted_index[token])

        # Remove from term frequencies
        if doc_id in self.term_freqs:
            del self.term_freqs[doc_id]

        # Remove document
        del self.documents[doc_id]
        del self.doc_lengths[doc_id]

        # Update statistics
        self.num_docs = len(self.documents)
        if self.num_docs > 0:
            self.avgdl = sum(self.doc_lengths.values()) / self.num_docs
        else:
            self.avgdl = 0.0

    def _idf(self, token: str) -> float:
        """Calculate inverse document frequency for a token.

        Uses the BM25 IDF formula: log((N - n + 0.5) / (n + 0.5) + 1)
        where N = total docs, n = docs containing token
        """
        n = self.doc_freqs.get(token, 0)
        if n == 0:
            return 0.0
        return math.log((self.num_docs - n + 0.5) / (n + 0.5) + 1)

    def _bm25_score(self, doc_id: str, query_tokens: list[str]) -> float:
        """Calculate BM25 score for a document given query tokens."""
        if doc_id not in self.documents:
            return 0.0

        doc_length = self.doc_lengths[doc_id]
        score = 0.0

        for token in query_tokens:
            tf = self.term_freqs[doc_id].get(token, 0)
            if tf == 0:
                continue

            idf = self._idf(token)
            if idf == 0:
                continue

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avgdl)
            score += idf * numerator / denominator

        return score

    def _find_phrase_matches(self, doc_id: str, phrase_tokens: list[str]) -> list[PhraseMatch]:
        """Find all occurrences of a phrase in a document."""
        if doc_id not in self.documents or not phrase_tokens:
            return []

        doc = self.documents[doc_id]
        matches = []

        first_token = phrase_tokens[0]
        if first_token not in doc.token_positions:
            return []

        # For each position of the first token, check if phrase matches
        for start_pos in doc.token_positions[first_token]:
            match = True
            for i, token in enumerate(phrase_tokens):
                expected_pos = start_pos + i
                if token not in doc.token_positions:
                    match = False
                    break
                if expected_pos not in doc.token_positions[token]:
                    match = False
                    break

            if match:
                phrase_text = ' '.join(phrase_tokens)
                matches.append(PhraseMatch(
                    doc_id=doc_id,
                    start_pos=start_pos,
                    end_pos=start_pos + len(phrase_tokens) - 1,
                    phrase=phrase_text,
                    score_boost=self.phrase_boost,
                ))

        return matches

    def _proximity_score(
        self,
        doc_id: str,
        query_tokens: list[str],
    ) -> float:
        """Calculate proximity score based on how close query terms appear.

        Returns higher scores when query terms appear close together.
        """
        if doc_id not in self.documents or len(query_tokens) < 2:
            return 0.0

        doc = self.documents[doc_id]

        # Get all positions for each query token
        all_positions = []
        for token in query_tokens:
            if token in doc.token_positions:
                all_positions.append(doc.token_positions[token])

        if len(all_positions) < 2:
            return 0.0

        # Calculate minimum span containing all query terms
        # This is a simplified version - we look for pairs of terms
        min_spans = []
        for i, positions1 in enumerate(all_positions):
            for j, positions2 in enumerate(all_positions[i+1:], i+1):
                for p1 in positions1:
                    for p2 in positions2:
                        span = abs(p2 - p1)
                        if span <= self.proximity_window:
                            min_spans.append(span)

        if not min_spans:
            return 0.0

        # Score inversely proportional to min span
        min_span = min(min_spans)
        return 1.0 / (1.0 + min_span / self.proximity_window)

    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_doc_ids: set[str] | None = None,
        use_phrases: bool = True,
        use_proximity: bool = True,
    ) -> list[BM25Result]:
        """Search the index with BM25 scoring.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_doc_ids: Optional set of doc IDs to search within
            use_phrases: Enable phrase matching boost
            use_proximity: Enable proximity scoring

        Returns:
            List of BM25Result sorted by score descending
        """
        # Tokenize query
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        # Find candidate documents (containing at least one query term)
        candidate_docs: set[str] = set()
        for token in query_tokens:
            if token in self.inverted_index:
                candidate_docs.update(self.inverted_index[token])

        # Apply filter if provided
        if filter_doc_ids is not None:
            candidate_docs &= filter_doc_ids

        if not candidate_docs:
            return []

        # Score documents
        results = []
        for doc_id in candidate_docs:
            # Base BM25 score
            score = self._bm25_score(doc_id, query_tokens)

            # Phrase matching
            phrase_matches = []
            if use_phrases and len(query_tokens) > 1:
                # Check for phrase matches (sliding window of 2-4 tokens)
                for phrase_len in range(min(4, len(query_tokens)), 1, -1):
                    for i in range(len(query_tokens) - phrase_len + 1):
                        phrase_tokens = query_tokens[i:i + phrase_len]
                        matches = self._find_phrase_matches(doc_id, phrase_tokens)
                        if matches:
                            phrase_matches.extend(matches)
                            score += sum(m.score_boost for m in matches) * 0.5

            # Proximity scoring
            proximity_score = 0.0
            if use_proximity and len(query_tokens) > 1:
                proximity_score = self._proximity_score(doc_id, query_tokens)
                score += proximity_score * 0.3

            if score > 0:
                results.append(BM25Result(
                    doc_id=doc_id,
                    score=score,
                    phrase_matches=phrase_matches,
                    proximity_score=proximity_score,
                    metadata=self.documents[doc_id].metadata.copy(),
                ))

        # Sort by score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def search_phrase(
        self,
        phrase: str,
        top_k: int = 20,
        filter_doc_ids: set[str] | None = None,
    ) -> list[BM25Result]:
        """Search for exact phrase matches.

        Args:
            phrase: Phrase to search for (will be tokenized)
            top_k: Number of results
            filter_doc_ids: Optional filter

        Returns:
            Documents containing the exact phrase, ranked by frequency
        """
        phrase_tokens = self._tokenize(phrase)

        if not phrase_tokens:
            return []

        # Find documents containing first token
        first_token = phrase_tokens[0]
        if first_token not in self.inverted_index:
            return []

        candidates = self.inverted_index[first_token]
        if filter_doc_ids is not None:
            candidates &= filter_doc_ids

        results = []
        for doc_id in candidates:
            matches = self._find_phrase_matches(doc_id, phrase_tokens)
            if matches:
                # Score based on number of phrase occurrences
                score = len(matches) * self.phrase_boost
                results.append(BM25Result(
                    doc_id=doc_id,
                    score=score,
                    phrase_matches=matches,
                    metadata=self.documents[doc_id].metadata.copy(),
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        return {
            "num_documents": self.num_docs,
            "vocabulary_size": len(self.inverted_index),
            "avg_doc_length": round(self.avgdl, 2),
            "total_terms": sum(self.doc_lengths.values()),
            "k1": self.k1,
            "b": self.b,
            "phrase_boost": self.phrase_boost,
            "proximity_window": self.proximity_window,
        }


class BM25HybridRetriever:
    """Hybrid retriever combining BM25 with dense vectors.

    This class provides a unified interface for hybrid retrieval
    that combines BM25 sparse retrieval with dense vector search.
    """

    def __init__(
        self,
        bm25_index: BM25Index | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.bm25_index = bm25_index or BM25Index(settings=self.settings)

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add document to BM25 index."""
        self.bm25_index.add_document(doc_id, text, metadata)

    def remove_document(self, doc_id: str) -> None:
        """Remove document from BM25 index."""
        self.bm25_index.remove_document(doc_id)

    async def search(
        self,
        query: str,
        top_k: int = 20,
        filter_doc_ids: set[str] | None = None,
    ) -> list[BM25Result]:
        """Search using BM25."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.bm25_index.search(query, top_k, filter_doc_ids),
        )

    async def search_phrase(
        self,
        phrase: str,
        top_k: int = 20,
        filter_doc_ids: set[str] | None = None,
    ) -> list[BM25Result]:
        """Search for exact phrase."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.bm25_index.search_phrase(phrase, top_k, filter_doc_ids),
        )

    def get_stats(self) -> dict[str, Any]:
        """Get BM25 index stats."""
        return self.bm25_index.get_stats()
