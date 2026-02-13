"""Entity normalization for deduplication and canonical form mapping.

Entity normalization is critical for knowledge graph quality:
- Same entity mentioned with different names ("H. pylori" vs "Helicobacter pylori")
- Abbreviations vs full names ("CHC" vs "carcinoma hepatocelular")
- Spelling variations and typos
- Multi-language variants

This module provides:
- Alias mapping for known entities
- Fuzzy matching for potential duplicates
- Canonical form selection
- Entity merging/canonicalization
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)


@dataclass
class EntityAlias:
    """Alias for an entity with metadata."""
    alias: str
    canonical_id: str
    alias_type: str  # "abbreviation", "synonym", "variant", "translation"
    confidence: float
    source: str = ""  # Where this alias came from


@dataclass
class NormalizationResult:
    """Result of entity normalization."""
    original: str
    canonical: str
    canonical_id: str
    confidence: float
    aliases_matched: list[str]
    needs_review: bool


class EntityNormalizer:
    """Normalize entity names to canonical forms.

    Uses multiple strategies:
    1. Exact match in alias dictionary
    2. Fuzzy matching with edit distance
    3. Acronym expansion
    4. LLM-based disambiguation (optional)
    """

    def __init__(
        self,
        llm_gateway: Any = None,
        settings: Settings | None = None,
    ):
        self.llm = llm_gateway
        self.settings = settings or get_settings()

        # Alias dictionary: alias -> canonical_id
        self._aliases: dict[str, str] = {}

        # Canonical entities: canonical_id -> canonical_name
        self._canonicals: dict[str, str] = {}

        # Alias metadata: canonical_id -> list of EntityAlias
        self._alias_metadata: dict[str, list[EntityAlias]] = {}

        # Common medical abbreviations
        self._medical_abbreviations = {
            # Spanish medical terms
            "chc": "carcinoma hepatocelular",
            "hcc": "hepatocellular carcinoma",
            "h. pylori": "Helicobacter pylori",
            "hp": "Helicobacter pylori",
            "ibp": "inhibidor de bomba de protones",
            "ppi": "proton pump inhibitor",
            "tnm": "tumor nodes metastasis",
            "vih": "virus de inmunodeficiencia humana",
            "hiv": "human immunodeficiency virus",
            "vhc": "virus de la hepatitis C",
            "hcv": "hepatitis C virus",
            "vhb": "virus de la hepatitis B",
            "hbv": "hepatitis B virus",
            "aeeh": "Asociación Española para el Estudio del Hígado",
            # Common conditions
            "cirrosis": "cirrosis hepática",
            "esteatosis": "esteatosis hepática",
            "hígado graso": "esteatosis hepática",
            "fatty liver": "esteatosis hepática",
            # Procedures
            "th": "trasplante hepático",
            "lt": "liver transplant",
            # Measurements
            "afp": "alfa-fetoproteína",
            "afp": "alpha-fetoprotein",
            "inr": "international normalized ratio",
            "alt": "alanine aminotransferase",
            "gpt": "alanine aminotransferase",
            "ast": "aspartate aminotransferase",
            "got": "aspartate aminotransferase",
            "ggt": "gamma-glutamyl transferase",
            "faf": "fosfatasa alcalina",
            "alp": "alkaline phosphatase",
        }

        # Initialize with abbreviations
        self._initialize_abbreviations()

    def _initialize_abbreviations(self) -> None:
        """Initialize alias dictionary with known abbreviations."""
        for abbr, full_form in self._medical_abbreviations.items():
            # Add abbreviation -> full form
            self._aliases[abbr.lower()] = full_form.lower()
            self._aliases[abbr.upper()] = full_form.lower()

            # Add full form as canonical
            canonical_id = self._make_canonical_id(full_form)
            self._canonicals[canonical_id] = full_form

            # Add metadata
            self._alias_metadata[canonical_id] = [
                EntityAlias(
                    alias=abbr,
                    canonical_id=canonical_id,
                    alias_type="abbreviation",
                    confidence=0.95,
                    source="predefined",
                ),
                EntityAlias(
                    alias=full_form,
                    canonical_id=canonical_id,
                    alias_type="canonical",
                    confidence=1.0,
                    source="predefined",
                ),
            ]

    def _make_canonical_id(self, name: str) -> str:
        """Create a canonical ID from entity name."""
        # Normalize: lowercase, remove punctuation, replace spaces with underscores
        normalized = re.sub(r'[^a-záéíóúñü0-9]', '_', name.lower())
        normalized = re.sub(r'_+', '_', normalized).strip('_')
        return f"entity:{normalized}"

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower().strip()
        # Normalize unicode
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def add_alias(
        self,
        alias: str,
        canonical_name: str,
        alias_type: str = "synonym",
        confidence: float = 0.8,
        source: str = "manual",
    ) -> None:
        """Add an alias for a canonical entity."""
        normalized_alias = self._normalize_text(alias)
        canonical_id = self._make_canonical_id(canonical_name)

        self._aliases[normalized_alias] = canonical_id
        self._canonicals[canonical_id] = canonical_name

        if canonical_id not in self._alias_metadata:
            self._alias_metadata[canonical_id] = []

        self._alias_metadata[canonical_id].append(EntityAlias(
            alias=alias,
            canonical_id=canonical_id,
            alias_type=alias_type,
            confidence=confidence,
            source=source,
        ))

    def normalize(self, entity_name: str) -> NormalizationResult:
        """Normalize an entity name to its canonical form.

        Args:
            entity_name: The entity name to normalize

        Returns:
            NormalizationResult with canonical form and confidence
        """
        normalized = self._normalize_text(entity_name)

        # 1. Check exact match in aliases
        if normalized in self._aliases:
            canonical_id = self._aliases[normalized]
            canonical = self._canonicals.get(canonical_id, entity_name)
            return NormalizationResult(
                original=entity_name,
                canonical=canonical,
                canonical_id=canonical_id,
                confidence=0.95,
                aliases_matched=[normalized],
                needs_review=False,
            )

        # 2. Check if it's already a canonical form
        canonical_id = self._make_canonical_id(entity_name)
        if canonical_id in self._canonicals:
            return NormalizationResult(
                original=entity_name,
                canonical=self._canonicals[canonical_id],
                canonical_id=canonical_id,
                confidence=1.0,
                aliases_matched=[],
                needs_review=False,
            )

        # 3. Try acronym expansion
        if len(normalized) <= 5 and normalized in self._medical_abbreviations:
            expanded = self._medical_abbreviations[normalized]
            expanded_id = self._make_canonical_id(expanded)
            return NormalizationResult(
                original=entity_name,
                canonical=expanded,
                canonical_id=expanded_id,
                confidence=0.9,
                aliases_matched=[normalized],
                needs_review=False,
            )

        # 4. Fuzzy matching (simple edit distance)
        fuzzy_match = self._fuzzy_match(normalized)
        if fuzzy_match:
            return NormalizationResult(
                original=entity_name,
                canonical=fuzzy_match["canonical"],
                canonical_id=fuzzy_match["canonical_id"],
                confidence=fuzzy_match["confidence"],
                aliases_matched=fuzzy_match["matched"],
                needs_review=fuzzy_match["confidence"] < 0.8,
            )

        # 5. No match - return as-is
        return NormalizationResult(
            original=entity_name,
            canonical=entity_name,
            canonical_id=self._make_canonical_id(entity_name),
            confidence=0.5,
            aliases_matched=[],
            needs_review=True,
        )

    def _fuzzy_match(
        self,
        normalized: str,
        threshold: float = 0.8,
    ) -> dict[str, Any] | None:
        """Try fuzzy matching against known aliases."""
        best_match = None
        best_score = 0

        for alias, canonical_id in self._aliases.items():
            score = self._similarity(normalized, alias)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = {
                    "canonical": self._canonicals.get(canonical_id, alias),
                    "canonical_id": canonical_id,
                    "confidence": score,
                    "matched": [alias],
                }

        return best_match

    def _similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings.

        Uses a combination of:
        - Jaccard similarity of tokens
        - Character-level similarity
        """
        # Token-level Jaccard
        tokens1 = set(s1.split())
        tokens2 = set(s2.split())

        if tokens1 and tokens2:
            jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2)
        else:
            jaccard = 0

        # Character-level similarity (simple)
        if len(s1) == 0 or len(s2) == 0:
            char_sim = 0
        else:
            # Simple edit distance approximation
            common = sum(1 for c in s1 if c in s2)
            char_sim = common / max(len(s1), len(s2))

        # Combine
        return 0.6 * jaccard + 0.4 * char_sim

    async def normalize_with_llm(
        self,
        entity_name: str,
        context: str = "",
    ) -> NormalizationResult:
        """Use LLM for disambiguation when fuzzy matching is uncertain.

        Args:
            entity_name: Entity to normalize
            context: Additional context (surrounding text)

        Returns:
            NormalizationResult with LLM-assisted disambiguation
        """
        # First try rule-based
        result = self.normalize(entity_name)

        if result.confidence >= 0.8 or not self.llm:
            return result

        # Use LLM for disambiguation
        prompt = f"""Given this entity mention, determine its canonical form.

Entity: "{entity_name}"
Context: {context[:200] if context else "No additional context"}

Known entities in our knowledge base:
{chr(10).join(f"- {c}" for c in list(self._canonicals.values())[:20])}

If the entity matches one of the known forms, respond with:
CANONICAL: [canonical form]
CONFIDENCE: [0.0-1.0]

If it's a new entity, respond with:
NEW_ENTITY: [suggested canonical form]
CONFIDENCE: [0.0-1.0]

Keep responses brief."""

        try:
            response = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
            )

            # Parse response
            canonical = entity_name
            confidence = 0.6

            for line in response.strip().split('\n'):
                if line.startswith("CANONICAL:"):
                    canonical = line.split(":", 1)[1].strip()
                elif line.startswith("NEW_ENTITY:"):
                    canonical = line.split(":", 1)[1].strip()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":")[1].strip())
                    except ValueError:
                        pass

            return NormalizationResult(
                original=entity_name,
                canonical=canonical,
                canonical_id=self._make_canonical_id(canonical),
                confidence=confidence,
                aliases_matched=result.aliases_matched,
                needs_review=True,
            )

        except Exception as e:
            logger.warning("LLM normalization failed", error=str(e))
            return result

    def get_aliases_for_entity(self, canonical_id: str) -> list[EntityAlias]:
        """Get all aliases for a canonical entity."""
        return self._alias_metadata.get(canonical_id, [])

    def get_stats(self) -> dict[str, int]:
        """Get normalizer statistics."""
        return {
            "total_aliases": len(self._aliases),
            "total_canonicals": len(self._canonicals),
            "predefined_abbreviations": len(self._medical_abbreviations),
        }


class EntityDeduplicator:
    """Deduplicate entities based on normalization."""

    def __init__(self, normalizer: EntityNormalizer):
        self.normalizer = normalizer

    def deduplicate(
        self,
        entities: list[dict[str, Any]],
        merge_strategy: str = "keep_first",
    ) -> list[dict[str, Any]]:
        """Deduplicate entities by normalizing to canonical forms.

        Args:
            entities: List of entity dicts with 'name' or 'text' field
            merge_strategy: How to handle duplicates
                - "keep_first": Keep first occurrence
                - "keep_highest_score": Keep entity with highest confidence
                - "merge": Merge metadata from all duplicates

        Returns:
            Deduplicated list of entities
        """
        seen_canonicals: dict[str, dict[str, Any]] = {}

        for entity in entities:
            name = entity.get("name") or entity.get("text", "")
            if not name:
                continue

            result = self.normalizer.normalize(name)
            canonical_id = result.canonical_id

            if canonical_id not in seen_canonicals:
                # New entity
                entity["canonical_name"] = result.canonical
                entity["canonical_id"] = canonical_id
                entity["normalization_confidence"] = result.confidence
                seen_canonicals[canonical_id] = entity

            else:
                # Duplicate
                if merge_strategy == "keep_first":
                    # Add alias to existing
                    existing = seen_canonicals[canonical_id]
                    if "aliases" not in existing:
                        existing["aliases"] = []
                    existing["aliases"].append(name)

                elif merge_strategy == "keep_highest_score":
                    existing = seen_canonicals[canonical_id]
                    existing_score = existing.get("confidence", 0)
                    new_score = entity.get("confidence", 0)
                    if new_score > existing_score:
                        entity["canonical_name"] = result.canonical
                        entity["canonical_id"] = canonical_id
                        entity["normalization_confidence"] = result.confidence
                        entity["aliases"] = existing.get("aliases", []) + [existing.get("name", "")]
                        seen_canonicals[canonical_id] = entity

                elif merge_strategy == "merge":
                    existing = seen_canonicals[canonical_id]
                    # Merge metadata
                    for key, value in entity.items():
                        if key not in existing:
                            existing[key] = value
                        elif isinstance(value, list) and isinstance(existing.get(key), list):
                            existing[key] = list(set(existing[key] + value))
                    # Add alias
                    if "aliases" not in existing:
                        existing["aliases"] = []
                    existing["aliases"].append(name)

        return list(seen_canonicals.values())


async def normalize_entities(
    entities: list[str],
    llm_gateway: Any = None,
) -> dict[str, str]:
    """Convenience function to normalize a list of entity names.

    Args:
        entities: List of entity names
        llm_gateway: Optional LLM for disambiguation

    Returns:
        Dict mapping original names to canonical forms
    """
    normalizer = EntityNormalizer(llm_gateway=llm_gateway)
    results = {}

    for entity in entities:
        result = normalizer.normalize(entity)
        results[entity] = result.canonical

    return results
