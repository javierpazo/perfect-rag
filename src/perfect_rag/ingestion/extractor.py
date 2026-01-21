"""Entity and relation extraction using NER and LLM."""

import re
from typing import Any

import structlog

from perfect_rag.config import Settings, get_settings
from perfect_rag.models.entity import Entity, EntityType
from perfect_rag.models.relation import Relation, RelationType

logger = structlog.get_logger(__name__)


class EntityExtractor:
    """Extract named entities from text using NER and/or LLM."""

    def __init__(
        self,
        settings: Settings | None = None,
        use_spacy: bool = True,
        use_llm: bool = False,
        llm_gateway: Any = None,
    ):
        self.settings = settings or get_settings()
        self.use_spacy = use_spacy
        self.use_llm = use_llm
        self.llm_gateway = llm_gateway
        self._nlp = None

    async def initialize(self) -> None:
        """Load NER models."""
        if self.use_spacy:
            try:
                import spacy

                # Try to load a multilingual model
                try:
                    self._nlp = spacy.load("xx_ent_wiki_sm")
                except OSError:
                    # Fallback to English
                    try:
                        self._nlp = spacy.load("en_core_web_sm")
                    except OSError:
                        logger.warning(
                            "No spaCy model found. Run: python -m spacy download en_core_web_sm"
                        )
                        self._nlp = None
            except ImportError:
                logger.warning("spaCy not installed. Install with: pip install spacy")
                self._nlp = None
            except Exception as e:
                # Handle Pydantic compatibility issues with Python 3.14
                logger.warning(
                    "spaCy initialization failed (may be Python version incompatibility)",
                    error=str(e),
                )
                self._nlp = None

    async def extract(
        self,
        text: str,
        chunk_id: str,
        doc_id: str,
    ) -> list[Entity]:
        """Extract entities from text.

        Args:
            text: Text to extract entities from
            chunk_id: Source chunk ID
            doc_id: Source document ID

        Returns:
            List of extracted Entity objects
        """
        entities = []

        # SpaCy NER
        if self.use_spacy and self._nlp:
            spacy_entities = await self._extract_spacy(text, chunk_id, doc_id)
            entities.extend(spacy_entities)

        # LLM extraction (more accurate but slower/expensive)
        if self.use_llm and self.llm_gateway:
            llm_entities = await self._extract_llm(text, chunk_id, doc_id)
            entities.extend(llm_entities)

        # Deduplicate by normalized name
        entities = self._deduplicate(entities)

        return entities

    async def _extract_spacy(
        self,
        text: str,
        chunk_id: str,
        doc_id: str,
    ) -> list[Entity]:
        """Extract entities using spaCy NER."""
        import asyncio

        loop = asyncio.get_event_loop()
        doc = await loop.run_in_executor(None, self._nlp, text)

        entities = []
        seen = set()

        # Map spaCy entity types to our types
        type_mapping = {
            "PERSON": EntityType.PERSON,
            "PER": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "DATE": EntityType.DATE,
            "TIME": EntityType.DATE,
            "MONEY": EntityType.CONCEPT,
            "PRODUCT": EntityType.CONCEPT,
            "EVENT": EntityType.EVENT,
            "WORK_OF_ART": EntityType.CONCEPT,
            "LAW": EntityType.CONCEPT,
            "LANGUAGE": EntityType.CONCEPT,
            "FAC": EntityType.LOCATION,
            "NORP": EntityType.ORGANIZATION,
            "MISC": EntityType.OTHER,
        }

        for ent in doc.ents:
            # Skip very short entities
            if len(ent.text.strip()) < 2:
                continue

            # Normalize name
            normalized = self._normalize_name(ent.text)
            if normalized in seen:
                continue
            seen.add(normalized)

            entity_type = type_mapping.get(ent.label_, EntityType.OTHER)

            entities.append(
                Entity(
                    id=f"ent_{doc_id}_{len(entities)}",
                    name=ent.text.strip(),
                    normalized_name=normalized,
                    entity_type=entity_type,
                    source_chunks=[chunk_id],
                    confidence=0.8,  # Default confidence for spaCy
                    aliases=[],
                    metadata={
                        "spacy_label": ent.label_,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                    },
                )
            )

        return entities

    async def _extract_llm(
        self,
        text: str,
        chunk_id: str,
        doc_id: str,
    ) -> list[Entity]:
        """Extract entities using LLM."""
        if not self.llm_gateway:
            return []

        prompt = f"""Extract all named entities from the following text.
For each entity, provide:
- name: The entity name as it appears in text
- type: One of [PERSON, ORGANIZATION, LOCATION, DATE, CONCEPT, EVENT, OTHER]
- confidence: Your confidence (0.0 to 1.0)

Text:
{text}

Respond in JSON format:
{{"entities": [{{"name": "...", "type": "...", "confidence": 0.9}}]}}
"""

        try:
            response = await self.llm_gateway.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
            )

            # Parse JSON response
            import json

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                entities = []

                for ent_data in data.get("entities", []):
                    entity_type = EntityType(ent_data.get("type", "other").lower())
                    entities.append(
                        Entity(
                            id=f"ent_{doc_id}_{len(entities)}",
                            name=ent_data["name"],
                            normalized_name=self._normalize_name(ent_data["name"]),
                            entity_type=entity_type,
                            source_chunks=[chunk_id],
                            confidence=float(ent_data.get("confidence", 0.7)),
                            metadata={"source": "llm"},
                        )
                    )

                return entities

        except Exception as e:
            logger.warning("LLM entity extraction failed", error=str(e))

        return []

    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for deduplication."""
        # Lowercase, strip whitespace, remove extra spaces
        normalized = " ".join(name.lower().split())
        # Remove common prefixes/suffixes
        prefixes = ["the ", "a ", "an "]
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        return normalized

    def _deduplicate(self, entities: list[Entity]) -> list[Entity]:
        """Deduplicate entities by normalized name."""
        seen = {}
        for entity in entities:
            key = entity.normalized_name
            if key in seen:
                # Merge: add chunk to existing entity
                existing = seen[key]
                existing.source_chunks.extend(entity.source_chunks)
                existing.source_chunks = list(set(existing.source_chunks))
                # Keep higher confidence
                existing.confidence = max(existing.confidence, entity.confidence)
            else:
                seen[key] = entity

        return list(seen.values())


class RelationExtractor:
    """Extract relations between entities using LLM."""

    # Predefined relation patterns (for pattern-based extraction)
    RELATION_PATTERNS = [
        # Employment
        (r"(\w+)\s+(?:works?|worked|working)\s+(?:at|for)\s+(\w+)", "works_at"),
        (r"(\w+)\s+(?:is|was)\s+(?:the\s+)?(?:CEO|CTO|CFO|founder|president)\s+of\s+(\w+)", "leads"),
        # Location
        (r"(\w+)\s+(?:is|was)\s+(?:located|based)\s+in\s+(\w+)", "located_in"),
        (r"(\w+)\s+(?:headquarters?|HQ)\s+in\s+(\w+)", "headquartered_in"),
        # Relationships
        (r"(\w+)\s+(?:is|was)\s+(?:part|member)\s+of\s+(\w+)", "part_of"),
        (r"(\w+)\s+(?:acquired|bought|purchased)\s+(\w+)", "acquired"),
        (r"(\w+)\s+(?:created|founded|established)\s+(\w+)", "created"),
    ]

    def __init__(
        self,
        settings: Settings | None = None,
        use_patterns: bool = True,
        use_llm: bool = True,
        llm_gateway: Any = None,
    ):
        self.settings = settings or get_settings()
        self.use_patterns = use_patterns
        self.use_llm = use_llm
        self.llm_gateway = llm_gateway

    async def extract(
        self,
        text: str,
        entities: list[Entity],
        chunk_id: str,
    ) -> list[Relation]:
        """Extract relations between entities.

        Args:
            text: Text containing the entities
            entities: List of entities found in text
            chunk_id: Source chunk ID

        Returns:
            List of extracted Relation objects
        """
        relations = []

        # Pattern-based extraction
        if self.use_patterns:
            pattern_relations = self._extract_patterns(text, entities, chunk_id)
            relations.extend(pattern_relations)

        # LLM-based extraction
        if self.use_llm and self.llm_gateway and entities:
            llm_relations = await self._extract_llm(text, entities, chunk_id)
            relations.extend(llm_relations)

        # Deduplicate
        relations = self._deduplicate(relations)

        return relations

    def _extract_patterns(
        self,
        text: str,
        entities: list[Entity],
        chunk_id: str,
    ) -> list[Relation]:
        """Extract relations using regex patterns."""
        relations = []
        entity_names = {e.name.lower(): e for e in entities}
        entity_normalized = {e.normalized_name: e for e in entities}

        for pattern, rel_type in self.RELATION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                head_name = match.group(1).lower()
                tail_name = match.group(2).lower()

                # Find matching entities
                head = entity_names.get(head_name) or entity_normalized.get(head_name)
                tail = entity_names.get(tail_name) or entity_normalized.get(tail_name)

                if head and tail and head.id != tail.id:
                    try:
                        relation_type = RelationType(rel_type)
                    except ValueError:
                        relation_type = RelationType.RELATED_TO

                    relations.append(
                        Relation(
                            id=f"rel_{chunk_id}_{len(relations)}",
                            head_id=head.id,
                            tail_id=tail.id,
                            relation_type=relation_type,
                            evidence_chunk_id=chunk_id,
                            confidence=0.7,
                            metadata={"source": "pattern", "pattern": pattern},
                        )
                    )

        return relations

    async def _extract_llm(
        self,
        text: str,
        entities: list[Entity],
        chunk_id: str,
    ) -> list[Relation]:
        """Extract relations using LLM."""
        if not self.llm_gateway or len(entities) < 2:
            return []

        entity_list = "\n".join(f"- {e.name} ({e.entity_type.value})" for e in entities)

        prompt = f"""Given the following entities and text, identify relationships between the entities.

Entities:
{entity_list}

Text:
{text}

For each relationship, provide:
- head: The source entity name
- tail: The target entity name
- relation: One of [works_at, located_in, part_of, created_by, related_to, same_as, leads, acquired, founded]
- confidence: Your confidence (0.0 to 1.0)

Respond in JSON format:
{{"relations": [{{"head": "...", "tail": "...", "relation": "...", "confidence": 0.8}}]}}
"""

        try:
            response = await self.llm_gateway.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
            )

            import json

            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                relations = []

                # Create entity lookup
                entity_by_name = {e.name.lower(): e for e in entities}
                entity_by_normalized = {e.normalized_name: e for e in entities}

                for rel_data in data.get("relations", []):
                    head_name = rel_data.get("head", "").lower()
                    tail_name = rel_data.get("tail", "").lower()

                    head = entity_by_name.get(head_name) or entity_by_normalized.get(head_name)
                    tail = entity_by_name.get(tail_name) or entity_by_normalized.get(tail_name)

                    if head and tail and head.id != tail.id:
                        try:
                            relation_type = RelationType(rel_data.get("relation", "related_to"))
                        except ValueError:
                            relation_type = RelationType.RELATED_TO

                        relations.append(
                            Relation(
                                id=f"rel_{chunk_id}_{len(relations)}",
                                head_id=head.id,
                                tail_id=tail.id,
                                relation_type=relation_type,
                                evidence_chunk_id=chunk_id,
                                confidence=float(rel_data.get("confidence", 0.7)),
                                metadata={"source": "llm"},
                            )
                        )

                return relations

        except Exception as e:
            logger.warning("LLM relation extraction failed", error=str(e))

        return []

    def _deduplicate(self, relations: list[Relation]) -> list[Relation]:
        """Deduplicate relations."""
        seen = {}
        for rel in relations:
            key = (rel.head_id, rel.tail_id, rel.relation_type)
            if key in seen:
                # Keep higher confidence
                if rel.confidence > seen[key].confidence:
                    seen[key] = rel
            else:
                seen[key] = rel

        return list(seen.values())


class GraphBuilder:
    """Build knowledge graph from extracted entities and relations."""

    def __init__(
        self,
        entity_extractor: EntityExtractor,
        relation_extractor: RelationExtractor,
    ):
        self.entity_extractor = entity_extractor
        self.relation_extractor = relation_extractor

    async def build_from_chunks(
        self,
        chunks: list[dict[str, Any]],
        doc_id: str,
    ) -> tuple[list[Entity], list[Relation]]:
        """Build graph from document chunks.

        Args:
            chunks: List of chunk dicts with 'id' and 'content'
            doc_id: Document ID

        Returns:
            Tuple of (entities, relations)
        """
        all_entities = []
        all_relations = []

        for chunk in chunks:
            chunk_id = chunk["id"]
            content = chunk["content"]

            # Extract entities
            entities = await self.entity_extractor.extract(content, chunk_id, doc_id)
            all_entities.extend(entities)

            # Extract relations
            if entities:
                relations = await self.relation_extractor.extract(
                    content, entities, chunk_id
                )
                all_relations.extend(relations)

        # Global deduplication across chunks
        all_entities = self._merge_entities(all_entities)
        all_relations = self._update_relation_ids(all_relations, all_entities)

        return all_entities, all_relations

    def _merge_entities(self, entities: list[Entity]) -> list[Entity]:
        """Merge entities across chunks by normalized name."""
        merged = {}

        for entity in entities:
            key = entity.normalized_name

            if key in merged:
                existing = merged[key]
                # Merge source chunks
                existing.source_chunks.extend(entity.source_chunks)
                existing.source_chunks = list(set(existing.source_chunks))
                # Keep higher confidence
                existing.confidence = max(existing.confidence, entity.confidence)
                # Merge aliases
                if entity.name != existing.name:
                    existing.aliases.append(entity.name)
                    existing.aliases = list(set(existing.aliases))
            else:
                merged[key] = entity

        return list(merged.values())

    def _update_relation_ids(
        self,
        relations: list[Relation],
        merged_entities: list[Entity],
    ) -> list[Relation]:
        """Update relation entity IDs after entity merging."""
        # Create mapping from old IDs to merged entity IDs
        # This is simplified - in practice you'd track ID mappings during merge
        return relations
