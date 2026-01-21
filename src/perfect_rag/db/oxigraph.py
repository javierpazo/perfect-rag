"""Oxigraph SPARQL client for RDF operations."""

from typing import Any

import httpx
import structlog
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from perfect_rag.config import Settings, get_settings
from perfect_rag.models.entity import Entity
from perfect_rag.models.relation import Relation

logger = structlog.get_logger(__name__)

# Namespaces
RAG = Namespace("http://perfect-rag.local/ontology#")
ENTITY = Namespace("http://perfect-rag.local/entity/")
CHUNK = Namespace("http://perfect-rag.local/chunk/")
DOC = Namespace("http://perfect-rag.local/document/")


class OxigraphClient:
    """Async Oxigraph SPARQL client."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._http_client: httpx.AsyncClient | None = None
        self.base_url = self.settings.oxigraph_url

    async def connect(self) -> None:
        """Initialize HTTP client."""
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0,
        )
        logger.info("Connected to Oxigraph", url=self.base_url)

    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            logger.info("Disconnected from Oxigraph")

    @property
    def client(self) -> httpx.AsyncClient:
        """Get HTTP client."""
        if self._http_client is None:
            raise RuntimeError("Oxigraph client not connected. Call connect() first.")
        return self._http_client

    # =========================================================================
    # SPARQL Query Operations
    # =========================================================================

    async def query(self, sparql: str) -> list[dict[str, Any]]:
        """Execute a SPARQL SELECT query."""
        response = await self.client.post(
            "/query",
            content=sparql,
            headers={
                "Content-Type": "application/sparql-query",
                "Accept": "application/sparql-results+json",
            },
        )
        response.raise_for_status()

        data = response.json()
        results = []

        for binding in data.get("results", {}).get("bindings", []):
            row = {}
            for var, val in binding.items():
                if val["type"] == "uri":
                    row[var] = val["value"]
                elif val["type"] == "literal":
                    row[var] = val["value"]
                elif val["type"] == "typed-literal":
                    row[var] = val["value"]
            results.append(row)

        return results

    async def update(self, sparql: str) -> None:
        """Execute a SPARQL UPDATE query."""
        response = await self.client.post(
            "/update",
            content=sparql,
            headers={"Content-Type": "application/sparql-update"},
        )
        response.raise_for_status()

    async def insert_triples(self, triples: list[tuple[str, str, str]]) -> None:
        """Insert RDF triples."""
        if not triples:
            return

        # Build INSERT DATA query
        triple_strings = []
        for s, p, o in triples:
            # Wrap URIs in angle brackets, literals in quotes
            if o.startswith("http://") or o.startswith("https://"):
                triple_strings.append(f"<{s}> <{p}> <{o}> .")
            else:
                # Escape quotes in literals
                escaped = o.replace("\\", "\\\\").replace('"', '\\"')
                triple_strings.append(f'<{s}> <{p}> "{escaped}" .')

        sparql = f"""
        INSERT DATA {{
            {chr(10).join(triple_strings)}
        }}
        """

        await self.update(sparql)
        logger.debug("Inserted triples", count=len(triples))

    async def delete_triples(self, subject: str | None = None) -> None:
        """Delete triples matching subject."""
        if subject:
            sparql = f"""
            DELETE WHERE {{
                <{subject}> ?p ?o .
            }}
            """
        else:
            sparql = "DELETE WHERE { ?s ?p ?o . }"

        await self.update(sparql)

    # =========================================================================
    # Entity & Relation Sync
    # =========================================================================

    async def sync_entity(self, entity: Entity) -> None:
        """Sync an entity to RDF store."""
        subject = f"{ENTITY}{entity.id}"
        entity_type_uri = f"{RAG}{entity.entity_type.value.capitalize()}"

        triples = [
            (subject, str(RDF.type), str(RAG.Entity)),
            (subject, str(RDF.type), entity_type_uri),
            (subject, str(RAG.name), entity.name),
            (subject, str(RAG.normalizedName), entity.normalized_name),
            (subject, str(RAG.entityType), entity.entity_type.value),
            (subject, str(RAG.confidence), str(entity.confidence)),
        ]

        # Add aliases
        for alias in entity.aliases:
            triples.append((subject, str(RAG.alias), alias))

        # Add source chunk references
        for chunk_id in entity.source_chunks:
            chunk_uri = f"{CHUNK}{chunk_id}"
            triples.append((subject, str(RAG.mentionedIn), chunk_uri))

        await self.insert_triples(triples)

    async def sync_relation(self, relation: Relation) -> None:
        """Sync a relation to RDF store."""
        head_uri = f"{ENTITY}{relation.head_id}"
        tail_uri = f"{ENTITY}{relation.tail_id}"
        rel_uri = f"{RAG}{relation.relation_type.value}"

        triples = [
            (head_uri, rel_uri, tail_uri),
        ]

        # Add reified properties if we need to attach confidence/evidence
        if relation.evidence_chunk_id:
            evidence_uri = f"{CHUNK}{relation.evidence_chunk_id}"
            # For simplicity, attach evidence to head entity
            triples.append((head_uri, str(RAG.hasEvidence), evidence_uri))

        await self.insert_triples(triples)

    async def sync_entities_batch(self, entities: list[Entity]) -> None:
        """Sync multiple entities in batch."""
        triples = []

        for entity in entities:
            subject = f"{ENTITY}{entity.id}"
            entity_type_uri = f"{RAG}{entity.entity_type.value.capitalize()}"

            triples.extend([
                (subject, str(RDF.type), str(RAG.Entity)),
                (subject, str(RDF.type), entity_type_uri),
                (subject, str(RAG.name), entity.name),
                (subject, str(RAG.normalizedName), entity.normalized_name),
                (subject, str(RAG.entityType), entity.entity_type.value),
            ])

            for chunk_id in entity.source_chunks:
                chunk_uri = f"{CHUNK}{chunk_id}"
                triples.append((subject, str(RAG.mentionedIn), chunk_uri))

        await self.insert_triples(triples)
        logger.info("Synced entities to RDF", count=len(entities))

    async def sync_relations_batch(self, relations: list[Relation]) -> None:
        """Sync multiple relations in batch."""
        triples = []

        for rel in relations:
            head_uri = f"{ENTITY}{rel.head_id}"
            tail_uri = f"{ENTITY}{rel.tail_id}"
            rel_uri = f"{RAG}{rel.relation_type.value}"
            triples.append((head_uri, rel_uri, tail_uri))

        await self.insert_triples(triples)
        logger.info("Synced relations to RDF", count=len(relations))

    # =========================================================================
    # Reasoning Queries
    # =========================================================================

    async def find_paths(
        self,
        start_entity_id: str,
        end_entity_id: str,
        max_length: int = 3,
    ) -> list[list[dict[str, str]]]:
        """Find paths between two entities (up to max_length hops)."""
        start_uri = f"{ENTITY}{start_entity_id}"
        end_uri = f"{ENTITY}{end_entity_id}"

        # Build path query for different lengths
        paths = []

        # Direct connection (length 1)
        sparql = f"""
        SELECT ?rel WHERE {{
            <{start_uri}> ?rel <{end_uri}> .
            FILTER(STRSTARTS(STR(?rel), "{RAG}"))
        }}
        """
        results = await self.query(sparql)
        for r in results:
            paths.append([{"from": start_entity_id, "rel": r["rel"].split("#")[-1], "to": end_entity_id}])

        if max_length >= 2:
            # Length 2 paths
            sparql = f"""
            SELECT ?mid ?rel1 ?rel2 WHERE {{
                <{start_uri}> ?rel1 ?mid .
                ?mid ?rel2 <{end_uri}> .
                FILTER(STRSTARTS(STR(?rel1), "{RAG}"))
                FILTER(STRSTARTS(STR(?rel2), "{RAG}"))
                FILTER(STRSTARTS(STR(?mid), "{ENTITY}"))
            }}
            LIMIT 10
            """
            results = await self.query(sparql)
            for r in results:
                mid_id = r["mid"].split("/")[-1]
                paths.append([
                    {"from": start_entity_id, "rel": r["rel1"].split("#")[-1], "to": mid_id},
                    {"from": mid_id, "rel": r["rel2"].split("#")[-1], "to": end_entity_id},
                ])

        return paths

    async def get_entity_neighborhood(
        self,
        entity_id: str,
        max_hops: int = 1,
    ) -> dict[str, Any]:
        """Get entity and its immediate neighborhood."""
        entity_uri = f"{ENTITY}{entity_id}"

        # Get outgoing relations
        sparql = f"""
        SELECT ?rel ?target ?targetName WHERE {{
            <{entity_uri}> ?rel ?target .
            FILTER(STRSTARTS(STR(?rel), "{RAG}"))
            FILTER(STRSTARTS(STR(?target), "{ENTITY}"))
            OPTIONAL {{ ?target <{RAG}name> ?targetName . }}
        }}
        """
        outgoing = await self.query(sparql)

        # Get incoming relations
        sparql = f"""
        SELECT ?source ?sourceName ?rel WHERE {{
            ?source ?rel <{entity_uri}> .
            FILTER(STRSTARTS(STR(?rel), "{RAG}"))
            FILTER(STRSTARTS(STR(?source), "{ENTITY}"))
            OPTIONAL {{ ?source <{RAG}name> ?sourceName . }}
        }}
        """
        incoming = await self.query(sparql)

        return {
            "entity_id": entity_id,
            "outgoing": [
                {
                    "relation": r["rel"].split("#")[-1],
                    "target_id": r["target"].split("/")[-1],
                    "target_name": r.get("targetName"),
                }
                for r in outgoing
            ],
            "incoming": [
                {
                    "source_id": r["source"].split("/")[-1],
                    "source_name": r.get("sourceName"),
                    "relation": r["rel"].split("#")[-1],
                }
                for r in incoming
            ],
        }

    async def get_entities_by_type(
        self,
        entity_type: str,
        limit: int = 100,
    ) -> list[dict[str, str]]:
        """Get all entities of a specific type."""
        type_uri = f"{RAG}{entity_type.capitalize()}"

        sparql = f"""
        SELECT ?entity ?name WHERE {{
            ?entity a <{type_uri}> .
            ?entity <{RAG}name> ?name .
        }}
        LIMIT {limit}
        """
        results = await self.query(sparql)

        return [
            {
                "id": r["entity"].split("/")[-1],
                "name": r["name"],
            }
            for r in results
        ]

    # =========================================================================
    # Bulk Load
    # =========================================================================

    async def load_ontology(self, ontology_path: str) -> None:
        """Load ontology from Turtle file."""
        with open(ontology_path, "r") as f:
            turtle_content = f.read()

        response = await self.client.post(
            "/store",
            content=turtle_content,
            headers={"Content-Type": "text/turtle"},
        )
        response.raise_for_status()
        logger.info("Loaded ontology", path=ontology_path)

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> bool:
        """Check Oxigraph connectivity."""
        try:
            response = await self.client.get("/")
            return response.status_code == 200
        except Exception:
            return False


# Global client instance
_client: OxigraphClient | None = None


async def get_oxigraph() -> OxigraphClient:
    """Get or create Oxigraph client."""
    global _client
    if _client is None:
        _client = OxigraphClient()
        await _client.connect()
    return _client
