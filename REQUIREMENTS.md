INGESTA OFFLINE  (una sola vez o incremental)
───────────────────────────────────────────────

1. Documentos brutos
   ↓
2. Fragmentación (chunking)
   ↓
3. Por cada chunk se guarda:
   • chunk_id
   • doc_id
   • offsets (start, end)
   • timestamps
   • permisos / ACL
   • etc.

4. Procesamiento de cada chunk
   ├── Tokenización + Embeddings densos  →  bge-large, E5, etc.
   └── Tokens → BM25 / inverted index

5. Almacenamiento principal
   ├── Índice vectorial de chunks     → Faiss / HNSW / Weaviate / Qdrant / ...
   └── Tabla de chunks (metadatos)    → Parquet

6. Extracción estructurada
   ├── NER + RE (Named Entity + Relation Extraction)
   ├── Normalización (aliases, entity linking)
   └── Por cada relación se guarda:
       (head, relation, tail, confidence, chunk_id, doc_id, span)

7. Output intermedio → formato tabular / Parquet

8. Construcción del grafo  (idealmente doble representación)

   8.1 Property Graph  (operacional / rápido para GraphRAG)
       • Motores: Neo4j, Neptune, JanusGraph, TigerGraph, ...
       • Nodos principales: Entity, Chunk, Document
       • Aristas típicas:
         - MENTIONED_IN
         - EVIDENCE_FOR
         - RELATED_TO
         - SAME_AS / ALIAS_OF
         - etc.

   8.2 RDF / Triple store  (interoperabilidad, ontologías, SPARQL)
       • Convertir Parquet → Turtle / N-Triples
       • Cargar en triple-store (GraphDB, Blazegraph, RDFox, Jena, ...)

9. Embeddings estructurales
   ├── Embeddings de entidades
   ├── Embeddings de comunidades / subgrafos
   │   (node2vec, GraphSAGE, text-aggregated-by-entity, etc.)
   └── Índice vectorial de nodos/entidades/comunidades

10. Opcional: Community detection
    • Louvain, Leiden, etc.
    → “paquetes” de contexto semántico

11. Mantenimiento
    • (Re)construir / actualizar índices cuando cambian datos
    • BM25, vectorial-chunks, vectorial-nodos, estadísticas de grafo

12. Cache-Augmented Generation (CAG) – prewarm
    • chunks más consultados
    • “mini-subgrafos calientes” (entidades + vecinos + evidencias)
    • Tamaño objetivo: ~150–250k tokens (ajustar según modelo)

───────────────────────────────────────────────
FLUJO DE CONSULTA ONLINE  (tiempo real)
───────────────────────────────────────────────

Usuario → q

1. Reescritura / expansión de query (LLM)
   • Multi-query, self-query
   • Extracción de constraints (fechas, tipos, jurisdicción, etc.)

2. Embedding de la query reescrita

3. Context Awareness Gate  (embedding + clasificador ligero)
   ├─ Hit  → CAG (contexto precargado + mini-subgrafo) → LLM responde
   └─ Miss → continuar con recuperación completa

4. Recuperación híbrida (texto)
   • BM25 + Dense vectors (chunks)
   • + filtros metadatos (fechas, permisos, doc_type, ...)
   → top_k_chunks

5. Arranque del grafo (GraphRAG “real”)
   Semillas:
   • Entidades detectadas en la query
   • Entidades mencionadas en top_k_chunks

   Expansión del subgrafo:
   • k-hops
   • Tipos de relación permitidos
   • Umbral de confianza
   • Filtros de permisos

   Opcional: “Graph vector search”
   → Recuperar nodos similares (índice vectorial de nodos)

6. Consulta al grafo
   • Property Graph → Cypher / Gremlin
   • RDF          → SPARQL
   → Resultado: subgraph = {entities, relations, evidence chunks}

7. Reranking (LLM o cross-encoder)
   • Rerank chunks (texto)
   • Rerank triples / rutas / paths del subgrafo
   → Reducir a contexto compacto:
     top_chunks + top_facts + top_paths

8. Generación final con citas
   LLM responde usando:
   • Pasajes citables (chunks)
   • Hechos del grafo (grounded a evidencias)
   Citas: doc_id / chunk_id / span / página / párrafo

9. Post-feedback y aprendizaje continuo (RA-GAS style)
   Actualiza:
   • Cache CAG (texto + subgrafos)
   • Umbral del Context Awareness Gate
   • Pesos del retrieval híbrido (BM25 vs vector vs grafo)
   • Prior / reglas de expansión del grafo
     (qué relaciones / hops / tipos funcionan mejor)