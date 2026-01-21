-- Perfect RAG - SurrealDB Schema
-- Run this on database initialization

-- =============================================================================
-- Namespace and Database
-- =============================================================================
DEFINE NAMESPACE IF NOT EXISTS rag;
USE NS rag;
DEFINE DATABASE IF NOT EXISTS main;
USE DB main;

-- =============================================================================
-- Document Table
-- =============================================================================
DEFINE TABLE document SCHEMAFULL;

DEFINE FIELD id ON document TYPE string ASSERT $value != NONE;
DEFINE FIELD source ON document TYPE string ASSERT $value != NONE;
DEFINE FIELD format ON document TYPE string ASSERT $value != NONE;
DEFINE FIELD content ON document TYPE string;
DEFINE FIELD metadata ON document TYPE object DEFAULT {};
DEFINE FIELD metadata.title ON document TYPE option<string>;
DEFINE FIELD metadata.author ON document TYPE option<string>;
DEFINE FIELD metadata.created_date ON document TYPE option<datetime>;
DEFINE FIELD metadata.modified_date ON document TYPE option<datetime>;
DEFINE FIELD metadata.language ON document TYPE option<string>;
DEFINE FIELD metadata.page_count ON document TYPE option<int>;
DEFINE FIELD metadata.word_count ON document TYPE option<int>;
DEFINE FIELD metadata.file_size ON document TYPE option<int>;
DEFINE FIELD metadata.mime_type ON document TYPE option<string>;
DEFINE FIELD metadata.tags ON document TYPE array<string> DEFAULT [];
DEFINE FIELD metadata.custom ON document TYPE object DEFAULT {};
DEFINE FIELD acl ON document TYPE array<string> DEFAULT ["*"];
DEFINE FIELD created_at ON document TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON document TYPE datetime DEFAULT time::now();
DEFINE FIELD content_hash ON document TYPE string ASSERT $value != NONE;
DEFINE FIELD chunk_count ON document TYPE int DEFAULT 0;

DEFINE INDEX idx_document_hash ON document FIELDS content_hash UNIQUE;
DEFINE INDEX idx_document_source ON document FIELDS source;
DEFINE INDEX idx_document_format ON document FIELDS format;
DEFINE INDEX idx_document_created ON document FIELDS created_at;

-- =============================================================================
-- Chunk Table
-- =============================================================================
DEFINE TABLE chunk SCHEMAFULL;

DEFINE FIELD id ON chunk TYPE string ASSERT $value != NONE;
DEFINE FIELD doc_id ON chunk TYPE string ASSERT $value != NONE;
DEFINE FIELD content ON chunk TYPE string ASSERT $value != NONE;
DEFINE FIELD offset_start ON chunk TYPE int ASSERT $value >= 0;
DEFINE FIELD offset_end ON chunk TYPE int ASSERT $value >= 0;
DEFINE FIELD token_count ON chunk TYPE int ASSERT $value >= 1;
DEFINE FIELD chunk_index ON chunk TYPE int DEFAULT 0;
DEFINE FIELD metadata ON chunk TYPE object DEFAULT {};
DEFINE FIELD acl ON chunk TYPE array<string> DEFAULT ["*"];

DEFINE INDEX idx_chunk_doc ON chunk FIELDS doc_id;
DEFINE INDEX idx_chunk_index ON chunk FIELDS doc_id, chunk_index;

-- =============================================================================
-- Entity Table
-- =============================================================================
DEFINE TABLE entity SCHEMAFULL;

DEFINE FIELD id ON entity TYPE string ASSERT $value != NONE;
DEFINE FIELD name ON entity TYPE string ASSERT $value != NONE;
DEFINE FIELD normalized_name ON entity TYPE string ASSERT $value != NONE;
DEFINE FIELD entity_type ON entity TYPE string ASSERT $value != NONE;
DEFINE FIELD aliases ON entity TYPE array<string> DEFAULT [];
DEFINE FIELD confidence ON entity TYPE float DEFAULT 1.0 ASSERT $value >= 0.0 AND $value <= 1.0;
DEFINE FIELD source_chunks ON entity TYPE array<string> DEFAULT [];
DEFINE FIELD mention_count ON entity TYPE int DEFAULT 1;
DEFINE FIELD metadata ON entity TYPE object DEFAULT {};

DEFINE INDEX idx_entity_normalized ON entity FIELDS normalized_name;
DEFINE INDEX idx_entity_type ON entity FIELDS entity_type;
DEFINE INDEX idx_entity_name_type ON entity FIELDS normalized_name, entity_type UNIQUE;

-- =============================================================================
-- Graph Relations
-- =============================================================================

-- Entity mentioned in Chunk
DEFINE TABLE mentioned_in SCHEMAFULL TYPE RELATION IN entity OUT chunk;
DEFINE FIELD confidence ON mentioned_in TYPE float DEFAULT 1.0;
DEFINE FIELD span_start ON mentioned_in TYPE option<int>;
DEFINE FIELD span_end ON mentioned_in TYPE option<int>;

-- Entity related to Entity
DEFINE TABLE related_to SCHEMAFULL TYPE RELATION IN entity OUT entity;
DEFINE FIELD relation_type ON related_to TYPE string ASSERT $value != NONE;
DEFINE FIELD confidence ON related_to TYPE float DEFAULT 1.0 ASSERT $value >= 0.0 AND $value <= 1.0;
DEFINE FIELD evidence_chunk ON related_to TYPE option<string>;
DEFINE FIELD span_start ON related_to TYPE option<int>;
DEFINE FIELD span_end ON related_to TYPE option<int>;
DEFINE FIELD metadata ON related_to TYPE object DEFAULT {};

DEFINE INDEX idx_related_type ON related_to FIELDS relation_type;
DEFINE INDEX idx_related_confidence ON related_to FIELDS confidence;

-- Same entity (for deduplication/linking)
DEFINE TABLE same_as SCHEMAFULL TYPE RELATION IN entity OUT entity;
DEFINE FIELD confidence ON same_as TYPE float DEFAULT 1.0;

-- Document hierarchy (for multi-part documents)
DEFINE TABLE part_of SCHEMAFULL TYPE RELATION IN document OUT document;
DEFINE FIELD order ON part_of TYPE int DEFAULT 0;

-- =============================================================================
-- User & ACL Tables
-- =============================================================================
DEFINE TABLE user SCHEMAFULL;

DEFINE FIELD id ON user TYPE string ASSERT $value != NONE;
DEFINE FIELD username ON user TYPE string ASSERT $value != NONE;
DEFINE FIELD email ON user TYPE option<string>;
DEFINE FIELD password_hash ON user TYPE string;
DEFINE FIELD roles ON user TYPE array<string> DEFAULT ["user"];
DEFINE FIELD api_key ON user TYPE option<string>;
DEFINE FIELD created_at ON user TYPE datetime DEFAULT time::now();
DEFINE FIELD last_login ON user TYPE option<datetime>;
DEFINE FIELD is_active ON user TYPE bool DEFAULT true;
DEFINE FIELD metadata ON user TYPE object DEFAULT {};

DEFINE INDEX idx_user_username ON user FIELDS username UNIQUE;
DEFINE INDEX idx_user_email ON user FIELDS email UNIQUE;
DEFINE INDEX idx_user_api_key ON user FIELDS api_key UNIQUE;

-- =============================================================================
-- Query Log Table (for learning)
-- =============================================================================
DEFINE TABLE query_log SCHEMAFULL;

DEFINE FIELD id ON query_log TYPE string ASSERT $value != NONE;
DEFINE FIELD query ON query_log TYPE string ASSERT $value != NONE;
DEFINE FIELD user_id ON query_log TYPE option<string>;
DEFINE FIELD query_embedding_hash ON query_log TYPE option<string>;
DEFINE FIELD retrieved_chunk_ids ON query_log TYPE array<string> DEFAULT [];
DEFINE FIELD retrieved_entity_ids ON query_log TYPE array<string> DEFAULT [];
DEFINE FIELD response ON query_log TYPE string;
DEFINE FIELD citations ON query_log TYPE array<object> DEFAULT [];
DEFINE FIELD model ON query_log TYPE string;
DEFINE FIELD tokens_used ON query_log TYPE int DEFAULT 0;
DEFINE FIELD latency_ms ON query_log TYPE float DEFAULT 0.0;
DEFINE FIELD timestamp ON query_log TYPE datetime DEFAULT time::now();
DEFINE FIELD feedback ON query_log TYPE option<string>;
DEFINE FIELD feedback_text ON query_log TYPE option<string>;
DEFINE FIELD dense_weight ON query_log TYPE float DEFAULT 0.5;
DEFINE FIELD sparse_weight ON query_log TYPE float DEFAULT 0.3;
DEFINE FIELD graph_weight ON query_log TYPE float DEFAULT 0.2;

DEFINE INDEX idx_query_log_user ON query_log FIELDS user_id;
DEFINE INDEX idx_query_log_timestamp ON query_log FIELDS timestamp;
DEFINE INDEX idx_query_log_embedding ON query_log FIELDS query_embedding_hash;

-- =============================================================================
-- Feedback Table
-- =============================================================================
DEFINE TABLE feedback SCHEMAFULL;

DEFINE FIELD id ON feedback TYPE string ASSERT $value != NONE;
DEFINE FIELD query_id ON feedback TYPE string ASSERT $value != NONE;
DEFINE FIELD feedback_type ON feedback TYPE string ASSERT $value != NONE;
DEFINE FIELD text ON feedback TYPE option<string>;
DEFINE FIELD corrected_answer ON feedback TYPE option<string>;
DEFINE FIELD cited_chunks ON feedback TYPE array<string> DEFAULT [];
DEFINE FIELD timestamp ON feedback TYPE datetime DEFAULT time::now();

DEFINE INDEX idx_feedback_query ON feedback FIELDS query_id;
DEFINE INDEX idx_feedback_type ON feedback FIELDS feedback_type;

-- =============================================================================
-- Cache Table (for CAG)
-- =============================================================================
DEFINE TABLE cache SCHEMAFULL;

DEFINE FIELD key ON cache TYPE string ASSERT $value != NONE;
DEFINE FIELD chunk_ids ON cache TYPE array<string> DEFAULT [];
DEFINE FIELD entity_ids ON cache TYPE array<string> DEFAULT [];
DEFINE FIELD subgraph ON cache TYPE object DEFAULT {};
DEFINE FIELD context_tokens ON cache TYPE int DEFAULT 0;
DEFINE FIELD access_count ON cache TYPE int DEFAULT 0;
DEFINE FIELD last_accessed ON cache TYPE datetime DEFAULT time::now();
DEFINE FIELD created_at ON cache TYPE datetime DEFAULT time::now();
DEFINE FIELD ttl ON cache TYPE datetime;

DEFINE INDEX idx_cache_key ON cache FIELDS key UNIQUE;
DEFINE INDEX idx_cache_ttl ON cache FIELDS ttl;
DEFINE INDEX idx_cache_access ON cache FIELDS access_count DESC;

-- =============================================================================
-- Usage Statistics Table
-- =============================================================================
DEFINE TABLE usage_stats SCHEMAFULL;

DEFINE FIELD id ON usage_stats TYPE string ASSERT $value != NONE;
DEFINE FIELD provider ON usage_stats TYPE string ASSERT $value != NONE;
DEFINE FIELD model ON usage_stats TYPE string ASSERT $value != NONE;
DEFINE FIELD date ON usage_stats TYPE datetime ASSERT $value != NONE;
DEFINE FIELD requests ON usage_stats TYPE int DEFAULT 0;
DEFINE FIELD prompt_tokens ON usage_stats TYPE int DEFAULT 0;
DEFINE FIELD completion_tokens ON usage_stats TYPE int DEFAULT 0;
DEFINE FIELD total_tokens ON usage_stats TYPE int DEFAULT 0;
DEFINE FIELD cost_usd ON usage_stats TYPE float DEFAULT 0.0;
DEFINE FIELD avg_latency_ms ON usage_stats TYPE float DEFAULT 0.0;
DEFINE FIELD errors ON usage_stats TYPE int DEFAULT 0;

DEFINE INDEX idx_usage_provider ON usage_stats FIELDS provider, date;
DEFINE INDEX idx_usage_date ON usage_stats FIELDS date;

-- =============================================================================
-- Functions for common operations
-- =============================================================================

-- Get entity with all its relations
DEFINE FUNCTION fn::get_entity_with_relations($entity_id: string) {
    LET $entity = SELECT * FROM entity WHERE id = $entity_id;
    LET $outgoing = SELECT
        out.id AS target_id,
        out.name AS target_name,
        relation_type,
        confidence,
        evidence_chunk
    FROM related_to WHERE in.id = $entity_id;
    LET $incoming = SELECT
        in.id AS source_id,
        in.name AS source_name,
        relation_type,
        confidence,
        evidence_chunk
    FROM related_to WHERE out.id = $entity_id;
    LET $mentions = SELECT
        out.id AS chunk_id,
        confidence,
        span_start,
        span_end
    FROM mentioned_in WHERE in.id = $entity_id;

    RETURN {
        entity: $entity[0],
        outgoing_relations: $outgoing,
        incoming_relations: $incoming,
        mentions: $mentions
    };
};

-- Expand subgraph from seed entities
DEFINE FUNCTION fn::expand_subgraph($seeds: array<string>, $max_hops: int, $min_confidence: float) {
    LET $entities = SELECT * FROM entity WHERE id IN $seeds;
    LET $relations = SELECT * FROM related_to
        WHERE (in.id IN $seeds OR out.id IN $seeds)
        AND confidence >= $min_confidence;
    LET $evidence = SELECT * FROM chunk
        WHERE id IN (SELECT evidence_chunk FROM $relations WHERE evidence_chunk != NONE);

    RETURN {
        entities: $entities,
        relations: $relations,
        evidence_chunks: $evidence
    };
};

-- Clean expired cache entries
DEFINE FUNCTION fn::clean_expired_cache() {
    DELETE FROM cache WHERE ttl < time::now();
};
