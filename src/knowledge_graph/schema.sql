CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS kg_entities (
    canonical_key VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    aliases JSONB DEFAULT '[]'::jsonb,
    description TEXT,
    doc_ids JSONB DEFAULT '[]'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS kg_relationships (
    id SERIAL PRIMARY KEY,
    source_entity VARCHAR(255) REFERENCES kg_entities(canonical_key) ON DELETE CASCADE,
    target_entity VARCHAR(255) REFERENCES kg_entities(canonical_key) ON DELETE CASCADE,
    relation_type VARCHAR(50) NOT NULL,
    description TEXT,
    confidence FLOAT DEFAULT 1.0,
    doc_ids JSONB DEFAULT '[]'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (source_entity, target_entity, relation_type)
);

CREATE TABLE IF NOT EXISTS kg_entity_chunks (
    entity_key VARCHAR(255) REFERENCES kg_entities(canonical_key) ON DELETE CASCADE,
    chunk_id VARCHAR(255) NOT NULL,
    doc_id VARCHAR(255) NOT NULL,
    PRIMARY KEY (entity_key, chunk_id)
);

CREATE TABLE IF NOT EXISTS kg_relationship_chunks (
    relationship_id INTEGER REFERENCES kg_relationships(id) ON DELETE CASCADE,
    chunk_id VARCHAR(255) NOT NULL,
    doc_id VARCHAR(255) NOT NULL,
    PRIMARY KEY (relationship_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_kg_entities_name_trgm ON kg_entities USING gin (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_kg_entities_type ON kg_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_kg_entities_doc_ids ON kg_entities USING gin (doc_ids);
CREATE INDEX IF NOT EXISTS idx_kg_relationships_source ON kg_relationships(source_entity);
CREATE INDEX IF NOT EXISTS idx_kg_relationships_target ON kg_relationships(target_entity);
CREATE INDEX IF NOT EXISTS idx_kg_relationships_doc_ids ON kg_relationships USING gin (doc_ids);
CREATE INDEX IF NOT EXISTS idx_kg_entity_chunks_doc_id ON kg_entity_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_kg_relationship_chunks_doc_id ON kg_relationship_chunks(doc_id);
