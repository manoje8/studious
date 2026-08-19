import json
from pathlib import Path

import logfire
from psycopg_pool import AsyncConnectionPool

from medici.knowledge_graph.models import KGExtractionResult


class KGStore:
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def setup(self) -> None:
        """Sets up the database tables by executing schema.sql."""
        schema_path = Path(__file__).parent / "schema.sql"
        try:
            schema_sql = schema_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logfire.error("schema.sql not found at {path}", path=str(schema_path))
            raise

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(schema_sql)
            await conn.commit()
            logfire.info("KGStore setup completed successfully.")

    async def upsert_extraction(self, result: KGExtractionResult) -> None:
        """Upserts an extraction result into the graph, handling conflicts."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                # Upsert Entities
                for entity in result.entities:
                    await cur.execute(
                        """
                        INSERT INTO kg_entities
                            (canonical_key, name, entity_type, aliases, description, doc_ids, metadata)
                        VALUES (%s, %s, %s, %s::jsonb, %s, %s::jsonb, %s::jsonb)
                        ON CONFLICT (canonical_key) DO UPDATE SET
                            aliases = kg_entities.aliases || EXCLUDED.aliases,
                            doc_ids = kg_entities.doc_ids || EXCLUDED.doc_ids,
                            description = COALESCE(NULLIF(kg_entities.description, ''), EXCLUDED.description),
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (
                            entity.canonical_key,
                            entity.name,
                            entity.entity_type.value,
                            json.dumps(entity.aliases),
                            entity.description,
                            json.dumps([result.doc_id]),
                            json.dumps(entity.metadata),
                        ),
                    )

                    # Insert Entity Chunks mapping
                    await cur.execute(
                        """
                        INSERT INTO kg_entity_chunks (entity_key, chunk_id, doc_id)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (entity_key, chunk_id) DO NOTHING
                        """,
                        (entity.canonical_key, result.chunk_id, result.doc_id),
                    )

                # Upsert Relationships
                for rel in result.relationships:
                    await cur.execute(
                        """
                        INSERT INTO kg_relationships
                            (source_entity, target_entity, relation_type, description, confidence, doc_ids, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                        ON CONFLICT (source_entity, target_entity, relation_type) DO UPDATE SET
                            doc_ids = kg_relationships.doc_ids || EXCLUDED.doc_ids,
                            description = COALESCE(NULLIF(kg_relationships.description, ''), EXCLUDED.description),
                            confidence = GREATEST(kg_relationships.confidence, EXCLUDED.confidence),
                            updated_at = CURRENT_TIMESTAMP
                        RETURNING id
                        """,
                        (
                            rel.source_entity,
                            rel.target_entity,
                            rel.relation_type.value,
                            rel.description,
                            rel.confidence,
                            json.dumps([result.doc_id]),
                            json.dumps(rel.metadata),
                        ),
                    )
                    rel_row = await cur.fetchone()
                    if rel_row:
                        rel_id = rel_row[0]
                        # Insert Relationship Chunks mapping
                        await cur.execute(
                            """
                            INSERT INTO kg_relationship_chunks (relationship_id, chunk_id, doc_id)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (relationship_id, chunk_id) DO NOTHING
                            """,
                            (rel_id, result.chunk_id, result.doc_id),
                        )

            await conn.commit()
            logfire.debug("Upserted KG extraction for chunk {chunk_id}", chunk_id=result.chunk_id)

    async def find_entities(self, query: str, limit: int = 10) -> list[dict]:
        """Finds entities by name similarity using pg_trgm."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT canonical_key, name, entity_type, aliases, description,
                           similarity(name, %s) as sim
                    FROM kg_entities
                    WHERE name %% %s OR %s = ANY(ARRAY(SELECT jsonb_array_elements_text(aliases)))
                    ORDER BY sim DESC
                    LIMIT %s
                    """,
                    (query, query, query, limit),
                )
                rows = await cur.fetchall()
                cols = [desc[0] for desc in cur.description]
                return [dict(zip(cols, row, strict=False)) for row in rows]

    async def get_neighbors(self, entity_keys: list[str], max_hops: int = 2) -> list[dict]:
        """Gets neighbor entities via recursive CTE for multi-hop traversal in both directions."""
        if not entity_keys:
            return []

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    WITH RECURSIVE traverse AS (
                        SELECT
                            canonical_key,
                            0 AS depth,
                            ARRAY[canonical_key]::varchar[] AS path
                        FROM kg_entities
                        WHERE canonical_key = ANY(%s)

                        UNION

                        SELECT
                            e.canonical_key,
                            t.depth + 1,
                            t.path || e.canonical_key
                        FROM traverse t
                        JOIN kg_relationships r ON (r.source_entity = t.canonical_key OR r.target_entity = t.canonical_key)
                        JOIN kg_entities e ON (e.canonical_key = CASE WHEN r.source_entity = t.canonical_key THEN r.target_entity ELSE r.source_entity END)
                        WHERE t.depth < %s
                          AND NOT e.canonical_key = ANY(t.path)
                    )
                    SELECT
                        t.canonical_key,
                        e.name,
                        e.entity_type,
                        MIN(t.depth) as depth
                    FROM traverse t
                    JOIN kg_entities e ON e.canonical_key = t.canonical_key
                    GROUP BY t.canonical_key, e.name, e.entity_type
                    ORDER BY depth
                    """,
                    (entity_keys, max_hops),
                )
                rows = await cur.fetchall()
                cols = [desc[0] for desc in cur.description]
                return [dict(zip(cols, row, strict=False)) for row in rows]

    async def get_entity_chunks(self, entity_keys: list[str]) -> list[str]:
        """Returns unique chunk_ids for given entity keys."""
        if not entity_keys:
            return []

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT DISTINCT chunk_id
                    FROM kg_entity_chunk
                    WHERE entity_key = ANY(%s)
                    """,
                    (entity_keys,),
                )
                rows = await cur.fetchall()
                return [row[0] for row in rows]

    async def get_subgraph(self, entity_keys: list[str], max_hops: int = 2) -> dict:
        """Combines get_neighbors, fetches relationships, and chunk_ids."""
        neighbors = await self.get_neighbors(entity_keys, max_hops)
        if not neighbors:
            return {"entities": [], "relationships": [], "chunk_ids": []}

        all_keys = [n["canonical_key"] for n in neighbors]

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT source_entity, target_entity, relation_type, description, confidence
                    FROM kg_relationships
                    WHERE source_entity = ANY(%s) AND target_entity = ANY(%s)
                    """,
                    (all_keys, all_keys),
                )
                rows = await cur.fetchall()
                cols = [desc[0] for desc in cur.description]
                relationships = [dict(zip(cols, row, strict=False)) for row in rows]

        chunk_ids = await self.get_entity_chunks(all_keys)

        return {"entities": neighbors, "relationships": relationships, "chunk_ids": chunk_ids}

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """Deletes all entities, relationships, and chunk mappings for a given document."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM kg_entity_chunks WHERE doc_id = %s", (doc_id,))
                deleted_chunks = cur.rowcount

                await cur.execute("DELETE FROM kg_relationship_chunks WHERE doc_id = %s", (doc_id,))

                # Removing doc_id from entities and relationships doc_ids array might be complex,
                # but if they only belong to this doc, they should be deleted.
                # For simplicity, we delete entities that only appear in this doc_id
                await cur.execute(
                    """
                    DELETE FROM kg_entities
                    WHERE doc_ids = %s::jsonb
                    """,
                    (f'["{doc_id}"]',),
                )
                deleted_entities = cur.rowcount

                await cur.execute(
                    """
                    DELETE FROM kg_relationships
                    WHERE doc_ids = %s::jsonb
                    """,
                    (f'["{doc_id}"]',),
                )
            await conn.commit()

        return deleted_chunks + deleted_entities
