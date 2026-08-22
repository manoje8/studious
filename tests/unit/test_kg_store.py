"""
Unit tests for KGStore — PostgreSQL-backed knowledge graph storage.

Uses AsyncMock to simulate psycopg pool/connection/cursor without a real database.
"""

from unittest.mock import AsyncMock, mock_open, patch

import pytest

from medici.knowledge_graph.models import (
    Entity,
    EntityType,
    KGExtractionResult,
    Relationship,
    RelationType,
)
from medici.knowledge_graph.store import KGStore


class MockCursor:
    """Simulates an async psycopg cursor."""

    def __init__(self, rows=None, description=None):
        self._rows = rows or []
        self.description = description or []
        self.rowcount = 0
        self.execute = AsyncMock()
        self.fetchall = AsyncMock(return_value=self._rows)
        self.fetchone = AsyncMock(return_value=self._rows[0] if self._rows else None)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockConnection:
    """Simulates an async psycopg connection."""

    def __init__(self, cursor):
        self._cursor = cursor
        self.commit = AsyncMock()

    def cursor(self):
        return self._cursor

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockPool:
    """Simulates a psycopg AsyncConnectionPool."""

    def __init__(self, cursor):
        self._conn = MockConnection(cursor)

    def connection(self):
        return self._conn


@pytest.fixture
def mock_cursor():
    return MockCursor()


@pytest.fixture
def mock_pool(mock_cursor):
    return MockPool(mock_cursor)


@pytest.fixture
def store(mock_pool):
    return KGStore(pool=mock_pool)


@pytest.mark.asyncio
async def test_setup_executes_schema(store, mock_cursor):
    """setup() reads and executes schema.sql."""
    with patch("builtins.open", mock_open(read_data="CREATE TABLE test;")):
        with patch("pathlib.Path.read_text", return_value="CREATE TABLE test;"):
            await store.setup()

    mock_cursor.execute.assert_called_once()
    sql_arg = mock_cursor.execute.call_args[0][0]
    assert "CREATE TABLE" in sql_arg


@pytest.mark.asyncio
async def test_upsert_extraction_calls_insert(store, mock_cursor):
    """upsert_extraction() inserts entities and relationships."""
    mock_cursor.fetchone = AsyncMock(return_value=(42,))  # relationship id

    entity = Entity(
        name="Alice",
        entity_type=EntityType.PERSON,
        description="A person",
    )
    rel = Relationship(
        source_entity=entity.canonical_key,
        target_entity="organization::acme",
        relation_type=RelationType.WORKS_AT,
        confidence=0.9,
    )
    result = KGExtractionResult(
        entities=[entity],
        relationships=[rel],
        chunk_id="doc1:0",
        doc_id="doc1",
    )

    await store.upsert_extraction(result)

    # Should have called execute for: entity INSERT, entity_chunks INSERT,
    # relationship INSERT, relationship_chunks INSERT
    assert mock_cursor.execute.call_count >= 4


@pytest.mark.asyncio
async def test_find_entities_returns_dicts(mock_cursor):
    """find_entities() returns list of dicts with column names."""
    mock_cursor._rows = [
        ("person::alice", "Alice", "person", "[]", "A person", 0.95),
    ]
    mock_cursor.fetchall = AsyncMock(return_value=mock_cursor._rows)
    mock_cursor.description = [
        ("canonical_key",),
        ("name",),
        ("entity_type",),
        ("aliases",),
        ("description",),
        ("sim",),
    ]

    pool = MockPool(mock_cursor)
    store = KGStore(pool=pool)

    results = await store.find_entities("Alice", limit=5)

    assert len(results) == 1
    assert results[0]["canonical_key"] == "person::alice"
    assert results[0]["sim"] == 0.95


@pytest.mark.asyncio
async def test_get_neighbors_empty_keys():
    """get_neighbors() with empty keys returns empty list."""
    pool = MockPool(MockCursor())
    store = KGStore(pool=pool)

    result = await store.get_neighbors([], max_hops=2)
    assert result == []


@pytest.mark.asyncio
async def test_get_neighbors_returns_traversal(mock_cursor):
    """get_neighbors() executes recursive CTE and returns results."""
    mock_cursor._rows = [
        ("person::alice", "Alice", "person", 0),
        ("organization::acme", "Acme", "organization", 1),
    ]
    mock_cursor.fetchall = AsyncMock(return_value=mock_cursor._rows)
    mock_cursor.description = [
        ("canonical_key",),
        ("name",),
        ("entity_type",),
        ("depth",),
    ]

    pool = MockPool(mock_cursor)
    store = KGStore(pool=pool)

    result = await store.get_neighbors(["person::alice"], max_hops=2)

    assert len(result) == 2
    assert result[0]["canonical_key"] == "person::alice"
    assert result[1]["depth"] == 1


@pytest.mark.asyncio
async def test_get_entity_chunks_returns_ids(mock_cursor):
    """get_entity_chunks() returns unique chunk IDs."""
    mock_cursor._rows = [("doc1:0",), ("doc1:3",)]
    mock_cursor.fetchall = AsyncMock(return_value=mock_cursor._rows)

    pool = MockPool(mock_cursor)
    store = KGStore(pool=pool)

    result = await store.get_entity_chunks(["person::alice"])

    assert result == ["doc1:0", "doc1:3"]


@pytest.mark.asyncio
async def test_get_entity_chunks_empty_keys():
    """get_entity_chunks() with empty keys returns empty list."""
    pool = MockPool(MockCursor())
    store = KGStore(pool=pool)

    result = await store.get_entity_chunks([])
    assert result == []


@pytest.mark.asyncio
async def test_get_subgraph_combines_data(mock_cursor):
    """get_subgraph() combines neighbors, relationships, and chunk_ids."""
    # We need to handle multiple calls to the cursor
    # First call: get_neighbors (recursive CTE)
    # Second call: get relationships
    # Third call: get_entity_chunks
    neighbor_rows = [("person::alice", "Alice", "person", 0)]
    rel_rows = [("person::alice", "organization::acme", "works_at", "works there", 0.9)]
    chunk_rows = [("doc1:0",)]

    call_count = 0

    async def side_effect_fetchall():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return neighbor_rows
        elif call_count == 2:
            return rel_rows
        elif call_count == 3:
            return chunk_rows
        return []

    mock_cursor.fetchall = side_effect_fetchall

    # Set description for each call
    descriptions = [
        [("canonical_key",), ("name",), ("entity_type",), ("depth",)],
        [
            ("source_entity",),
            ("target_entity",),
            ("relation_type",),
            ("description",),
            ("confidence",),
        ],
    ]
    desc_count = 0

    class DescriptionProperty:
        def __get__(self, obj, objtype=None):
            nonlocal desc_count
            if desc_count < len(descriptions):
                d = descriptions[desc_count]
                desc_count += 1
                return d
            return []

    # Use a simpler approach: just test that get_subgraph calls the right methods
    pool = MockPool(mock_cursor)
    store = KGStore(pool=pool)

    # Patch individual methods for cleaner testing
    store.get_neighbors = AsyncMock(
        return_value=[
            {"canonical_key": "person::alice", "name": "Alice", "entity_type": "person", "depth": 0}
        ]
    )
    store.get_entity_chunks = AsyncMock(return_value=["doc1:0"])

    # Mock the relationship query via the cursor
    mock_cursor.fetchall = AsyncMock(
        return_value=[("person::alice", "organization::acme", "works_at", "works there", 0.9)]
    )
    mock_cursor.description = [
        ("source_entity",),
        ("target_entity",),
        ("relation_type",),
        ("description",),
        ("confidence",),
    ]

    result = await store.get_subgraph(["person::alice"], max_hops=2)

    assert "entities" in result
    assert "relationships" in result
    assert "chunk_ids" in result
    assert len(result["entities"]) == 1
    assert result["chunk_ids"] == ["doc1:0"]


@pytest.mark.asyncio
async def test_delete_by_doc_id(store, mock_cursor):
    """delete_by_doc_id() executes delete queries."""
    mock_cursor.rowcount = 3

    await store.delete_by_doc_id("doc1")

    # Should have called execute for:
    # DELETE kg_entity_chunks, DELETE kg_relationship_chunks,
    # DELETE kg_entities, DELETE kg_relationships
    assert mock_cursor.execute.call_count >= 4
