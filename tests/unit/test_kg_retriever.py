"""
Unit tests for KGRetriever — query-time KG-augmented retrieval.
"""

from unittest.mock import AsyncMock

import pytest

from medici.knowledge_graph.retriever import KGRetriever


class MockLLMResponse:
    def __init__(self, parsed_json=None, text=""):
        self.parsed_json = parsed_json
        self.text = text


class MockLLM:
    def __init__(self, responses=None):
        self._responses = responses or []
        self._call_count = 0

    async def complete(self, prompt, stage_tag=None, json_mode=False):
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
        else:
            resp = MockLLMResponse(parsed_json={"entities": []})
        self._call_count += 1
        return resp


@pytest.fixture
def mock_kg_store():
    store = AsyncMock()
    store.find_entities = AsyncMock(return_value=[])
    store.get_subgraph = AsyncMock(
        return_value={"entities": [], "relationships": [], "chunk_ids": []}
    )
    store.get_entity_chunks = AsyncMock(return_value=[])
    return store


@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    storage.get_chunks_by_ids = AsyncMock(return_value=[])
    return storage


@pytest.mark.asyncio
async def test_extract_query_entities_valid():
    """extract_query_entities returns entity names from LLM response."""
    llm = MockLLM([MockLLMResponse(parsed_json={"entities": ["Alice", "Acme Corp"]})])
    retriever = KGRetriever(
        llm_client=llm,
        kg_store=AsyncMock(),
        storage_service=AsyncMock(),
    )

    entities = await retriever.extract_query_entities("Who does Alice work for at Acme Corp?")

    assert entities == ["Alice", "Acme Corp"]


@pytest.mark.asyncio
async def test_extract_query_entities_empty():
    """extract_query_entities returns empty list when LLM finds no entities."""
    llm = MockLLM([MockLLMResponse(parsed_json={"entities": []})])
    retriever = KGRetriever(
        llm_client=llm,
        kg_store=AsyncMock(),
        storage_service=AsyncMock(),
    )

    entities = await retriever.extract_query_entities("Hello world")

    assert entities == []


@pytest.mark.asyncio
async def test_extract_query_entities_llm_failure():
    """extract_query_entities returns empty list on LLM failure."""

    class FailLLM:
        async def complete(self, *args, **kwargs):
            raise RuntimeError("Timeout")

    retriever = KGRetriever(
        llm_client=FailLLM(),
        kg_store=AsyncMock(),
        storage_service=AsyncMock(),
    )

    entities = await retriever.extract_query_entities("test")
    assert entities == []


@pytest.mark.asyncio
async def test_retrieve_kg_context_no_entities(mock_kg_store, mock_storage):
    """retrieve_kg_context returns empty when no query entities found."""
    llm = MockLLM([MockLLMResponse(parsed_json={"entities": []})])
    retriever = KGRetriever(
        llm_client=llm,
        kg_store=mock_kg_store,
        storage_service=mock_storage,
    )

    result = await retriever.retrieve_kg_context("What is the meaning of life?")

    assert result["entities"] == []
    assert result["chunks"] == []
    assert result["paths"] == []


@pytest.mark.asyncio
async def test_retrieve_kg_context_no_kg_matches(mock_kg_store, mock_storage):
    """retrieve_kg_context returns empty when entities don't match KG."""
    llm = MockLLM([MockLLMResponse(parsed_json={"entities": ["Unknown Entity"]})])
    mock_kg_store.find_entities = AsyncMock(return_value=[])  # No matches

    retriever = KGRetriever(
        llm_client=llm,
        kg_store=mock_kg_store,
        storage_service=mock_storage,
    )

    result = await retriever.retrieve_kg_context("Tell me about Unknown Entity")

    assert result["entities"] == []
    assert result["chunks"] == []


@pytest.mark.asyncio
async def test_retrieve_kg_context_full_pipeline(mock_kg_store, mock_storage):
    """retrieve_kg_context runs the full pipeline: extract → match → traverse → fetch."""
    llm = MockLLM([MockLLMResponse(parsed_json={"entities": ["Alice"]})])

    # KG store returns a match
    mock_kg_store.find_entities = AsyncMock(
        return_value=[{"canonical_key": "person::alice", "name": "Alice", "sim": 0.95}]
    )
    mock_kg_store.get_subgraph = AsyncMock(
        return_value={
            "entities": [{"canonical_key": "person::alice"}],
            "relationships": [{"source_entity": "person::alice", "target_entity": "org::acme"}],
            "chunk_ids": ["doc1:0", "doc1:3"],
        }
    )

    # Storage returns chunk payloads
    mock_storage.get_chunks_by_ids = AsyncMock(
        return_value=[
            {"text": "Alice joined Acme in 2020", "doc_id": "doc1", "chunk_index": 0, "score": 1.0},
            {"text": "Alice manages the team", "doc_id": "doc1", "chunk_index": 3, "score": 1.0},
        ]
    )

    retriever = KGRetriever(
        llm_client=llm,
        kg_store=mock_kg_store,
        storage_service=mock_storage,
        max_kg_chunks=5,
    )

    result = await retriever.retrieve_kg_context("What does Alice do?")

    assert len(result["entities"]) == 1
    assert result["entities"][0] == "person::alice"
    assert len(result["chunks"]) == 2


@pytest.mark.asyncio
async def test_retrieve_kg_context_doc_id_filter(mock_kg_store, mock_storage):
    """retrieve_kg_context filters chunks by doc_id when filter is set."""
    llm = MockLLM([MockLLMResponse(parsed_json={"entities": ["Alice"]})])

    mock_kg_store.find_entities = AsyncMock(return_value=[{"canonical_key": "person::alice"}])
    mock_kg_store.get_subgraph = AsyncMock(
        return_value={
            "entities": [{"canonical_key": "person::alice"}],
            "relationships": [],
            "chunk_ids": ["doc1:0", "doc2:1"],
        }
    )

    mock_storage.get_chunks_by_ids = AsyncMock(
        return_value=[
            {"text": "chunk from doc1", "doc_id": "doc1", "chunk_index": 0, "score": 1.0},
            {"text": "chunk from doc2", "doc_id": "doc2", "chunk_index": 1, "score": 1.0},
        ]
    )

    retriever = KGRetriever(
        llm_client=llm,
        kg_store=mock_kg_store,
        storage_service=mock_storage,
    )

    result = await retriever.retrieve_kg_context("Alice?", doc_id_filter="doc1")

    # Only doc1 chunks should remain
    assert len(result["chunks"]) == 1
    assert result["chunks"][0]["doc_id"] == "doc1"


@pytest.mark.asyncio
async def test_retrieve_kg_context_max_chunks_limit(mock_kg_store, mock_storage):
    """retrieve_kg_context limits chunks to max_kg_chunks."""
    llm = MockLLM([MockLLMResponse(parsed_json={"entities": ["X"]})])

    mock_kg_store.find_entities = AsyncMock(return_value=[{"canonical_key": "concept::x"}])
    mock_kg_store.get_subgraph = AsyncMock(
        return_value={
            "entities": [{"canonical_key": "concept::x"}],
            "relationships": [],
            "chunk_ids": [f"doc1:{i}" for i in range(10)],
        }
    )

    mock_storage.get_chunks_by_ids = AsyncMock(
        return_value=[
            {"text": f"chunk {i}", "doc_id": "doc1", "chunk_index": i, "score": 1.0}
            for i in range(10)
        ]
    )

    retriever = KGRetriever(
        llm_client=llm,
        kg_store=mock_kg_store,
        storage_service=mock_storage,
        max_kg_chunks=3,
    )

    result = await retriever.retrieve_kg_context("Tell me about X")

    assert len(result["chunks"]) == 3
