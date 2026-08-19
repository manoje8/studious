"""
Unit tests for the kg_retrieve graph node function.
"""

from unittest.mock import AsyncMock

import pytest

from medici.agents.graph.nodes import kg_retrieve


@pytest.fixture
def base_state():
    """Minimal state dict matching the State TypedDict shape."""
    return {
        "current_query": "Who does Alice report to?",
        "doc_id_filter": None,
        "accepted_chunks": [],
        "kg_entities_found": [],
        "kg_chunks": [],
        "kg_traversal_paths": [],
    }


@pytest.mark.asyncio
async def test_kg_retrieve_disabled(base_state):
    """When kg_retriever is None (KG disabled), returns empty KG state."""
    result = await kg_retrieve(base_state, kg_retriever=None)

    assert result["kg_entities_found"] == []
    assert result["kg_chunks"] == []
    assert result["kg_traversal_paths"] == []
    # Should NOT touch accepted_chunks when disabled
    assert "accepted_chunks" not in result


@pytest.mark.asyncio
async def test_kg_retrieve_with_results(base_state):
    """When KG retriever returns chunks, they are tagged and pre-seeded."""
    mock_retriever = AsyncMock()
    mock_retriever.retrieve_kg_context.return_value = {
        "entities": ["person::alice"],
        "chunks": [
            {"text": "Alice reports to Bob", "doc_id": "doc1", "chunk_index": 0, "score": 1.0},
            {"text": "Bob is VP of Eng", "doc_id": "doc1", "chunk_index": 5, "score": 1.0},
        ],
        "paths": [{"source_entity": "person::alice", "target_entity": "person::bob"}],
    }

    result = await kg_retrieve(base_state, kg_retriever=mock_retriever)

    # Entities found
    assert result["kg_entities_found"] == ["person::alice"]

    # Chunks returned and tagged
    assert len(result["kg_chunks"]) == 2
    for chunk in result["kg_chunks"]:
        assert chunk["retrieval_source"] == "knowledge_graph"

    # Traversal paths preserved
    assert len(result["kg_traversal_paths"]) == 1

    # Chunks pre-seeded into accepted_chunks
    assert len(result["accepted_chunks"]) == 2

    # Verify the retriever was called correctly
    mock_retriever.retrieve_kg_context.assert_called_once_with(
        query="Who does Alice report to?",
        doc_id_filter=None,
    )


@pytest.mark.asyncio
async def test_kg_retrieve_no_results(base_state):
    """When KG retriever returns no chunks, KG state is empty."""
    mock_retriever = AsyncMock()
    mock_retriever.retrieve_kg_context.return_value = {
        "entities": ["person::unknown"],
        "chunks": [],
        "paths": [],
    }

    result = await kg_retrieve(base_state, kg_retriever=mock_retriever)

    # Entities may be found even if no chunks
    assert result["kg_entities_found"] == ["person::unknown"]
    assert result["kg_chunks"] == []
    assert result["kg_traversal_paths"] == []
    # No accepted_chunks key since there's nothing to add
    assert "accepted_chunks" not in result


@pytest.mark.asyncio
async def test_kg_retrieve_empty_result(base_state):
    """When KG retriever returns empty dict, state is empty."""
    mock_retriever = AsyncMock()
    mock_retriever.retrieve_kg_context.return_value = {
        "entities": [],
        "chunks": [],
        "paths": [],
    }

    result = await kg_retrieve(base_state, kg_retriever=mock_retriever)

    assert result["kg_entities_found"] == []
    assert result["kg_chunks"] == []


@pytest.mark.asyncio
async def test_kg_retrieve_appends_to_existing_accepted(base_state):
    """KG chunks are appended to already-existing accepted_chunks."""
    base_state["accepted_chunks"] = [
        {"text": "existing chunk", "doc_id": "doc1", "chunk_index": 99}
    ]

    mock_retriever = AsyncMock()
    mock_retriever.retrieve_kg_context.return_value = {
        "entities": ["concept::x"],
        "chunks": [
            {"text": "new KG chunk", "doc_id": "doc1", "chunk_index": 2, "score": 1.0},
        ],
        "paths": [],
    }

    result = await kg_retrieve(base_state, kg_retriever=mock_retriever)

    # Should have existing + new
    assert len(result["accepted_chunks"]) == 2


@pytest.mark.asyncio
async def test_kg_retrieve_with_doc_id_filter(base_state):
    """doc_id_filter is passed through to the retriever."""
    base_state["doc_id_filter"] = "specific-doc"

    mock_retriever = AsyncMock()
    mock_retriever.retrieve_kg_context.return_value = {
        "entities": [],
        "chunks": [],
        "paths": [],
    }

    await kg_retrieve(base_state, kg_retriever=mock_retriever)

    mock_retriever.retrieve_kg_context.assert_called_once_with(
        query="Who does Alice report to?",
        doc_id_filter="specific-doc",
    )
