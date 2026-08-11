"""
Unit tests for KGExtractor — LLM-based entity/relationship extraction.
"""

import pytest

from src.knowledge_graph.extractor import KGExtractor
from src.knowledge_graph.models import EntityType, RelationType


class MockLLMResponse:
    def __init__(self, parsed_json=None, text=""):
        self.parsed_json = parsed_json
        self.text = text


class MockLLM:
    """Configurable async LLM mock that returns canned responses."""

    def __init__(self, responses=None):
        self._responses = responses or []
        self._call_count = 0

    async def complete(self, prompt, stage_tag=None, json_mode=False):
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
        else:
            resp = MockLLMResponse(parsed_json={"entities": [], "relationships": []})
        self._call_count += 1
        return resp


VALID_EXTRACTION = {
    "entities": [
        {
            "name": "Sarah Chen",
            "type": "person",
            "aliases": ["Dr. Chen"],
            "description": "VP of Engineering",
        },
        {
            "name": "Acme Corp",
            "type": "organization",
            "aliases": [],
            "description": "A tech company",
        },
    ],
    "relationships": [
        {
            "source": "Sarah Chen",
            "target": "Acme Corp",
            "type": "works_at",
            "description": "Sarah works at Acme",
            "confidence": 0.95,
        },
    ],
}


@pytest.mark.asyncio
async def test_extract_from_chunk_valid():
    """Valid LLM response produces correct entities and relationships."""
    llm = MockLLM([MockLLMResponse(parsed_json=VALID_EXTRACTION)])
    extractor = KGExtractor(llm_client=llm)

    result = await extractor.extract_from_chunk(
        chunk_text="Sarah Chen is VP of Engineering at Acme Corp.",
        chunk_id="doc1:0",
        doc_id="doc1",
        source_file="report.pdf",
        section_title="Leadership",
    )

    assert len(result.entities) == 2
    assert result.entities[0].name == "Sarah Chen"
    assert result.entities[0].entity_type == EntityType.PERSON
    assert result.entities[0].aliases == ["Dr. Chen"]
    assert result.entities[1].entity_type == EntityType.ORGANIZATION

    assert len(result.relationships) == 1
    assert result.relationships[0].relation_type == RelationType.WORKS_AT
    assert result.relationships[0].confidence == 0.95
    assert result.chunk_id == "doc1:0"
    assert result.doc_id == "doc1"


@pytest.mark.asyncio
async def test_extract_from_chunk_malformed_json():
    """Malformed/None parsed_json returns empty result gracefully."""
    llm = MockLLM([MockLLMResponse(parsed_json=None)])
    extractor = KGExtractor(llm_client=llm)

    result = await extractor.extract_from_chunk(
        chunk_text="Some text",
        chunk_id="doc1:0",
        doc_id="doc1",
        source_file="test.pdf",
    )

    assert result.entities == []
    assert result.relationships == []
    assert result.chunk_id == "doc1:0"


@pytest.mark.asyncio
async def test_extract_from_chunk_empty_entities():
    """LLM returns empty entities list — result has no entities."""
    llm = MockLLM([MockLLMResponse(parsed_json={"entities": [], "relationships": []})])
    extractor = KGExtractor(llm_client=llm)

    result = await extractor.extract_from_chunk(
        chunk_text="No entities here.",
        chunk_id="doc1:5",
        doc_id="doc1",
        source_file="test.pdf",
    )

    assert result.entities == []
    assert result.relationships == []


@pytest.mark.asyncio
async def test_confidence_filtering():
    """Relationships below min_confidence are filtered out."""
    extraction = {
        "entities": [
            {"name": "A", "type": "concept", "aliases": [], "description": ""},
            {"name": "B", "type": "concept", "aliases": [], "description": ""},
        ],
        "relationships": [
            {
                "source": "A",
                "target": "B",
                "type": "related_to",
                "description": "low confidence",
                "confidence": 0.3,
            },
        ],
    }
    llm = MockLLM([MockLLMResponse(parsed_json=extraction)])
    extractor = KGExtractor(llm_client=llm, min_confidence=0.5)

    result = await extractor.extract_from_chunk(
        chunk_text="A and B",
        chunk_id="doc1:0",
        doc_id="doc1",
        source_file="test.pdf",
    )

    assert len(result.entities) == 2
    assert len(result.relationships) == 0  # filtered out


@pytest.mark.asyncio
async def test_confidence_filtering_keeps_high():
    """Relationships at or above min_confidence are kept."""
    extraction = {
        "entities": [
            {"name": "A", "type": "concept", "aliases": [], "description": ""},
            {"name": "B", "type": "concept", "aliases": [], "description": ""},
        ],
        "relationships": [
            {
                "source": "A",
                "target": "B",
                "type": "related_to",
                "description": "high confidence",
                "confidence": 0.8,
            },
        ],
    }
    llm = MockLLM([MockLLMResponse(parsed_json=extraction)])
    extractor = KGExtractor(llm_client=llm, min_confidence=0.5)

    result = await extractor.extract_from_chunk(
        chunk_text="A and B",
        chunk_id="doc1:0",
        doc_id="doc1",
        source_file="test.pdf",
    )

    assert len(result.relationships) == 1


@pytest.mark.asyncio
async def test_extract_from_chunks_batch():
    """Batch extraction processes multiple Chunk objects."""
    from dataclasses import dataclass

    @dataclass
    class FakeChunk:
        text: str
        chunk_index: int
        source_file: str = "test.pdf"
        section_title: str = ""

    chunks = [
        FakeChunk(text="Chunk one", chunk_index=0),
        FakeChunk(text="Chunk two", chunk_index=1),
    ]

    extraction = {
        "entities": [{"name": "X", "type": "concept", "aliases": [], "description": ""}],
        "relationships": [],
    }
    llm = MockLLM(
        [MockLLMResponse(parsed_json=extraction), MockLLMResponse(parsed_json=extraction)]
    )
    extractor = KGExtractor(llm_client=llm, batch_size=5)

    results = await extractor.extract_from_chunks(chunks, doc_id="doc1")

    assert len(results) == 2
    assert results[0].chunk_id == "doc1:0"
    assert results[1].chunk_id == "doc1:1"
    assert len(results[0].entities) == 1


@pytest.mark.asyncio
async def test_invalid_entity_type_skipped():
    """Invalid entity type string is skipped gracefully."""
    extraction = {
        "entities": [
            {"name": "X", "type": "unknown_type_xyz", "aliases": [], "description": ""},
        ],
        "relationships": [],
    }
    llm = MockLLM([MockLLMResponse(parsed_json=extraction)])
    extractor = KGExtractor(llm_client=llm)

    result = await extractor.extract_from_chunk(
        chunk_text="test", chunk_id="d:0", doc_id="d", source_file="t.pdf"
    )

    # Invalid type should be skipped
    assert len(result.entities) == 0


@pytest.mark.asyncio
async def test_invalid_relation_type_skipped():
    """Invalid relation type string is skipped gracefully."""
    extraction = {
        "entities": [
            {"name": "A", "type": "concept", "aliases": [], "description": ""},
            {"name": "B", "type": "concept", "aliases": [], "description": ""},
        ],
        "relationships": [
            {
                "source": "A",
                "target": "B",
                "type": "some_invalid_relation",
                "description": "",
                "confidence": 1.0,
            },
        ],
    }
    llm = MockLLM([MockLLMResponse(parsed_json=extraction)])
    extractor = KGExtractor(llm_client=llm)

    result = await extractor.extract_from_chunk(
        chunk_text="test", chunk_id="d:0", doc_id="d", source_file="t.pdf"
    )

    assert len(result.entities) == 2
    assert len(result.relationships) == 0  # invalid type skipped


@pytest.mark.asyncio
async def test_llm_exception_returns_empty():
    """If the LLM raises an exception, an empty result is returned."""

    class FailingLLM:
        async def complete(self, *args, **kwargs):
            raise RuntimeError("API timeout")

    extractor = KGExtractor(llm_client=FailingLLM())

    result = await extractor.extract_from_chunk(
        chunk_text="test", chunk_id="d:0", doc_id="d", source_file="t.pdf"
    )

    assert result.entities == []
    assert result.relationships == []
    assert result.chunk_id == "d:0"
