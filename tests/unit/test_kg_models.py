"""
Unit tests for Knowledge Graph data models.
"""

import pytest

from medici.knowledge_graph.models import (
    Entity,
    EntityType,
    KGExtractionResult,
    Relationship,
    RelationType,
)


class TestEntityType:
    """Verify EntityType enum completeness."""

    def test_core_types_exist(self):
        assert EntityType.PERSON == "person"
        assert EntityType.ORGANIZATION == "organization"
        assert EntityType.LOCATION == "location"
        assert EntityType.CONCEPT == "concept"
        assert EntityType.EVENT == "event"
        assert EntityType.TECHNOLOGY == "technology"
        assert EntityType.OTHER == "other"

    def test_all_types_are_strings(self):
        for member in EntityType:
            assert isinstance(member.value, str)


class TestRelationType:
    """Verify RelationType enum completeness."""

    def test_core_relations_exist(self):
        assert RelationType.REPORTS_TO == "reports_to"
        assert RelationType.MANAGES == "manages"
        assert RelationType.WORKS_AT == "works_at"
        assert RelationType.CAUSES == "causes"
        assert RelationType.RELATED_TO == "related_to"
        assert RelationType.PART_OF == "part_of"
        assert RelationType.CONTRADICTS == "contradicts"

    def test_all_relations_are_strings(self):
        for member in RelationType:
            assert isinstance(member.value, str)


class TestEntity:
    """Verify Entity dataclass and canonical_key logic."""

    def test_canonical_key_basic(self):
        e = Entity(name="Sarah Chen", entity_type=EntityType.PERSON)
        assert e.canonical_key == "person::sarah chen"

    def test_canonical_key_strips_whitespace(self):
        e = Entity(name="  Elon Musk  ", entity_type=EntityType.PERSON)
        assert e.canonical_key == "person::elon musk"

    def test_canonical_key_lowercases(self):
        e = Entity(name="NASA", entity_type=EntityType.ORGANIZATION)
        assert e.canonical_key == "organization::nasa"

    def test_canonical_key_different_types(self):
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        e2 = Entity(name="Python", entity_type=EntityType.CONCEPT)
        assert e1.canonical_key != e2.canonical_key
        assert e1.canonical_key == "technology::python"
        assert e2.canonical_key == "concept::python"

    def test_default_values(self):
        e = Entity(name="Test", entity_type=EntityType.CONCEPT)
        assert e.aliases == []
        assert e.description == ""
        assert e.source_chunk_ids == []
        assert e.doc_ids == []
        assert e.metadata == {}

    def test_with_all_fields(self):
        e = Entity(
            name="Google",
            entity_type=EntityType.ORGANIZATION,
            aliases=["Alphabet", "GOOGL"],
            description="A technology company",
            source_chunk_ids=["doc1:0", "doc1:3"],
            doc_ids=["doc1"],
            metadata={"founded": 1998},
        )
        assert e.name == "Google"
        assert len(e.aliases) == 2
        assert e.metadata["founded"] == 1998


class TestRelationship:
    """Verify Relationship dataclass defaults."""

    def test_default_values(self):
        r = Relationship(
            source_entity="person::alice",
            target_entity="organization::acme",
            relation_type=RelationType.WORKS_AT,
        )
        assert r.description == ""
        assert r.confidence == 1.0
        assert r.source_chunk_ids == []
        assert r.doc_ids == []
        assert r.metadata == {}

    def test_with_confidence(self):
        r = Relationship(
            source_entity="concept::a",
            target_entity="concept::b",
            relation_type=RelationType.CAUSES,
            confidence=0.75,
        )
        assert r.confidence == 0.75


class TestKGExtractionResult:
    """Verify KGExtractionResult structure."""

    def test_requires_all_fields(self):
        """KGExtractionResult requires entities, relationships, chunk_id, doc_id."""
        with pytest.raises(TypeError):
            KGExtractionResult()  # type: ignore

    def test_basic_construction(self):
        result = KGExtractionResult(
            entities=[],
            relationships=[],
            chunk_id="doc1:0",
            doc_id="doc1",
        )
        assert result.entities == []
        assert result.relationships == []
        assert result.chunk_id == "doc1:0"
        assert result.doc_id == "doc1"

    def test_with_data(self):
        e = Entity(name="Entity1", entity_type=EntityType.CONCEPT)
        r = Relationship(
            source_entity="concept::entity1",
            target_entity="concept::entity2",
            relation_type=RelationType.RELATED_TO,
        )
        result = KGExtractionResult(
            entities=[e],
            relationships=[r],
            chunk_id="doc1:0",
            doc_id="doc1",
        )
        assert len(result.entities) == 1
        assert len(result.relationships) == 1
