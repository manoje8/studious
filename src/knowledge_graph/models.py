from dataclasses import dataclass, field
from enum import Enum


class EntityType(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    DOCUMENT = "document"
    SECTION = "section"
    DATE = "date"
    METRIC = "metric"
    TECHNOLOGY = "technology"
    POLICY = "policy"
    PRODUCT = "product"
    OTHER = "other"


class RelationType(str, Enum):
    REPORTS_TO = "reports_to"
    MANAGES = "manages"
    WORKS_AT = "works_at"
    MEMBER_OF = "member_of"
    CAUSES = "causes"
    DEPENDS_ON = "depends_on"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    IMPLEMENTS = "implements"
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    OCCURS_DURING = "occurs_during"
    MENTIONS = "mentions"
    DEFINED_IN = "defined_in"
    REFERENCES = "references"
    PART_OF = "part_of"
    RELATED_TO = "related_to"


@dataclass
class Entity:
    name: str
    entity_type: EntityType
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    source_chunk_ids: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def canonical_key(self) -> str:
        return f"{self.entity_type.value}::{self.name.lower().strip()}"


@dataclass
class Relationship:
    source_entity: str  # canonical_key
    target_entity: str  # canonical_key
    relation_type: RelationType
    description: str = ""
    confidence: float = 1.0
    source_chunk_ids: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class KGExtractionResult:
    entities: list[Entity]
    relationships: list[Relationship]
    chunk_id: str
    doc_id: str
