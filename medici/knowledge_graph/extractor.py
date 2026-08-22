import asyncio

import logfire

from medici.knowledge_graph.models import (
    Entity,
    EntityType,
    KGExtractionResult,
    Relationship,
    RelationType,
)


class KGExtractor:
    def __init__(self, llm_client, batch_size: int = 5, min_confidence: float = 0.5):
        self.llm = llm_client
        self.batch_size = batch_size
        self.min_confidence = min_confidence

    async def extract_from_chunk(
        self, chunk_text: str, chunk_id: str, doc_id: str, source_file: str, section_title: str = ""
    ) -> KGExtractionResult:
        """Extracts entities and relationships from a text chunk using LLM."""
        prompt = f"""
Given the following text chunk from document "{source_file}", section "{section_title}":

---
{chunk_text}
---

Extract all entities and relationships. Return JSON:
{{
  "entities": [
    {{
      "name": "canonical name",
      "type": "person|organization|location|concept|event|document|section|date|metric|technology|policy|product|other",
      "aliases": ["alternate names mentioned in text"],
      "description": "one-sentence description from context"
    }}
  ],
  "relationships": [
    {{
      "source": "source entity name (must match an entity name above)",
      "target": "target entity name (must match an entity name above)",
      "type": "reports_to|manages|works_at|member_of|causes|depends_on|contradicts|supports|implements|precedes|follows|occurs_during|mentions|defined_in|references|part_of|related_to",
      "description": "natural language description of the relationship",
      "confidence": 0.9
    }}
  ]
}}

Rules:
- Only extract entities and relationships explicitly stated or clearly implied in the text
- Use consistent canonical names (e.g., "Sarah Chen" not "Ms. Chen")
- Set confidence < 0.7 for implied/inferred relationships
- If no entities or relationships are found, return empty lists
- Every entity in a relationship must appear in the entities list
"""
        with logfire.span("Extract KG from chunk", chunk_id=chunk_id):
            try:
                response = await self.llm.complete(
                    prompt, stage_tag="kg_extraction", json_mode=True
                )
                data = response.parsed_json or {"entities": [], "relationships": []}
            except Exception as e:
                logfire.warning(
                    "Failed to extract KG for chunk {chunk_id}: {error}",
                    chunk_id=chunk_id,
                    error=str(e),
                )
                return KGExtractionResult(
                    entities=[], relationships=[], chunk_id=chunk_id, doc_id=doc_id
                )

            entities = []
            entity_map = {}
            for e_data in data.get("entities", []):
                try:
                    ent = Entity(
                        name=e_data["name"],
                        entity_type=EntityType(e_data["type"].lower()),
                        aliases=e_data.get("aliases", []),
                        description=e_data.get("description", ""),
                        source_chunk_ids=[chunk_id],
                        doc_ids=[doc_id],
                    )
                    entities.append(ent)
                    entity_map[ent.name.lower()] = ent.canonical_key
                except Exception as e:
                    logfire.warning(
                        "Skipping invalid entity {entity_data}: {error}",
                        entity_data=e_data,
                        error=str(e),
                    )

            relationships = []
            for r_data in data.get("relationships", []):
                if r_data.get("confidence", 1.0) < self.min_confidence:
                    continue
                try:
                    source_name = r_data["source"].lower()
                    target_name = r_data["target"].lower()
                    if source_name in entity_map and target_name in entity_map:
                        rel = Relationship(
                            source_entity=entity_map[source_name],
                            target_entity=entity_map[target_name],
                            relation_type=RelationType(r_data["type"].lower()),
                            description=r_data.get("description", ""),
                            confidence=r_data.get("confidence", 1.0),
                            source_chunk_ids=[chunk_id],
                            doc_ids=[doc_id],
                        )
                        relationships.append(rel)
                except Exception as e:
                    logfire.warning(
                        "Skipping invalid relationship {rel_data}: {error}",
                        rel_data=r_data,
                        error=str(e),
                    )

            return KGExtractionResult(
                entities=entities, relationships=relationships, chunk_id=chunk_id, doc_id=doc_id
            )

    async def extract_from_chunks(self, chunks: list, doc_id: str) -> list[KGExtractionResult]:
        """Processes chunks concurrently to extract KG elements."""
        semaphore = asyncio.Semaphore(self.batch_size)

        async def process_chunk(chunk) -> KGExtractionResult:
            async with semaphore:
                chunk_id = f"{doc_id}:{chunk.chunk_index}"
                return await self.extract_from_chunk(
                    chunk_text=chunk.text,
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    source_file=getattr(chunk, "source_file", ""),
                    section_title=getattr(chunk, "section_title", ""),
                )

        with logfire.span("Extract KG from {count} chunks", count=len(chunks)):
            tasks = [process_chunk(chunk) for chunk in chunks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            valid_results = []
            for res in results:
                if isinstance(res, Exception):
                    logfire.error("Error processing chunk: {error}", error=str(res))
                else:
                    valid_results.append(res)

            return valid_results
