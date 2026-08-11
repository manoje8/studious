import logfire

from src.knowledge_graph.store import KGStore


class KGRetriever:
    def __init__(
        self,
        llm_client,
        kg_store: KGStore,
        storage_service,
        max_hops: int = 2,
        max_kg_chunks: int = 5,
    ):
        self.llm = llm_client
        self.kg_store = kg_store
        self.storage = storage_service
        self.max_hops = max_hops
        self.max_kg_chunks = max_kg_chunks

    async def extract_query_entities(self, query: str) -> list[str]:
        """Extracts entity mentions from a user query using the LLM."""
        prompt = f"""
Extract entity mentions from the user query.
Query: "{query}"

Return JSON:
{{
  "entities": ["entity1", "entity2"]
}}
"""
        with logfire.span("Extract entities from query", query=query):
            try:
                response = await self.llm.complete(
                    prompt, stage_tag="kg_query_extraction", json_mode=True
                )
                data = response.parsed_json or {}
                return data.get("entities", [])
            except Exception as e:
                logfire.warning("Failed to extract entities from query: {error}", error=str(e))
                return []

    async def retrieve_kg_context(self, query: str, doc_id_filter: str | None = None) -> dict:
        """Retrieves KG-augmented context for a query."""
        with logfire.span("Retrieve KG context", query=query):
            empty_result = {"entities": [], "chunks": [], "paths": []}

            # Extract entities from query
            query_entities = await self.extract_query_entities(query)
            if not query_entities:
                return empty_result

            # Fuzzy-match each entity to KG and collect canonical_keys
            matched_entity_keys = set()
            for entity_mention in query_entities:
                matches = await self.kg_store.find_entities(entity_mention, limit=3)
                for match in matches:
                    matched_entity_keys.add(match["canonical_key"])

            if not matched_entity_keys:
                return empty_result

            matched_keys_list = list(matched_entity_keys)

            # Get subgraph via multi-hop traversal
            subgraph = await self.kg_store.get_subgraph(matched_keys_list, max_hops=self.max_hops)
            if not subgraph["entities"]:
                return empty_result

            # Get chunk_ids from traversed entities
            chunk_ids = subgraph["chunk_ids"]
            if not chunk_ids:
                return empty_result

            # Fetch chunk payloads from Qdrant storage_service
            # Note: Assuming storage.get_chunks_by_ids exists, if not we fall back to a helper method
            # that we'll implement later in QdrantStorageService
            try:
                # We expect QdrantStorageService to have get_chunks_by_ids
                # If not, we can implement _fetch_chunks_by_ids using scroll/filter
                chunks = await self._fetch_chunks_by_ids(chunk_ids)
            except AttributeError:
                logfire.warning(
                    "storage_service does not have get_chunks_by_ids yet. Ensure it's implemented."
                )
                chunks = []

            # Filter by doc_id if provided
            if doc_id_filter:
                chunks = [c for c in chunks if c.get("doc_id") == doc_id_filter]

            # Limit to max_kg_chunks
            chunks = chunks[: self.max_kg_chunks]

            return {
                "entities": [e["canonical_key"] for e in subgraph["entities"]],
                "chunks": chunks,
                "paths": subgraph["relationships"],
            }

    async def _fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        """
        Fetch chunk payloads from Qdrant by doc_id:chunk_index identifiers.
        Uses storage_service methods.
        """
        if not chunk_ids:
            return []

        try:
            if hasattr(self.storage, "get_chunks_by_ids"):
                return await self.storage.get_chunks_by_ids(chunk_ids)

            # Fallback if get_chunks_by_ids is not implemented
            # This is a placeholder for the logic that uses Qdrant's scroll with payload filter
            # Parse chunk_ids into (doc_id, chunk_index) pairs
            # Since we can't implement the exact Qdrant scroll here without knowing the client,
            # we log and return empty.
            logfire.error("Please implement get_chunks_by_ids in the storage service.")
            return []
        except Exception as e:
            logfire.error("Error fetching chunks by ids: {error}", error=str(e))
            return []
