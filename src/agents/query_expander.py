import logfire


class QueryExpander:
    def __init__(self, llm_client):
        self.llm = llm_client

    async def expand(self, query: str) -> list[str]:
        prompt = f"""
        Generate 2 alternative phrasings of this search query.
        Each phrasing should use different vocabulary but seek the same information.
        Keep each phrasing concise — under 15 words.
        Return a JSON array of strings only. Do not include the original query.

        Original query: {query}

        Example:
        Query: "What was the memory usage in Q3?"
        Output: [
            "RAM consumption levels third quarter",
            "heap utilization Q3 report figures",
            "memory allocation statistics July August September"
        ]
        """

        response = await self.llm.complete(prompt)
        alternatives = response.parsed_json

        all_queries = [query] + alternatives

        logfire.info(f"Query expanded to {len(all_queries)} variants")
        for q in all_queries:
            logfire.info(f"  - {q}")

        return all_queries
