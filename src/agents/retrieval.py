import json

from src.agents.agent_model import AgentState, RetrievalRound, RetrievalDecision
from src.agents.hybrid_search import HybridSearch
from src.agents.query_expander import QueryExpander
from src.services.reranker import Reranker


class RetrievalAgent:
    def __init__(
        self,
        llm_client,
        hybrid_search: HybridSearch,
        reranker: Reranker,
        query_expand: QueryExpander,
    ):
        self.llm = llm_client
        self.hybrid_search = hybrid_search
        self.reranker = reranker
        self.query_expand = query_expand

    async def _evaluate_results(
        self, query: str, original_question: str, results: list[dict]
    ):
        context_preview = "\n".join(
            f"Chunk {i}: {r['text'][:300]}..." for i, r in enumerate(results)
        )

        prompt = f"""
        Original question: {original_question}
        Search query used: {query}

        Retrieved chunks:
        {context_preview}

        Evaluate the retrieval and respond in JSON only:
        {{
            "scores": [0.0-1.0 per chunk],
            "decision": "sufficient" | "refine_query" | "expand_search" | "exhausted",
            "reasoning": "why",
            "refined_query": "better query if decision is refine_query, else null"
        }}

        Use "sufficient" if chunks clearly answer the question.
        Use "refine_query" if chunks are partially relevant but a better query would help.
        Use "expand_search" if you need additional chunks on a different aspect.
        Use "exhausted" if the information is likely not in this document.
        """

        response = await self.llm.complete(prompt)
        return response.parsed_json

    async def retrieve(
        self, query: str, original_question: str, doc_id_filter: str | None = None
    ) -> list[dict]:
        expanded_queries = await self.query_expand.expand(query)

        candidates = await self.hybrid_search.search(
            queries=expanded_queries, doc_id_filter=doc_id_filter
        )

        rerank = await self.reranker.rerank(
            query=original_question, candidates=candidates
        )

        return rerank

    async def retrieve_and_evaluate(
        self, query: str, original_question: str, state: AgentState
    ) -> RetrievalRound:
        results = await self.retrieve(query, original_question, state.doc_id_filter)

        print(json.dumps(results, indent=2, default=str))

        if not results:
            return RetrievalRound(
                query_used=query,
                chunk_retrieved=[],
                relevance_score=[],
                decision=RetrievalDecision.REFINE_QUERY,
                reasoning="No results after hybrid search and reranking",
            )

        evaluation = await self._evaluate_results(
            query=query, original_question=original_question, results=results
        )

        return RetrievalRound(
            query_used=query,
            chunk_retrieved=results,
            relevance_score=[r["rrf_score"] for r in results],
            decision=RetrievalDecision(evaluation["decision"]),
            reasoning=evaluation["reasoning"],
        )

    async def generate_refined_query(
        self, original_question: str, previous_rounds: list[RetrievalRound]
    ) -> str:
        history = "\n".join(
            f"Round {i+1}: query='{r.query_used}' -> {r.reasoning}"
            for i, r in enumerate(previous_rounds)
        )

        prompt = f"""
        Original question: {original_question}

        Previous retrieval attempts:
        {history}

        Generate a better search query that avoids what didn't work.
        Return only the query string, nothing else.
        """

        response = await self.llm.complete(prompt)

        print(response.text)

        return response.text
