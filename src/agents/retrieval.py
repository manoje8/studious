from src.agents.agent_model import RetrievalRound, RetrievalDecision
from src.agents.graph.state import State
from src.agents.hybrid_search import HybridSearch
from src.agents.query_expander import QueryExpander
from src.services.reranker import Reranker
import logfire

HIGH_CONFIDENCE_RRF_THRESHOLD = 0.85


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

        logfire.info(
            "retrieval_agent_initialized",
            has_llm=llm_client is not None,
            has_hybrid_search=hybrid_search is not None,
            has_reranker=reranker is not None,
            has_query_expander=query_expand is not None,
        )

    async def _evaluate_results(
        self, query: str, original_question: str, results: list[dict]
    ):
        logfire.debug(
            "evaluating_retrieval_results",
            query=query,
            original_question=(
                original_question[:100] + "..."
                if len(original_question) > 100
                else original_question
            ),
            num_results=len(results),
        )

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
        self,
        query: str,
        original_question: str,
        doc_id_filter: str | None = None,
        round_no: int = 0,
    ) -> list[dict]:
        if round_no == 0:
            expanded_queries = await self.query_expand.expand(query)
        else:
            expanded_queries = [query]
            logfire.debug(
                "using_original_query_for_subsequent_round", round_no=round_no
            )

        candidates = await self.hybrid_search.search(
            queries=expanded_queries, doc_id_filter=doc_id_filter
        )

        rerank = await self.reranker.rerank(
            query=original_question, candidates=candidates
        )

        return rerank

    async def retrieve_and_evaluate(
        self, query: str, original_question: str, state: State, round_no: int = 0
    ) -> RetrievalRound:
        results = await self.retrieve(
            query, original_question, state["doc_id_filter"], round_no=round_no
        )

        logfire.info(
            "retrieval_round_start",
            round_no=round_no,
            query=query,
            original_question=(
                original_question[:100] + "..."
                if len(original_question) > 100
                else original_question
            ),
        )

        if not results:
            return RetrievalRound(
                query_used=query,
                chunk_retrieved=[],
                relevance_score=[],
                decision=RetrievalDecision.REFINE_QUERY,
                reasoning="No results after hybrid search and reranking",
            )

        top_score = results[0].get("rrf_score", 0.0)
        if top_score > HIGH_CONFIDENCE_RRF_THRESHOLD:
            logfire.info(
                "high_confidence_retrieval_skipping_evaluator",
                top_score=top_score,
                threshold=HIGH_CONFIDENCE_RRF_THRESHOLD,
                num_results=len(results),
                round_no=round_no,
                query=query,
            )
            return RetrievalRound(
                query_used=query,
                chunk_retrieved=results,
                relevance_score=[r["rrf_sccore"] for r in results],
                decision=RetrievalDecision.SUFFICIENT,
                reasoning="Top rerank score {top_score:.2f} exceeded confidence threshold",
            )

        evaluation = await self._evaluate_results(
            query=query, original_question=original_question, results=results
        )
        decision = RetrievalDecision(evaluation["decision"])

        logfire.info(
            "retrieval_round_complete",
            round_no=round_no,
            decision=decision.value,
            reasoning=(
                evaluation["reasoning"][:200] + "..."
                if len(evaluation["reasoning"]) > 200
                else evaluation["reasoning"]
            ),
            num_results=len(results),
            top_score=results[0].get("rrf_score", 0.0) if results else None,
            has_refined_query=evaluation.get("refined_query") is not None,
        )

        return RetrievalRound(
            query_used=query,
            chunk_retrieved=results,
            relevance_score=[r["rrf_score"] for r in results],
            decision=decision,
            reasoning=evaluation["reasoning"],
            refined_query=evaluation.get("refined_query"),
        )

    async def generate_refined_query(
        self, original_question: str, previous_rounds: list[dict]
    ) -> str:
        recent_rounds = previous_rounds[-2:]

        history = "\n".join(
            f"Round {i + 1}: query='{r['query']}' -> {r['reasoning']}"
            for i, r in enumerate(recent_rounds)
        )

        prompt = f"""
        Original question: {original_question}

        Previous retrieval attempts:
        {history}

        Generate a better search query that avoids what didn't work.
        Return only the query string, nothing else.
        """

        response = await self.llm.complete(prompt)

        return response.text
