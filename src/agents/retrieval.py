import logfire

from src.agents.agent_model import RetrievalDecision, RetrievalRound
from src.agents.graph.state import State
from src.agents.hybrid_search import HybridSearch
from src.agents.query_expander import QueryExpander
from src.services.reranker import Reranker

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

    async def _evaluate_results(self, query: str, original_question: str, results: list[dict]):
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

        with logfire.span("llm_retrieval_evaluation_result", query=query, num_results=len(results)):
            response = await self.llm.complete(prompt)
            parsed_result = response.parsed_json
            logfire.info(
                "retrieval_evaluation_complete",
                decision=parsed_result.get("decision"),
                reasoning=(
                    parsed_result.get("reasoning")[:100] + "..."
                    if parsed_result.get("reasoning")
                    else None
                ),
                has_refined_query=parsed_result.get("refined_query") is not None,
            )

            return parsed_result

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
            logfire.debug("using_original_query_for_subsequent_round", round_no=round_no)

        with logfire.span("hybrid_search", num_queries=len(expanded_queries)):
            candidates = await self.hybrid_search.search(
                queries=expanded_queries, doc_id_filter=doc_id_filter
            )

        with logfire.span("reranking", num_candidates=len(candidates)):
            rerank_results = await self.reranker.rerank(
                query=original_question, candidates=candidates
            )

            logfire.debug(
                "reranking_complete",
                num_results=len(rerank_results),
                top_score=(rerank_results[0].get("rrf_score", 0.0) if rerank_results else None),
            )

        return rerank_results

    async def retrieve_and_evaluate(
        self, query: str, original_question: str, state: State, round_no: int = 0
    ) -> RetrievalRound:
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

        results = await self.retrieve(
            query, original_question, state["doc_id_filter"], round_no=round_no
        )

        if not results:
            logfire.warning(
                "no_results_found",
                query=query,
                round_no=round_no,
                doc_id_filter=state.get("doc_id_filter"),
            )
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
                relevance_score=[r["rrf_score"] for r in results],
                decision=RetrievalDecision.SUFFICIENT,
                reasoning="Top rerank score {top_score:.2f} exceeded confidence threshold",
            )

        logfire.debug(
            "below_confidence_threshold_proceeding_to_evaluation",
            top_score=top_score,
            threshold=HIGH_CONFIDENCE_RRF_THRESHOLD,
            round_no=round_no,
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

        with logfire.span("llm_refined_query_generation", num_previous_rounds=len(previous_rounds)):
            response = await self.llm.complete(prompt)
            refined_query = response.text.strip()

            logfire.info(
                "refined_query_generated",
                original_question=(
                    original_question[:100] + "..."
                    if len(original_question) > 100
                    else original_question
                ),
                refined_query=refined_query,
                previous_attempts=len(previous_rounds),
            )

            return refined_query
