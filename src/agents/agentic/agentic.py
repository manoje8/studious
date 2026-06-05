import json

import logfire

from src.agents.agent_model import AgentState, RetrievalDecision
from src.agents.agentic.grader import GraderAgent
from src.agents.agentic.planner import PlannerAgent
from src.agents.agentic.router import RouterAgent
from src.agents.agentic.synthesizer import SynthesizerAgent
from src.agents.retrieval import RetrievalAgent


class AgenticRAG:
    def __init__(
        self,
        llm_client,
        retrieval_agent: RetrievalAgent,
    ):
        self.router = RouterAgent(llm_client)
        self.planner = PlannerAgent(llm_client)
        self.retriever = retrieval_agent
        self.grader = GraderAgent(llm_client)
        self.synthesizer = SynthesizerAgent(llm_client)

    async def run(
        self, question: str, doc_id_filter: str | None = None, max_rounds: int = 1
    ) -> dict:
        logfire.info(f"Agentic RAG started: '{question}'")

        state = AgentState(
            original_question=question,
            doc_id_filter=doc_id_filter,
            max_retrieval_rounds=max_rounds,
        )

        classification = await self.router.classify(question)
        logfire.info(f"Router: {classification['category']}")

        state.sub_questions = await self.planner.decompose(
            question, classification["category"]
        )

        logfire.info(f"Planner sub questions: {json.dumps(state.sub_questions)}")

        for sub_question in state.sub_questions:
            current_query = sub_question
            round_for_this_subq = 0
            last_round_chunks: list[dict] = []

            while round_for_this_subq < state.max_retrieval_rounds:
                logfire.info(
                    f"Retrieval round {state.current_round + 1} "
                    f"(sub-question {state.sub_questions.index(sub_question) + 1}): '{current_query}'"
                )

                round_result = await self.retriever.retrieve_and_evaluate(
                    query=current_query, original_question=question, state=state
                )

                state.retrieval_rounds.append(round_result)
                state.current_round += 1
                round_for_this_subq += 1

                # # Always track the latest retrieved chunks as a fallback
                if round_result.chunk_retrieved:
                    last_round_chunks = round_result.chunk_retrieved

                logfire.info(
                    f"Round result retrieval decision: {round_result.decision}"
                )

                if round_result.decision == RetrievalDecision.SUFFICIENT:
                    state.accepted_chunks.extend(round_result.chunk_retrieved)
                    logfire.info("Retrieval sufficient, moving on")
                    break

                elif round_result.decision == RetrievalDecision.REFINE_QUERY:
                    if round_for_this_subq >= state.max_retrieval_rounds:
                        # No more rounds left — accept best chunks found so far
                        logfire.warning(
                            f"Max rounds reached for '{sub_question}', "
                            f"accepting best {len(last_round_chunks)} chunks as fallback"
                        )
                        state.accepted_chunks.extend(last_round_chunks)
                        break

                    current_query = await self.retriever.generate_refined_query(
                        original_question=sub_question,
                        previous_rounds=state.retrieval_rounds,
                    )
                    logfire.info(f"Query refined to '{current_query}'")

                elif round_result.decision == RetrievalDecision.EXPAND_SEARCH:
                    state.accepted_chunks.extend(round_result.chunk_retrieved)
                    logfire.info("Expanding search — adding chunks and continuing")

                elif round_result.decision == RetrievalDecision.EXHAUSTED:
                    # Still accept whatever we found rather than nothing
                    if last_round_chunks:
                        logfire.warning(
                            f"Information exhausted for '{sub_question}', "
                            f"accepting {len(last_round_chunks)} best-effort chunks"
                        )
                        state.accepted_chunks.extend(last_round_chunks)
                    else:
                        logfire.warning(
                            f"Information exhausted for: '{sub_question}' — no chunks found"
                        )
                    break

        logfire.info(f"Grading {len(state.accepted_chunks)} total chunks")

        state.accepted_chunks = await self.grader.grade_chunks(
            state.accepted_chunks, question
        )

        logfire.info("Synthesizing final answer")
        state.final_answer = await self.synthesizer.synthesize(state)

        state.is_complete = True

        state.source_used = list({c["source"] for c in state.accepted_chunks})

        return {
            "answer": state.final_answer,
            "sources": state.source_used,
            "retrieval_rounds": state.current_round,
            "chunks_used": len(state.accepted_chunks),
            "sub_questions": state.sub_questions,
        }
