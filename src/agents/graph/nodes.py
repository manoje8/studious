from src.agents.agent_model import AgentState
from src.agents.graph.state import State


# query Rewriter
async def rewrite_query(state: State, short_term, rewriter) -> dict:
    session = await short_term.get_session(state["session_id"])
    result = await rewriter.rewrite(state["original_message"], session)

    return {
        "current_query": result["rewritten_query"],
        "effective_query": result["rewritten_query"],
        "was_rewritten": result["was_rewritten"],
        "resolved_references": result["resolved_references"],
    }


# Router
async def route(state: State, router) -> dict:
    classification = await router.classify(state["effective_query"])

    return {"question_category": classification["category"]}


# Planner
async def plan(state: State, planner) -> dict:
    sub_qs = await planner.decompose(
        state["effective_query"], state["question_category"]
    )

    return {
        "sub_questions": sub_qs,
        "current_sub_question_idx": 0,
        "retrieval_round": 0,
    }


# Retrieve & evaluate
async def retrieve(state: State, retrieval_agent) -> dict:
    sub_q = state["sub_questions"][state["current_sub_question_idx"]]

    round_result = await retrieval_agent.retrieve_and_evaluate(
        query=state["current_query"],
        original_question=sub_q,
        state=AgentState,
    )

    return {
        "retrieval_history": state["retrieval_history"]
        + [
            {
                "query": round_result.query_used,
                "decision": round_result.decision.value,
                "reasoning": round_result.reasoning,
                "chunks": round_result.chunk_retrieved,
            }
        ],
        "retrieval_round": state["retrieval_round"] + 1,
    }


# Refine Query
async def refine_query(state: State, retrieval_agent) -> dict:
    new_query = await retrieval_agent.generate_refined_query(
        original_question=state["sub_questions"][state["current_sub_question_idx"]],
        previous_rounds=state["retrieval_round"],
    )

    return {"current_query": new_query}


# Sub question
async def next_sub_question(state: State) -> dict:
    last = state["retrieval_history"][-1]

    return {
        "accepted_chunks": state["accepted_chunks"] + last["chunks"],
        "current_sub_question_idx": state["current_sub_question_idx"] + 1,
        "retrieval_round": 0,
        "current_query": state["sub_questions"][state["current_sub_question_idx"] + 1],
    }


# Grader
async def grade(state: State, grader) -> dict:
    graded = await grader.grade_chunks(
        state["accepted_chunks"], state["effective_query"]
    )

    return {"accepted_chunks": graded}


# Synthesizer
async def synthesize(state: State, synthesizer) -> dict:
    answer = await synthesizer.synthesize_from_state(state)
    sources = list({c["source"] for c in state["accepted_chunks"]})

    return {"final_answer": answer, "sources": sources}
