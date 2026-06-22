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
    classification = await router.classify(
        state["effective_query"], conversation_history=state["retrieval_history"]
    )

    return {
        "question_category": classification["primary_category"],
        "classification": classification,
    }


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
        state=state,
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
    next_idx = state["current_sub_question_idx"] + 1
    sub_questions = state["sub_questions"]

    update: dict = {
        "accepted_chunks": (state["accepted_chunks"] or []) + (last["chunks"] or []),
        "current_sub_question_idx": next_idx,
        "retrieval_round": 0,
    }

    if next_idx < len(sub_questions):
        update["current_query"] = sub_questions[next_idx]

    return update


# Grader
async def grade(state: State, grader) -> dict:
    graded = await grader.grade_chunks(
        state["accepted_chunks"], state["effective_query"]
    )

    return {"accepted_chunks": graded}


# Synthesizer
async def synthesize(state: State, synthesizer) -> dict:
    answer = await synthesizer.synthesize(state)

    accepted_chunks = state.get("accepted_chunks") or []
    sources = list({c["source"] for c in accepted_chunks if c.get("source")})

    return {"final_answer": answer, "sources": sources}


async def direct_synthesize(state: State, synthesizer) -> dict:
    """Direct synthesis for simple queries (bypasses planning/grading loop)."""
    result = await synthesizer.direct_synthesize(state)

    return {
        "final_answer": result["final_answer"],
        "sources": result.get("sources", []),
    }


async def handle_simple_response(state: State, synthesizer) -> dict:
    """Handle chitchat/meta queries without retrieval."""
    result = await synthesizer.handle_simple_response(state)

    return {
        "final_answer": result["final_answer"],
        "sources": [],
    }
