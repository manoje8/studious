from src.agents.graph.state import State


def route_after_classify(state: State) -> str:
    category = state.get("question_category", "factual").lower()
    classify = state.get("classification", {})

    if category in ["chitchat", "meta"]:
        return "simple_response"
    if category == "factual":
        complexity = classify.get("complexity_level", 1)
        if complexity <= 2:
            return "direct_synthesize"
    if category == "summarization":
        msg = state.get("original_message", "").lower()
        if any(
            word in msg
            for word in [
                "previous",
                "discussed",
                "conversation",
                "recap",
                "summarize our",
            ]
        ):
            return "synthesize"

    return "plan"


def route_after_retrieve(state: State) -> str:
    last = state["retrieval_history"][-1]
    decision = last["decision"]
    round_no = state["retrieval_round"]
    max_rounds = state["max_retrieval_rounds"]
    sub_q_idx = state["current_sub_question_idx"]
    total_sub_qs = len(state["sub_questions"])

    if decision == "sufficient":
        if sub_q_idx + 1 < total_sub_qs:
            return "next_sub_question"
        return "grade"

    elif decision == "refine_query" and round_no < max_rounds:
        return "refine_query"

    elif decision == "expand_search":
        if sub_q_idx + 1 < total_sub_qs:
            return "next_sub_question"
        return "grade"

    else:
        if sub_q_idx + 1 < total_sub_qs:
            return "next_sub_question"
        return "grade"


def route_after_next_sub_question(state: State) -> str:
    sub_questions = state.get("sub_questions", [])
    current_idx = state.get("current_sub_question_idx", 0)

    if current_idx < len(sub_questions):
        return "retrieve"

    return "grade"
