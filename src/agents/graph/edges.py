from src.agents.graph.state import State


def route_after_classify(state: State) -> str:
    if state["question_category"] == "chitchat":
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
    return "retrieve"
