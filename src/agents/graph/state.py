from typing_extensions import TypedDict


class State(TypedDict):
    session_id: str
    user_id: str
    original_message: str
    effective_query: str
    was_rewritten: bool

    # planning
    question_category: str
    sub_questions: list[str]

    # retrieval loop
    current_sub_question_idx: int
    current_query: str
    retrieval_round: int
    max_retrieval_rounds: int
    retrieval_history: list[dict]
    accepted_chunks: list[dict]

    final_answer: str
    sources: list[str]
    doc_id_filter: str | None
