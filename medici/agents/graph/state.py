from typing_extensions import TypedDict


class State(TypedDict):
    session_id: str
    user_id: str
    original_message: str
    effective_query: str
    was_rewritten: bool

    # planning
    question_category: str
    hop_questions: list[str]
    classification: dict

    # retrieval loop
    current_hop: int
    max_hops: int
    current_query: str
    retrieval_round: int
    total_retrieval_steps: int
    max_retrieval_rounds: int
    retrieval_history: list[dict]
    accepted_chunks: list[dict]
    hop_decision: str  # "retrieve_again" | "sufficient" | "exhausted"

    final_answer: str
    sources: list[str]
    images: list[dict]
    doc_id_filter: str | None

    resolved_references: list

    # memory
    conversational_history: list[dict]
    episodic_context: str

    # grading / refinement loop
    needs_refinement: bool
    refinement_loops: int
    skip_grading: bool

    # post-synthesis faithfulness gate
    faithfulness_score: float
    faithfulness_passed: bool
    faithfulness_skipped: bool
