import logfire

from medici.agents.graph.state import State

GLOBAL_MAX_RETRIEVAL_STEPS = 6
_MAX_REFINEMENT_LOOPS = 1


def route_after_classify(state: State) -> str:
    """Route the query after classification.

    Routing table
    -------------
    chitchat / meta -> simple_response -> (no retrieval)
    summarization (conv.) -> synthesize -> (history only, no retrieval)
    everything else -> plan -> (full retrieve → grade → synthesize)
    """
    category = state.get("question_category", "factual").lower()

    if category in ["chitchat", "meta", "conversational"]:
        return "simple_response"
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


def route_after_hop_check(state: State) -> str:
    """
    Route after the hop_check node.

    Reads ``hop_decision`` from state:
    - ``"retrieve_again"`` → loop back to ``retrieve`` (covers both rephrase and new_sub_question)
    - ``"sufficient"`` or ``"exhausted"`` → proceed to ``grade``

    Also enforces the global retrieval step budget as a safety net.
    """
    hop_decision = state.get("hop_decision", "sufficient")
    total_steps = state.get("total_retrieval_steps", 0)

    logfire.debug(
        "route_after_hop_check",
        hop_decision=hop_decision,
        current_hop=state.get("current_hop", 0),
        max_hops=state.get("max_hops", 4),
        total_steps=total_steps,
    )

    if total_steps >= GLOBAL_MAX_RETRIEVAL_STEPS:
        logfire.warn("global retrieval step budget exceeded, forcing grade")
        return "grade"

    if hop_decision == "retrieve_again":
        return "retrieve"

    if state.get("skip_grading"):
        return "synthesize"

    return "grade"


def route_after_grade(state: State) -> str:
    """
    Route from 'grade' to either a refinement loop or the final synthesizer.

    When the grader signals needs_refinement=True and the refinement budget
    has not been exhausted, the graph loops back through retrieve so the
    synthesizer is only called when retrieval quality is acceptable (or the
    budget is spent).
    """
    needs_refinement = state.get("needs_refinement", False)
    refinement_loops = state.get("refinement_loops", 0)

    logfire.debug(
        "route_after_grade",
        needs_refinement=needs_refinement,
        refinement_loops=refinement_loops,
        max_loops=_MAX_REFINEMENT_LOOPS,
    )

    if needs_refinement and refinement_loops < _MAX_REFINEMENT_LOOPS:
        logfire.info(
            f"Grader triggered refinement loop {refinement_loops + 1}/{_MAX_REFINEMENT_LOOPS}"
        )
        return "rewrite_for_refinement"

    return "synthesize"


def route_after_faithfulness(state: State) -> str:
    """
    Route from the ``faithfulness_check`` node to the terminal ``END``.

    Gate decision
    -------------
    - ``skipped=True``-> the answer came from a no-retrieval path
      (chitchat, simple response, …).  Pass straight through.
    - ``faithfulness_passed`` -> score met or exceeded the configured
      threshold.  Answer is clean - route to END.
    - ``not faithfulness_passed`` -> score below a threshold.  The final_answer
      is annotated in-state with a low-confidence prefix so the client can
      surface a disclaimer.  Still routes to END (soft-fail) to avoid
      silently dropping responses.

    Routing targets
    ---------------
    Both branches currently terminate at ``END``.  The distinction is kept
    as separate symbolic names (``"pass"`` / ``"fail_soft"``) so the graph
    can be extended later to add a hard-fail branch (e.g., re-synthesize or
    return a canned response) without changing the edge function signature.
    """
    skipped: bool = state.get("faithfulness_skipped", False)
    passed: bool = state.get("faithfulness_passed", True)
    score: float = state.get("faithfulness_score", 1.0)

    logfire.debug(
        "route_after_faithfulness",
        skipped=skipped,
        passed=passed,
        score=score,
    )

    if skipped or passed:
        return "pass"

    logfire.warning(
        "faithfulness_gate_failed",
        score=score,
        action="soft_fail_with_annotation",
    )
    return "fail_soft"
