import logfire

from medici.agents.adaptive_retrieval import AdaptiveRetrievalConfig
from medici.agents.graph.state import State
from medici.common.services.faithfulness_checker import FaithfulnessChecker
from medici.knowledge_graph.retriever import KGRetriever

_CHUNK_PREVIEW_MAX_CHARS = 300

_LOW_CONFIDENCE_PREFIX = (
    "*Low-confidence answer* — the response may not be fully grounded "
    "in the retrieved documents.\n\n"
)


def _build_structured_chunk_previews(chunks: list[dict]) -> str:
    """
    Build metadata-tagged raw chunk previews for the hop planner.

    Each chunk is truncated to ``_CHUNK_PREVIEW_MAX_CHARS`` and tagged
    with source/section metadata so the LLM can assess coverage without
    an expensive summarization step.
    """
    if not chunks:
        return "(no chunks retrieved yet)"

    previews = []
    for i, c in enumerate(chunks):
        source = c.get("source", "unknown")
        section = c.get("section", "unknown")
        text = c.get("text", "")[:_CHUNK_PREVIEW_MAX_CHARS]
        score = c.get("score", "?")
        previews.append(
            f"[Chunk {i} | source={source} | section={section} | score={score}]\n{text}"
        )
    return "\n\n".join(previews)


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
    sub_qs = await planner.decompose(state["effective_query"], state["question_category"])

    return {
        "hop_questions": sub_qs,
        "current_hop": 0,
        "retrieval_round": 0,
    }


async def kg_retrieve(state: State, kg_retriever: KGRetriever | None) -> dict:
    """
    Retrieve chunks via knowledge graph traversal.

    Runs before the vector-based ``retrieve`` node.  Extracted entities and
    their graph neighborhoods are resolved to source chunks, which are
    pre-seeded into ``accepted_chunks`` so the vector retriever can focus
    on filling remaining gaps.

    When *kg_retriever* is ``None`` (KG disabled), the node is a transparent
    pass-through that returns an empty KG state.
    """
    if kg_retriever is None:
        return {
            "kg_entities_found": [],
            "kg_chunks": [],
            "kg_traversal_paths": [],
        }

    query = state["current_query"]
    doc_id_filter = state.get("doc_id_filter")

    with logfire.span("kg_retrieve", query=query[:80]):
        kg_result = await kg_retriever.retrieve_kg_context(
            query=query,
            doc_id_filter=doc_id_filter,
        )

    if not kg_result or not kg_result.get("chunks"):
        logfire.info("kg_retrieve_no_results", query=query[:80])
        return {
            "kg_entities_found": kg_result.get("entities", []) if kg_result else [],
            "kg_chunks": [],
            "kg_traversal_paths": [],
        }

    # Tag KG-sourced chunks with provenance
    for chunk in kg_result["chunks"]:
        chunk["retrieval_source"] = "knowledge_graph"

    logfire.info(
        "kg_retrieve_complete",
        num_entities=len(kg_result.get("entities", [])),
        num_chunks=len(kg_result["chunks"]),
        num_paths=len(kg_result.get("paths", [])),
    )

    return {
        "kg_entities_found": kg_result.get("entities", []),
        "kg_chunks": kg_result["chunks"],
        "kg_traversal_paths": kg_result.get("paths", []),
        "accepted_chunks": (state.get("accepted_chunks") or []) + kg_result["chunks"],
    }


# Retrieve & evaluate
async def retrieve(state: State, retrieval_agent) -> dict:
    adaptive_config = AdaptiveRetrievalConfig.from_state(state)
    current_query = state["current_query"]
    hop_questions = state.get("hop_questions", [])
    # Use the last hop question as the "original question" for reranking
    original_question = hop_questions[-1] if hop_questions else current_query

    round_result = await retrieval_agent.retrieve_and_evaluate(
        query=current_query,
        original_question=original_question,
        state=state,
        round_no=state["retrieval_round"],
    )

    is_factual = state.get("question_category", "factual").lower() == "factual"
    top_score = (
        round_result.chunk_retrieved[0].get("score", 0.0) if round_result.chunk_retrieved else 0.0
    )
    skip_grading = is_factual and top_score > adaptive_config.confidence_threshold

    return {
        "retrieval_history": state["retrieval_history"]
        + [
            {
                "query": round_result.query_used,
                "decision": round_result.decision.value,
                "reasoning": round_result.reasoning,
                "chunks": round_result.chunk_retrieved,
                "refined_query": getattr(round_result, "refined_query", None),
            }
        ],
        "retrieval_round": state["retrieval_round"] + 1,
        "total_retrieval_steps": state.get("total_retrieval_steps", 0) + 1,
        "skip_grading": skip_grading,
        "max_hops": adaptive_config.max_hops,
    }


async def hop_check(state: State, planner) -> dict:
    """
    Assess retrieval sufficiency and decide the next action.

    After each retrieval round this node:
    1. Accumulates accepted chunks from the latest retrieval
    2. Builds structured raw-chunk previews of everything gathered so far
    3. Checks the hop budget
    4. Calls ``planner.plan_next_hop()`` to decide: rephrase / new_sub_question / sufficient
    5. Returns a state update with ``hop_decision`` for the edge router
    """

    last = state["retrieval_history"][-1]
    raw_accepted = (state.get("accepted_chunks") or []) + (last.get("chunks") or [])
    accepted = [c for c in raw_accepted if isinstance(c, dict)]
    current_hop = state.get("current_hop", 0)
    max_hops = state.get("max_hops", 4)
    hop_questions = list(state.get("hop_questions", []))

    # Budget check
    if current_hop >= max_hops:
        logfire.warning(
            "hop_budget_exhausted",
            current_hop=current_hop,
            max_hops=max_hops,
        )
        return {
            "accepted_chunks": accepted,
            "hop_decision": "exhausted",
        }

    if state.get("skip_grading"):
        logfire.info("skipping_hop_check_due_to_high_confidence", current_hop=current_hop)
        return {
            "accepted_chunks": accepted,
            "hop_decision": "sufficient",
        }

    context_preview = _build_structured_chunk_previews(accepted)

    hop_result = await planner.plan_next_hop(
        original_question=state.get("effective_query") or state.get("original_message", ""),
        hop_questions=hop_questions,
        retrieved_context=context_preview,
        question_category=state.get("question_category", "factual"),
    )

    next_action = hop_result.get("next_action", "sufficient")
    new_query = hop_result.get("query")
    reasoning = hop_result.get("reasoning", "")

    logfire.info(
        "hop_check_decision",
        next_action=next_action,
        current_hop=current_hop,
        max_hops=max_hops,
        num_accepted_chunks=len(accepted),
        reasoning=reasoning[:200] if reasoning else "",
    )

    update: dict = {
        "accepted_chunks": accepted,
    }

    if next_action in ("rephrase", "new_sub_question"):
        update["hop_decision"] = "retrieve_again"
        update["current_query"] = new_query
        update["current_hop"] = current_hop + 1
        update["retrieval_round"] = 0  # reset per-hop retrieval round counter
        if new_query:
            update["hop_questions"] = hop_questions + [new_query]
    else:
        # "sufficient" or any unrecognised value → proceed to grade
        update["hop_decision"] = "sufficient"

    return update


# Grader
async def grade(state: State, grader) -> dict:
    return await grader.grade(state)


# Synthesizer
async def synthesize(state: State, synthesizer) -> dict:
    answer = await synthesizer.synthesize(state)

    accepted_chunks = state.get("accepted_chunks") or []
    dict_chunks = [c for c in accepted_chunks if isinstance(c, dict)]

    sources = list(set(c["source"] for c in dict_chunks if c.get("source")))

    images = [
        {
            "image_path": c["image_path"],
            "caption": c.get("text", ""),
            "page_numbers": c.get("page_numbers", []),
            "score": c.get("score"),
            "source": c.get("source", ""),
        }
        for c in dict_chunks
        if c.get("content_type") == "image" and c.get("image_path")
    ]

    return {"final_answer": answer, "sources": sources, "images": images}


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


async def rewrite_for_refinement(state: State) -> dict:
    """
    Reset retrieval state for a grader-triggered re-retrieval pass.

    Called when route_after_grade decides the grader's needs_refinement flag
    warrants another full retrieval cycle.  Increments refinement_loops so the
    loop-guard in route_after_grade can enforce the ceiling, and resets all
    per-cycle counters so the graph re-runs the hop loop cleanly.

    Note: hop_questions is intentionally preserved — we re-retrieve for the
    same decomposed plan rather than re-running the planner.
    """
    return {
        "refinement_loops": state.get("refinement_loops", 0) + 1,
        "retrieval_round": 0,
        "current_hop": 0,
        "total_retrieval_steps": 0,
        "accepted_chunks": [],
        "retrieval_history": [],
        "hop_decision": "",
    }


async def faithfulness_check(state: State, checker: FaithfulnessChecker) -> dict:
    """
    Post-synthesis faithfulness gate.

    Scores the ``final_answer`` against the ``accepted_chunks`` using the
    Vectara hallucination evaluation NLI model.  Writes three fields back to
    state:

    ``faithfulness_score``
        Float in [0, 1] — how well the answer is grounded in the retrieved
        evidence.  Higher is better.

    ``faithfulness_passed``
        Boolean gate decision (``score >= threshold``).  The edge router uses
        this to decide whether to surface the answer or a fallback.

    ``faithfulness_skipped``
        ``True`` when there is no answer or no evidence chunks (e.g. chitchat
        paths), in which case the gate always passes.
    """
    result = await checker.check_from_state(state)

    logfire.info(
        "faithfulness_gate",
        score=result["score"],
        passed=result["passed"],
        skipped=result.get("skipped", False),
        threshold=result["threshold"],
    )

    return {
        "faithfulness_score": result["score"],
        "faithfulness_passed": result["passed"],
        "faithfulness_skipped": result.get("skipped", False),
    }


def annotate_low_confidence(state: State) -> dict:
    """
    Soft-fail annotation node.

    Prepends a disclaimer to the ``final_answer`` when the faithfulness gate
    score falls below the configured threshold.  This keeps the graph from
    silently dropping answers while still surfacing a client-visible signal.
    """
    answer = state.get("final_answer", "")
    if not answer.startswith(_LOW_CONFIDENCE_PREFIX):
        answer = _LOW_CONFIDENCE_PREFIX + answer
    return {"final_answer": answer}
