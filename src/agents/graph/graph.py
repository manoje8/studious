from functools import partial

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph

from src.agents.graph.edges import (
    route_after_classify,
    route_after_faithfulness,
    route_after_grade,
    route_after_hop_check,
)
from src.agents.graph.nodes import (
    annotate_low_confidence,
    direct_synthesize,
    faithfulness_check,
    grade,
    handle_simple_response,
    hop_check,
    kg_retrieve,
    plan,
    retrieve,
    rewrite_for_refinement,
    rewrite_query,
    route,
    synthesize,
)
from src.agents.graph.state import State


def build_rag_graph(
    short_term,
    rewriter,
    router,
    planner,
    retrieval_agent,
    grader,
    synthesizer,
    faithfulness_checker=None,
    kg_retriever=None,
):
    builder = StateGraph(State)

    builder.add_node(
        "rewrite_query",
        partial(rewrite_query, short_term=short_term, rewriter=rewriter),
    )
    builder.add_node("route", partial(route, router=router))
    builder.add_node("plan", partial(plan, planner=planner))
    builder.add_node("retrieve", partial(retrieve, retrieval_agent=retrieval_agent))
    builder.add_node("hop_check", partial(hop_check, planner=planner))
    builder.add_node("grade", partial(grade, grader=grader))
    builder.add_node("synthesize", partial(synthesize, synthesizer=synthesizer))
    builder.add_node("direct_synthesize", partial(direct_synthesize, synthesizer=synthesizer))
    builder.add_node(
        "handle_simple_response",
        partial(handle_simple_response, synthesizer=synthesizer),
    )
    builder.add_node("rewrite_for_refinement", partial(rewrite_for_refinement))
    builder.add_node("annotate_low_confidence", partial(annotate_low_confidence))

    if faithfulness_checker is not None:
        builder.add_node(
            "faithfulness_check",
            partial(faithfulness_check, checker=faithfulness_checker),
        )

    builder.set_entry_point("rewrite_query")
    builder.add_edge("rewrite_query", "route")
    builder.add_conditional_edges(
        "route",
        route_after_classify,
        {
            "plan": "plan",
            "simple_response": "handle_simple_response",
            "synthesize": "synthesize",
        },
    )

    builder.add_node("kg_retrieve", partial(kg_retrieve, kg_retriever=kg_retriever))
    builder.add_edge("plan", "kg_retrieve")
    builder.add_edge("kg_retrieve", "retrieve")
    builder.add_edge("retrieve", "hop_check")

    builder.add_conditional_edges(
        "hop_check",
        route_after_hop_check,
        {
            "retrieve": "kg_retrieve",
            "grade": "grade",
            "synthesize": "synthesize",
        },
    )

    builder.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "rewrite_for_refinement": "rewrite_for_refinement",
            "synthesize": "synthesize",
        },
    )
    builder.add_edge("rewrite_for_refinement", "kg_retrieve")

    if faithfulness_checker is not None:
        builder.add_edge("synthesize", "faithfulness_check")
        builder.add_edge("direct_synthesize", "faithfulness_check")
        builder.add_edge("handle_simple_response", "faithfulness_check")

        builder.add_conditional_edges(
            "faithfulness_check",
            route_after_faithfulness,
            {
                "pass": END,
                "fail_soft": "annotate_low_confidence",
            },
        )
        builder.add_edge("annotate_low_confidence", END)
    else:
        builder.add_edge("direct_synthesize", END)
        builder.add_edge("handle_simple_response", END)
        builder.add_edge("synthesize", END)

    return builder


async def compile_graph_with_postgres(pool, **agent_kwargs):
    builder = build_rag_graph(**agent_kwargs)

    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    return builder.compile(checkpointer=checkpointer)
