from functools import partial

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph, END

from src.agents.graph.edges import (
    route_after_retrieve,
    route_after_next_sub_question,
    route_after_classify,
)
from src.agents.graph.nodes import (
    rewrite_query,
    route,
    plan,
    retrieve,
    refine_query,
    next_sub_question,
    grade,
    synthesize,
    direct_synthesize,
    handle_simple_response,
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
    is_multi_retriever: bool = False,
):
    builder = StateGraph(State)

    builder.add_node(
        "rewrite_query",
        partial(rewrite_query, short_term=short_term, rewriter=rewriter),
    )
    builder.add_node("route", partial(route, router=router))
    builder.add_node("plan", partial(plan, planner=planner))
    builder.add_node("retrieve", partial(retrieve, retrieval_agent=retrieval_agent))
    builder.add_node(
        "refine_query", partial(refine_query, retrieval_agent=retrieval_agent)
    )
    builder.add_node("next_sub_question", partial(next_sub_question))
    builder.add_node("grade", partial(grade, grader=grader))
    builder.add_node("synthesize", partial(synthesize, synthesizer=synthesizer))
    builder.add_node(
        "direct_synthesize", partial(direct_synthesize, synthesizer=synthesizer)
    )
    builder.add_node(
        "handle_simple_response",
        partial(handle_simple_response, synthesizer=synthesizer),
    )
    # Edges
    builder.set_entry_point("rewrite_query")
    builder.add_edge("rewrite_query", "route")
    builder.add_conditional_edges(
        "route",
        route_after_classify,
        {
            "plan": "plan",
            "direct_synthesize": "direct_synthesize",
            "simple_response": "handle_simple_response",
            "synthesize": "synthesize",
        },
    )

    builder.add_edge("plan", "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "refine_query": "refine_query",
            "next_sub_question": "next_sub_question",
            "grade": "grade",
        },
    )

    builder.add_edge("refine_query", "retrieve")

    builder.add_conditional_edges(
        "next_sub_question",
        route_after_next_sub_question,
        {
            "retrieve": "retrieve",
            "grade": "grade",
        },
    )

    builder.add_edge("grade", "synthesize")
    builder.add_edge("direct_synthesize", END)
    builder.add_edge("handle_simple_response", END)
    builder.add_edge("synthesize", END)

    return builder


async def compile_graph_with_postgres(pool, **agent_kwargs):
    builder = build_rag_graph(**agent_kwargs)

    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    return builder.compile(checkpointer=checkpointer)
