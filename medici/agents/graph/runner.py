import traceback
import uuid

import logfire

from medici.agents.memory.conversation_model import ConversationSession
from medici.common.utils.config import config

_UNCACHEABLE_CATEGORIES = frozenset({"chitchat", "meta", "conversational", "summarization"})


class GraphPipeline:
    def __init__(self, graph, short_term_memory, semantic_cache=None, llm_clients=None):
        self.graph = graph
        self.short_term = short_term_memory
        self._semantic_cache = semantic_cache
        self._llm_clients: list = llm_clients or []

    async def chat(self, user_message: str, session_id: str, user_id: str) -> dict:

        result = {}

        if not session_id:
            logfire.info("Creating new session...")
            session_id = f"{user_id}_{uuid.uuid4()}"

        session: ConversationSession = await self.short_term.get_session(session_id)
        if not session:
            session = await self.short_term.create_session(user_id, session_id=session_id)

        history = session.to_history_dicts()

        await self.short_term.append_turn(
            session=session,
            role="user",
            content=user_message,
        )

        for client in self._llm_clients:
            client.reset_usage()

        if self._semantic_cache is not None:
            try:
                cached = await self._semantic_cache.lookup(user_message)
                if cached is not None:
                    logfire.info(
                        "SemanticCache served response",
                        similarity=cached.similarity,
                        session_id=session.session_id,
                    )
                    await self.short_term.append_turn(
                        session=session,
                        role="assistant",
                        content=cached.answer,
                        metadata={"sources": cached.sources},
                    )
                    return {
                        "answer": cached.answer,
                        "session_id": session.session_id,
                        "sources": cached.sources,
                        "query_was_rewritten": False,
                        "retrieval_rounds": 0,
                        "cache_hit": True,
                        "cache_similarity": round(cached.similarity, 4),
                        "token_usage": cached.token_usage,
                    }
            except Exception as e:
                logfire.warning(f"SemanticCache lookup failed, running full pipeline: {e}")

        graph_config = {"configurable": {"thread_id": session.session_id}}

        initial_state = {
            "session_id": session.session_id,
            "user_id": user_id,
            "original_message": user_message,
            "effective_query": user_message,
            "was_rewritten": False,
            "conversational_history": history,
            "question_category": "",
            "hop_questions": [],
            "current_hop": 0,
            "max_hops": config.MAX_HOPS,
            "current_query": user_message,
            "retrieval_round": 0,
            "total_retrieval_steps": 0,
            "max_retrieval_rounds": config.MAX_RETRIEVAL_ROUND,
            "retrieval_history": [],
            "accepted_chunks": [],
            "hop_decision": "",
            "kg_entities_found": [],
            "kg_chunks": [],
            "kg_traversal_paths": [],
            "final_answer": "",
            "sources": [],
            "images": [],
            "doc_id_filter": None,
            "episodic_context": "",
        }

        try:
            result = await self.graph.ainvoke(initial_state, config=graph_config)
        except Exception as e:
            logfire.warning(f"Graph invocation Error: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error processing your request. Please try again.",
                "session_id": session.session_id,
                "sources": [],
                "query_was_rewritten": False,
                "retrieval_hops": 0,
                "cache_hit": False,
                "token_usage": {},
                "error": str(e),
            }

        token_usage = {}
        total_calls = 0
        total_tokens = 0
        for client in self._llm_clients:
            snap = client.usage_snapshot()
            token_usage[snap["model"]] = snap
            total_calls += snap["calls"]
            total_tokens += snap["total_tokens"]

        logfire.info(
            "pipeline_token_budget",
            total_calls=total_calls,
            total_tokens=total_tokens,
            breakdown=token_usage,
            session_id=session.session_id,
        )

        await self.short_term.append_turn(
            session=session,
            role="assistant",
            content=result["final_answer"],
            metadata={"sources": result.get("sources", [])},
        )

        if self._semantic_cache is not None:
            category = result.get("question_category", "").lower()
            if category not in _UNCACHEABLE_CATEGORIES:
                try:
                    await self._semantic_cache.store(
                        query=user_message,
                        answer=result["final_answer"],
                        sources=result.get("sources", []),
                        token_usage={"total_calls": total_calls, "total_tokens": total_tokens},
                    )
                except Exception as e:
                    logfire.warning(f"SemanticCache store failed: {e}")
            else:
                logfire.info(
                    "SemanticCache store skipped (uncacheable category)",
                    category=category,
                )

        return {
            "answer": result["final_answer"],
            "session_id": session.session_id,
            "sources": result["sources"],
            "images": result.get("images", []),
            "query_was_rewritten": result["was_rewritten"],
            "retrieval_hops": result.get("current_hop", 0),
            "cache_hit": False,
            "token_usage": token_usage,
            "faithfulness": {
                "score": result.get("faithfulness_score"),
                "passed": result.get("faithfulness_passed"),
                "skipped": result.get("faithfulness_skipped"),
            },
        }

    async def chat_stream(self, user_message: str, session_id: str, user_id: str):
        """
        Async-generator that streams pipeline progress + answer tokens.

        Yields dicts shaped for SSE consumption:

        ``{"type": "progress", "node": "<node_name>", "message": "..."}``
            Emitted as each LangGraph node completes, giving the frontend
            real-time stage visibility (routing, retrieval, grading …).

        ``{"type": "token", "content": "<text>"}``
            Emitted token-by-token during synthesis via
            ``SynthesizerAgent.stream_synthesize()``.

        ``{"type": "done", "session_id": "...", "sources": [...], ...}``
            Final event carrying session metadata and token-usage stats.

        ``{"type": "error", "message": "..."}``
            Emitted on unrecoverable errors; the stream then ends.

        Strategy
        --------
        1. Run the **full graph** (minus the synthesize node) via
           ``graph.ainvoke()`` so retrieval, grading, and all routing
           logic runs to completion.  Graph-level progress events are
           surfaced via ``graph.astream_events()``.
        2. Once the pre-synthesis state is ready, call
           ``synthesizer.stream_synthesize()`` directly so answer tokens
           are pushed to the client the moment they arrive from the LLM.
        3. Semantic cache hit-path returns immediately as a single
           ``done`` event (no tokens to stream).
        """
        import uuid as _uuid

        from medici.common.utils.config import config as _cfg

        if not session_id:
            logfire.info("Creating new session (stream)...")
            session_id = f"{user_id}_{_uuid.uuid4()}"

        session = await self.short_term.get_session(session_id)
        if not session:
            session = await self.short_term.create_session(user_id, session_id=session_id)

        history = session.to_history_dicts()

        await self.short_term.append_turn(
            session=session,
            role="user",
            content=user_message,
        )

        for client in self._llm_clients:
            client.reset_usage()

        if self._semantic_cache is not None:
            try:
                cached = await self._semantic_cache.lookup(user_message)
                if cached is not None:
                    logfire.info(
                        "SemanticCache served response (stream)",
                        similarity=cached.similarity,
                        session_id=session.session_id,
                    )
                    await self.short_term.append_turn(
                        session=session,
                        role="assistant",
                        content=cached.answer,
                        metadata={"sources": cached.sources},
                    )
                    yield {
                        "type": "done",
                        "answer": cached.answer,
                        "session_id": session.session_id,
                        "sources": cached.sources,
                        "query_was_rewritten": False,
                        "cache_hit": True,
                        "cache_similarity": round(cached.similarity, 4),
                        "token_usage": cached.token_usage,
                    }
                    return
            except Exception as e:
                logfire.warning(f"SemanticCache lookup failed (stream): {e}")

        _NODE_LABELS: dict[str, str] = {
            "rewrite_query": "Rewriting query",
            "route": "Classifying question",
            "plan": "Planning retrieval",
            "kg_retrieve": "Searching knowledge graph",
            "retrieve": "Retrieving documents",
            "hop_check": "Evaluating retrieved context",
            "grade": "Grading relevance",
            "rewrite_for_refinement": "Refining query",
            "direct_synthesize": "Synthesizing answer",
            "handle_simple_response": "Generating response",
            "synthesize": "Synthesizing answer",
        }

        graph_config = {"configurable": {"thread_id": session.session_id}}

        initial_state = {
            "session_id": session.session_id,
            "user_id": user_id,
            "original_message": user_message,
            "effective_query": user_message,
            "was_rewritten": False,
            "conversational_history": history,
            "question_category": "",
            "hop_questions": [],
            "current_hop": 0,
            "max_hops": _cfg.MAX_HOPS,
            "current_query": user_message,
            "retrieval_round": 0,
            "total_retrieval_steps": 0,
            "max_retrieval_rounds": _cfg.MAX_RETRIEVAL_ROUND,
            "retrieval_history": [],
            "accepted_chunks": [],
            "hop_decision": "",
            "kg_entities_found": [],
            "kg_chunks": [],
            "kg_traversal_paths": [],
            "final_answer": "",
            "sources": [],
            "images": [],
            "doc_id_filter": None,
            "episodic_context": "",
        }

        final_state: dict = {}
        try:
            async for event in self.graph.astream_events(
                initial_state, config=graph_config, version="v2"
            ):
                kind = event.get("event")
                name = event.get("name", "")

                if kind == "on_chain_end" and name in _NODE_LABELS:
                    label = _NODE_LABELS[name]
                    yield {"type": "progress", "node": name, "message": label}

                    if name in ("synthesize", "direct_synthesize", "handle_simple_response"):
                        node_output = event.get("data", {}).get("output", {})
                        if isinstance(node_output, dict):
                            final_state.update(node_output)

        except Exception as exc:
            logfire.warning(f"Graph stream error: {exc}")
            logfire.warning(f"Full traceback: {traceback.format_exc()}")
            yield {"type": "error", "message": str(exc)}
            return

        if not final_state:
            try:
                final_state = await self.graph.ainvoke(initial_state, config=graph_config)
            except Exception as exc:
                logfire.warning(f"Graph ainvoke fallback error: {exc}")
                yield {"type": "error", "message": str(exc)}
                return

        answer = final_state.get("final_answer", "")
        sources = final_state.get("sources", [])
        images = final_state.get("images", [])
        was_rewritten = final_state.get("was_rewritten", False)

        if not answer and final_state.get("accepted_chunks"):
            # synthesizer = self.graph.nodes.get("synthesize")
            _synthesizer_agent = None
            try:
                _node_fn = self.graph.nodes["synthesize"].func  # type: ignore[attr-defined]
                _synthesizer_agent = _node_fn.keywords.get("synthesizer")
            except (AttributeError, KeyError):
                pass

            if _synthesizer_agent is not None:
                collected_tokens: list[str] = []
                try:
                    async for token in _synthesizer_agent.stream_synthesize(final_state):
                        collected_tokens.append(token)
                        yield {"type": "token", "content": token}
                    answer = "".join(collected_tokens)
                except Exception as exc:
                    logfire.warning(f"Streaming synthesis error: {exc}")
                    if not answer:
                        answer = final_state.get("final_answer", "")
            else:
                answer = final_state.get("final_answer", "")

        if answer:
            await self.short_term.append_turn(
                session=session,
                role="assistant",
                content=answer,
                metadata={"sources": sources},
            )

        token_usage: dict = {}
        total_calls = 0
        total_tokens = 0
        for client in self._llm_clients:
            snap = client.usage_snapshot()
            token_usage[snap["model"]] = snap
            total_calls += snap["calls"]
            total_tokens += snap["total_tokens"]

        logfire.info(
            "pipeline_stream_token_budget",
            total_calls=total_calls,
            total_tokens=total_tokens,
            session_id=session.session_id,
        )

        category = final_state.get("question_category", "").lower()
        if self._semantic_cache is not None and answer and category not in _UNCACHEABLE_CATEGORIES:
            try:
                await self._semantic_cache.store(
                    query=user_message,
                    answer=answer,
                    sources=sources,
                    token_usage={"total_calls": total_calls, "total_tokens": total_tokens},
                )
            except Exception as e:
                logfire.warning(f"SemanticCache store failed (stream): {e}")
        elif self._semantic_cache is not None and answer:
            logfire.info(
                "SemanticCache store skipped (uncacheable category, stream)",
                category=category,
            )

        yield {
            "type": "done",
            "answer": answer,
            "session_id": session.session_id,
            "sources": sources,
            "images": images,
            "query_was_rewritten": was_rewritten,
            "cache_hit": False,
            "token_usage": token_usage,
            "faithfulness": {
                "score": final_state.get("faithfulness_score"),
                "passed": final_state.get("faithfulness_passed"),
                "skipped": final_state.get("faithfulness_skipped"),
            },
        }
