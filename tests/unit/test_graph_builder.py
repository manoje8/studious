"""
Unit tests for build_rag_graph() in src/agents/graph/graph.py.

We test that the graph builder:
- Correctly wires nodes and edges (StateGraph structure)
- Does not require a live Postgres connection
- compile_graph_with_postgres is testable via mocks

Note: we do NOT call compile() since that needs a real checkpointer.
We only test the builder-level structure via build_rag_graph().
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from medici.agents.graph.graph import build_rag_graph


@pytest.fixture
def mock_agents():
    return {
        "short_term": MagicMock(),
        "rewriter": MagicMock(),
        "router": MagicMock(),
        "planner": MagicMock(),
        "retrieval_agent": MagicMock(),
        "grader": MagicMock(),
        "synthesizer": MagicMock(),
    }


class TestBuildRagGraph:
    def test_build_returns_state_graph_builder(self, mock_agents):
        """build_rag_graph should return a StateGraph builder (not compiled)."""
        builder = build_rag_graph(**mock_agents)
        # The builder should have compiled nodes registered
        assert builder is not None

    def test_all_expected_nodes_present(self, mock_agents):
        """All expected nodes must be registered in the builder."""
        builder = build_rag_graph(**mock_agents)
        expected_nodes = {
            "rewrite_query",
            "route",
            "plan",
            "retrieve",
            "hop_check",
            "grade",
            "synthesize",
            "direct_synthesize",
            "handle_simple_response",
            "rewrite_for_refinement",
        }
        registered = set(builder.nodes.keys())
        for node in expected_nodes:
            assert node in registered, f"Node '{node}' not found in graph"

    def test_builder_is_reusable(self, mock_agents):
        """build_rag_graph can be called multiple times without side effects."""
        builder1 = build_rag_graph(**mock_agents)
        builder2 = build_rag_graph(**mock_agents)
        assert set(builder1.nodes.keys()) == set(builder2.nodes.keys())

    # def test_entry_point_is_rewrite_query(self, mock_agents):
    #     """The entry point must be 'rewrite_query'."""
    #     builder = build_rag_graph(**mock_agents)
    #     assert builder.entry_point == "rewrite_query"

    async def test_compile_graph_with_postgres_calls_setup(self, mock_agents):
        """compile_graph_with_postgres should call checkpointer.setup()."""
        from medici.agents.graph.graph import compile_graph_with_postgres

        mock_pool = MagicMock()
        mock_checkpointer = MagicMock()
        mock_checkpointer.setup = AsyncMock()
        mock_compiled = MagicMock()
        mock_builder = MagicMock()
        mock_builder.compile = MagicMock(return_value=mock_compiled)

        with (
            patch(
                "medici.agents.graph.graph.AsyncPostgresSaver",
                return_value=mock_checkpointer,
            ),
            patch("medici.agents.graph.graph.build_rag_graph", return_value=mock_builder),
        ):
            result = await compile_graph_with_postgres(pool=mock_pool, **mock_agents)

        mock_checkpointer.setup.assert_called_once()
        assert result == mock_compiled
