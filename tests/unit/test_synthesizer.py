"""Unit tests for SynthesizerAgent hardening.

Tests cover:
- Token budget guard in _build_context()
- Citation enforcement via _ensure_citations()
- Unified no-context guard via _require_context()
- Comparative partial guard (<2 chunks)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.agentic.synthesizer import (
    SynthesizerAgent,
    _build_context,
    _get,
    _NO_CONTEXT_MSG,
)


def _make_chunk(source: str = "doc.pdf", section: str = "Intro", text: str = "x"):
    return {"source": source, "section": section, "text": text}


def _make_state(
    chunks: list[dict] | None = None,
    question_category: str = "factual",
    effective_query: str = "What is X?",
    original_message: str = "What is X?",
    **extra,
) -> dict:
    return {
        "accepted_chunks": chunks,
        "question_category": question_category,
        "effective_query": effective_query,
        "original_message": original_message,
        "retrieval_history": [],
        **extra,
    }


def _mock_llm(response_text: str = "LLM says hello") -> MagicMock:
    llm = MagicMock()
    resp = MagicMock()
    resp.text = response_text
    llm.complete = AsyncMock(return_value=resp)
    return llm


# 1. Token budget guard — _build_context()


class TestBuildContextTokenBudget:
    """Verify _build_context() truncates to stay within MAX_CONTEXT_CHARS."""

    def test_small_context_no_truncation(self):
        """All chunks fit within budget — nothing is dropped."""
        chunks = [_make_chunk(text=f"Chunk {i}") for i in range(3)]
        state = _make_state(chunks=chunks)

        result = _build_context(state, max_chars=100_000)

        assert "Chunk 0" in result
        assert "Chunk 1" in result
        assert "Chunk 2" in result

    def test_large_context_truncated(self):
        """When chunks exceed budget, later chunks are dropped."""
        # Each chunk text is ~100 chars + header ≈ 160 chars; separator = 7
        big_text = "A" * 200
        chunks = [_make_chunk(text=big_text, section=f"S{i}") for i in range(20)]
        state = _make_state(chunks=chunks)

        result = _build_context(state, max_chars=1_000)

        assert len(result) <= 1_000
        # At least the first chunk should be present
        assert "S0" in result
        # Not all 20 chunks should fit
        assert "S19" not in result

    def test_exact_budget_boundary(self):
        """A single chunk that exactly fills the budget is included."""
        text = "X" * 50
        chunk = _make_chunk(text=text, section="Only")
        state = _make_state(chunks=[chunk])

        # Calculate exact size of the single chunk string
        single = f"[Source: {chunk['source']} | Section: {chunk['section']}]\n{text}"
        budget = len(single)

        result = _build_context(state, max_chars=budget)
        assert result == single

    def test_budget_exceeded_by_one_char_drops_chunk(self):
        """A chunk that would exceed the budget by 1 char is dropped."""
        text = "Y" * 50
        chunk = _make_chunk(text=text, section="Drop")
        state = _make_state(chunks=[chunk])

        single = f"[Source: {chunk['source']} | Section: {chunk['section']}]\n{text}"
        budget = len(single) - 1  # 1 char short

        result = _build_context(state, max_chars=budget)
        assert result == ""  # chunk is dropped entirely

    def test_empty_chunks_returns_empty(self):
        state = _make_state(chunks=[])
        result = _build_context(state, max_chars=100_000)
        assert result == ""

    def test_none_chunks_returns_empty(self):
        state = _make_state(chunks=None)
        result = _build_context(state, max_chars=100_000)
        assert result == ""

    def test_uses_config_default_when_no_max_chars(self):
        """When max_chars is not passed, falls back to config.MAX_CONTEXT_CHARS."""
        chunks = [_make_chunk(text="short")]
        state = _make_state(chunks=chunks)

        with patch("src.agents.agentic.synthesizer.config") as mock_config:
            mock_config.MAX_CONTEXT_CHARS = 500
            result = _build_context(state)

        assert len(result) <= 500

    def test_truncation_logs_warning(self):
        """Verify a warning is logged when chunks are truncated."""
        big_text = "Z" * 200
        chunks = [_make_chunk(text=big_text, section=f"S{i}") for i in range(10)]
        state = _make_state(chunks=chunks)

        with patch("src.agents.agentic.synthesizer.logfire") as mock_logger:
            _build_context(state, max_chars=500)
            mock_logger.warning.assert_called_once()
            assert "budget" in mock_logger.warning.call_args[0][0].lower()


# 2. Citation enforcement — _ensure_citations()


class TestEnsureCitations:
    """Verify citation fallback footer behavior."""

    def setup_method(self):
        self.synth = SynthesizerAgent(llm_client=_mock_llm())

    def test_response_with_citations_untouched(self):
        """If LLM response already has [Section] citations, no footer added."""
        text = "The answer is 42 [Intro]. More detail [Chapter 2]."
        chunks = [_make_chunk(section="Intro"), _make_chunk(section="Chapter 2")]
        state = _make_state(chunks=chunks)

        result = self.synth._ensure_citations(text, state)
        assert result == text
        assert "Sources Used" not in result

    def test_response_without_citations_gets_footer(self):
        """If LLM omits citations, a Sources Used footer is appended."""
        text = "The answer is 42. No brackets here."
        chunks = [
            _make_chunk(source="doc.pdf", section="Intro"),
            _make_chunk(source="manual.pdf", section="Setup"),
        ]
        state = _make_state(chunks=chunks)

        result = self.synth._ensure_citations(text, state)
        assert "Sources Used" in result
        assert "doc.pdf — Intro" in result
        assert "manual.pdf — Setup" in result

    def test_footer_deduplicates_sections(self):
        """Duplicate source/section pairs appear only once in footer."""
        text = "No citations at all."
        chunks = [
            _make_chunk(source="a.pdf", section="S1"),
            _make_chunk(source="a.pdf", section="S1"),
            _make_chunk(source="b.pdf", section="S2"),
        ]
        state = _make_state(chunks=chunks)

        result = self.synth._ensure_citations(text, state)
        assert result.count("a.pdf — S1") == 1
        assert result.count("b.pdf — S2") == 1

    def test_no_chunks_no_footer(self):
        """If there are no chunks, no footer is added even without citations."""
        text = "Just a plain response."
        state = _make_state(chunks=[])

        result = self.synth._ensure_citations(text, state)
        assert result == text

    def test_markdown_brackets_count_as_citations(self):
        """Markdown links like [text](url) should count as bracket content."""
        text = "See [this page](http://example.com) for details."
        chunks = [_make_chunk()]
        state = _make_state(chunks=chunks)

        result = self.synth._ensure_citations(text, state)
        # [this page] matches the regex, so no footer
        assert "Sources Used" not in result


# 3. No-context guard — _require_context()


class TestRequireContext:
    """Verify _require_context returns fallback when chunks are empty."""

    def setup_method(self):
        self.synth = SynthesizerAgent(llm_client=_mock_llm())

    def test_empty_chunks_returns_fallback(self):
        state = _make_state(chunks=[])
        assert self.synth._require_context(state) == _NO_CONTEXT_MSG

    def test_none_chunks_returns_fallback(self):
        state = _make_state(chunks=None)
        assert self.synth._require_context(state) == _NO_CONTEXT_MSG

    def test_has_chunks_returns_none(self):
        state = _make_state(chunks=[_make_chunk()])
        assert self.synth._require_context(state) is None


# 4. Integration: no-context guard across all categories


class TestNoContextGuardAllCategories:
    """Each retrieval-dependent category should return _NO_CONTEXT_MSG
    when accepted_chunks is empty, without calling llm.complete()."""

    RETRIEVAL_CATEGORIES = [
        "factual",
        "procedural",
        "analytical",
        "summarization",
        "clarification",
        "comparative",
    ]

    @pytest.mark.parametrize("category", RETRIEVAL_CATEGORIES)
    @pytest.mark.asyncio
    async def test_empty_chunks_returns_fallback(self, category):
        llm = _mock_llm()
        synth = SynthesizerAgent(llm_client=llm)
        state = _make_state(chunks=[], question_category=category)

        result = await synth.synthesize(state)

        assert _NO_CONTEXT_MSG in result
        llm.complete.assert_not_called()

    @pytest.mark.parametrize("category", RETRIEVAL_CATEGORIES)
    @pytest.mark.asyncio
    async def test_with_chunks_calls_llm(self, category):
        """When chunks exist, the LLM should be called."""
        llm = _mock_llm("LLM answer [Intro]")
        synth = SynthesizerAgent(llm_client=llm)
        chunks = [
            _make_chunk(section="Intro"),
            _make_chunk(section="Details"),
            _make_chunk(section="Conclusion"),
        ]
        state = _make_state(chunks=chunks, question_category=category)

        result = await synth.synthesize(state)

        llm.complete.assert_called_once()
        assert "LLM answer" in result


class TestNoContextNotAppliedToNonRetrieval:
    """Chitchat and meta should NOT be affected by empty chunks."""

    @pytest.mark.asyncio
    async def test_chitchat_works_without_chunks(self):
        llm = _mock_llm("Hey there!")
        synth = SynthesizerAgent(llm_client=llm)
        state = _make_state(chunks=[], question_category="chitchat")

        result = await synth.synthesize(state)

        assert result == "Hey there!"
        llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_meta_works_without_chunks(self):
        llm = _mock_llm("I can help with documents.")
        synth = SynthesizerAgent(llm_client=llm)
        state = _make_state(chunks=[], question_category="meta")

        result = await synth.synthesize(state)

        assert result == "I can help with documents."
        llm.complete.assert_called_once()


# 5. Comparative partial guard (<2 chunks)


class TestComparativePartialGuard:
    """Comparative needs ≥2 chunks for a meaningful comparison."""

    @pytest.mark.asyncio
    async def test_one_chunk_returns_comparison_message(self):
        """With exactly 1 chunk, comparative returns its specific message."""
        llm = _mock_llm()
        synth = SynthesizerAgent(llm_client=llm)
        state = _make_state(chunks=[_make_chunk()], question_category="comparative")

        result = await synth.synthesize(state)

        assert "more information to make a meaningful comparison" in result
        llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_zero_chunks_returns_no_context(self):
        """With 0 chunks, the no-context guard fires before the <2 check."""
        llm = _mock_llm()
        synth = SynthesizerAgent(llm_client=llm)
        state = _make_state(chunks=[], question_category="comparative")

        result = await synth.synthesize(state)

        assert _NO_CONTEXT_MSG in result
        llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_two_chunks_proceeds_normally(self):
        llm = _mock_llm("Comparison result [S1] [S2]")
        synth = SynthesizerAgent(llm_client=llm)
        state = _make_state(
            chunks=[_make_chunk(section="S1"), _make_chunk(section="S2")],
            question_category="comparative",
        )

        result = await synth.synthesize(state)

        llm.complete.assert_called_once()
        assert "Comparison result" in result


# 6. Citation enforcement end-to-end


class TestCitationEnforcementEndToEnd:
    """Verify _ensure_citations is wired into synthesis methods."""

    @pytest.mark.asyncio
    async def test_factual_no_citations_gets_footer(self):
        llm = _mock_llm("Plain answer without brackets.")
        synth = SynthesizerAgent(llm_client=llm)
        state = _make_state(
            chunks=[_make_chunk(source="report.pdf", section="Summary")],
            question_category="factual",
        )

        result = await synth.synthesize(state)

        assert "Sources Used" in result
        assert "report.pdf — Summary" in result

    @pytest.mark.asyncio
    async def test_factual_with_citations_no_footer(self):
        llm = _mock_llm("Answer is 42 [Summary].")
        synth = SynthesizerAgent(llm_client=llm)
        state = _make_state(
            chunks=[_make_chunk(section="Summary")],
            question_category="factual",
        )

        result = await synth.synthesize(state)

        assert "Sources Used" not in result

    @pytest.mark.asyncio
    async def test_procedural_no_citations_gets_footer(self):
        llm = _mock_llm("Step 1: Do this. Step 2: Do that.")
        synth = SynthesizerAgent(llm_client=llm)
        state = _make_state(
            chunks=[_make_chunk(source="guide.pdf", section="Steps")],
            question_category="procedural",
        )

        result = await synth.synthesize(state)

        assert "Sources Used" in result
        assert "guide.pdf — Steps" in result


# 7. _get helper


class TestGetHelper:
    def test_dict_state(self):
        state = {"question_category": "factual"}
        assert _get(state, "question_category") == "factual"

    def test_dict_state_missing_key(self):
        state = {}
        assert _get(state, "question_category") is None

    def test_dataclass_state(self):
        from src.agents.agent_model import AgentState

        state = AgentState(original_question="test")
        assert _get(state, "original_question") == "test"

    def test_dataclass_attr_override(self):
        from src.agents.agent_model import AgentState

        state = AgentState(original_question="test")
        assert _get(state, "q", attr="original_question") == "test"
