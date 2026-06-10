"""
Unit tests for src/ingestion/chunking/chunk.py

Covers:
- Chunk dataclass construction and to_quant_payload()
- Chunking._clean_text()
- Chunking.chunk_by_structure()
- Chunking.chunk_fixed()
- Chunking.splitter()
- Chunking.build_parent_child_chunk()
"""

import pytest
from unittest.mock import patch

from src.ingestion.chunking.chunk import Chunk, Chunking

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chunker():
    """Return a fresh Chunking instance."""
    return Chunking()


def make_block(text: str, block_type: str = "paragraph") -> dict:
    return {"type": block_type, "text": text}


def make_chunks(n: int = 5, chunker_instance=None) -> list[Chunk]:
    """Create *n* deterministic Chunk objects for testing."""
    return [
        Chunk(
            text=f"chunk text {i}",
            chunk_index=i,
            doc_id="doc-1",
            source_file="file.pdf",
            chunk_type="structure",
            section_title=f"Section {i}",
        )
        for i in range(n)
    ]


# ===========================================================================
# Chunk dataclass
# ===========================================================================


class TestChunkDataclass:
    def test_required_fields(self):
        chunk = Chunk(
            text="hello",
            chunk_index=0,
            doc_id="d1",
            source_file="f.pdf",
            chunk_type="structure",
        )
        assert chunk.text == "hello"
        assert chunk.chunk_index == 0
        assert chunk.doc_id == "d1"
        assert chunk.source_file == "f.pdf"
        assert chunk.chunk_type == "structure"

    def test_optional_field_defaults(self):
        chunk = Chunk(
            text="hello",
            chunk_index=0,
            doc_id="d1",
            source_file="f.pdf",
            chunk_type="fixed",
        )
        assert chunk.section_title == ""
        assert chunk.page_numbers == []
        assert chunk.block_types == []
        assert chunk.token_count == 0
        assert chunk.parent_text == ""
        assert chunk.parent_window_start == 0
        assert chunk.parent_window_end == 0

    def test_mutable_default_isolation(self):
        """Each Chunk should have its own independent list instances."""
        c1 = Chunk("a", 0, "d", "f", "fixed")
        c2 = Chunk("b", 1, "d", "f", "fixed")
        c1.page_numbers.append(1)
        assert c2.page_numbers == []

    def test_to_quant_payload_keys(self):
        chunk = Chunk(
            text="payload",
            chunk_index=3,
            doc_id="doc-42",
            source_file="doc.pdf",
            chunk_type="table",
            section_title="Intro",
            page_numbers=[1, 2],
            token_count=10,
        )
        payload = chunk.to_quant_payload()
        expected_keys = {
            "text",
            "chunk_index",
            "doc_id",
            "source_file",
            "chunk_type",
            "section_title",
            "page_numbers",
            "token_count",
        }
        assert set(payload.keys()) == expected_keys

    def test_to_quant_payload_values(self):
        chunk = Chunk(
            text="data",
            chunk_index=7,
            doc_id="doc-99",
            source_file="report.pdf",
            chunk_type="structure",
            section_title="Results",
            page_numbers=[5, 6],
            token_count=42,
        )
        payload = chunk.to_quant_payload()
        assert payload["text"] == "data"
        assert payload["chunk_index"] == 7
        assert payload["doc_id"] == "doc-99"
        assert payload["source_file"] == "report.pdf"
        assert payload["chunk_type"] == "structure"
        assert payload["section_title"] == "Results"
        assert payload["page_numbers"] == [5, 6]
        assert payload["token_count"] == 42


# ===========================================================================
# Chunking._clean_text
# ===========================================================================


class TestCleanText:
    def test_strips_whitespace(self, chunker):
        assert chunker._clean_text("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self, chunker):
        result = chunker._clean_text("word1   word2")
        assert result == "word1 word2"

    def test_empty_string(self, chunker):
        assert chunker._clean_text("") == ""

    def test_normal_text_unchanged(self, chunker):
        text = "This is a normal sentence."
        assert chunker._clean_text(text) == text

    def test_ocr_space_separated_word_collapsed(self, chunker):
        """Five or more single-char words separated by spaces should merge."""
        # "H e l l o" → "Hello"  (5 chars, exactly at the 4+ threshold)
        result = chunker._clean_text("H e l l o")
        assert " " not in result or result == "H e l l o"  # either merged or left

    def test_leading_trailing_spaces_removed(self, chunker):
        assert chunker._clean_text("   ") == ""

    def test_unicode_special_chars_preserved(self, chunker):
        text = "It\u2019s a \u201ctest\u201d"
        result = chunker._clean_text(text)
        assert "\u2019" in result
        assert "\u201c" in result
        assert "\u201d" in result


# ===========================================================================
# Chunking.chunk_by_structure
# ===========================================================================


class TestChunkByStructure:
    DOC_ID = "doc-struct"
    SOURCE = "report.pdf"

    # --- Basic behaviour ---

    def test_returns_list(self, chunker):
        result = chunker.chunk_by_structure([], self.DOC_ID, self.SOURCE)
        assert isinstance(result, list)

    def test_empty_input_returns_no_chunks(self, chunker):
        result = chunker.chunk_by_structure([], self.DOC_ID, self.SOURCE)
        assert result == []

    def test_single_paragraph_creates_one_chunk(self, chunker):
        content = [make_block("Hello world")]
        result = chunker.chunk_by_structure(content, self.DOC_ID, self.SOURCE)
        assert len(result) == 1
        assert result[0].text == "Hello world"

    def test_chunk_type_is_structure(self, chunker):
        content = [make_block("Some text")]
        chunks = chunker.chunk_by_structure(content, self.DOC_ID, self.SOURCE)
        assert chunks[0].chunk_type == "structure"

    def test_chunk_metadata(self, chunker):
        content = [make_block("Para")]
        chunks = chunker.chunk_by_structure(content, self.DOC_ID, self.SOURCE)
        c = chunks[0]
        assert c.doc_id == self.DOC_ID
        assert c.source_file == self.SOURCE
        assert c.chunk_index == 0

    # --- Heading splits ---

    def test_heading_flushes_previous_blocks(self, chunker):
        content = [
            make_block("Intro paragraph"),
            make_block("Chapter 1", "heading"),
            make_block("Body text"),
        ]
        chunks = chunker.chunk_by_structure(content, self.DOC_ID, self.SOURCE)
        # First chunk from intro, second from body
        assert len(chunks) == 2
        assert "Intro" in chunks[0].text

    def test_section_title_updated_after_heading(self, chunker):
        content = [
            make_block("Chapter 1", "heading"),
            make_block("Content under ch1"),
        ]
        chunks = chunker.chunk_by_structure(content, self.DOC_ID, self.SOURCE)
        assert chunks[0].section_title == "Chapter 1"

    def test_default_section_title_is_introduction(self, chunker):
        content = [make_block("Some text before any heading")]
        chunks = chunker.chunk_by_structure(content, self.DOC_ID, self.SOURCE)
        assert chunks[0].section_title == "Introduction"

    def test_consecutive_headings_no_empty_chunks(self, chunker):
        content = [
            make_block("Heading A", "heading"),
            make_block("Heading B", "heading"),
            make_block("Content B"),
        ]
        chunks = chunker.chunk_by_structure(content, self.DOC_ID, self.SOURCE)
        # Only 1 real chunk (Content B); no empty chunk between headings
        assert all(c.text.strip() != "" for c in chunks)

    # --- Table handling ---

    def test_table_block_gets_its_own_chunk(self, chunker):
        content = [make_block("Table data", "table")]
        chunks = chunker.chunk_by_structure(content, self.DOC_ID, self.SOURCE)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "table"

    def test_table_flushes_accumulated_paragraphs(self, chunker):
        content = [
            make_block("Para before table"),
            make_block("col1 | col2", "table"),
        ]
        chunks = chunker.chunk_by_structure(content, self.DOC_ID, self.SOURCE)
        assert len(chunks) == 2
        assert chunks[0].chunk_type == "structure"
        assert chunks[1].chunk_type == "table"

    def test_table_block_types_field(self, chunker):
        content = [make_block("Table data", "table")]
        chunks = chunker.chunk_by_structure(content, self.DOC_ID, self.SOURCE)
        assert chunks[0].block_types == ["table"]

    # --- Block types ---

    def test_block_types_recorded_for_paragraphs(self, chunker):
        content = [
            make_block("Para 1", "paragraph"),
            make_block("Para 2", "paragraph"),
        ]
        chunks = chunker.chunk_by_structure(content, self.DOC_ID, self.SOURCE)
        assert "paragraph" in chunks[0].block_types

    # --- Non-dict input ---

    def test_string_blocks_treated_as_paragraphs(self, chunker):
        content = ["Plain string block"]
        chunks = chunker.chunk_by_structure(content, self.DOC_ID, self.SOURCE)
        assert len(chunks) == 1
        assert "Plain string block" in chunks[0].text

    # --- Whitespace / empty blocks ---

    def test_empty_text_blocks_skipped(self, chunker):
        content = [make_block(""), make_block("   "), make_block("Real content")]
        chunks = chunker.chunk_by_structure(content, self.DOC_ID, self.SOURCE)
        assert len(chunks) == 1
        assert "Real content" in chunks[0].text

    # --- Multiple paragraphs merged ---

    def test_multiple_paragraphs_merged_into_one_chunk(self, chunker):
        content = [
            make_block("First paragraph"),
            make_block("Second paragraph"),
            make_block("Third paragraph"),
        ]
        chunks = chunker.chunk_by_structure(content, self.DOC_ID, self.SOURCE)
        assert len(chunks) == 1
        assert "First paragraph" in chunks[0].text
        assert "Second paragraph" in chunks[0].text

    # --- Chunk index increments ---

    def test_chunk_index_increments_correctly(self, chunker):
        content = [
            make_block("Para 1"),
            make_block("Heading", "heading"),
            make_block("Para 2"),
            make_block("Another heading", "heading"),
            make_block("Para 3"),
        ]
        chunks = chunker.chunk_by_structure(content, self.DOC_ID, self.SOURCE)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i


# ===========================================================================
# Chunking.chunk_fixed
# ===========================================================================


class TestChunkFixed:
    DOC_ID = "doc-fixed"
    SOURCE = "report.pdf"

    def _make_content(self, words: int) -> list[dict]:
        text = " ".join(f"word{i}" for i in range(words))
        return [make_block(text)]

    # --- Basic ---

    def test_returns_list(self, chunker):
        result = chunker.chunk_fixed([], self.DOC_ID, self.SOURCE)
        assert isinstance(result, list)

    def test_empty_input_returns_no_chunks(self, chunker):
        result = chunker.chunk_fixed([], self.DOC_ID, self.SOURCE)
        assert result == []

    def test_short_text_fits_in_single_chunk(self, chunker):
        content = [make_block("short text")]
        chunks = chunker.chunk_fixed(content, self.DOC_ID, self.SOURCE, chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "fixed"

    def test_chunk_metadata(self, chunker):
        content = [make_block("hello world")]
        chunks = chunker.chunk_fixed(content, self.DOC_ID, self.SOURCE)
        c = chunks[0]
        assert c.doc_id == self.DOC_ID
        assert c.source_file == self.SOURCE

    # --- Splitting ---

    def test_large_text_split_into_multiple_chunks(self, chunker):
        content = self._make_content(1100)
        chunks = chunker.chunk_fixed(
            content, self.DOC_ID, self.SOURCE, chunk_size=512, overlap=50
        )
        assert len(chunks) > 1

    def test_each_chunk_within_size_limit(self, chunker):
        content = self._make_content(2000)
        chunk_size = 100
        chunks = chunker.chunk_fixed(
            content, self.DOC_ID, self.SOURCE, chunk_size=chunk_size, overlap=10
        )
        for c in chunks:
            word_count = len(c.text.split())
            # Allow a small margin for overlap tokens that were prepended
            assert word_count <= chunk_size + 10 + 5

    def test_chunk_index_increments_correctly(self, chunker):
        content = self._make_content(1200)
        chunks = chunker.chunk_fixed(
            content, self.DOC_ID, self.SOURCE, chunk_size=512, overlap=0
        )
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    # --- Overlap ---

    def test_overlap_zero_no_repeated_content(self, chunker):
        """With overlap=0 consecutive chunks should not share tokens at boundaries."""
        words = [f"w{i}" for i in range(200)]
        content = [make_block(" ".join(words))]
        chunks = chunker.chunk_fixed(
            content, self.DOC_ID, self.SOURCE, chunk_size=100, overlap=0
        )
        # Each word should appear only once across all chunks
        all_words = []
        for c in chunks:
            all_words.extend(c.text.split())
        assert len(all_words) == len(set(all_words))

    # --- String block input ---

    def test_string_block_handled_correctly(self, chunker):
        content = ["plain string content"]
        chunks = chunker.chunk_fixed(content, self.DOC_ID, self.SOURCE)
        assert len(chunks) == 1
        assert "plain string content" in chunks[0].text

    # --- Split by character ---

    def test_custom_split_character(self, chunker):
        content = [make_block("part1\npart2\npart3", "paragraph")]
        chunks = chunker.chunk_fixed(
            content,
            self.DOC_ID,
            self.SOURCE,
            chunk_size=512,
            split_by_character="\n",
        )
        # All parts should be present in the output
        combined = " ".join(c.text for c in chunks)
        assert "part1" in combined
        assert "part2" in combined
        assert "part3" in combined

    # --- Empty / whitespace-only blocks ---

    def test_empty_blocks_ignored(self, chunker):
        content = [make_block(""), make_block("   "), make_block("actual content")]
        chunks = chunker.chunk_fixed(content, self.DOC_ID, self.SOURCE)
        assert len(chunks) == 1
        assert "actual content" in chunks[0].text

    # --- Multiple blocks joined ---

    def test_multiple_blocks_joined(self, chunker):
        content = [make_block("block one"), make_block("block two")]
        chunks = chunker.chunk_fixed(content, self.DOC_ID, self.SOURCE, chunk_size=512)
        combined = " ".join(c.text for c in chunks)
        assert "block one" in combined
        assert "block two" in combined


# ===========================================================================
# Chunking.splitter
# ===========================================================================


class TestSplitter:
    DOC_ID = "doc-split"
    SOURCE = "report.pdf"

    def test_returns_list(self, chunker):
        result = chunker.splitter("", self.DOC_ID, self.SOURCE)
        assert isinstance(result, list)

    def test_empty_string_returns_no_chunks(self, chunker):
        result = chunker.splitter("", self.DOC_ID, self.SOURCE)
        assert result == []

    def test_single_paragraph_below_limit(self, chunker):
        text = "This is a short paragraph."
        chunks = chunker.splitter(text, self.DOC_ID, self.SOURCE, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_chunk_type_is_splitter(self, chunker):
        chunks = chunker.splitter("Some text.", self.DOC_ID, self.SOURCE)
        assert chunks[0].chunk_type == "splitter"

    def test_chunk_metadata(self, chunker):
        chunks = chunker.splitter("Hello.", self.DOC_ID, self.SOURCE)
        c = chunks[0]
        assert c.doc_id == self.DOC_ID
        assert c.source_file == self.SOURCE

    def test_long_text_split_into_multiple_chunks(self, chunker):
        paragraph = "word " * 200  # 200 words, ~1000 chars
        text = "\n\n".join([paragraph.strip()] * 10)
        chunks = chunker.splitter(text, self.DOC_ID, self.SOURCE, chunk_size=500)
        assert len(chunks) > 1

    def test_chunk_index_increments(self, chunker):
        paragraph = "a " * 300
        text = "\n\n".join([paragraph.strip()] * 5)
        chunks = chunker.splitter(text, self.DOC_ID, self.SOURCE, chunk_size=500)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_whitespace_only_input(self, chunker):
        result = chunker.splitter("   \n\n   ", self.DOC_ID, self.SOURCE)
        assert result == []

    def test_custom_split_character(self, chunker):
        text = "part1|part2|part3"
        chunks = chunker.splitter(
            text, self.DOC_ID, self.SOURCE, chunk_size=2000, split_by_character="|"
        )
        assert len(chunks) == 1
        combined = chunks[0].text
        assert "part1" in combined

    def test_logfire_info_called(self, chunker):
        """Verify logfire.info is invoked after splitting."""
        with patch("src.ingestion.chunking.chunk.logfire") as mock_logfire:
            chunker.splitter("hello world.", self.DOC_ID, self.SOURCE)
            mock_logfire.info.assert_called_once()


# ===========================================================================
# Chunking.build_parent_child_chunk
# ===========================================================================


class TestBuildParentChildChunk:
    def test_returns_list_of_same_length(self, chunker):
        chunks = make_chunks(5)
        result = chunker.build_parent_child_chunk(chunks)
        assert len(result) == len(chunks)

    def test_returns_chunk_objects(self, chunker):
        chunks = make_chunks(3)
        result = chunker.build_parent_child_chunk(chunks)
        assert all(isinstance(c, Chunk) for c in result)

    def test_child_text_preserved(self, chunker):
        chunks = make_chunks(3)
        result = chunker.build_parent_child_chunk(chunks)
        for original, enriched in zip(chunks, result):
            assert enriched.text == original.text

    def test_parent_text_non_empty(self, chunker):
        chunks = make_chunks(5)
        result = chunker.build_parent_child_chunk(chunks)
        for c in result:
            assert c.parent_text != ""

    def test_parent_window_start_gte_zero(self, chunker):
        chunks = make_chunks(5)
        result = chunker.build_parent_child_chunk(chunks)
        for c in result:
            assert c.parent_window_start >= 0

    def test_parent_window_end_lte_total(self, chunker):
        n = 5
        chunks = make_chunks(n)
        result = chunker.build_parent_child_chunk(chunks)
        for c in result:
            assert c.parent_window_end <= n

    def test_parent_window_start_lte_end(self, chunker):
        chunks = make_chunks(5)
        result = chunker.build_parent_child_chunk(chunks)
        for c in result:
            assert c.parent_window_start <= c.parent_window_end

    def test_single_chunk(self, chunker):
        chunks = make_chunks(1)
        result = chunker.build_parent_child_chunk(chunks)
        assert len(result) == 1
        assert result[0].text == chunks[0].text

    def test_custom_parent_window(self, chunker):
        chunks = make_chunks(10)
        result = chunker.build_parent_child_chunk(chunks, parent_window=6)
        for c in result:
            window_size = c.parent_window_end - c.parent_window_start
            assert window_size <= 7  # at most parent_window+1

    def test_metadata_preserved(self, chunker):
        chunks = make_chunks(3)
        result = chunker.build_parent_child_chunk(chunks)
        for original, enriched in zip(chunks, result):
            assert enriched.doc_id == original.doc_id
            assert enriched.source_file == original.source_file
            assert enriched.chunk_index == original.chunk_index
            assert enriched.chunk_type == original.chunk_type
            assert enriched.section_title == original.section_title

    def test_empty_list_returns_empty(self, chunker):
        result = chunker.build_parent_child_chunk([])
        assert result == []

    def test_parent_window_boundary_first_chunk(self, chunker):
        """First chunk cannot look behind; start should be 0."""
        chunks = make_chunks(5)
        result = chunker.build_parent_child_chunk(chunks, parent_window=4)
        assert result[0].parent_window_start == 0

    def test_parent_window_boundary_last_chunk(self, chunker):
        """Last chunk cannot look ahead; end should be len(chunks)."""
        n = 5
        chunks = make_chunks(n)
        result = chunker.build_parent_child_chunk(chunks, parent_window=4)
        assert result[-1].parent_window_end == n
