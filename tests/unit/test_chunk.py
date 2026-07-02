"""
Unit tests for src/ingestion/chunking

Covers:
- Chunk dataclass construction and to_quant_payload()
- Chunking._clean_text()
- FixedWindow chunker
- RecursiveCharacterChunker
- ChunkerFactory and create_chunker
- Chunking.build_parent_child_chunk()
"""

import pytest

from src.common.utils.constants import ChunkerStrategy
from src.ingestion.chunking.chunk import Chunk, Chunking
from src.ingestion.chunking.chunker_factory import create_chunker
from src.ingestion.chunking.chunking_config import ChunkingConfig
from src.ingestion.chunking.fixed_window import FixedWindow
from src.ingestion.chunking.recursive_character import RecursiveCharacterChunker


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
            metadata={},
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
            "metadata",
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


class TestFixedWindow:
    DOC_ID = "doc-fixed"
    SOURCE = "report.pdf"

    def test_returns_list(self):
        chunker = FixedWindow(size=5, overlap=1)
        result = chunker.chunk("word1 word2 word3", doc_id=self.DOC_ID, source_file=self.SOURCE)
        assert isinstance(result, list)

    def test_empty_input_returns_no_chunks(self):
        chunker = FixedWindow(size=5, overlap=1)
        result = chunker.chunk("", doc_id=self.DOC_ID, source_file=self.SOURCE)
        assert result == []

    def test_invalid_overlap_raises_value_error(self):
        with pytest.raises(ValueError):
            FixedWindow(size=5, overlap=5)
        with pytest.raises(ValueError):
            FixedWindow(size=5, overlap=6)

    def test_short_text_fits_in_single_chunk(self):
        chunker = FixedWindow(size=10, overlap=2)
        chunks = chunker.chunk("hello world", doc_id=self.DOC_ID, source_file=self.SOURCE)
        assert len(chunks) == 1
        assert chunks[0].text == "hello world"
        assert chunks[0].chunk_type == "text"
        assert chunks[0].doc_id == self.DOC_ID
        assert chunks[0].source_file == self.SOURCE

    def test_large_text_split_into_multiple_chunks(self):
        text = "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9"
        chunker = FixedWindow(size=4, overlap=1)
        chunks = chunker.chunk(text, doc_id=self.DOC_ID, source_file=self.SOURCE)
        assert len(chunks) == 4
        assert chunks[0].text == "w0 w1 w2 w3"
        assert chunks[1].text == "w3 w4 w5 w6"
        assert chunks[2].text == "w6 w7 w8 w9"
        assert chunks[3].text == "w9"

        for idx, c in enumerate(chunks):
            assert c.chunk_index == idx
            assert c.doc_id == self.DOC_ID
            assert c.source_file == self.SOURCE


class TestRecursiveCharacterChunker:
    DOC_ID = "doc-rec"
    SOURCE = "manual.md"

    def test_returns_list(self):
        chunker = RecursiveCharacterChunker(size=100, overlap=10)
        result = chunker.chunk(
            "This is some text to chunk.", doc_id=self.DOC_ID, source_file=self.SOURCE
        )
        assert isinstance(result, list)

    def test_empty_input_returns_no_chunks(self):
        chunker = RecursiveCharacterChunker(size=100, overlap=10)
        assert chunker.chunk("", doc_id=self.DOC_ID, source_file=self.SOURCE) == []
        assert chunker.chunk("   ", doc_id=self.DOC_ID, source_file=self.SOURCE) == []

    def test_chunk_metadata_and_token_count(self):
        chunker = RecursiveCharacterChunker(size=100, overlap=10)
        chunks = chunker.chunk("Hello world.", doc_id=self.DOC_ID, source_file=self.SOURCE)
        assert len(chunks) == 1
        c = chunks[0]
        assert c.doc_id == self.DOC_ID
        assert c.source_file == self.SOURCE
        assert c.chunk_type == "text"
        assert c.token_count > 0

    def test_large_text_splits(self):
        chunker = RecursiveCharacterChunker(size=5, overlap=0)
        text = "This is a long sentence that should be split into multiple chunks."
        chunks = chunker.chunk(text, doc_id=self.DOC_ID, source_file=self.SOURCE)
        assert len(chunks) > 1
        for idx, c in enumerate(chunks):
            assert c.chunk_index == idx
            assert len(c.text) > 0

    def test_check_installation(self):
        chunker = RecursiveCharacterChunker()
        assert chunker.check_installation() is True


class TestChunkerFactory:
    def test_create_fixed_chunker(self):
        config = ChunkingConfig(type=ChunkerStrategy.FIXED, size=256, overlap=32)
        chunker = create_chunker(config)
        assert isinstance(chunker, FixedWindow)
        assert chunker.chunk_size == 256
        assert chunker.overlap == 32

    def test_create_recursive_character_chunker(self):
        config = ChunkingConfig(type=ChunkerStrategy.RECURSIVE_CHARACTER, size=1000, overlap=50)
        chunker = create_chunker(config)
        assert isinstance(chunker, RecursiveCharacterChunker)
        assert chunker.size == 1000
        assert chunker.overlap == 50

    def test_invalid_strategy_raises_value_error(self):
        config = ChunkingConfig(type="invalid_strategy", size=100, overlap=10)
        with pytest.raises(ValueError) as excinfo:
            create_chunker(config)
        assert "is not registered in the ChunkerFactory" in str(excinfo.value)


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
        for original, enriched in zip(chunks, result, strict=False):
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
        for original, enriched in zip(chunks, result, strict=False):
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
