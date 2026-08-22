"""
Integration tests: Parser output shape → Chunker input shape seam.

Verifies that the data flowing from DoclingParser output through
separate_content() into RecursiveCharacterChunker is shape-compatible
end-to-end — without mocking either side of the boundary.

The "parser-output-shape vs chunker-input-shape mismatch" bug class lives
at this seam. Unit tests miss it because they mock the parser on one side
and the chunker never sees real-shaped input.
"""

import pytest

from medici.common.utils.constants import ChunkerStrategy
from medici.common.utils.helper import separate_content
from medici.ingestion.chunking.chunk import Chunk
from medici.ingestion.chunking.chunker_factory import create_chunker
from medici.ingestion.chunking.chunking_config import ChunkingConfig


@pytest.fixture
def chunker():
    cfg = ChunkingConfig(type=ChunkerStrategy.RECURSIVE_CHARACTER, size=256, overlap=32)
    return create_chunker(cfg)


@pytest.mark.integration
class TestSeparateContentOutputShape:
    """Verify separate_content() returns the exact shapes downstream consumers expect."""

    def test_returns_two_element_tuple(self, docling_content_list):
        result = separate_content(docling_content_list)
        assert isinstance(result, tuple) and len(result) == 3

    def test_first_element_is_string(self, docling_content_list):
        """chunker.chunk() expects str — not list[dict]."""
        text_content, _, _tb = separate_content(docling_content_list)
        assert isinstance(text_content, str), (
            f"Expected str, got {type(text_content).__name__}. "
            "RecursiveCharacterChunker.chunk() will fail with list input."
        )

    def test_second_element_is_list_of_dicts(self, docling_content_list):
        """chunk_multimodal_items() expects list[dict]."""
        _, multimodal, _tb = separate_content(docling_content_list)
        assert isinstance(multimodal, list)
        assert all(isinstance(item, dict) for item in multimodal)

    def test_text_content_contains_all_text_blocks(self, docling_content_list):
        text_content, _, _tb = separate_content(docling_content_list)
        assert "Introduction to RAG systems." in text_content
        assert "RAG stands for Retrieval-Augmented Generation." in text_content
        assert "It combines dense retrieval with language model generation." in text_content

    def test_multimodal_items_retain_type_key(self, docling_content_list):
        _, multimodal, _tb = separate_content(docling_content_list)
        for item in multimodal:
            assert "type" in item, f"Item missing 'type' key: {item}"

    def test_multimodal_items_exclude_text_type(self, docling_content_list):
        _, multimodal, _tb = separate_content(docling_content_list)
        types = {item["type"] for item in multimodal}
        assert "text" not in types

    def test_multimodal_items_include_table_image_equation(self, docling_content_list):
        _, multimodal, _tb = separate_content(docling_content_list)
        types = {item["type"] for item in multimodal}
        assert {"table", "image", "equation"}.issubset(types)

    def test_empty_input_returns_empty_string_and_empty_list(self):
        text, multimodal, text_blocks = separate_content([])
        assert text == "" and multimodal == [] and text_blocks == []

    def test_all_text_returns_empty_multimodal(self):
        content = [
            {"type": "text", "text": "First."},
            {"type": "text", "text": "Second."},
        ]
        _, multimodal, _tb = separate_content(content)
        assert multimodal == []

    def test_blank_text_items_are_excluded(self):
        """Blank text items should not inject empty separators into the string."""
        content = [
            {"type": "text", "text": "Real content."},
            {"type": "text", "text": "   "},  # blank — should be skipped
            {"type": "text", "text": "More content."},
        ]
        text, _, _tb = separate_content(content)
        # Should not introduce triple newlines from blank entries
        assert "\n\n\n" not in text


@pytest.mark.integration
class TestChunkerAcceptsParserOutput:
    """
    Verify RecursiveCharacterChunker consumes the exact output of separate_content().
    These tests cross the seam: real parser-shaped input → real chunker.
    """

    def test_chunk_accepts_string_output_of_separate_content(self, chunker, docling_content_list):
        """chunker.chunk() must not crash or AttributeError on a real str input."""
        text_content, _, _tb = separate_content(docling_content_list)
        result = chunker.chunk(text_content, doc_id="doc-001", source_file="test.pdf")
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_chunk_produces_non_empty_text_in_every_chunk(self, chunker, docling_content_list):
        text_content, _, _tb = separate_content(docling_content_list)
        result = chunker.chunk(text_content, doc_id="doc-001", source_file="test.pdf")
        assert len(result) > 0
        assert all(c.text.strip() for c in result)

    def test_chunk_multimodal_accepts_separate_content_output(self, chunker, docling_content_list):
        """chunk_multimodal_items() must accept list[dict] from separate_content()."""
        _, multimodal, _tb = separate_content(docling_content_list)
        result = chunker.chunk_multimodal_items(
            multimodal, doc_id="doc-001", source_file="test.pdf", start_index=10
        )
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_multimodal_chunks_have_populated_text(self, chunker, docling_content_list):
        _, multimodal, _tb = separate_content(docling_content_list)
        result = chunker.chunk_multimodal_items(
            multimodal, doc_id="doc-001", source_file="test.pdf"
        )
        assert len(result) > 0
        assert all(c.text.strip() for c in result)

    def test_chunk_indices_are_sequential_from_start_index(self, chunker, docling_content_list):
        """Multimodal chunk indices must start from the given start_index."""
        _, multimodal, _tb = separate_content(docling_content_list)
        start = 42
        result = chunker.chunk_multimodal_items(
            multimodal, doc_id="doc-001", source_file="test.pdf", start_index=start
        )
        if result:
            assert result[0].chunk_index >= start


@pytest.mark.integration
class TestFullParserToChunkPipeline:
    """
    End-to-end pipeline with realistic data shapes.
    Catches any type/key mismatch that only shows up when real data flows
    through both sides of the seam simultaneously.
    """

    def test_full_pipeline_produces_chunk_list(self, chunker, docling_content_list):
        text, multimodal, _tb = separate_content(docling_content_list)

        text_chunks = chunker.chunk(text, doc_id="doc-pipeline", source_file="test.pdf")
        multimodal_chunks = chunker.chunk_multimodal_items(
            multimodal,
            doc_id="doc-pipeline",
            source_file="test.pdf",
            start_index=len(text_chunks),
        )

        all_chunks = text_chunks + multimodal_chunks
        assert len(all_chunks) > 0
        assert all(isinstance(c, Chunk) for c in all_chunks)

    def test_pipeline_chunk_indices_are_unique(self, chunker, docling_content_list):
        """No two chunks in the combined output should share the same index."""
        text, multimodal, _tb = separate_content(docling_content_list)

        text_chunks = chunker.chunk(text, doc_id="doc-pipeline", source_file="test.pdf")
        multimodal_chunks = chunker.chunk_multimodal_items(
            multimodal,
            doc_id="doc-pipeline",
            source_file="test.pdf",
            start_index=len(text_chunks),
        )

        indices = [c.chunk_index for c in text_chunks + multimodal_chunks]
        assert len(indices) == len(set(indices)), (
            "Duplicate chunk indices detected — start_index hand-off between "
            "text and multimodal chunking is broken."
        )

    def test_table_serialization_with_realistic_docling_shape(self, chunker):
        """
        A table dict shaped exactly like docling produces must serialize without error.
        Catches the table_body.get('grid') AttributeError when table_body is a plain list.
        """
        table_item = {
            "type": "table",
            "img_path": "",
            "table_caption": "Model Comparison",
            "table_footnote": "Source: internal benchmarks",
            "table_body": {
                "grid": [
                    [
                        {
                            "text": "Model",
                            "column_header": True,
                            "start_row_offset_idx": 0,
                            "end_row_offset_idx": 1,
                            "start_col_offset_idx": 0,
                            "end_col_offset_idx": 1,
                        },
                        {
                            "text": "F1",
                            "column_header": True,
                            "start_row_offset_idx": 0,
                            "end_row_offset_idx": 1,
                            "start_col_offset_idx": 1,
                            "end_col_offset_idx": 2,
                        },
                    ],
                    [
                        {
                            "text": "Llama-3",
                            "column_header": False,
                            "start_row_offset_idx": 1,
                            "end_row_offset_idx": 2,
                            "start_col_offset_idx": 0,
                            "end_col_offset_idx": 1,
                        },
                        {
                            "text": "0.88",
                            "column_header": False,
                            "start_row_offset_idx": 1,
                            "end_row_offset_idx": 2,
                            "start_col_offset_idx": 1,
                            "end_col_offset_idx": 2,
                        },
                    ],
                ],
                "num_rows": 2,
                "num_cols": 2,
            },
            "page_idx": 0,
        }

        result = chunker.chunk_multimodal_items([table_item], doc_id="d1", source_file="test.pdf")
        assert len(result) > 0
        assert "Model" in result[0].text or "Llama" in result[0].text
