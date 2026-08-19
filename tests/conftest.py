"""
Root-level pytest conftest.py — shared fixtures for unit and integration tests.

Provides realistic data shapes that cross component boundaries so tests can
verify the actual seam contracts rather than mocking both sides independently.
"""

import pytest

from medici.ingestion.chunking.chunk import Chunk


@pytest.fixture
def docling_content_list():
    """
    A list[dict] mirroring the exact shape DoclingParser._read_from_block()
    produces. Use in tests that exercise the parser → chunker seam without a
    real Docling installation.

    Shape contract:
        - Every item has a "type" key
        - "text" items have a "text" key (str) and "page_idx" (int)
        - "table" items have "table_body" (dict with "grid"), "table_caption",
          "table_footnote", "img_path", "page_idx"
        - "image" items have "img_path", "image_caption", "page_idx"
        - "equation" items have "text", "img_path", "text_format", "page_idx"
    """
    return [
        {"type": "text", "text": "Introduction to RAG systems.", "page_idx": 0},
        {
            "type": "text",
            "text": "RAG stands for Retrieval-Augmented Generation.",
            "page_idx": 0,
        },
        {
            "type": "text",
            "text": "It combines dense retrieval with language model generation.",
            "page_idx": 1,
        },
        {
            "type": "table",
            "img_path": "",
            "table_caption": "Comparison of retrieval methods",
            "table_footnote": "",
            "table_body": {
                "grid": [
                    [
                        {
                            "text": "Method",
                            "column_header": True,
                            "start_row_offset_idx": 0,
                            "end_row_offset_idx": 1,
                            "start_col_offset_idx": 0,
                            "end_col_offset_idx": 1,
                        },
                        {
                            "text": "Accuracy",
                            "column_header": True,
                            "start_row_offset_idx": 0,
                            "end_row_offset_idx": 1,
                            "start_col_offset_idx": 1,
                            "end_col_offset_idx": 2,
                        },
                    ],
                    [
                        {
                            "text": "BM25",
                            "column_header": False,
                            "start_row_offset_idx": 1,
                            "end_row_offset_idx": 2,
                            "start_col_offset_idx": 0,
                            "end_col_offset_idx": 1,
                        },
                        {
                            "text": "0.75",
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
            "page_idx": 1,
        },
        {
            "type": "image",
            "img_path": "/tmp/test_image.png",
            "image_caption": "Architecture diagram of the RAG pipeline",
            "image_footnote": "",
            "page_idx": 2,
        },
        {
            "type": "equation",
            "img_path": "",
            "text": "P(answer | query, context) = softmax(W * h)",
            "text_format": "unknown",
            "page_idx": 2,
        },
    ]


@pytest.fixture
def scroll_chunk_dicts():
    """
    A list[dict] mirroring the exact shape QdrantStorageService.scroll_all_chunks()
    produces. Use in tests that exercise the storage → sparse_index seam.

    Shape contract: keys are text, doc_id, chunk_index, section_title, source_file.
    """
    return [
        {
            "text": "Retrieval-Augmented Generation combines dense retrieval with LLMs.",
            "doc_id": "doc-001",
            "chunk_index": 0,
            "section_title": "Introduction",
            "source_file": "rag_overview.pdf",
        },
        {
            "text": "Vector databases store high-dimensional embeddings for semantic search.",
            "doc_id": "doc-001",
            "chunk_index": 1,
            "section_title": "Vector Databases",
            "source_file": "rag_overview.pdf",
        },
        {
            "text": "BM25 is a sparse retrieval algorithm based on term frequency.",
            "doc_id": "doc-002",
            "chunk_index": 0,
            "section_title": "Sparse Retrieval",
            "source_file": "retrieval_methods.pdf",
        },
        {
            "text": "Hybrid search combines dense and sparse retrieval for better results.",
            "doc_id": "doc-002",
            "chunk_index": 1,
            "section_title": "Hybrid Methods",
            "source_file": "retrieval_methods.pdf",
        },
    ]


@pytest.fixture
def make_chunk():
    """Factory fixture for Chunk dataclass instances."""

    def _make(
        text: str = "sample chunk text",
        doc_id: str = "doc-001",
        chunk_index: int = 0,
        source_file: str = "test.pdf",
        chunk_type: str = "text",
        section_title: str = "",
        token_count: int = 10,
    ) -> Chunk:
        return Chunk(
            text=text,
            doc_id=doc_id,
            chunk_index=chunk_index,
            source_file=source_file,
            chunk_type=chunk_type,
            section_title=section_title,
            token_count=token_count,
        )

    return _make
