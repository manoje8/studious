import re
from dataclasses import dataclass, field

from src.common.utils.tokenizer import TikTokenTokenizer, Tokenizer


@dataclass
class Chunk:
    """
    A single text chunk produced by the chunking pipeline.

    Attributes
    ----------
    text:
        Raw text content of the chunk.
    chunk_index:
        Zero-based position within the source document.
    doc_id:
        Unique identifier of the parent document.
    source_file:
        Path to the originating file.
    chunk_type:
        Strategy that produced this chunk (``"structure"`` or ``"fixed"``).
    section_title:
        Nearest ancestor heading text (empty when unavailable).
    page_numbers:
        Sorted, deduplicated list of contributing page numbers.
    block_types:
        Ordered list of content-block types merged into this chunk.
    token_count:
        Approximate token count of *text*.
    """

    text: str
    chunk_index: int
    doc_id: str
    source_file: str
    chunk_type: str
    section_title: str = ""
    page_numbers: list[int] = field(default_factory=list)
    block_types: list[str] = field(default_factory=list)
    token_count: int = 0
    parent_text: str = ""
    parent_token_count: int = 0
    parent_window_start: int = 0
    parent_window_end: int = 0
    metadata: dict = field(default_factory=dict)

    def to_quant_payload(self) -> dict:
        return {
            "text": self.text,
            "chunk_index": self.chunk_index,
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "chunk_type": self.chunk_type,
            "section_title": self.section_title,
            "page_numbers": self.page_numbers,
            "token_count": self.token_count,
        }


class Chunking:
    def __init__(self, tokenizer: Tokenizer = None):
        self.tokenizer = tokenizer or TikTokenTokenizer(model_name="gpt-4o-mini")

    @staticmethod
    def _clean_text(text: str) -> str:
        """Fix common OCR artifacts such as space-separated characters."""
        cleaned = re.sub(
            r"\b(?:(?<=\s)|(?<=^))([A-Za-z\u2013\u2014\u2018\u2019\u201c\u201d]"
            r"(?: [A-Za-z\u2013\u2014\u2018\u2019\u201c\u201d]){4,})\b",
            lambda m: m.group(0).replace(" ", ""),
            text,
        )
        cleaned = re.sub(r" {2,}", " ", cleaned)
        return cleaned.strip()

    def _make_chunk(
        self,
        text: str,
        chunk_index: int,
        doc_id: str,
        source_file: str,
        chunk_type: str,
        section_title: str = "",
        block_types: list[str] = None,
        page_numbers: list[int] = None,
    ) -> Chunk:
        """Helper that constructs a Chunk and populates token_count immediately."""
        return Chunk(
            text=text,
            chunk_index=chunk_index,
            doc_id=doc_id,
            source_file=source_file,
            chunk_type=chunk_type,
            section_title=section_title,
            block_types=block_types or [],
            page_numbers=page_numbers or [],
            token_count=self.tokenizer.count(text),
        )

    def build_parent_child_chunk(self, chunks: list[Chunk], parent_window: int = 3) -> list[Chunk]:
        """
        Enrich each child chunk with a broader parent context window.

        Each chunk gains a ``parent_text`` field containing the concatenated
        text of itself and its neighboring chunks. The ``parent_token_count``
        field is also populated for budget-aware retrieval.

        Parameters
        ----------
        chunks:
            List of Chunk objects to enrich.
        parent_window:
            Total number of chunks to include in the parent window
            (centered on the current chunk).
        """

        enriched = []
        for i, chunk in enumerate(chunks):
            start = max(0, i - parent_window // 2)
            end = min(len(chunks), i + parent_window // 2 + 1)

            parent_text = " ".join(c.text for c in chunks[start:end])
            parent_token_count = self.tokenizer.count(parent_text)

            enriched.append(
                Chunk(
                    text=chunk.text,
                    chunk_index=chunk.chunk_index,
                    doc_id=chunk.doc_id,
                    source_file=chunk.source_file,
                    chunk_type=chunk.chunk_type,
                    section_title=chunk.section_title,
                    page_numbers=chunk.page_numbers,
                    block_types=chunk.block_types,
                    token_count=chunk.token_count,
                    parent_text=parent_text,
                    parent_token_count=parent_token_count,
                    parent_window_start=start,
                    parent_window_end=end,
                )
            )

        return enriched
