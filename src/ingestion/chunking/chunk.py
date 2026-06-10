import re
from dataclasses import dataclass, field
import logfire


@dataclass
class Chunk:
    """A single text chunk produced by the chunking pipeline.

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
    parent_window_start: int = 0
    parent_window_end: int = 0

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
    # TODO: Need to add more chunking process

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

    def chunk_by_structure(
        self,
        content_list: list[dict],
        doc_id: str,
        source_file: str,
        max_token: int = 512,
    ) -> list[Chunk]:
        chunks = []
        current_blocks = []
        current_title = "Introduction"
        chunk_index = 0

        for block in content_list:
            if isinstance(block, dict):
                block_type = block.get("type", "paragraph")
                text = self._clean_text(block.get("text", "").strip())
            else:
                block_type = "paragraph"
                text = self._clean_text(str(block).strip())
                block = {"type": block_type, "text": text}

            if not text:
                continue

            if block_type == "heading":
                if current_blocks:
                    chunk_text = " ".join(b.get("text", "") for b in current_blocks)
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            chunk_index=chunk_index,
                            doc_id=doc_id,
                            source_file=source_file,
                            chunk_type="structure",
                            section_title=current_title,
                            block_types=[b.get("type") for b in current_blocks],
                        )
                    )
                    chunk_index += 1
                    current_blocks = []

                current_title = text

            elif block_type == "table":
                if current_blocks:
                    chunk_text = " ".join(b.get("text", "") for b in current_blocks)
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            chunk_index=chunk_index,
                            doc_id=doc_id,
                            source_file=source_file,
                            chunk_type="structure",
                            section_title=current_title,
                            block_types=[b.get("type") for b in current_blocks],
                        )
                    )
                    chunk_index += 1
                    current_blocks = []

                chunks.append(
                    Chunk(
                        text=text,
                        chunk_index=chunk_index,
                        doc_id=doc_id,
                        source_file=source_file,
                        chunk_type="table",
                        section_title=current_title,
                        block_types=["table"],
                    )
                )
                chunk_index += 1

            else:
                current_blocks.append(block)

        # Flush whatever remains
        if current_blocks:
            chunk_text = " ".join(b.get("text", "") for b in current_blocks)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    doc_id=doc_id,
                    source_file=source_file,
                    chunk_type="structure",
                    section_title=current_title,
                    block_types=[b.get("type") for b in current_blocks],
                )
            )

        return chunks

    def chunk_fixed(
        self,
        content_list: list[dict],
        doc_id: str,
        source_file: str,
        chunk_size: int = 512,
        max_token: int = 512,
        overlap: int = 50,
        split_by_character: str = "\n\n",
    ) -> list[Chunk]:
        texts = []
        for block in content_list:
            if isinstance(block, dict):
                text = self._clean_text(block.get("text", ""))
            else:
                text = self._clean_text(str(block))

            if text:
                texts.append(text)

        full_text = split_by_character.join(texts)

        segments = full_text.split(split_by_character)

        chunks = []
        current_tokens = []
        chunk_index = 0

        def flush():
            nonlocal chunk_index
            if current_tokens:
                chunks.append(
                    Chunk(
                        text=" ".join(current_tokens),
                        chunk_index=chunk_index,
                        doc_id=doc_id,
                        source_file=source_file,
                        chunk_type="fixed",
                    )
                )
                chunk_index += 1

        for segment in segments:
            words = segment.split()

            while len(words) > chunk_size:
                available = chunk_size - len(current_tokens)
                current_tokens.extend(words[:available])
                words = words[available:]
                flush()
                current_tokens = current_tokens[-overlap:]

            if len(current_tokens) + len(words) > chunk_size:
                flush()
                current_tokens = current_tokens[-overlap:]

            current_tokens.extend(words)

        flush()
        return chunks

    def splitter(
        self,
        content_list,
        doc_id: str,
        source_file: str,
        chunk_size: int = 1500,
        split_by_character: str = "\n\n",
    ):
        paragraphs = content_list.split(split_by_character)
        chunks = []
        current_chunk = ""
        chunk_index = 0

        for p in paragraphs:
            if len(current_chunk) + len(p) < chunk_size:
                current_chunk += p + split_by_character
            else:
                if current_chunk.strip():
                    chunks.append(
                        Chunk(
                            text="".join(current_chunk.strip()),
                            chunk_index=chunk_index,
                            doc_id=doc_id,
                            source_file=source_file,
                            chunk_type="splitter",
                        )
                    )
                    chunk_index += 1
                current_chunk = p + split_by_character

        if current_chunk.strip():
            chunks.append(
                Chunk(
                    text="".join(current_chunk.strip()),
                    chunk_index=chunk_index,
                    doc_id=doc_id,
                    source_file=source_file,
                    chunk_type="splitter",
                )
            )

        logfire.info(f"Generated {len(chunks)} chunks")

        return chunks

    def build_parent_child_chunk(
        self, chunks: list[dict], parent_window: int = 3
    ) -> list[dict]:
        """
        Each chunk gets a parent_id pointing to a broader context window.
        Retrieve by child, re-rank or expand using parent at query time.

        :param chunks:
        :param parent_window:
        :return:
        """

        enriched = []
        for i, chunk in enumerate(chunks):
            start = max(0, i - parent_window // 2)
            end = min(len(chunks), i + parent_window // 2 + 1)

            parent_text = " ".join(c.get("text", "") for c in chunks[start:end])

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
                    parent_window_start=start,
                    parent_window_end=end,
                )
            )

        return enriched
