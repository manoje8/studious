from dataclasses import dataclass, field


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
    @staticmethod
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
            block_type = block.get("type", "paragraph")
            text = block.get("text", "").strip()

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

    @staticmethod
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
        full_text = split_by_character.join(
            block.get("text", "") for block in content_list if block.get("text")
        )

        segments = full_text.split(split_by_character)

        chunks = []
        current_tokens = []
        chunk_index = 0

        for segment in segments:
            words = segment.split()

            if len(current_tokens) + len(words) > chunk_size:
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
                    current_tokens = current_tokens[-overlap:]

            current_tokens.extend(words)

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

        return chunks
