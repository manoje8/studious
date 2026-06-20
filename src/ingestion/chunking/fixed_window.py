from typing import Callable, List, Optional

from src.ingestion.chunking.Chunker import Chunker
from src.ingestion.chunking.chunk import Chunk


class FixedWindow(Chunker):
    def __init__(
        self,
        size: int = 512,
        overlap: int = 64,
        token_len_fn: Optional[Callable[[str], int]] = None,
    ) -> None:
        if overlap >= size:
            raise ValueError(
                "Overlap tokens must be smaller than the chunk size tokens"
            )
        self.chunk_size = size
        self.overlap = overlap
        self.token_len_fn = token_len_fn or (lambda s: len(s.split()))

    def chunk(
        self, text: str, transform: Callable[[str], str] | None = None, **kwargs
    ) -> List[Chunk]:
        doc_id: str = kwargs.get("doc_id", "")
        source_file: str = kwargs.get("source_file", "")

        words = text.split()
        if not words:
            return []

        offsets = []
        cursor = 0

        for w in words:
            start = text.index(w, cursor)
            end = start + len(w)
            offsets.append((start, end))
            cursor = end

        chunk: list[Chunk] = []
        step = self.chunk_size - self.overlap
        idx = 0
        i = 0
        while i < len(words):
            window_words = words[i : i + self.chunk_size]
            if not window_words:
                break

            char_start = offsets[i][0]
            char_end = offsets[min(i + len(window_words) - 1, len(offsets) - 1)][1]
            content = text[char_start:char_end]

            chunk.append(
                Chunk(
                    text=content,
                    chunk_index=idx,
                    doc_id=doc_id,
                    source_file=source_file,
                    chunk_type="text",
                )
            )

            idx += 1
            i += step

        return chunk
