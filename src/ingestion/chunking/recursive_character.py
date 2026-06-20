import logfire
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.chunking.Chunker import Chunker
from src.ingestion.chunking.chunk import Chunk
from src.utils.tokenizer import Tokenizer, TikTokenTokenizer


class RecursiveCharacterChunker(Chunker):
    """
    For articles, books, reports
    """

    def __init__(
        self, size: int = 1200, overlap: int = 100, tokenizer: Tokenizer = None
    ):
        self.size = size
        self.overlap = overlap
        self.tokenizer = tokenizer or TikTokenTokenizer(model_name="gpt-4o-mini")

    def check_installation(self):
        try:
            from langchain_text_splitters import (  # noqa: F401
                RecursiveCharacterTextSplitter,
            )

            return True
        except ImportError:
            logfire.error(
                "langchain_text_splitters is not installed. Install it with: pip install langchain-text-splitters"
            )
            return False

    def chunk(
        self,
        text: str,
        separator: list[str] | None = None,
        **kwargs,
    ) -> list[Chunk]:
        doc_id: str = kwargs.get("doc_id", "")
        source_file: str = kwargs.get("source_file", "")

        if not text or not text.strip():
            return []

        splitter_kwargs = {
            "chunk_size": max(int(self.size), 1),
            "chunk_overlap": max(int(self.overlap), 0),
            "length_function": lambda s: len(self.tokenizer.encode(s)),
        }

        if separator is not None:
            splitter_kwargs["separators"] = list(separator)

        splitter = RecursiveCharacterTextSplitter(**splitter_kwargs)

        pieces = splitter.split_text(text)

        results: list[Chunk] = []

        for idx, piece in enumerate(pieces):
            body = piece.strip()
            if not body:
                continue
            results.append(
                Chunk(
                    text=body,
                    chunk_index=idx,
                    doc_id=doc_id,
                    source_file=source_file,
                    chunk_type="text",
                    token_count=len(self.tokenizer.encode(body)),
                )
            )

        if not results:
            logfire.warn(f"Splitter produced no non-empty chunks: {len(text)}")

            body = text.strip()
            if body:
                results.append(
                    Chunk(
                        text=body,
                        chunk_index=0,
                        doc_id=doc_id,
                        source_file=source_file,
                        chunk_type="text",
                        token_count=len(self.tokenizer.encode(body)),
                    )
                )

        return results
