from typing import Any

import logfire
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.chunking.Chunker import Chunker
from src.utils.tokenizer import Tokenizer, TikTokenTokenizer


class RecursiveCharacterChunker(Chunker):
    """
    For articles, books, reports
    """

    def __init__(self, tokenizer: Tokenizer = None):
        self.tokenizer = tokenizer or TikTokenTokenizer(model_name="gpt-4o-mini")

    def check_installation(self):
        try:
            from langchain.text_splitter import (  # noqa: F401
                RecursiveCharacterTextSplitter,
            )

            return True
        except ImportError:
            logfire.error(
                "langchain is not installed. Install it with: pip install langchain"
            )
            return False

    def chunk(
        self,
        content: str,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        separator: list[str] | None = None,
    ):
        if not content or content.strip():
            return []

        splitter_kwargs = {
            "chunk_size": max(int(chunk_size), 1),
            "chunk_overlap": max(int(chunk_overlap), 0),
            "length_function": lambda s: len(self.tokenizer.encode(s)),
        }

        if separator is not None:
            splitter_kwargs["separators"] = list(separator)

        splitter = RecursiveCharacterTextSplitter(**splitter_kwargs)

        pieces = splitter.split_text(content)

        results = list[dict[str, Any]] = []

        for piece in pieces:
            body = piece.strip()
            if not body:
                continue
            results.append(
                {
                    "token": len(self.tokenizer.encode(body)),
                    "content": body,
                    "chunk_order_index": len(results),
                }
            )

        if not results:
            logfire.warn(f"Splitter produced no non-empty chunks: {len(content)}")

            body = content.strip()
            if body:
                results.append(
                    {
                        "token": len(self.tokenizer.encode(body)),
                        "content": body,
                        "chunk_order_index": 0,
                    }
                )

        return results
