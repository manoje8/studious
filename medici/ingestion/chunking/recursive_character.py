from typing import Any

import logfire
from langchain_text_splitters import RecursiveCharacterTextSplitter

from medici.common.storage.base_storage import BaseStorage
from medici.common.utils.tokenizer import TikTokenTokenizer, Tokenizer
from medici.ingestion.chunking.chunk import (
    Chunk,
    extract_page_numbers,
    serialize_image_caption,
    serialize_multimodal_item,
)
from medici.ingestion.chunking.Chunker import Chunker


class RecursiveCharacterChunker(Chunker):
    # Splits on separators (\n\n, \n, ., space) in priority order.
    # Does NOT parse Markdown headers or PDF section structure.
    """
    For articles, books, reports
    """

    def __init__(
        self, size: int = 1200, overlap: int = 100, tokenizer: Tokenizer | None = None, **kwargs
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
        **kwargs,
    ) -> list[Chunk]:
        doc_id: str = kwargs.get("doc_id", "")
        source_file: str = kwargs.get("source_file", "")
        separators: list[str] = kwargs.get("separator", ["```", "| ---", "\n\n", "\n", " "])

        if not text or not text.strip():
            return []

        splitter_kwargs = {
            "chunk_size": max(int(self.size), 1),
            "chunk_overlap": max(int(self.overlap), 0),
            "length_function": lambda s: len(self.tokenizer.encode(s)),
            "separators": list(separators),
        }

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

    def chunk_multimodal_items(
        self,
        items: list[dict[str, Any]],
        doc_id: str,
        source_file: str,
        start_index: int = 0,
        storage: "BaseStorage | None" = None,
    ) -> list[Chunk]:
        results: list[Chunk] = []
        pending_images: list[dict[str, Any]] = []
        idx = start_index

        for item in items:
            item_type = item.get("type", "unknown")
            image_path = ""

            if item_type == "image":
                caption = serialize_image_caption(item)

                image_bytes: bytes | None = item.get("image_data") or item.get("image_bytes")
                if image_bytes and storage is not None:
                    img_key = f"images/{doc_id}/chunk_{idx}.png"
                    try:
                        image_path = storage.save_bytes(img_key, image_bytes)
                        logfire.debug(
                            "image_saved_to_storage",
                            key=img_key,
                            size_bytes=len(image_bytes),
                        )
                    except Exception as exc:
                        logfire.warning("image_save_failed", key=img_key, error=str(exc))

                if caption is None:
                    pending_images.append(
                        {
                            "doc_id": doc_id,
                            "img_path": item.get("img_path", ""),
                            "page_idx": item.get("page_idx"),
                            "content_list_index": item.get("_content_list_index"),
                        }
                    )
                    page = item.get("page_idx")
                    page_str = f" (page {page + 1})" if isinstance(page, int) else ""
                    serialized = f"[Figure{page_str} — awaiting caption]"
                else:
                    serialized = caption
            else:
                serialized = serialize_multimodal_item(item)

            if not serialized:
                logfire.warn(
                    f"Skipping multimodal item with no serializable text: type={item_type}"
                )
                continue

            page_numbers = extract_page_numbers(item)
            metadata = {k: v for k, v in item.items() if k not in ("type", "text")}
            token_count = len(self.tokenizer.encode(serialized))

            if token_count <= 512:
                results.append(
                    Chunk(
                        text=serialized,
                        chunk_index=idx,
                        doc_id=doc_id,
                        source_file=source_file,
                        chunk_type=item_type,
                        page_numbers=page_numbers,
                        block_types=[item_type],
                        token_count=token_count,
                        metadata=metadata,
                        image_path=image_path,
                    )
                )
                idx += 1
                continue

            # Oversized item (e.g. a huge table) — split but keep chunk_type
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=max(512, 1),
                chunk_overlap=max(64, 0),
                length_function=lambda s: len(self.tokenizer.encode(s)),
                separators=["\n\n", "\n", "</tr>", " "],
            )
            for piece in splitter.split_text(serialized):
                body = piece.strip()
                if not body:
                    continue
                results.append(
                    Chunk(
                        text=body,
                        chunk_index=idx,
                        doc_id=doc_id,
                        source_file=source_file,
                        chunk_type=item_type,
                        page_numbers=page_numbers,
                        block_types=[item_type],
                        token_count=len(self.tokenizer.encode(body)),
                        metadata=metadata,
                        # Only the first split piece carries the image_path
                        image_path=image_path if idx == start_index else "",
                    )
                )
                idx += 1

        logfire.info(
            f"Multimodal chunking complete: {idx - start_index} chunks from {len(items)} items"
        )
        return results
