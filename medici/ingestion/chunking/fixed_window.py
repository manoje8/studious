from collections.abc import Callable
from typing import Any

import logfire

from medici.common.storage.base_storage import BaseStorage
from medici.ingestion.chunking.chunk import (
    Chunk,
    extract_page_numbers,
    serialize_image_caption,
    serialize_multimodal_item,
)
from medici.ingestion.chunking.Chunker import Chunker


class FixedWindow(Chunker):
    # Splits on raw word/character boundaries. Does NOT respect sentence,
    # Markdown-header, or PDF-section structure.
    def __init__(
        self,
        size: int = 512,
        overlap: int = 64,
        token_len_fn: Callable[[str], int] | None = None,
        **kwargs,
    ) -> None:
        if overlap >= size:
            raise ValueError("Overlap tokens must be smaller than the chunk size tokens")
        self.chunk_size = size
        self.overlap = overlap
        self.token_len_fn = token_len_fn or (lambda s: len(s.split()))

    def chunk(self, text: str, **kwargs) -> list[Chunk]:
        doc_id: str = kwargs.get("doc_id", "")
        source_file: str = kwargs.get("source_file", "")

        pieces = self._split_oversized(text)
        return [
            Chunk(
                text=piece,
                chunk_index=idx,
                doc_id=doc_id,
                source_file=source_file,
                chunk_type="text",
            )
            for idx, piece in enumerate(pieces)
        ]

    def chunk_multimodal_items(
        self,
        items: list[dict[str, Any]],
        doc_id: str,
        source_file: str,
        start_index: int = 0,
        storage: "BaseStorage | None" = None,
    ) -> tuple[list[Chunk], list[dict[str, Any]]]:
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
            token_count = self.token_len_fn(serialized)

            if token_count <= self.chunk_size:
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

            # Oversized (e.g. a huge table serialized to Markdown) — reuse
            # the same word-offset fixed-window splitting as chunk(), but
            # keep chunk_type instead of forcing "text".
            for piece in self._split_oversized(serialized):
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
                        token_count=self.token_len_fn(body),
                        metadata=metadata,
                        image_path=image_path if idx == start_index else "",
                    )
                )
                idx += 1

        logfire.info(
            f"Multimodal chunking complete: {idx - start_index} chunks, "
            f"{len(pending_images)} images flagged for captioning"
        )
        return results

    def _split_oversized(self, text: str) -> list[str]:
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

        pieces = []
        step = self.chunk_size - self.overlap
        i = 0
        while i < len(words):
            window_words = words[i : i + self.chunk_size]
            if not window_words:
                break
            char_start = offsets[i][0]
            char_end = offsets[min(i + len(window_words) - 1, len(offsets) - 1)][1]
            pieces.append(text[char_start:char_end])
            i += step

        return pieces
