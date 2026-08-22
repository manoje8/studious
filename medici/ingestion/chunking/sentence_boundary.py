"""Sentence-boundary chunker.

Splits raw text on sentence endings (``[.!?]``) only.
No Markdown headers. No PDF section structure. No semantic embeddings.
"""

import re
from typing import Any, Literal

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

_ABBREV: frozenset[str] = frozenset(
    {
        "Mr",
        "Mrs",
        "Ms",
        "Dr",
        "Prof",
        "Sr",
        "Jr",
        "vs",
        "etc",
        "al",
        "Fig",
        "Figs",
        "No",
        "St",
        "Lt",
        "Sgt",
        "Cpl",
        "Pvt",
        "Gen",
        "Gov",
        "Pres",
        "Sen",
        "Rep",
        "Dept",
        "Approx",
        "approx",
    }
)


class SentenceBoundaryChunker(Chunker):
    """
    Splits text on sentence boundaries only (``[.!?]``).

    Does NOT parse Markdown headers or PDF section structure.
    Does NOT compute semantic embeddings.

    Sentences are batched until adding the next sentence would exceed
    *size* (measured in tokens when ``size_mode="tokens"`` — the
    default — or characters when ``size_mode="characters"``).

    The last *overlap* **complete sentences** from chunk N are prepended
    verbatim to chunk N+1.

    If a single sentence exceeds *size* on its own it is forwarded to
    ``RecursiveCharacterTextSplitter`` as a safety net.  After that the
    buffer is **reset to empty** — the oversized sentence is NOT carried
    as an overlap seed so it cannot inflate subsequent chunks.

    Parameters
    ----------
    size : int
        Maximum chunk size.  Interpreted as token count when
        ``size_mode="tokens"`` (default) or character count when
        ``size_mode="characters"``.
    overlap : int
        Number of complete sentences carried from the end of chunk N
        into the beginning of chunk N+1.  Must be >= 0.
    size_mode : {"tokens", "characters"}
        Unit used when measuring *size*.  ``"tokens"`` aligns with
        embedding-model context limits; ``"characters"`` is faster
        because it avoids tokenizer calls.
    tokenizer : Tokenizer | None
        Used only when ``size_mode="tokens"``.  Falls back to
        ``TikTokenTokenizer`` when *None*.

    Notes
    -----
    Abbreviation handling: a curated set of common abbreviations
    (``_ABBREV``) prevents false sentence boundaries after e.g.
    "Dr. Smith" or "Fig. 3".  For production-grade NLP accuracy, swap
    ``_split_sentences`` for spaCy's sensitizers — the public interface
    is unchanged.
    """

    def __init__(
        self,
        size: int = 512,
        overlap: int = 1,
        size_mode: Literal["tokens", "characters"] = "tokens",
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if size_mode not in ("tokens", "characters"):
            raise ValueError("size_mode must be 'tokens' or 'characters'")

        self.size = size
        self.overlap = overlap
        self.size_mode = size_mode
        self.tokenizer: Tokenizer = tokenizer or TikTokenTokenizer(model_name="gpt-4o-mini")

    def chunk(self, text: str, **kwargs) -> list[Chunk]:
        """Split *text* into sentence-boundary chunks."""
        doc_id: str = kwargs.get("doc_id", "")
        source_file: str = kwargs.get("source_file", "")

        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)
        raw_pieces = self._batch_sentences(sentences)

        results: list[Chunk] = []
        for idx, piece in enumerate(raw_pieces):
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
                    token_count=self._measure(body),
                )
            )

        if not results:
            logfire.warn(
                f"SentenceBoundaryChunker produced no chunks for text of {len(text)} chars"
            )
            body = text.strip()
            if body:
                results.append(
                    Chunk(
                        text=body,
                        chunk_index=0,
                        doc_id=doc_id,
                        source_file=source_file,
                        chunk_type="text",
                        token_count=self._measure(body),
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
        """Chunk a list of multimodal content-list items."""
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
            token_count = self._measure(serialized)

            if token_count <= self.size:
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

            # Oversized item — re-split using sentence batching, keep chunk_type.
            for piece in self._batch_sentences(self._split_sentences(serialized)):
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
                        token_count=self._measure(body),
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

    def _measure(self, text: str) -> int:
        """Return the size of *text* in the configured unit."""
        if self.size_mode == "tokens":
            return len(self.tokenizer.encode(text))
        return len(text)  # characters

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split *text* into individual sentences on ``[.!?]`` boundaries.

        Runs a post-pass to re-merge fragments that follow a known
        abbreviation so that e.g. "Dr. Smith arrived." is not broken
        after "Dr.".
        """
        raw: list[str] = re.split(r"(?<=[.!?])\s+", text.strip())
        merged: list[str] = []
        for piece in raw:
            if merged:
                last_word = merged[-1].rstrip().rsplit(None, 1)[-1].rstrip(".")
                if last_word in _ABBREV:
                    merged[-1] += " " + piece
                    continue
            merged.append(piece)
        return [s.strip() for s in merged if s.strip()]

    def _fallback_split(self, text: str) -> list[str]:
        """
        Split a single oversized sentence using ``RecursiveCharacterTextSplitter``.

        This is a safety net only — it produces sub-sentence splits so
        the pipeline does not drop content that cannot be represented in
        a single chunk.  The surrounding ``_batch_sentences`` loop resets
        its buffer to empty after calling this method.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(self.size, 1),
            chunk_overlap=0,
            length_function=self._measure,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return [p for p in splitter.split_text(text) if p.strip()]

    def _batch_sentences(self, sentences: list[str]) -> list[str]:
        """
        Group *sentences* into chunks bounded by ``self.size``.

        Overlap
        -------
        The last ``self.overlap`` complete sentences from chunk N are
        prepended verbatim to chunk N+1.

        Oversized sentence handling
        ---------------------------
        1. Flush the current buffer as a chunk.
        2. Forward the oversized sentence to ``_fallback_split`` and
           append all resulting sub-pieces directly to the output.
        3. **Reset the buffer to empty** — the oversized sentence is NOT
           used as an overlap seed so it cannot inflate the next chunk.
        """
        if not sentences:
            return []

        chunks: list[str] = []
        buffer: list[str] = []
        buffer_size: int = 0

        def _flush(buf: list[str]) -> None:
            joined = " ".join(buf).strip()
            if joined:
                chunks.append(joined)

        for sentence in sentences:
            s_size = self._measure(sentence)

            # Oversized single sentence
            if s_size > self.size:
                _flush(buffer)
                for piece in self._fallback_split(sentence):
                    chunks.append(piece.strip())
                buffer = []
                buffer_size = 0
                continue

            # Adding this sentence would overflow the current chunk
            if buffer and buffer_size + s_size > self.size:
                _flush(buffer)
                overlap_sents = buffer[-self.overlap :] if self.overlap > 0 else []
                buffer = list(overlap_sents)
                buffer_size = sum(self._measure(s) for s in buffer)

            buffer.append(sentence)
            buffer_size += s_size

        _flush(buffer)
        return chunks
