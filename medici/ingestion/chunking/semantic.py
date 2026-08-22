"""Semantic chunker — production-grade implementation.

Splits text where the *meaning* changes rather than on fixed boundaries.

Algorithm
---------
1. Tokenize the input into individual sentences with NLTK's Punkt tokeniser.
2. Encode every sentence with a lightweight ``SentenceTransformer`` model
   (``BAAI/bge-small-en-v1.5`` by default; swap for any MTEB-compatible model).
3. Accumulate sentences into a buffer.  When the cosine similarity between the
   *last sentence added* and the *next sentence* drops below
   ``similarity_threshold`` **AND** the current buffer has grown past
   ``min_chunk_size``, flush the buffer as a new chunk.
4. Also flush when the buffer would exceed ``size`` regardless of similarity.
5. Carry the last ``overlap_sentences`` sentences of each chunk into the next
   buffer so context bleeds naturally across boundaries.
6. Any chunk that still exceeds ``size`` is forwarded to
   ``RecursiveCharacterTextSplitter`` as a safety net — identical to the
   pattern used by ``SentenceBoundaryChunker`` and ``RecursiveCharacterChunker``.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import logfire
import nltk
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from medici.common.storage.base_storage import BaseStorage
from medici.common.utils.tokenizer import TikTokenTokenizer, Tokenizer
from medici.ingestion.chunking.chunk import (
    Chunk,
    extract_page_numbers,
    serialize_image_caption,
    serialize_multimodal_item,
)
from medici.ingestion.chunking.Chunker import Chunker

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


class SemanticChunker(Chunker):
    """Semantic chunker that splits on meaning shifts rather than fixed sizes.

    Parameters
    ----------
    size:
        Maximum chunk size (tokens when ``size_mode="tokens"``, characters
        otherwise).  Chunks that exceed this limit are force-split with a
        ``RecursiveCharacterTextSplitter`` fallback.
    min_chunk_size:
        Minimum buffer size before a semantic break is allowed to flush.
        Prevents tiny chunks when adjacent sentences diverge early.
    overlap_sentences:
        Number of complete sentences from the end of chunk *N* that are
        prepended verbatim to chunk *N+1* to preserve cross-boundary context.
    similarity_threshold:
        Cosine-similarity cutoff below which a topic shift is declared.
        Range: [0, 1].  Lower → more splits; higher → fewer, larger chunks.
    embedding_model:
        HuggingFace model identifier for ``SentenceTransformer``.
    device:
        Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    size_mode:
        ``"tokens"`` (default) or ``"characters"``.
    tokenizer:
        Used only when ``size_mode="tokens"``.  Falls back to
        ``TikTokenTokenizer`` when *None*.
    verbose:
        Enable ``show_progress_bar`` on the sentence encoder (useful for
        debugging large documents; keep *False* in production).
    """

    def __init__(
        self,
        size: int = 1200,
        min_chunk_size: int = 100,
        overlap_sentences: int = 1,
        similarity_threshold: float = 0.75,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        size_mode: Literal["tokens", "characters"] = "tokens",
        tokenizer: Tokenizer | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        if similarity_threshold < 0 or similarity_threshold > 1:
            raise ValueError("similarity_threshold must be in [0, 1]")
        if size_mode not in ("tokens", "characters"):
            raise ValueError("size_mode must be 'tokens' or 'characters'")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0")

        self.size = size
        self.min_chunk_size = min_chunk_size
        self.overlap_sentences = overlap_sentences
        self.similarity_threshold = similarity_threshold
        self.size_mode = size_mode
        self.verbose = verbose
        self.tokenizer: Tokenizer = tokenizer or TikTokenTokenizer(model_name="gpt-4o-mini")

        logfire.info(
            "SemanticChunker initialising",
            embedding_model=embedding_model,
            device=device,
            similarity_threshold=similarity_threshold,
            size=size,
            size_mode=size_mode,
        )
        self._encoder = SentenceTransformer(embedding_model, device=device)
        self._sent_tokenize = nltk.sent_tokenize

    def chunk(self, text: str, **kwargs: Any) -> list[Chunk]:
        """Split *text* into semantically coherent chunks.

        Keyword arguments are forwarded from the processor layer:
        - ``doc_id``     – document identifier stored in each ``Chunk``.
        - ``source_file``– origin path stored in each ``Chunk``.
        - ``metadata``   – arbitrary dict merged into each ``Chunk.metadata``.
        """
        doc_id: str = kwargs.get("doc_id", "")
        source_file: str = kwargs.get("source_file", "")
        metadata: dict = kwargs.get("metadata", {})

        if not text or not text.strip():
            return []

        chunks = self._semantic_chunk(
            text, doc_id=doc_id, source_file=source_file, metadata=metadata
        )

        final: list[Chunk] = []
        chunk_index = 0
        for ch in chunks:
            if self._measure(ch.text) > self.size:
                logfire.debug(
                    "semantic_chunk_oversized_fallback",
                    chunk_index=ch.chunk_index,
                    size=self._measure(ch.text),
                    limit=self.size,
                )
                for piece in self._fallback_split(ch.text):
                    body = piece.strip()
                    if not body:
                        continue
                    final.append(
                        Chunk(
                            text=body,
                            chunk_index=chunk_index,
                            doc_id=doc_id,
                            source_file=source_file,
                            chunk_type="text",
                            token_count=self._measure(body),
                            metadata=metadata,
                        )
                    )
                    chunk_index += 1
            else:
                ch.chunk_index = chunk_index
                final.append(ch)
                chunk_index += 1

        if not final:
            logfire.warn(
                "SemanticChunker produced no chunks",
                text_len=len(text),
                doc_id=doc_id,
            )
            body = text.strip()
            if body:
                final.append(
                    Chunk(
                        text=body,
                        chunk_index=0,
                        doc_id=doc_id,
                        source_file=source_file,
                        chunk_type="text",
                        token_count=self._measure(body),
                        metadata=metadata,
                    )
                )

        logfire.info(
            "semantic_chunking_complete",
            doc_id=doc_id,
            chunks=len(final),
            input_chars=len(text),
        )
        return final

    def chunk_multimodal_items(
        self,
        items: list[dict[str, Any]],
        doc_id: str,
        source_file: str,
        start_index: int = 0,
        storage: BaseStorage | None = None,
    ) -> list[Chunk]:
        """
        Chunk a list of multimodal content-list items.

        Images, tables, and equations are serialized to text (same as the
        other chunker implementations) and then size-guarded with the
        ``RecursiveCharacterTextSplitter`` fallback if needed.
        """
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
                    "semantic_multimodal_skip_no_text",
                    item_type=item_type,
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

            for piece in self._fallback_split(serialized):
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
            "semantic_multimodal_chunking_complete",
            chunks=idx - start_index,
            pending_image_captions=len(pending_images),
        )
        return results

    def _semantic_chunk(
        self,
        text: str,
        doc_id: str,
        source_file: str,
        metadata: dict,
    ) -> list[Chunk]:
        """Embed sentences and split on similarity drops."""
        sentences = self._sent_tokenize(text.strip())

        if len(sentences) <= 1:
            body = text.strip()
            return (
                [
                    Chunk(
                        text=body,
                        chunk_index=0,
                        doc_id=doc_id,
                        source_file=source_file,
                        chunk_type="text",
                        token_count=self._measure(body),
                        metadata=metadata,
                    )
                ]
                if body
                else []
            )

        embeddings: np.ndarray = self._encoder.encode(
            sentences,
            batch_size=64,
            show_progress_bar=self.verbose,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        chunks: list[Chunk] = []
        buffer: list[str] = []
        buffer_size: int = 0
        chunk_index: int = 0

        for i, sentence in enumerate(sentences):
            s_size = self._measure(sentence)

            if not buffer:
                # Start of a new buffer, always accept the first sentence
                buffer.append(sentence)
                buffer_size += s_size
                continue

            sim: float = float(np.dot(embeddings[i - 1], embeddings[i]))

            would_overflow = (buffer_size + s_size) > self.size

            topic_shift = sim < self.similarity_threshold and buffer_size >= self.min_chunk_size

            if would_overflow or topic_shift:
                body = " ".join(buffer).strip()
                if body:
                    chunks.append(
                        Chunk(
                            text=body,
                            chunk_index=chunk_index,
                            doc_id=doc_id,
                            source_file=source_file,
                            chunk_type="text",
                            token_count=self._measure(body),
                            metadata=metadata,
                        )
                    )
                    chunk_index += 1

                overlap_seed = (
                    buffer[-self.overlap_sentences :] if self.overlap_sentences > 0 else []
                )
                buffer = list(overlap_seed) + [sentence]
                buffer_size = sum(self._measure(s) for s in buffer)
            else:
                buffer.append(sentence)
                buffer_size += s_size

        if buffer:
            body = " ".join(buffer).strip()
            if body:
                chunks.append(
                    Chunk(
                        text=body,
                        chunk_index=chunk_index,
                        doc_id=doc_id,
                        source_file=source_file,
                        chunk_type="text",
                        token_count=self._measure(body),
                        metadata=metadata,
                    )
                )

        return chunks

    def _measure(self, text: str) -> int:
        """Return the size of *text* in the configured unit."""
        if self.size_mode == "tokens":
            return len(self.tokenizer.encode(text))
        return len(text)  # characters

    def _fallback_split(self, text: str) -> list[str]:
        """
        Re-split an oversized chunk using ``RecursiveCharacterTextSplitter``.

        This is a safety net only, it produces sub-sentence splits so the
        pipeline never drops content.  The caller resets the buffer after
        invoking this method.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(self.size, 1),
            chunk_overlap=0,
            length_function=self._measure,
            separators=["```", "| ---", "\n\n", "\n", ". ", " ", ""],
        )
        return [p for p in splitter.split_text(text) if p.strip()]
