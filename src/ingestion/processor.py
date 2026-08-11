import asyncio
import time
from pathlib import Path
from typing import Any

import logfire
from tqdm import tqdm
from treant.treant import ParseMethod, process_document

from src.common.services.qdrant import QdrantStorageService
from src.common.storage.storage_factory import StorageFactory
from src.common.utils.config import config
from src.common.utils.constants import (
    HTML_FORMATS,
    OFFICE_FORMATS,
    TEXT_FORMATS,
    ChunkerStrategy,
    StorageType,
)
from src.common.utils.helper import separate_content, supported_extensions_list
from src.common.utils.tokenizer import TikTokenTokenizer, Tokenizer
from src.ingestion.chunking.chunk import BatchProcess, Chunk, build_parent_child_chunk
from src.ingestion.chunking.chunker_factory import create_chunker
from src.ingestion.chunking.chunking_config import ChunkingConfig
from src.ingestion.embedding import EmbeddingService


class Processor:
    def __init__(
        self,
        tokenizer: Tokenizer,
        embedding_service: EmbeddingService,
        storage_service: QdrantStorageService,
        max_concurrency: int = 4,
        kg_extractor=None,
        kg_store=None,
    ):
        self.embedding_service = embedding_service
        self.storage_service = storage_service
        self._kg_extractor = kg_extractor
        self._kg_store = kg_store

        self.tokenizer = tokenizer or TikTokenTokenizer(model_name="gpt-4o-mini")

        local_config = {
            "type": StorageType.LOCAL.value,
            "base_dir": config.STORAGE_BASE_DIR,
        }

        # cloud_config = {
        #     "type": StorageType.GCS.value,
        #     "bucket": config.GCP_PROCESSED_BUCKET,
        # }

        self.in_storage = StorageFactory.create(local_config)
        self._max_concurrency = max_concurrency

    def _filter_supported_files(self, file_paths: list[str], recursive: bool = False):
        supported_extensions = supported_extensions_list()
        supported_files = []

        for file_path in file_paths:
            path = Path(file_path)
            if path.is_dir():
                if recursive:
                    for inside_path in path.rglob("*"):
                        if (
                            inside_path.is_file()
                            and inside_path.suffix.lower() in supported_extensions
                        ):
                            supported_files.append(str(inside_path))
                else:
                    for inside_path in path.glob("*"):
                        if (
                            inside_path.is_file()
                            and inside_path.suffix.lower() in supported_extensions
                        ):
                            supported_files.append(str(inside_path))
            elif path.is_file():
                if path.suffix.lower() in supported_extensions:
                    supported_files.append(str(path))
                else:
                    logfire.warn(f"Unsupported file format : {file_path}")

            else:
                logfire.warn(f"Path doesn't exist: {path}")

        return supported_files

    def _select_chunking_strategy(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()

        if suffix in HTML_FORMATS | TEXT_FORMATS:
            return ChunkerStrategy.SENTENCE_BOUNDARY

        if suffix in OFFICE_FORMATS or suffix == ".pdf":
            return ChunkerStrategy.RECURSIVE_CHARACTER

        return ChunkerStrategy.FIXED

    def _interleave_chunks(
        self,
        text_chunks: list[Chunk],
        multimodal_chunks: list[Chunk],
        text_blocks: list[tuple[str, int]],
    ) -> list[Chunk]:
        """
        Merge text_chunks and multimodal_chunks in true source-document order
        so that ``build_parent_child_chunk`` windows across both types correctly.

        Document-order positions
        ------------------------
        * Multimodal chunks carry ``metadata["_content_list_index"]`` — the item's
          exact index in the original parsed content list.
        * Text chunks are derived from the joined text string; we recover their
          approximate document positions from *text_blocks*, a list of
          ``(text, original_index)`` pairs returned by ``separate_content``.  Each
          Chunk produced by the text splitter is attributed to the text block whose
          cumulative character boundary it falls within.

        After sorting the merged list, ``chunk_index`` is reassigned 0…N so that
        downstream code that relies on a contiguous integer index stays correct.
        """

        # Assign a doc-order key to every text chunk
        if text_blocks and text_chunks:
            cumulative: list[tuple[int, int]] = []  # (end_char_offset, original_index)
            offset = 0
            sep = "\n\n"
            for i, (block_text, original_idx) in enumerate(text_blocks):
                offset += len(block_text)
                cumulative.append((offset, original_idx))
                if i < len(text_blocks) - 1:
                    offset += len(sep)

            cursor = 0
            for chunk in text_chunks:
                chunk_len = len(chunk.text)
                while cursor < len(cumulative) - 1 and cumulative[cursor][0] < chunk_len:
                    cursor += 1
                chunk.metadata["_doc_order"] = cumulative[cursor][1]
        else:
            # No text blocks or no text chunks — assign sequential fallback keys
            for i, chunk in enumerate(text_chunks):
                chunk.metadata["_doc_order"] = i

        # Assign doc-order key to multimodal chunks
        for chunk in multimodal_chunks:
            chunk.metadata.setdefault(
                "_doc_order",
                chunk.metadata.get("_content_list_index", float("inf")),
            )

        # merge, sort, re-index
        merged = sorted(
            text_chunks + multimodal_chunks,
            key=lambda c: (c.metadata.get("_doc_order", float("inf")), c.chunk_index),
        )
        for new_idx, chunk in enumerate(merged):
            chunk.chunk_index = new_idx

        return merged

    async def _chunk_doc_content(
        self,
        file_path: Path,
        content_list: str,
        multimodal_items: list[dict[str, Any]],
        doc_id: str,
        parse_method: ParseMethod,
        split_by_character: str | None = None,
        text_blocks: list[tuple[str, int]] | None = None,
    ):
        chunking_strategy = self._select_chunking_strategy(file_path)
        logfire.info(f"Starting chunking with strategy: {chunking_strategy} - {parse_method}")

        chunking_config = ChunkingConfig(
            type=chunking_strategy, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP
        )

        chunker = create_chunker(chunking_config)

        text_chunks = chunker.chunk(content_list, doc_id=doc_id, source_file=str(file_path))

        if multimodal_items:
            multimodal_chunks = chunker.chunk_multimodal_items(
                multimodal_items,
                doc_id=doc_id,
                source_file=str(file_path),
                start_index=0,
                storage=self.in_storage,
            )
            chunks = self._interleave_chunks(text_chunks, multimodal_chunks, text_blocks or [])
        else:
            chunks = text_chunks

        logfire.info(
            f"Chunking complete: {len(chunks)} chunks produced from {len(content_list)} blocks"
        )

        enriched = build_parent_child_chunk(chunks, self.tokenizer)
        logfire.info(f"Build parent child chunk: {len(enriched)}")

        return enriched

    async def _process_one_guarded(
        self,
        file_path: str,
        parse_method: ParseMethod,
        semaphore: asyncio.Semaphore,
        **kwargs,
    ):
        async with semaphore:
            try:
                content_list, doc_id = await process_document(
                    file_path=file_path, parse_method=parse_method, **kwargs
                )
                return file_path, True, content_list, doc_id, None
            except Exception as e:
                logfire.error(f"parser failed to process {file_path}: {str(e)}")
                return file_path, False, None, None, str(e)

    async def process_document_batch(
        self,
        file_paths: list[str],
        parse_method: ParseMethod,
        recursive: bool = False,
        show_progress: bool = True,
        output_dir: str | None = None,
        **kwargs,
    ) -> tuple[BatchProcess, dict[str, tuple[list, str]]]:
        """
        Parse many files concurrently using process_document as the single
        worker (no separate pass/fail-only parse, no double parsing).

        Returns (BatchProcess summary, {file_path: (content_list, doc_id)}
        for successful files).
        """
        start_time = time.time()
        supported_files = self._filter_supported_files(file_paths, recursive)

        if not supported_files:
            return (
                BatchProcess(
                    successful_files=[],
                    failed_files=[],
                    total_files=0,
                    processing_time=0.0,
                    errors={},
                    output_dir=output_dir,
                ),
                {},
            )

        logfire.info(f"Found {len(supported_files)} file to process")

        semaphore = asyncio.Semaphore(self._max_concurrency)
        tasks = [
            self._process_one_guarded(fp, parse_method, semaphore, **kwargs)
            for fp in supported_files
        ]

        pbar = (
            tqdm(total=len(supported_files), desc=f"processing files {parse_method}", unit="file")
            if show_progress
            else None
        )

        success_files, failed_files, errors, results = [], [], {}, {}
        try:
            for coro in asyncio.as_completed(tasks):
                file_path, ok, content_list, doc_id, error_msg = await coro
                if ok:
                    success_files.append(file_path)
                    results[file_path] = (content_list, doc_id)
                else:
                    failed_files.append(file_path)
                    errors[file_path] = error_msg
                if pbar:
                    pbar.update(1)
        finally:
            if pbar:
                pbar.close()

        processing_time = time.time() - start_time

        output = BatchProcess(
            successful_files=success_files,
            failed_files=failed_files,
            total_files=len(supported_files),
            processing_time=processing_time,
            errors=errors,
            output_dir=output_dir,
        )
        logfire.info(output.summary())

        return output, results

    async def ingest_document(
        self,
        file_path: str | Path,
        parse_method: ParseMethod = ParseMethod.DOCLING,
        doc_id: str | None = None,
        split_by_character: str = "\n\n",
    ):
        multimodal_items = None
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content_list, doc_id = await process_document(
            file_path=file_path,
            doc_id=doc_id,
            split_by_character=split_by_character,
            parse_method=parse_method,
        )
        logfire.info(f"Stage 1 complete: {len(content_list)} content list")

        text_blocks: list[tuple[str, int]] = []
        if parse_method == ParseMethod.DOCLING:
            content_list, multimodal_items, text_blocks = separate_content(content_list)

        return await self._chunk_embed_store(
            file_path,
            content_list,
            multimodal_items,
            doc_id,
            parse_method,
            text_blocks=text_blocks,
        )

    async def ingest_documents(
        self,
        file_paths: list[str],
        parse_method: ParseMethod,
        recursive: bool = False,
    ) -> dict:
        """
        Batch version of ingest_document: parses all files concurrently via
        process_document_batch, then chunks/embeds/stores each successfully
        parsed document.
        """

        batch_result, parsed = await self.process_document_batch(
            file_paths=file_paths, parse_method=parse_method, recursive=recursive
        )
        logfire.info(batch_result.summary())
        multimodal_items = None
        results = {}
        for file_path, (content_list, doc_id) in parsed.items():
            path = Path(file_path)
            text_blocks: list[tuple[str, int]] = []
            if parse_method == ParseMethod.DOCLING:
                content_list, multimodal_items, text_blocks = separate_content(content_list)
            try:
                results[file_path] = await self._chunk_embed_store(
                    path,
                    content_list,
                    multimodal_items,
                    doc_id,
                    parse_method,
                    text_blocks=text_blocks,
                )
            except Exception as e:
                logfire.error(f"Failed to chunk/embed/store {file_path}: {str(e)}")
                batch_result.failed_files.append(file_path)
                batch_result.errors[file_path] = str(e)

        return {"batch": batch_result, "results": results}

    async def _chunk_embed_store(
        self,
        file_path: Path,
        content_list,
        multimodal_items: list[dict[str, Any]],
        doc_id: str,
        parse_method: ParseMethod,
        text_blocks: list[tuple[str, int]] | None = None,
    ) -> dict:

        if len(content_list) <= 1:
            logfire.warn(f"Content is empty: {file_path}")
            return {"doc_id": doc_id, "chunks_produced": 0, "vectors_stored": 0}

        chunks = await self._chunk_doc_content(
            file_path=file_path,
            content_list=content_list,
            multimodal_items=multimodal_items,
            doc_id=doc_id,
            parse_method=parse_method,
            text_blocks=text_blocks,
        )
        self.in_storage.upload(key="chunks", data=chunks)
        logfire.info(f"Document chunking completed: {len(chunks)}")

        embedded_chunks, dead_letter = await self.embedding_service.embed_chunks(chunks)
        self.in_storage.upload(key="embedded_chunks", data=embedded_chunks)
        logfire.info(f"Stage 3 complete: {len(embedded_chunks)} vectors")

        if dead_letter:
            logfire.warn(
                f"Stage 3: {len(dead_letter)} chunk(s) could not be embedded and were "
                f"dead-lettered for {file_path}. Indices: "
                + str([d["chunk_index"] for d in dead_letter])
            )

        await self.storage_service.upsert_embedded_chunks(embedded_chunks)
        logfire.info("Stage 4 complete: stored in Qdrant")

        kg_entities_count = 0
        kg_relationships_count = 0
        if self._kg_extractor is not None and self._kg_store is not None:
            try:
                kg_results = await self._kg_extractor.extract_from_chunks(chunks, doc_id)
                for result in kg_results:
                    await self._kg_store.upsert_extraction(result)
                kg_entities_count = sum(len(r.entities) for r in kg_results)
                kg_relationships_count = sum(len(r.relationships) for r in kg_results)
                logfire.info(
                    f"Stage 5 complete: KG extraction — "
                    f"{kg_entities_count} entities, "
                    f"{kg_relationships_count} relationships"
                )
            except Exception as e:
                logfire.error(f"Stage 5 KG extraction failed (non-fatal): {e}")

        return {
            "doc_id": doc_id,
            "chunks_produced": len(chunks),
            "vectors_stored": len(embedded_chunks),
            "dead_letter_count": len(dead_letter),
            "kg_entities": kg_entities_count,
            "kg_relationships": kg_relationships_count,
            "dead_letter_chunks": [
                {
                    "chunk_index": d["chunk_index"],
                    "source_file": d["chunk"].source_file,
                    "error": d["error"],
                }
                for d in dead_letter
            ],
        }
