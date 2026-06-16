import asyncio
import hashlib
import json
from pathlib import Path
from typing import Dict

import logfire

from src.ingestion.parser.docling_parser import DoclingParser
from src.ingestion.parser.google_doc_ai import GoogleDocAI
from src.ingestion.chunking.chunk import Chunking
from src.ingestion.embedding import EmbeddingService
from src.services.qdrant import QdrantStorageService
from src.services.sparse_index import SparseSearchIndex
from src.storage.storage_factory import StorageFactory
from src.utils.constants import (
    HTML_FORMATS,
    OFFICE_FORMATS,
    TEXT_FORMATS,
    StorageType,
    ChunkingType,
    ParseMethod,
)
from src.utils.config import config
from src.utils.doc_cache import DocumentCache
from src.utils.helper import bootstrap_sparse_index


class Processor:
    def __init__(self, cache_dir: str | None = None):
        self.kwags = {"parser_method": "auto"}

        self.embedding_service = EmbeddingService(
            model_name=config.EMBEDDING_MODEL_NAME,
            dimensions=config.EMBEDDING_DIMENSIONS,
            batch_size=config.EMBEDDING_BATCH_SIZE,
        )
        self.storage_service = QdrantStorageService(
            url=config.QDRANT_CLUSTER_ENDPOINT,
            collection_name=config.QDRANT_COLLECTION_NAME,
            vector_size=self.embedding_service.vector_size,
        )

        self._cache = DocumentCache(
            cache_dir=(
                Path(cache_dir) if hasattr(config, "cache_dir") else config.CACHE_DIR
            )
        )

        self.sparse_index = SparseSearchIndex()

        local_config = {
            "type": StorageType.LOCAL.value,
            "base_dir": config.STORAGE_BASE_DIR,
        }

        # cloud_config = {
        #     "type": StorageType.GCS.value,
        #     "bucket": config.GCP_PROCESSED_BUCKET,
        # }

        self.in_storage = StorageFactory.create(local_config)

    def _get_parser(self, parser_type: str):
        parser_name = parser_type.strip().lower()

        if parser_name == ParseMethod.GOOGLE_DOC_AI.value:
            return GoogleDocAI()
        elif parser_name == ParseMethod.DOCLING.value:
            return DoclingParser()
        else:
            raise ValueError(f"Unsupported Parser type: {parser_type}")

    def _select_chunking_strategy(
        self, file_path: Path, parse_method: ParseMethod
    ) -> str:
        suffix = file_path.suffix.lower()

        if parse_method == ParseMethod.GOOGLE_DOC_AI:
            return ChunkingType.SPLITTER.value
        else:
            if suffix in HTML_FORMATS | TEXT_FORMATS:
                return ChunkingType.STRUCTURE.value
            elif suffix in OFFICE_FORMATS or suffix == ".pdf":
                return ChunkingType.FIXED.value
            else:
                return ChunkingType.SPLITTER.value

    def _generate_cache_key(self, file_path: Path, parse_method: str) -> str:
        mtime = file_path.stat().st_mtime

        config_dict = {
            "file_path": str(file_path),
            "mtime": mtime,
            "parser": self.kwags.get("parser", "google_doc_ai"),
            "parse_method": parse_method,
        }

        config_str = json.dumps(config_dict, sort_keys=True)
        cache_key = hashlib.md5(config_str.encode()).hexdigest()

        return cache_key

    def _get_cached_result(self, cache_key: str, file_path: Path, parse_method: str):
        return self._cache.get(cache_key)

    def _store_cache_result(
        self,
        cache_key: str,
        content_list: list[dict],
        file_path: Path,
        parser: str,
        parse_method: str = None,
    ):
        self._cache.store(
            cache_key, content_list, file_path, parse_method=parse_method, parser=parser
        )

    def _generate_doc_id(self, file_path: str | Path, read_bytes: int = 8192) -> str:
        path = Path(file_path).resolve()
        stat = path.stat()
        hasher = hashlib.sha256()
        hasher.update(str(stat.st_size).encode())

        with open(path, "rb") as f:
            hasher.update(f.read(read_bytes))

        return f"doc={hasher.hexdigest()[:24]}"

    async def _chunk_doc_content(
        self,
        file_path: Path,
        content_list: list[dict],
        doc_id: str,
        chunking_strategy: ChunkingType,
        split_by_character: str | None = None,
    ):
        chunking = Chunking()

        if not chunking_strategy:
            chunking_strategy = self._select_chunking_strategy(file_path)
            logfire.info(f"Starting chunking with strategy: {chunking_strategy}")

        if chunking_strategy == ChunkingType.STRUCTURE.value:
            chunks = chunking.chunk_by_structure(
                content_list=content_list,
                doc_id=doc_id or file_path.stem,
                source_file=str(file_path),
            )
        elif chunking_strategy == ChunkingType.FIXED.value:
            chunks = chunking.chunk_fixed(
                content_list=content_list,
                doc_id=doc_id or file_path.stem,
                source_file=str(file_path),
                split_by_character=split_by_character or "\n\n",
            )
        else:
            chunks = chunking.splitter(
                content_list=content_list,
                doc_id=doc_id or file_path.stem,
                source_file=str(file_path),
            )

        logfire.info(
            f"Chunking complete: {len(chunks)} chunks produced from {len(content_list)} blocks"
        )

        enriched = chunking.build_parent_child_chunk(chunks)
        logfire.info(f"Build parent child chunk: {len(enriched)}")

        return enriched

    async def process_document_complete(
        self,
        file_path: str | Path,
        parse_method: ParseMethod,
        parser: str | None = None,
        display_stats: bool = None,
        split_by_character: str | None = None,
        split_by_character_only: str | None = None,
        doc_id: str | None = None,
        file_name: str | None = None,
        **kwargs,
    ):
        logfire.info(f"Starting document parsing: {parse_method.value} - {file_path}")

        ext = file_path.suffix.lower()

        # Todo check cache
        cache_key = self._generate_cache_key(file_path, parse_method.value)

        cache_result = self._get_cached_result(cache_key, file_path, parse_method.value)

        if cache_result is not None:
            logfire.info(f"Cache HIT - Returning cached result for {file_path}")
            doc_id = self._generate_doc_id(file_path)
            return cache_result, doc_id

        try:
            doc_parser = self._get_parser(parser_type=parse_method.value)

            if not doc_parser.check_installation():
                raise ImportError("Required package is not installed")

            if ext in [".pdf"]:
                logfire.info("Detected PDF file, parsing the pdf...")

                content_list = await asyncio.to_thread(
                    doc_parser.parse_pdf,
                    file_path=file_path,
                    method=parse_method.value,
                    **kwargs,
                )
            elif ext in HTML_FORMATS:
                logfire.info("Detected HTML file, parsing html...")
                content_list = await asyncio.to_thread(
                    doc_parser.parse_html,
                    file_path=file_path,
                    method=parse_method.value,
                    **kwargs,
                )

            elif ext in OFFICE_FORMATS:
                logfire.info("Detected office file, parsing document...")
                content_list = await asyncio.to_thread(
                    doc_parser.parse_doc,
                    file_path=file_path,
                    method=parse_method.value,
                    **kwargs,
                )
            else:
                raise ValueError(
                    f"Unsupported file format: {ext}. "
                    f"Only supports PDF files, Office formats ({', '.join(OFFICE_FORMATS)}) "
                    f"and HTML formats ({', '.join(HTML_FORMATS)})"
                )

        except Exception as e:
            logfire.error(f"Error during parsing: {str(e)}")
            raise

        msg = f"Parsing {file_path} completed! Extracted {len(content_list)} content block"
        logfire.info(msg)

        if len(content_list) == 0:
            raise ValueError("Parsing failed: No content extracted")

        self._store_cache_result(
            cache_key, content_list, file_path, parse_method.value, parser
        )

        doc_id = self._generate_doc_id(file_path)

        if display_stats:
            logfire.info("\n Content information: ")
            logfire.info(f"* Total content in list: {len(content_list)}")

            block_types: Dict[str, int] = {}
            for block in content_list:
                if isinstance(block, dict):
                    block_type = block.get("type", "Unknown")
                    if isinstance(block_type, str):
                        block_types[block_type] = block_types.get(block_type, 0) + 1

            logfire.info("* Content block types: ")

            for block_type, count in block_types.items():
                logfire.info(f" - {block_type} : {count}")

        return content_list, doc_id

    async def ingest_document(
        self,
        file_path: str,
        parse_method: ParseMethod,
        chunking_strategy: ChunkingType,
        doc_id: str | None = None,
        split_by_character: str = "\n\n",
        parser: str | None = "auto",
    ):
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content_list, doc_id = await self.process_document_complete(
            file_path=file_path,
            doc_id=doc_id,
            split_by_character=split_by_character,
            parse_method=parse_method,
        )
        logfire.info(f"Stage 1 complete: {len(content_list)} content list")

        chunks = await self._chunk_doc_content(
            content_list=content_list,
            file_path=file_path,
            doc_id=doc_id,
            chunking_strategy=chunking_strategy,
        )
        self.in_storage.upload(key="chunks", data=chunks)
        logfire.info(f"Document chunking completed: {len(chunks)}")

        embedded_chunks = await self.embedding_service.embed_chunks(chunks)
        self.in_storage.upload(key="embedded_chunks", data=embedded_chunks)
        logfire.info(f"Stage 3 complete: {len(embedded_chunks)} vectors")

        # Rebuild sparse index to stay in sync
        logfire.info("Rebuilding sparse index...")
        await bootstrap_sparse_index(self.storage_service, self.sparse_index)

        await self.storage_service.upsert_embedded_chunks(embedded_chunks)
        logfire.info("Stage 4 complete: stored in Qdrant")

        return {
            "doc_id": doc_id,
            "chunks_produced": len(chunks),
            "vectors_stored": len(embedded_chunks),
        }

    async def query(
        self, question: str, top_k: int = 5, doc_id_filter: str | None = None
    ):
        if not question:
            raise ValueError("Please enter your question!")

        query_vector = await self.embedding_service.embed_single(question)

        results = await self.storage_service.search(
            query_vector=query_vector, top_k=top_k, doc_id_filter=doc_id_filter
        )

        return results
