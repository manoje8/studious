import asyncio
import hashlib
import json
from pathlib import Path
from typing import Dict

import logfire

from src.ingestion.parser.google_doc_ai import GoogleDocAI
from src.ingestion.chunk import Chunking
from src.ingestion.embedding import EmbeddingService
from src.services.qdrant import QdrantStorageService
from src.utils.constants import GOOGLE_DOC_AI, HTML_FORMATS, OFFICE_FORMATS
from src.utils.config import config


class Processor:
    def __init__(self):
        logfire.configure(service_name=self.__class__.__name__)

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

    def _get_parser(self, parser_type: str):
        parser_name = parser_type.strip().lower()

        if parser_name == GOOGLE_DOC_AI:
            return GoogleDocAI()
        else:
            raise ValueError(f"Unsupported Parser type: {parser_type}")

    def _generate_cache_key(self, file_path: Path, parse_method: str = None):
        m_time = file_path.stat().st_mtime

        config_dict = {
            "file_path": file_path,
            "m_time": m_time,
            "parse_method": parse_method,
        }

        config_str = json.dumps(config_dict, sort_keys=True)
        cache_key = hashlib.md5(config_str.encode()).hexdigest()

        return cache_key

    def _get_cached_result(
        self, cache_key: str, file_path: Path, parse_method: str = None
    ):
        pass

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
        chunking_strategy: str,
        split_by_character: str | None = None,
    ):
        logfire.info(f"Starting chunking with strategy: {chunking_strategy}")

        chunking = Chunking()

        if chunking_strategy == "structure":
            chunks = chunking.chunk_by_structure(
                content_list=content_list,
                doc_id=doc_id or file_path.stem,
                source_file=str(file_path),
            )
        elif chunking_strategy == "fixed":
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

        return chunks

    async def process_document_complete(
        self,
        file_path: str | Path,
        output_dir: str = None,
        parse_method: str = None,
        display_stats: bool = None,
        split_by_character: str | None = None,
        split_by_character_only: str | None = None,
        doc_id: str | None = None,
        file_name: str | None = None,
        **kwargs,
    ):
        logfire.info(f"Starting document parsing: {file_path}")

        ext = file_path.suffix.lower()

        try:
            doc_parser = self._get_parser(GOOGLE_DOC_AI)
            if ext in [".pdf"]:
                logfire.info("Detected PDF file, parsing the pdf...")

                content_list = await asyncio.to_thread(
                    doc_parser.parse_pdf,
                    pdf_path=file_path,
                    method=parse_method,
                    **kwargs,
                )
            elif ext in HTML_FORMATS:
                logfire.info("Detected HTML file, parsing html...")
                content_list = await asyncio.to_thread(
                    doc_parser.parse_html,
                    file_path=file_path,
                    method=parse_method,
                    **kwargs,
                )

            elif ext in OFFICE_FORMATS:
                logfire.info("Detected office file, parsing document...")
                content_list = await asyncio.to_thread(
                    doc_parser.parse_doc,
                    file_path=file_path,
                    method=parse_method,
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

        # TODO: Implement Cache result

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
        doc_id: str | None = None,
        chunking_strategy: str | None = None,
        split_by_character: str = "\n\n",
    ):
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content_list, doc_id = await self.process_document_complete(
            file_path=file_path,
            doc_id=doc_id,
            chunking_strategy=chunking_strategy,
            split_by_character=split_by_character,
        )
        logfire.info(f"Stage 1 complete: {len(content_list)} content list")

        chunks = await self._chunk_doc_content(
            content_list=content_list,
            file_path=file_path,
            doc_id=doc_id,
            chunking_strategy=chunking_strategy,
        )

        embedded_chunks = await self.embedding_service.embed_chunks(chunks)
        logfire.info(f"Stage 3 complete: {len(embedded_chunks)} vectors")

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
