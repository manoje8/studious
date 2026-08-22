import argparse
import asyncio
import json
from pathlib import Path

import logfire

from medici.common.cache.embedding_cache import EmbeddingCache
from medici.common.services.qdrant import QdrantStorageService
from medici.common.utils.config import config
from medici.common.utils.constants import ParseMethod
from medici.common.utils.tokenizer import TikTokenTokenizer
from medici.ingestion.embedding import EmbeddingService
from medici.ingestion.processor import Processor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse, ingest, and query documents (PDF, Office, HTML)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser(
        "parse",
        help="Parse a document and extract content blocks (no embedding/storage).",
    )
    parse_parser.add_argument("file_path", type=str, help="Path to the input document.")
    parse_parser.add_argument("--output-dir", type=str, default=None)
    parse_parser.add_argument("--parse-method", type=str, default=ParseMethod.DOCLING)
    parse_parser.add_argument("--display-stats", action="store_true")
    parse_parser.add_argument("--split-by-character", type=str, default=None)
    parse_parser.add_argument("--split-by-character-only", type=str, default=None)
    parse_parser.add_argument("--doc-id", type=str, default=None)
    parse_parser.add_argument("--file-name", type=str, default=None)

    # Ingestion
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Parse, embed, and store a document in Qdrant.",
    )
    ingest_parser.add_argument("file_path", type=str, help="Path to the input document.")
    ingest_parser.add_argument("--parse-method", type=str, default=ParseMethod.DOCLING)
    ingest_parser.add_argument("--split-by-character", type=str, default=None)

    ingest_parser.add_argument(
        "--doc-id", type=str, default=None, help="Optional document identifier."
    )
    ingest_parser.add_argument(
        "--chunking-strategy",
        type=str,
        default="structure",
        help="Chunking strategy passed to process_document_complete (default: structure).",
    )

    # Query
    query_parser = subparsers.add_parser(
        "query",
        help="Embed a question and retrieve the most relevant chunks from Qdrant.",
    )
    query_parser.add_argument("question", type=str, help="Question to search for.")
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )
    query_parser.add_argument(
        "--doc-id-filter",
        type=str,
        default=None,
        help="Restrict search to a specific document ID.",
    )

    return parser.parse_args()


async def main():
    logfire.configure(service_name="PROCESS CLI")
    args = parse_args()

    emb_cache = await EmbeddingCache.create(dsn=config.POSTGRES_CONN_STRING, max_entries=50_000)

    embedding_service = EmbeddingService(
        model_name=config.EMBEDDING_MODEL_NAME,
        dimensions=config.EMBEDDING_DIMENSIONS,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        cache=emb_cache,
    )

    storage_service = QdrantStorageService(
        url=config.QDRANT_CLUSTER_ENDPOINT,
        vector_size=embedding_service.vector_size,
        collection_name=config.QDRANT_COLLECTION_NAME,
    )
    tokenizer = TikTokenTokenizer()
    processor = Processor(tokenizer, embedding_service, storage_service)

    if args.command == "parse":
        output_dir = str(Path(args.output_dir)) if args.output_dir else None

        content_list = await processor.process_document(
            file_path=Path(args.file_path),
            output_dir=output_dir,
            parse_method=args.parse_method,
            display_stats=args.display_stats,
            split_by_character=args.split_by_character,
            split_by_character_only=args.split_by_character_only,
            doc_id=args.doc_id,
            file_name=args.file_name,
        )

        if output_dir and content_list:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            output_file = out_path / f"{Path(args.file_path).stem}_content.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(content_list, f, indent=2, default=str)
            print(f"Saved {len(content_list)} content blocks to {output_file}")
        else:
            print(f"Extracted {len(content_list)} content blocks.")

    elif args.command == "ingest":
        result = await processor.ingest_document(
            file_path=args.file_path,
            doc_id=args.doc_id,
            parse_method=args.parse_method,
        )
        print(
            f"Ingestion complete — "
            f"doc_id={result['doc_id']}, "
            f"chunks={result['chunks_produced']}, "
            f"vectors stored={result['vectors_stored']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
