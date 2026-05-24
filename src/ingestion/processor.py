import asyncio
import hashlib
import json
import argparse
from pathlib import Path
from typing import Dict

import logfire

from parser.google_doc_ai import GoogleDocAI
from src.ingestion.chunk import Chunking
from utils.constants import GOOGLE_DOC_AI, HTML_FORMATS, OFFICE_FORMATS


class Processor:
    def __init__(self):
        logfire.configure(service_name=self.__class__.__name__)

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

    def _chunk_doc_content(
        self,
        file_path: Path,
        content_list: list[dict],
        doc_id: str,
        split_by_character,
        chunking_strategy: str,
    ):
        logfire.info(f"Starting chunking with strategy: {chunking_strategy}")

        if chunking_strategy == "structure":
            chunks = Chunking.chunk_by_structure(
                content_list,
                doc_id=doc_id or file_path.stem,
                source_file=str(file_path),
            )
        elif chunking_strategy == "fixed":
            chunks = Chunking.chunk_fixed(
                content_list,
                doc_id=doc_id or file_path.stem,
                source_file=str(file_path),
                split_by_character=split_by_character or "\n\n",
            )
        else:
            return content_list

        logfire.info(
            f"Chunking complete: {len(chunks)} chunks produced from {len(content_list)} blocks"
        )

        return chunks

    async def process_document_complete(
        self,
        file_path: str,
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

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse a document (PDF, Office, HTML) and extract content blocks."
    )
    parser.add_argument("file_path", type=str, help="Path to the input document.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save extracted content (optional).",
    )
    parser.add_argument(
        "--parse-method",
        type=str,
        default=None,
        help="Parsing method (passed to underlying parser).",
    )
    parser.add_argument(
        "--display-stats",
        action="store_true",
        help="Show statistics about extracted content blocks.",
    )
    parser.add_argument(
        "--split-by-character",
        type=str,
        default=None,
        help="Character to split content by (passed via **kwargs).",
    )
    parser.add_argument(
        "--split-by-character-only",
        type=str,
        default=None,
        help="Character to split content by, only this method (passed via **kwargs).",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Document identifier (passed via **kwargs).",
    )
    parser.add_argument(
        "--file-name",
        type=str,
        default=None,
        help="Override original file name (passed via **kwargs).",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    extra_kwargs = {
        k: v
        for k, v in vars(args).items()
        if k in ["split_by_character", "split_by_character_only", "doc_id", "file_name"]
        and v is not None
    }

    processor = Processor()

    output_dir = args.output_dir
    if output_dir is not None:
        output_dir = str(Path(output_dir))

    content_list = await processor.process_document_complete(
        file_path=args.file_path,
        output_dir=output_dir,
        parse_method=args.parse_method,
        display_stats=args.display_stats,
        split_by_character=extra_kwargs.get("split_by_character"),
        split_by_character_only=extra_kwargs.get("split_by_character_only"),
        doc_id=extra_kwargs.get("doc_id"),
        file_name=extra_kwargs.get("file_name"),
        **extra_kwargs,
    )

    if output_dir and content_list:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        output_file = out_path / f"{Path(args.file_path).stem}_content.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(content_list, f, indent=2, default=str)
        print(f"Saved extracted content to {output_file}")
    else:
        print(f"Extracted {len(content_list)} content blocks.")
        if args.display_stats:
            pass


if __name__ == "__main__":
    asyncio.run(main())
