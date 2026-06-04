import base64
from pathlib import Path
from typing import Union, Optional, Dict, Tuple, Any, List
import threading

import logfire
from docling.datamodel.pipeline_options import TableFormerMode

from src.ingestion.parser.parser import Parser
from src.utils.config import config
from src.utils.constants import HTML_FORMATS, OFFICE_FORMATS


class DoclingParser(Parser):
    """
    Docling document parsing utility class.

    TODO: Read blocks from document response
    """

    def __init__(self):
        super().__init__()
        self._converter_cache: Dict[Tuple, Any] = {}
        self._converter_cache_lock = threading.Lock()

    def _get_converter(self):
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        table_mode = str(config.TABLE_MODE).lower()
        do_tables = config.DO_TABLES
        do_ocr = config.DO_OCR

        cache_key = (table_mode, do_tables, do_ocr)
        cached = self._converter_cache.get(cache_key)

        if cached is not None:
            return cached

        pipeline_options = PdfPipelineOptions()

        if hasattr(pipeline_options, "do_ocr"):
            pipeline_options.do_ocr = do_ocr

        if hasattr(pipeline_options, "do_table_structure"):
            pipeline_options.do_table_structure = do_tables

        if hasattr(pipeline_options, "table_structure_options"):
            try:
                pipeline_options.table_structure_options.mode = (
                    TableFormerMode.ACCURATE
                    if table_mode == "accurate"
                    else TableFormerMode.FAST
                )
            except Exception as e:
                logfire.debug(f"Could not set TableFormer mode '{table_mode}': {e}")

        if hasattr(pipeline_options, "generate_picture_images"):
            pipeline_options.generate_picture_images = True

        if hasattr(pipeline_options, "images_scale"):
            pipeline_options.images_scale = 2.0

        with self._converter_cache_lock:
            cached = self._converter_cache.get(cache_key)

            if cached is not None:
                return cached

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

            self._converter_cache[cache_key] = converter

            return converter

    def read_from_block_recursive(
        self,
        block,
        type: str,
        cnt: int,
        num: str,
        docling_content: Dict[str, Any],
        output_dir: Path = "data",
    ) -> List[Dict[str, Any]]:
        content_list = []
        if not block.get("children"):
            cnt += 1
            content_list.append(self.read_from_block(block, type, output_dir, cnt, num))
        else:
            if type not in ["groups", "body"]:
                cnt += 1
                content_list.append(
                    self.read_from_block(block, type, output_dir, cnt, num)
                )
            members = block["children"]
            for member in members:
                cnt += 1
                member_tag = member["$ref"]
                # JSON References follow the form "#/<type>/<index>" (e.g. "#/body/0")
                ref_parts = member_tag.split("/")
                if len(ref_parts) < 3:
                    self.logger.warning(
                        f"Unexpected $ref format (expected #/<type>/<index>): {member_tag!r}"
                    )
                    continue
                member_type = ref_parts[1]
                member_num = ref_parts[2]
                try:
                    member_block = docling_content[member_type][int(member_num)]
                except (KeyError, ValueError, IndexError) as e:
                    self.logger.warning(f"Could not resolve $ref {member_tag!r}: {e}")
                    continue
                content_list.extend(
                    self.read_from_block_recursive(
                        member_block,
                        member_type,
                        output_dir,
                        cnt,
                        member_num,
                        docling_content,
                    )
                )
        return content_list

    def read_from_block(
        self, block, type: str, output_dir: Path, cnt: int, num: str
    ) -> Dict[str, Any]:
        if type == "texts":
            if block["label"] == "formula":
                return {
                    "type": "equation",
                    "img_path": "",
                    "text": block["orig"],
                    "text_format": "unknown",
                    "page_idx": cnt // 10,
                }
            else:
                return {
                    "type": "text",
                    "text": block["orig"],
                    "page_idx": cnt // 10,
                }
        elif type == "pictures":
            try:
                base64_uri = block["image"]["uri"]
                # base64 data URIs have the form "data:<mime>;base64,<data>"
                # but some exporters may omit the prefix
                parts = base64_uri.split(",", 1)
                base64_str = parts[1] if len(parts) == 2 else parts[0]
                # Create images directory within the docling subdirectory
                image_dir = output_dir / "images"
                image_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                image_path = image_dir / f"image_{num}.png"
                with open(image_path, "wb") as f:
                    f.write(base64.b64decode(base64_str))
                return {
                    "type": "image",
                    "img_path": str(image_path.resolve()),  # Convert to absolute path
                    "image_caption": block.get("caption", ""),
                    "image_footnote": block.get("footnote", ""),
                    "page_idx": cnt // 10,
                }
            except Exception as e:
                self.logger.warning(f"Failed to process image {num}: {e}")
                return {
                    "type": "text",
                    "text": f"[Image processing failed: {block.get('caption', '')}]",
                    "page_idx": cnt // 10,
                }
        else:
            try:
                return {
                    "type": "table",
                    "img_path": "",
                    "table_caption": block.get("caption", ""),
                    "table_footnote": block.get("footnote", ""),
                    "table_body": block.get("data", []),
                    "page_idx": cnt // 10,
                }
            except Exception as e:
                self.logger.warning(f"Failed to process table {num}: {e}")
                return {
                    "type": "text",
                    "text": f"[Table processing failed: {block.get('caption', '')}]",
                    "page_idx": cnt // 10,
                }

    def check_installation(self) -> bool:
        try:
            from docling.document_converter import DocumentConverter  # noqa: F401

            return True
        except ImportError:
            logfire.debug(
                "Docling Python package is not installed. "
                "Install it with: pip install docling"
            )
            return False

    def parse_doc(
        self,
        file_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs,
    ):
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Document file does not exist: {file_path}")

            ext = file_path.suffix.lower()

            if ext == ".pdf":
                return self.parse_pdf(file_path)
            elif ext in HTML_FORMATS:
                return self.parse_html(file_path)
            elif ext in OFFICE_FORMATS:
                pass

            else:
                raise ValueError(
                    f"Unsupported file format: {ext}. "
                    f"Docling only supports PDF files, Office formats ({', '.join(OFFICE_FORMATS)}) "
                    f"and HTML formats ({', '.join(HTML_FORMATS)})"
                )

            converter = self._get_converter()
            result = converter.convert(str(file_path))

            return result
        except Exception as e:
            logfire.error(f"Error in parse file_path: {str(e)}")
            raise

    def parse_pdf(
        self,
        file_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs,
    ):
        try:
            pdf_path = Path(file_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

            converter = self._get_converter()
            result = converter.convert(str(pdf_path))

            return result
        except Exception as e:
            logfire.error(f"Error in parse pdf: {str(e)}")
            raise

    def parse_html(self, file_path: Path, output_dir: Optional[str] = None, **kwargs):
        try:
            html_path = Path(file_path)
            if not html_path.exists():
                raise FileNotFoundError(f"HTML file does not exist: {html_path}")

            if html_path.suffix.lower() not in HTML_FORMATS:
                raise ValueError(f"Unsupported HTML format: {html_path.suffix}")

            converter = self._get_converter()
            result = converter.convert(str(html_path))

            return result
        except Exception as e:
            logfire.error(f"Error in parse pdf: {str(e)}")
            raise
