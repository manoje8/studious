from pathlib import Path
from typing import Union, Optional, Dict, Tuple, Any
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
        pdf_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs,
    ):
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

            converter = self._get_converter()
            result = converter.convert(str(pdf_path))

            return result
        except Exception as e:
            logfire.error(f"Error in parse pdf: {str(e)}")
            raise

    def parse_html(self, html_path: Path, output_dir: Optional[str] = None, **kwargs):
        try:
            html_path = Path(html_path)
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
