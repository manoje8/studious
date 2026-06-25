import io
from pathlib import Path
from typing import Union, Optional

import logfire
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
from google.cloud import documentai
from pypdf import PdfReader, PdfWriter

from .parser import Parser
from src.utils.config import config
from src.utils.constants import GOOGLE_MIME_TYPES, HTML_FORMATS, OFFICE_FORMATS


class GoogleDocAI(Parser):
    def __init__(self):
        super().__init__()
        self.client = documentai.DocumentProcessorServiceClient()

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
                msg = f"PDF file doesn't exist: {pdf_path}"
                logfire.error(msg)
                raise FileNotFoundError(msg)

            content = self._process_pdf(pdf_path)
            return content

        except (FileNotFoundError, ValueError):
            raise
        except Exception as e:
            logfire.error(f"Error in parsing pdf: {str(e)}")
            raise

    def parse_doc(
        self,
        file_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs,
    ):
        doc_path = Path(file_path)

        if not doc_path.exists():
            raise FileNotFoundError(f"Office document doesn't exist: {doc_path}")

        if doc_path.suffix.lower() not in OFFICE_FORMATS:
            raise ValueError(f"Unsupported Office format: {doc_path.suffix}")

        name_without_suff = doc_path.stem

        logfire.info(f"Parsing {name_without_suff} document")

        ext = doc_path.suffix

        with open(doc_path, "rb") as f:
            content = f.read()

        content = self._process_with_doc_ai(content, ext)
        return content

    def parse_html(
        self,
        file_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs,
    ):
        try:
            html_path = Path(file_path)

            if not html_path.exists():
                raise FileNotFoundError(f"HTML file doesn't exist: {html_path}")

            if html_path.suffix.lower() not in HTML_FORMATS:
                raise ValueError(f"Unsupported HTML format: {html_path.suffix}")

            name_without_suff = html_path.stem

            logfire.info(f"Parsing {name_without_suff} html")

            with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            content_list = self.extract_html_content(content)

            return content_list

        except Exception as e:
            logfire.error(f"Error in parsing HTML: {str(e)}")
            raise

    def check_installation(self) -> bool:
        try:
            from google.cloud import documentai  # noqa: F401

            return True
        except ImportError:
            logfire.error(
                "google-cloud-documentai is not installed. Install it with: pip install google-cloud-documentai"
            )
            return False

    def _process_pdf(self, file_path: Optional[Path | str]):
        name_without_suffix = file_path.stem
        ext = file_path.suffix

        reader = PdfReader(file_path)
        pages = reader.pages
        total_pages = len(pages)
        max_page: int = config.MAX_PAGE_PER_PARSE

        logfire.debug(f"{name_without_suffix} PDF with {total_pages}'s pages")

        text = ""

        if total_pages <= max_page:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            text = self._process_with_doc_ai(file_bytes, ext)
        else:
            logfire.info(f"PDF exceeds {max_page} pages and splitting into chunks...")

            for i in range(0, total_pages, max_page):
                writer = PdfWriter()

                split_end = min(i + max_page, total_pages)

                for page_n in range(i, split_end):
                    writer.add_page(reader[page_n])

                with io.BytesIO() as bs:
                    writer.write(bs)
                    file_bytes = bs.getvalue()

                with logfire.span(f"Processing pages {i+1} to {split_end}"):
                    split_text = self._process_with_doc_ai(file_bytes, ext)

                    text += split_text + "\n"

        return text

    def _process_with_doc_ai(self, content: bytes, ext: str) -> str:
        try:
            ext_key = ext.lower().lstrip(".")
            mime_type = GOOGLE_MIME_TYPES.get(ext_key)

            if not mime_type:
                raise ValueError(f"No MIME type mapping for extension: {ext}")

            processor_name = self.client.processor_path(
                config.PROJECT_ID,
                config.GCP_DOC_AI_LOCATION,
                config.GCP_DOC_AI_PROCESSOR_ID,
            )

            raw_doc = documentai.RawDocument(content=content, mime_type=mime_type)
            request = documentai.ProcessRequest(
                name=processor_name, raw_document=raw_doc
            )

            result = self.client.process_document(request=request)
            return result.document.text

        except ResourceExhausted:
            logfire.error("Doc AI quota exhausted")
            raise
        except ServiceUnavailable as e:
            logfire.warning(f"Doc AI transiently unavailable: {e}")
            raise
