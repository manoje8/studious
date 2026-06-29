import hashlib
from pathlib import Path

import trafilatura
from bs4 import BeautifulSoup

from src.utils.constants import BLOCK_TAGS, HEADING_TAGS, SKIP_TAGS, ParseMethod


class Parser:
    @classmethod
    def _parse_inline_markdown(cls, text: str):
        """Process inline markdown formatting (bold, italic, code, links)"""
        import re

        text = text.replace("&", "&amp").replace("<", "&lt;").replace(">", "&gt;")

        # Bold text: **text** or __text__
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"__(.*?)__", r"<b>\1</b>", text)

        # Italic text: *text* or _text_ (but not in the middle of words)
        text = re.sub(r"(?<!\w)\*([^*\n]+?)\*(?!\w)", r"<i>\1</i>", text)
        text = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"<i>\1</i>", text)

        # Inline code: `code`
        text = re.sub(
            r"`([^`]+?)`",
            r'<font name="Courier" size="9" color="darkred">\1</font>',
            text,
        )

        # Links: [text](url) - convert to text with URL annotation
        def link_replacer(match):
            link_text = match.group(1)
            url = match.group(2)
            return f'<link href="{url}" color="blue"><u>{link_text}</u></link>'

        text = re.sub(r"\[([^\]]+?)\]\(([^)]+?)\)", link_replacer, text)

        # Strikethrough: ~~text~~
        text = re.sub(r"~~(.*?)~~", r"<strike>\1</strike>", text)

        return text

    @staticmethod
    def _unique_output_dir(base_dir: str | Path, file_path: str | Path) -> Path:
        """
        Create a unique output subdirectory for a file to prevent same-name collisions
        """
        file_path = Path(file_path).resolve()
        stem = file_path.stem
        path_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        return Path(base_dir) / f"{stem}_{path_hash}"

    def extract_html_content(self, html: str) -> list[dict]:
        content = trafilatura.extract(
            html,
            include_tables=True,
            include_links=False,
            include_images=False,
            no_fallback=False,
            output_format="xml",
        )

        nodes = self._parse_structure(content)
        return nodes

    def parse_pdf(
        self,
        pdf_path: str | Path,
        output_dir: str | None = None,
        method: ParseMethod = ParseMethod.DOCLING.value,
        lang: str | None = None,
        **kwargs,
    ):
        """Abstract method to parse PDF document"""
        raise NotImplementedError("parse_pdf must be implemented by sub-classes")

    def parse_doc(
        self,
        file_path: str | Path,
        output_dir: str | None = None,
        method: str = "auto",
        lang: str | None = None,
        **kwargs,
    ):
        raise NotImplementedError("parse_office_doc must be implemented by sub-classes")

    def check_installation(self) -> bool:
        raise NotImplementedError("check_installation must be implemented by subclasses")

    def _get_clean_text(self, tag) -> str:
        return " ".join(tag.get_text(separator=" ").split())

    def _table_to_text(self, table_tag) -> str:
        """Convert HTML table to a row-per-line natural language format."""
        rows = []
        headers = [th.get_text(strip=True) for th in table_tag.find_all("th")]

        for _row in table_tag.find_all("tr"):
            cells = [td.get_text(strip=True) for td in table_tag.find_all("td")]

            if not cells:
                continue

            if headers:
                rows.append(" | ".join(f"{h}: {c}") for h, c in zip(headers, cells, strict=False))
            else:
                rows.append(" | ".join(cells))

        return "\n".join(rows)

    def _parse_structure(self, html: str) -> list[dict]:
        """Returns a list of nodes: {type, level, text, path}"""

        soup = BeautifulSoup(html, "lxml")

        for tag in soup(list(SKIP_TAGS)):
            tag.decompose()

        nodes = []
        heading_stack = []  # Track current heading breadcrumb

        for tag in soup.find_all(True):
            name = tag.name

            if name in HEADING_TAGS:
                level = int(name[1])

                text = self._get_clean_text(tag)

                heading_stack = [h for h in heading_stack if h["level"] < level]
                heading_stack.append({"level": level, "text": text})

                nodes.append(
                    {
                        "type": "heading",
                        "level": level,
                        "text": text,
                        "breadcrumb": " > ".join(h["text"] for h in heading_stack[:-1]),
                    }
                )
            elif name in BLOCK_TAGS:
                text = self._get_clean_text(tag)

                if len(text) < 20:
                    continue

                nodes.append(
                    {
                        "type": "block",
                        "text": text,
                        "breadcrumb": " > ".join(h["text"] for h in heading_stack),
                    }
                )

            elif name == "table":
                nodes.append(
                    {
                        "type": "table",
                        "text": self._table_to_text(tag),
                        "breadcrumb": " > ".join(h["text"] for h in heading_stack),
                    }
                )

        return nodes
