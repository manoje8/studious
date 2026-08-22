import hashlib
from pathlib import Path

import trafilatura
from bs4 import BeautifulSoup

from medici.common.utils.constants import BLOCK_TAGS, HEADING_TAGS, SKIP_TAGS, ParseMethod


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

    def _parse_structure(self, html: str) -> str:
        """Returns a list of nodes: {type, level, text, path}"""

        soup = BeautifulSoup(html, "lxml")

        for tag in soup(list(SKIP_TAGS)):
            tag.decompose()

        nodes = []
        heading_stack = []

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

        return self._node_to_string(nodes)

    def _get_clean_text(self, tag) -> str:
        return " ".join(tag.get_text(separator=" ").split())

    def _table_to_text(self, table_tag) -> str:
        rows_tags = table_tag.find_all(["tr", "row"])

        def is_header_cell(cell) -> bool:
            return cell.name == "th" or cell.get("role") == "head"

        headers = []
        if rows_tags:
            first_cells = rows_tags[0].find_all(["td", "th", "cell"])
            if any(is_header_cell(c) for c in first_cells):
                headers = [c.get_text(strip=True) for c in first_cells]

        out = []
        for row in rows_tags:
            cells = row.find_all(["td", "th", "cell"])
            values = [c.get_text(strip=True) for c in cells]
            if not values:
                continue
            if (
                headers
                and values
                == [
                    c.get_text(strip=True)
                    for c in row.find_all(["td", "th", "cell"])
                    if is_header_cell(c) or True
                ]
                and row is rows_tags[0]
                and any(is_header_cell(c) for c in cells)
            ):
                continue
            if headers and len(headers) == len(values):
                out.append(" | ".join(f"{h}: {v}" for h, v in zip(headers, values, strict=False)))
            else:
                out.append(" | ".join(values))

        return "\n".join(out)

    def _node_to_string(self, nodes: list[dict]) -> str:
        parts = []

        for node in nodes:
            text = node.get("text", "")
            if not text.strip():
                continue

            node_type = node["type"]

            if node_type == "heading":
                level = max(1, min(6, node.get("level", 1)))
                parts.append(f"{'#' * level} {text}")

            elif node_type == "block":
                breadcrumb = node.get("breadcrumb")
                prefix = f"[{breadcrumb}]\n" if breadcrumb else ""
                parts.append(f"{prefix}{text}")

            elif node_type == "table":
                breadcrumb = node.get("breadcrumb")
                prefix = f"[{breadcrumb}]\n" if breadcrumb else ""
                parts.append(f"{prefix}<table>\n{text}\n</table>")

            else:
                raise ValueError(f"Unhandled node type in _node_to_string: {node_type!r}")

        return "\n\n".join(parts)
