import re
from dataclasses import dataclass
from typing import Any

from medici.common.models import Chunk
from medici.common.utils.tokenizer import Tokenizer


@dataclass
class BatchProcess:
    successful_files: list[str]
    failed_files: list[str]
    total_files: int
    processing_time: float
    errors: dict[str, str]
    output_dir: str

    @property
    def success_rate(self) -> float:
        if self.total_files == 0:
            return 0.0
        return (len(self.successful_files) / self.total_files) * 100

    def summary(self) -> str:
        return (
            f"Batch Processing Summary:\n"
            f"  Total files: {self.total_files}\n"
            f"  Successful: {len(self.successful_files)} ({self.success_rate:.1f}%)\n"
            f"  Failed: {len(self.failed_files)}\n"
            f"  Processing time: {self.processing_time:.2f} seconds\n"
            f"  Output directory: {self.output_dir}\n"
        )


class Chunking:
    def _clean_text(self, text: str) -> str:
        """Fix common OCR artifacts such as space-separated characters."""
        cleaned = re.sub(
            r"\b(?:(?<=\s)|(?<=^))([A-Za-z\u2013\u2014\u2018\u2019\u201c\u201d]"
            r"(?: [A-Za-z\u2013\u2014\u2018\u2019\u201c\u201d]){4,})\b",
            lambda m: m.group(0).replace(" ", ""),
            text,
        )
        cleaned = re.sub(r" {2,}", " ", cleaned)
        return cleaned.strip()


def build_parent_child_chunk(
    chunks: list[Chunk], tokenizer: Tokenizer, parent_window: int = 3
) -> list[Chunk]:
    """
    Enrich each child chunk with a broader parent context window.

    Each chunk gains a ``parent_text`` field containing the concatenated
    text of itself and its neighboring chunks. The ``parent_token_count``
    field is also populated for budget-aware retrieval.

    Parameters
    ----------
    chunks:
        List of Chunk objects to enrich.
    parent_window:
        Total number of chunks to include in the parent window
        (centered on the current chunk).
        :param parent_window:
        :param chunks:
        :param tokenizer:
    """

    enriched = []
    for i, chunk in enumerate(chunks):
        start = max(0, i - parent_window // 2)
        end = min(len(chunks), i + parent_window // 2 + 1)

        parent_text = " ".join(c.text for c in chunks[start:end])
        parent_token_count = tokenizer.count(parent_text)

        enriched.append(
            Chunk(
                text=chunk.text,
                chunk_index=chunk.chunk_index,
                doc_id=chunk.doc_id,
                source_file=chunk.source_file,
                chunk_type=chunk.chunk_type,
                section_title=chunk.section_title,
                page_numbers=chunk.page_numbers,
                block_types=chunk.block_types,
                token_count=chunk.token_count,
                parent_text=parent_text,
                parent_token_count=parent_token_count,
                parent_window_start=start,
                parent_window_end=end,
                image_path=chunk.image_path,
            )
        )

    return enriched


def _as_text(val) -> str:
    if isinstance(val, list):
        return " ".join(v for v in val if isinstance(v, str)).strip()
    if isinstance(val, str):
        return val.strip()
    return ""


def _serialize_table(item: dict[str, Any]) -> str | None:
    table_body = item.get("table_body") or {}
    grid = table_body.get("grid")

    if not grid:
        return None

    def cell_text(cell: dict[str, Any]) -> str:
        text = (cell.get("text") or "").strip()
        return text.replace("|", "\\|").replace("\n", " ")

    num_rows = table_body.get("num_rows", len(grid))
    num_cols = table_body.get("num_cols", max((len(r) for r in grid), default=0))
    dense = [["" for _ in range(num_cols)] for _ in range(num_rows)]

    for row in grid:
        for cell in row:
            text = cell_text(cell)
            r0, r1 = cell.get("start_row_offset_idx", 0), cell.get("end_row_offset_idx", 1)
            c0, c1 = cell.get("start_col_offset_idx", 0), cell.get("end_col_offset_idx", 1)
            for r in range(r0, min(r1, num_rows)):
                for c in range(c0, min(c1, num_cols)):
                    dense[r][c] = text
    if not dense:
        return None

    header_row_indices = []
    for i, row in enumerate(grid):
        if row and all(c.get("column_header") for c in row):
            header_row_indices.append(i)
        else:
            break  # header rows are contiguous from the top

    header_idx = header_row_indices[0] if header_row_indices else 0
    header = dense[header_idx]
    body_rows = dense[header_idx + 1 :] if header_row_indices else dense[1:]

    lines = []
    caption = (
        (item.get("table_caption") or "").strip()
        if isinstance(item.get("table_caption"), str)
        else ""
    )
    if caption:
        lines.append(f"Table: {caption}")

    lines.append("| " + " | ".join(h or " " for h in header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in body_rows:
        lines.append("| " + " | ".join(c or " " for c in row) + " |")

    footnote = item.get("table_footnote")
    if isinstance(footnote, str) and footnote.strip():
        lines.append(f"\nNote: {footnote.strip()}")
    elif isinstance(footnote, list) and footnote:
        joined = " ".join(f for f in footnote if isinstance(f, str)).strip()
        if joined:
            lines.append(f"\nNote: {joined}")

    return "\n".join(lines)


def _serialize_equation(item: dict[str, Any]) -> str | None:
    text = item.get("text") or item.get("latex")
    return f"Equation: {text.strip()}" if isinstance(text, str) and text.strip() else None


def serialize_image_caption(item: dict[str, Any]) -> str | None:
    caption = _as_text(item.get("img_caption") or item.get("image_caption") or item.get("caption"))
    return f"Image caption: {caption}" if caption else None


def extract_page_numbers(item: dict[str, Any]) -> list[int]:
    page_idx = item.get("page_idx")
    if isinstance(page_idx, int):
        return [page_idx]
    pages = item.get("page_numbers")
    if isinstance(pages, list):
        return [p for p in pages if isinstance(p, int)]
    return []


def serialize_multimodal_item(item: dict[str, Any]) -> str | None:
    """
    Convert a multimodal content_list item (table, equation, image, ...)
    into a text serialization suitable for embedding.
    Returns None if there's nothing meaningful to embed.
    """
    item_type = item.get("type", "unknown")

    if item_type == "table":
        return _serialize_table(item)
    if item_type == "equation":
        return _serialize_equation(item)
    if item_type == "image":
        return serialize_image_caption(item)

    # Fallback for unknown multimodal types
    for key in ("text", "content", "caption"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None
