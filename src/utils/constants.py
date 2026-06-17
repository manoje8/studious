from enum import Enum

SUPPORTED_PARSERS = "DocumentAI"
HTML_FORMATS = {".html", ".htm", ".xhtml"}
OFFICE_FORMATS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
IMAGE_FORMATS = {".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
TEXT_FORMATS = {".txt", ".md"}

GOOGLE_MIME_TYPES = {
    "html": "text/html",
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "jpg": "image/jpeg",
    "png": "image/png",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

SKIP_TAGS = {
    "script",
    "style",
    "noscript",
    "head",
    "meta",
    "link",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "button",
}

BLOCK_TAGS = {"p", "li", "blockquote", "pre", "td", "th", "dd", "dt"}

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


GOOGLE_DOC_AI = "google_doc_ai"


CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MIN_CHUNK_SIZE = 50


class StorageType(Enum):
    LOCAL = "local"
    GCS = "gcs"


class ChunkingType(str, Enum):
    STRUCTURE = "structure"
    FIXED = "fixed"
    SPLITTER = "splitter"


class ParseMethod(str, Enum):
    GOOGLE_DOC_AI = "google_doc_ai"
    DOCLING = "docling"


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"
    EQUATION = "equation"


if __name__ == "__main__":
    print(ChunkType.EQUATION.value)
