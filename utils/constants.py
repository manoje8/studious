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