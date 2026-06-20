import os.path
import sys
from typing import List, Any, Tuple, Dict

import logfire
from ascii_colors import ASCIIColors


def check_env():
    env_path = ".env"

    if not os.path.exists(env_path):
        warning_msg = "Warning: Startup directory must contain .env file for multi-instance support."
        ASCIIColors.yellow(warning_msg)

        if sys.stdin.isatty():
            response = input("Do you want to continue? (yes/NO): ")
            if response.lower() != "yes":
                ASCIIColors.red("Server startup cancelled")
                return False
        return True
    return True


async def bootstrap_sparse_index(storage_service, sparse_index):
    all_chunks = await storage_service.scroll_all_chunks()

    if not all_chunks:
        logfire.warning("No chunks available to build sparse index")
        return

    logfire.info(f"Building BM25 index with {len(all_chunks)} chunks...")
    sparse_index.build(all_chunks)


def separate_content(
    content_list: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Separate the content after parsing
    :param content_list:
    :return:
    """
    text_parts = []
    multimodal_items = []

    for index, item in enumerate(content_list):
        content_type = item.get("type", "text")

        if content_type == "text":
            # Text content
            text = item.get("text", "")
            if text.strip():
                text_parts.append(text)
        else:
            # Multimodal content (image, table, equation, etc.)
            multimodal_item = dict(item)
            multimodal_item.setdefault("_content_list_index", index)
            multimodal_items.append(multimodal_item)

    text_content = "\n\n".join(text_parts)

    logfire.info("Content separation complete:")
    logfire.info(f"  - Text content length: {len(text_content)} characters")
    logfire.info(f"  - Multimodal items count: {len(multimodal_items)}")

    modal_types = {}
    for item in multimodal_items:
        modal_type = item.get("type", "unknown")
        modal_types[modal_type] = modal_types.get(modal_type, 0) + 1

    if modal_types:
        logfire.info(f"  - Multimodal type distribution: {modal_types}")

    return text_content, multimodal_items
