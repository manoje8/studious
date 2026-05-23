import hashlib
import json
from pathlib import Path

import logfire


class Processor:

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
