import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import logfire

from src.common.utils.config import config


class DocumentCache:
    """
    Filesystem-backed cache for parsed document content.
    """

    def __init__(self, cache_dir: Path = config.CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._manifest: dict = self._load_manifest()

    def _load_manifest(self) -> dict:
        if config.CACHE_MANIFEST.exists():
            try:
                return json.loads(config.CACHE_MANIFEST.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logfire.error("Cache - Manifest corrupted, starting fresh")

        return {}

    def _evict(self, cache_key: str) -> None:
        entry = self._manifest.pop(cache_key, None)

        if entry:
            stale_file = self.cache_dir / entry["filename"]
            stale_file.unlink(missing_ok=True)
            self._save_manifest()

    def _save_manifest(self) -> None:
        config.CACHE_MANIFEST.write_text(
            json.dumps(self._manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def get(self, cache_key: str) -> list[dict] | None:
        entry = self._manifest.get(cache_key)

        if entry is None:
            return None

        cache_file = self.cache_dir / entry["filename"]

        if not cache_file.exists():
            self._evict(cache_key)
            return None

        try:
            with gzip.open(cache_file, "rt", encoding="utf-8") as f:
                content_list = json.load(f)

            logfire.info(f"Cache HIT - key={cache_key[:8]}… file={entry['source_file']}")
            return content_list

        except (OSError, json.JSONDecodeError) as e:
            logfire.warning(f"Cache - Corrupted entry {cache_key[:8]}…, evicting. Error: {e}")
            self._evict(cache_key)
            return None

    def store(
        self,
        cache_key: str,
        content_list: list[dict],
        file_path: str | Path,
        parse_method: str,
        parser: str = "auto",
    ):
        filename = f"{cache_key}.json.gz"
        cache_file = self.cache_dir / filename

        try:
            with gzip.open(cache_file, "wt", encoding="utf-8") as f:
                json.dump(content_list, f, ensure_ascii=False)

        except OSError as e:
            logfire.error(f"Cache - Failed to write {cache_file}: {e}")
            return

        self._manifest[cache_key] = {
            "filename": filename,
            "source_file": str(file_path),
            "parse_method": parse_method,
            "parser": parser,
            "block_count": len(content_list),
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }

        self._save_manifest()
        logfire.info(f"Cache STORE - key={cache_key[:8]}… blocks={len(content_list)}")

    def invalidate(self, file_path: str | Path) -> int:
        """
        Remove all cache entries for a given source file.
        Useful when a file is re-uploaded or explicitly re-processed.
        Returns the number of entries removed.
        """

        state_keys = [
            k for k, v in self._manifest.items() if v.get("source_file") == str(file_path)
        ]

        for key in state_keys:
            self._evict(key)

        return len(state_keys)

    def stats(self) -> dict:
        total_bytes = sum(
            (self.cache_dir / v["filename"]).stat().st_size
            for v in self._manifest.values()
            if (self.cache_dir / v["filename"]).exists()
        )

        return {
            "entries": len(self._manifest),
            "size_mb": round(total_bytes / 1024 / 1024, 2),
            "cache_dir": str(self.cache_dir),
        }
