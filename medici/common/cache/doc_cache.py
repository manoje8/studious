import gzip
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import logfire

from medici.common.utils.config import config


class DocumentCache:
    """
    Filesystem-backed cache for parsed document content.
    """

    def __init__(self, cache_dir: Path = config.CACHE_DIR, manifest_path: Path | None = None):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = manifest_path or (self.cache_dir / "manifest.json")
        self._manifest: dict = self._load_manifest()

    def _load_manifest(self) -> dict:
        if self._manifest_path.exists():
            try:
                return json.loads(self._manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logfire.error("Cache - Manifest corrupted, starting fresh")

        return {}

    @staticmethod
    def _hash_file_content(file_path: Path) -> str:
        """
        Return a BLAKE2b-256 hex digest of the full file bytes.
        Chunked reads keep memory usage constant regardless of file size.
        """
        hasher = hashlib.blake2b(digest_size=32)
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65_536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _evict(self, cache_key: str) -> None:
        entry = self._manifest.pop(cache_key, None)

        if entry:
            stale_file = self.cache_dir / entry["filename"]
            stale_file.unlink(missing_ok=True)
            self._save_manifest()

    def _save_manifest(self) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(
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

        stored_hash = entry.get("content_hash")
        stored_size = entry.get("file_size")
        source_path = Path(entry["source_file"])

        if stored_hash is None or stored_size is None:
            logfire.warning(
                f"Cache - Entry {cache_key[:8]}\u2026 missing fingerprint fields, evicting "
                f"(written before staleness-guard was added)"
            )
            self._evict(cache_key)
            return None

        try:
            current_size = source_path.stat().st_size
        except OSError:
            logfire.warning(f"Cache - Source file gone for {cache_key[:8]}\u2026, evicting")
            self._evict(cache_key)
            return None

        if current_size != stored_size:
            logfire.warning(
                f"Cache STALE (size changed) - key={cache_key[:8]}\u2026 "
                f"file={entry['source_file']}"
            )
            self._evict(cache_key)
            return None

        current_hash = self._hash_file_content(source_path)
        if current_hash != stored_hash:
            logfire.warning(
                f"Cache STALE (content changed, mtime preserved) - key={cache_key[:8]}\u2026 "
                f"file={entry['source_file']}"
            )
            self._evict(cache_key)
            return None

        try:
            with gzip.open(cache_file, "rt", encoding="utf-8") as f:
                content_list = json.load(f)

            logfire.info(f"Cache HIT - key={cache_key[:8]}\u2026 file={entry['source_file']}")
            return content_list

        except (OSError, json.JSONDecodeError) as e:
            logfire.warning(f"Cache - Corrupted entry {cache_key[:8]}\u2026, evicting. Error: {e}")
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
        file_path = Path(file_path)
        filename = f"{cache_key}.json.gz"
        cache_file = self.cache_dir / filename

        try:
            with gzip.open(cache_file, "wt", encoding="utf-8") as f:
                json.dump(content_list, f, ensure_ascii=False)

        except OSError as e:
            logfire.error(f"Cache - Failed to write {cache_file}: {e}")
            return

        content_hash = self._hash_file_content(file_path)
        file_size = file_path.stat().st_size

        self._manifest[cache_key] = {
            "filename": filename,
            "source_file": str(file_path),
            "parse_method": parse_method,
            "parser": parser,
            "block_count": len(content_list),
            "cached_at": datetime.now(UTC).isoformat(),
            "content_hash": content_hash,
            "file_size": file_size,
        }

        self._save_manifest()
        logfire.info(f"Cache STORE - key={cache_key[:8]}\u2026 blocks={len(content_list)}")

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
