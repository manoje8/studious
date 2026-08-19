"""
tests/unit/test_doc_cache.py
============================
Unit tests for DocumentCache.

Key regression guarded here:
    A file whose *content* changes but whose *mtime* is preserved
    (e.g. `rsync -a`, `docker COPY`, `touch -t`) must NOT yield a
    cache HIT — the automatic staleness guard introduced in
    ``doc_cache.py`` must detect the change and return None.
"""

import gzip
import json
import os
from pathlib import Path

from medici.common.cache.doc_cache import DocumentCache

DUMMY_CONTENT = [{"type": "text", "text": "hello world"}]
PARSE_METHOD = "docling"
PARSER = "docling"


def _write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _make_cache(tmp_path: Path) -> DocumentCache:
    """Return a DocumentCache whose manifest and .gz files live in *tmp_path*."""
    return DocumentCache(cache_dir=tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDocumentCacheRoundtrip:
    def test_store_and_get_roundtrip(self, tmp_path):
        """store() followed by get() returns the original content list."""
        src = tmp_path / "doc.pdf"
        _write_file(src, b"PDF bytes v1")

        cache = _make_cache(tmp_path / "cache")
        key = "roundtrip_key"

        cache.store(key, DUMMY_CONTENT, src, parse_method=PARSE_METHOD, parser=PARSER)
        result = cache.get(key)

        assert result == DUMMY_CONTENT

    def test_get_returns_none_on_miss(self, tmp_path):
        """get() returns None for a key that was never stored."""
        cache = _make_cache(tmp_path / "cache")
        assert cache.get("nonexistent_key") is None

    def test_get_evicts_on_missing_cache_file(self, tmp_path):
        """If the manifest entry exists but the .gz file is gone, get() evicts and returns None."""
        src = tmp_path / "doc.pdf"
        _write_file(src, b"PDF bytes")

        cache = _make_cache(tmp_path / "cache")
        key = "orphan_key"
        cache.store(key, DUMMY_CONTENT, src, parse_method=PARSE_METHOD, parser=PARSER)

        # Delete the backing .gz file manually
        gz_path = tmp_path / "cache" / cache._manifest[key]["filename"]
        gz_path.unlink()

        assert cache.get(key) is None
        # Entry must also have been removed from the manifest
        assert key not in cache._manifest


class TestStalenessDetection:
    def test_staleness_detected_same_mtime_different_content(self, tmp_path):
        """
        REGRESSION TEST — the primary bug this PR fixes.

        Scenario: a file is stored in cache, then its *content* is replaced
        while its mtime is artificially kept identical (as rsync -a and
        Docker COPY can do). get() must detect the change and return None.
        """
        src = tmp_path / "doc.pdf"
        original_bytes = b"original PDF content"
        _write_file(src, original_bytes)

        cache = _make_cache(tmp_path / "cache")
        key = "stale_mtime_key"
        cache.store(key, DUMMY_CONTENT, src, parse_method=PARSE_METHOD, parser=PARSER)

        # Replace content with something different but restore the original mtime
        original_stat = src.stat()
        new_bytes = b"completely different PDF content - same length padded!!!"
        assert len(new_bytes) != len(original_bytes) or new_bytes != original_bytes

        _write_file(src, new_bytes)
        os.utime(src, (original_stat.st_atime, original_stat.st_mtime))  # restore mtime

        # The cache must NOT serve the stale result
        result = cache.get(key)
        assert result is None, (
            "Cache returned stale content even though file bytes changed "
            "(mtime-only staleness check would have missed this)"
        )

    def test_staleness_detected_different_size(self, tmp_path):
        """Fast-path: if file size changes, we evict without computing the hash."""
        src = tmp_path / "doc.pdf"
        _write_file(src, b"short")

        cache = _make_cache(tmp_path / "cache")
        key = "size_change_key"
        cache.store(key, DUMMY_CONTENT, src, parse_method=PARSE_METHOD, parser=PARSER)

        # Grow the file (size changes → fast eviction path)
        _write_file(src, b"much longer content that changes the byte count significantly")

        assert cache.get(key) is None
        assert key not in cache._manifest

    def test_no_staleness_on_identical_content(self, tmp_path):
        """An unmodified file must still produce a cache HIT (no false positives)."""
        src = tmp_path / "doc.pdf"
        _write_file(src, b"stable PDF bytes")

        cache = _make_cache(tmp_path / "cache")
        key = "stable_key"
        cache.store(key, DUMMY_CONTENT, src, parse_method=PARSE_METHOD, parser=PARSER)

        # No modifications — second get() should be a HIT
        result = cache.get(key)
        assert result == DUMMY_CONTENT

    def test_legacy_entry_without_fingerprint_is_evicted(self, tmp_path):
        """
        Entries written by the old code (no content_hash / file_size) must be
        treated as stale and evicted, not served as-is.
        """
        src = tmp_path / "doc.pdf"
        _write_file(src, b"legacy document")

        cache = _make_cache(tmp_path / "cache")
        key = "legacy_key"

        # Manually inject a manifest entry that lacks the new fingerprint fields
        filename = f"{key}.json.gz"
        gz_path = tmp_path / "cache" / filename
        gz_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(gz_path, "wt", encoding="utf-8") as fh:
            json.dump(DUMMY_CONTENT, fh)

        cache._manifest[key] = {
            "filename": filename,
            "source_file": str(src),
            "parse_method": PARSE_METHOD,
            "parser": PARSER,
            "block_count": 1,
            # content_hash and file_size deliberately absent
        }
        cache._save_manifest()

        result = cache.get(key)
        assert result is None
        assert key not in cache._manifest


class TestInvalidate:
    def test_invalidate_removes_all_entries_for_file(self, tmp_path):
        """invalidate(path) removes every manifest entry whose source_file matches."""
        src = tmp_path / "doc.pdf"
        _write_file(src, b"document bytes")

        cache = _make_cache(tmp_path / "cache")

        # Store two entries for the same source file (different parse methods)
        cache.store("key_a", DUMMY_CONTENT, src, parse_method="docling", parser="docling")
        cache.store("key_b", DUMMY_CONTENT, src, parse_method="google", parser="google")

        removed = cache.invalidate(src)

        assert removed == 2
        assert "key_a" not in cache._manifest
        assert "key_b" not in cache._manifest

    def test_invalidate_returns_zero_for_unknown_file(self, tmp_path):
        cache = _make_cache(tmp_path / "cache")
        assert cache.invalidate(tmp_path / "ghost.pdf") == 0


class TestStats:
    def test_stats_reflects_stored_entries(self, tmp_path):
        src = tmp_path / "doc.pdf"
        _write_file(src, b"x" * 1024)

        cache = _make_cache(tmp_path / "cache")
        cache.store("stats_key", DUMMY_CONTENT, src, parse_method=PARSE_METHOD, parser=PARSER)

        stats = cache.stats()
        assert stats["entries"] == 1
        assert stats["size_mb"] >= 0
        assert "cache_dir" in stats
