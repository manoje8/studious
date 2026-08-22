"""
Integration tests: Dual SparseSearchIndex instance seam.

The problem: Processor creates its own SparseSearchIndex at ingestion time,
while the API's bootstrap_sparse_index() builds a *separate* instance from
Qdrant scroll data. These two instances never share state — changes in one
are invisible to the other. This test suite makes that contract explicit and
catches regressions if the two are accidentally coupled (or fail to be shared).

Additionally tests that SparseSearchIndex correctly consumes the exact dict
shape that QdrantStorageService.scroll_all_chunks() produces.
"""

from unittest.mock import AsyncMock, patch

import pytest

from medici.common.services.sparse_index import SparseSearchIndex
from medici.common.utils.helper import bootstrap_sparse_index


@pytest.mark.integration
class TestSparseIndexConsumeScrollShape:
    """
    SparseSearchIndex.build() must accept the exact dict shape that
    QdrantStorageService.scroll_all_chunks() returns.
    """

    def test_build_accepts_scroll_shaped_dicts(self, scroll_chunk_dicts):
        """build() must not KeyError or crash on real scroll output shape."""
        index = SparseSearchIndex()
        index.build(scroll_chunk_dicts)  # must not raise
        assert index.index is not None

    def test_search_returns_results_from_scroll_shaped_chunks(self, scroll_chunk_dicts):
        index = SparseSearchIndex()
        index.build(scroll_chunk_dicts)
        results = index.search("retrieval augmented generation")
        assert len(results) > 0

    def test_search_result_preserves_all_scroll_dict_keys(self, scroll_chunk_dicts):
        """Results must pass through all original payload keys intact."""
        index = SparseSearchIndex()
        index.build(scroll_chunk_dicts)
        results = index.search("BM25 sparse retrieval")
        assert len(results) > 0
        first = results[0]
        # All original keys from scroll_all_chunks() shape must be present
        for key in ("text", "doc_id", "chunk_index", "section_title", "source_file"):
            assert key in first, f"Missing key '{key}' in search result"

    def test_search_result_contains_bm25_score_field(self, scroll_chunk_dicts):
        """BM25 score must be injected by search(), not come from stored data."""
        index = SparseSearchIndex()
        index.build(scroll_chunk_dicts)
        results = index.search("hybrid search dense sparse")
        assert len(results) > 0
        assert all("bm25_score" in r for r in results)
        assert all(isinstance(r["bm25_score"], float) for r in results)

    def test_search_finds_correct_document_for_query(self, scroll_chunk_dicts):
        """Semantic integrity: a keyword query should find the right chunk."""
        index = SparseSearchIndex()
        index.build(scroll_chunk_dicts)
        results = index.search("BM25 term frequency sparse")
        assert len(results) > 0
        assert any("BM25" in r["text"] or "sparse" in r["text"].lower() for r in results)


@pytest.mark.integration
class TestDualSparseIndexIndependence:
    """
    Prove that two SparseSearchIndex instances are fully independent.
    """

    def test_two_instances_do_not_share_state(self, scroll_chunk_dicts):
        """Building one instance must not affect an uninitialised second instance."""
        index_a = SparseSearchIndex()
        index_b = SparseSearchIndex()

        index_a.build(scroll_chunk_dicts)

        # index_b was never built — must still return empty results
        results = index_b.search("retrieval augmented generation")
        assert results == [], (
            "index_b returned results despite never being built. "
            "This would indicate shared class-level state — a regression."
        )

    def test_building_index_b_does_not_affect_index_a(self, scroll_chunk_dicts):
        index_a = SparseSearchIndex()
        index_b = SparseSearchIndex()

        index_a.build(scroll_chunk_dicts)
        # 2-chunk corpus so BM25 has document variance
        new_chunks = [
            {
                "text": "astrophysics telescope nebula galaxy stellar",
                "doc_id": "d3",
                "chunk_index": 0,
            },
            {
                "text": "pasta recipe tomato sauce cheese oven",
                "doc_id": "d3",
                "chunk_index": 1,
            },
        ]
        index_b.build(new_chunks)

        # index_a must still find the original chunks (unaffected by index_b's build)
        results_a = index_a.search("retrieval augmented generation")
        assert any("retrieval" in r["text"].lower() or "RAG" in r["text"] for r in results_a), (
            "index_a lost its chunks after index_b was built — shared state regression."
        )

        # index_b must NOT contain index_a's chunks
        assert index_b.chunks == new_chunks, (
            "index_b.chunks contains index_a's data — shared state detected."
        )
        # The astrophysics chunk must not appear in index_a
        assert all(c["doc_id"] != "d3" for c in index_a.chunks), (
            "index_a.chunks contains index_b's doc_id 'd3' — shared state detected."
        )

    def test_rebuild_replaces_index_in_place(self, scroll_chunk_dicts):
        """Rebuilding the same instance should replace the previous index, not accumulate."""
        index = SparseSearchIndex()
        index.build(scroll_chunk_dicts)
        original_chunk_ids = {c["doc_id"] for c in index.chunks}

        new_chunks = [
            {
                "text": "astrophysics telescope nebula galaxy stellar",
                "doc_id": "d99",
                "chunk_index": 0,
            },
            {"text": "pasta recipe tomato sauce cheese oven", "doc_id": "d99", "chunk_index": 1},
        ]
        index.build(new_chunks)

        # After rebuild, .chunks must reflect the NEW corpus, not the old one
        new_chunk_ids = {c["doc_id"] for c in index.chunks}
        assert new_chunk_ids == {"d99"}, (
            f"After rebuild, index.chunks still contains old doc_ids {new_chunk_ids - {'d99'}}. "
            "build() appears to accumulate rather than replace the corpus."
        )
        assert original_chunk_ids.isdisjoint(new_chunk_ids), (
            "Original doc_ids still present after rebuild — index was not replaced."
        )


@pytest.mark.integration
class TestBootstrapSparseIndex:
    """
    bootstrap_sparse_index() must populate the *passed-in* SparseSearchIndex
    instance, not create a new one internally. If it created an internal
    instance, the caller's index would remain empty — the seam bug.
    """

    @pytest.mark.asyncio
    async def test_bootstrap_populates_provided_index(self, scroll_chunk_dicts):
        """After bootstrap, the same index object passed in must have chunks."""
        index = SparseSearchIndex()

        mock_storage = AsyncMock()
        mock_storage.chunk_count.return_value = len(scroll_chunk_dicts)
        mock_storage.scroll_all_chunks.return_value = scroll_chunk_dicts

        with patch.object(index, "load", return_value=False):
            await bootstrap_sparse_index(mock_storage, index)

        # The passed-in instance must now have chunks — not a hidden new instance
        assert index.chunks == scroll_chunk_dicts, (
            "bootstrap_sparse_index did not populate the provided SparseSearchIndex. "
            "If a new internal instance was created, searches on `index` will always return empty."
        )

    @pytest.mark.asyncio
    async def test_bootstrap_rebuilds_when_chunk_count_differs(self, scroll_chunk_dicts):
        """If cached index has stale chunk count, bootstrap must rebuild."""
        index = SparseSearchIndex()
        stale_chunks = scroll_chunk_dicts[:2]  # only 2 chunks cached

        def fake_load():
            index.chunks = stale_chunks
            index.index = object()  # non-None to simulate a loaded index
            return True

        mock_storage = AsyncMock()
        mock_storage.chunk_count.return_value = len(scroll_chunk_dicts)  # 4 total now
        mock_storage.scroll_all_chunks.return_value = scroll_chunk_dicts

        with patch.object(index, "load", side_effect=fake_load):
            await bootstrap_sparse_index(mock_storage, index)

        # Must have rebuilt with all 4 chunks
        assert len(index.chunks) == len(scroll_chunk_dicts)

    @pytest.mark.asyncio
    async def test_bootstrap_skips_rebuild_when_count_matches(self, scroll_chunk_dicts):
        """If chunk count matches cache, bootstrap must skip the rebuild."""
        index = SparseSearchIndex()

        def fake_load():
            index.chunks = scroll_chunk_dicts
            return True

        mock_storage = AsyncMock()
        mock_storage.chunk_count.return_value = len(scroll_chunk_dicts)
        mock_storage.scroll_all_chunks.return_value = scroll_chunk_dicts

        with patch.object(index, "load", side_effect=fake_load):
            await bootstrap_sparse_index(mock_storage, index)

        # scroll_all_chunks should NOT have been called (cache was valid)
        mock_storage.scroll_all_chunks.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_bootstrap_with_empty_storage_leaves_index_empty(self):
        """Empty storage → warning logged, index remains unbuilt."""
        index = SparseSearchIndex()

        mock_storage = AsyncMock()
        mock_storage.chunk_count.return_value = 0
        mock_storage.scroll_all_chunks.return_value = []

        with patch.object(index, "load", return_value=False):
            await bootstrap_sparse_index(mock_storage, index)

        assert index.index is None
        assert index.chunks == []
