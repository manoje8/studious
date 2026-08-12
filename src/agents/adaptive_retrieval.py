"""
Adaptive retrieval configuration.

Translates the router's ``retrieval_strategy`` dict into concrete
retrieval parameters that the :class:`RetrievalAgent` and graph
nodes consume at runtime.

"""

from __future__ import annotations

from dataclasses import dataclass

import logfire

_TOP_K_SEARCH_MIN, _TOP_K_SEARCH_MAX = 5, 60
_TOP_K_RERANK_MIN, _TOP_K_RERANK_MAX = 3, 20
_MAX_HOPS_MIN, _MAX_HOPS_MAX = 1, 5

# Keys: top_k_search, top_k_rerank, max_hops, use_query_expansion
_CATEGORY_DEFAULTS: dict[str, dict] = {
    "factual": {
        "top_k_search": 15,
        "top_k_rerank": 5,
        "max_hops": 1,
        "use_query_expansion": False,
    },
    "comparative": {
        "top_k_search": 30,
        "top_k_rerank": 8,
        "max_hops": 3,
        "use_query_expansion": True,
    },
    "analytical": {
        "top_k_search": 30,
        "top_k_rerank": 10,
        "max_hops": 4,
        "use_query_expansion": True,
    },
    "summarization": {
        "top_k_search": 25,
        "top_k_rerank": 8,
        "max_hops": 2,
        "use_query_expansion": True,
    },
    "procedural": {
        "top_k_search": 25,
        "top_k_rerank": 8,
        "max_hops": 3,
        "use_query_expansion": True,
    },
    "clarification": {
        "top_k_search": 15,
        "top_k_rerank": 5,
        "max_hops": 1,
        "use_query_expansion": False,
    },
}

_FALLBACK_DEFAULTS: dict = {
    "top_k_search": 20,
    "top_k_rerank": 5,
    "max_hops": 2,
    "use_query_expansion": True,
}

_CHUNKING_MULTIPLIERS: dict[str, float] = {
    "small": 1.5,
    "medium": 1.0,
    "large": 0.75,
}


@dataclass
class AdaptiveRetrievalConfig:
    """Concrete retrieval parameters derived from the router's strategy."""

    top_k_search: int
    """How many candidates to fetch from hybrid search."""

    top_k_rerank: int
    """How many results to keep after reranking."""

    max_hops: int
    """Maximum multi-hop retrieval iterations."""

    use_query_expansion: bool
    """Whether to expand queries via the QueryExpander."""

    use_reranking: bool
    """Whether to apply the reranker (vs. raw score truncation)."""

    confidence_threshold: float
    """Minimum rerank score to shortcut the evaluation step."""

    chunking_preference: str
    """Router's chunking size hint: ``"small"`` | ``"medium"`` | ``"large"``."""

    def __post_init__(self) -> None:
        """Clamp all numeric fields to safe operating bounds."""
        self.top_k_search = max(_TOP_K_SEARCH_MIN, min(self.top_k_search, _TOP_K_SEARCH_MAX))
        self.top_k_rerank = max(_TOP_K_RERANK_MIN, min(self.top_k_rerank, _TOP_K_RERANK_MAX))
        self.max_hops = max(_MAX_HOPS_MIN, min(self.max_hops, _MAX_HOPS_MAX))
        self.confidence_threshold = max(0.0, min(self.confidence_threshold, 1.0))

    @classmethod
    def from_state(cls, state: dict) -> AdaptiveRetrievalConfig:
        """Build an adaptive config from the current graph state.

        Resolution order (the highest priority wins):

        1. **Router explicit values** — ``state["classification"]["retrieval_strategy"]``
        2. **Category defaults** — looked up via ``state["question_category"]``
        3. **Module-level fallback** — ``_FALLBACK_DEFAULTS``

        After resolution, the ``chunking_strategy`` field applies a search-width
        multiplier to ``top_k_search``.
        """
        classification = state.get("classification") or {}
        strategy: dict = classification.get("retrieval_strategy") or {}
        category = (state.get("question_category") or "factual").lower()

        defaults = _CATEGORY_DEFAULTS.get(category, _FALLBACK_DEFAULTS)

        chunking_preference = strategy.get("chunking_strategy") or "medium"
        multiplier = _CHUNKING_MULTIPLIERS.get(chunking_preference, 1.0)

        raw_top_k_search = strategy.get("target_chunks")
        if raw_top_k_search and raw_top_k_search > 0:
            top_k_rerank = raw_top_k_search
            top_k_search = int(raw_top_k_search * 4 * multiplier)
        else:
            top_k_rerank = defaults["top_k_rerank"]
            top_k_search = int(defaults["top_k_search"] * multiplier)

        max_hops = strategy.get("max_retrieval_depth") or defaults["max_hops"]

        use_query_expansion = defaults["use_query_expansion"]
        if strategy.get("needs_multi_hop") is True:
            use_query_expansion = True

        use_reranking = strategy.get("needs_re_ranking", True)
        confidence_threshold = strategy.get("confidence_threshold") or 0.7

        config = cls(
            top_k_search=top_k_search,
            top_k_rerank=top_k_rerank,
            max_hops=max_hops,
            use_query_expansion=use_query_expansion,
            use_reranking=use_reranking,
            confidence_threshold=confidence_threshold,
            chunking_preference=chunking_preference,
        )

        logfire.info(
            "adaptive_retrieval_config_resolved",
            category=category,
            top_k_search=config.top_k_search,
            top_k_rerank=config.top_k_rerank,
            max_hops=config.max_hops,
            use_query_expansion=config.use_query_expansion,
            use_reranking=config.use_reranking,
            confidence_threshold=config.confidence_threshold,
            chunking_preference=config.chunking_preference,
            router_strategy_present=bool(strategy),
        )

        return config
