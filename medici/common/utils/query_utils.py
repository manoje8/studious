import logfire

from medici.common.utils.config import config
from medici.common.utils.tokenizer import TikTokenTokenizer

_tokenizer: TikTokenTokenizer | None = None


def get_tokenizer() -> TikTokenTokenizer:
    """Get or create a singleton TikTokenTokenizer instance."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = TikTokenTokenizer()
    return _tokenizer


def truncate_query(query: str, component_name: str = "Query") -> str:
    """Truncate *query* to ``MAX_QUERY_INPUT_TOKENS`` tokens if it is too long.

    Guards query processors against runaway inputs (e.g., pasted documents) that
    would consume the model's entire context window before the prompt template
    is added.

    Args:
        query: The input query to potentially truncate
        component_name: Name of the component for log messages (e.g., "QueryRewriter", "QueryExpander")

    Returns:
        The original query or a truncated version if it exceeds the token limit
    """
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(query)
    limit = config.MAX_QUERY_INPUT_TOKENS

    if len(tokens) > limit:
        logfire.warning(f"{component_name}: query truncated from {len(tokens)} to {limit} tokens")
        return tokenizer.decode(tokens[:limit])

    return query
