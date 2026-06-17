from collections.abc import Callable

from src.ingestion.chunking.Chunker import Chunker
from src.utils.factory import Factory, ServiceScope


class ChunkerFactory(Factory[Chunker]):
    """actory for creating Chunker instances."""


chunker_factory = ChunkerFactory()


def register_chunker(
    chunker_type: str,
    chunker_initializer: Callable[..., Chunker],
    scope: ServiceScope = "transient",
):
    chunker_factory.register(chunker_type, chunker_initializer, scope)


# Todo: Create chunker
