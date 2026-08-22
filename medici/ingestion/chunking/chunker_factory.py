from collections.abc import Callable

import logfire

from medici.common.utils.constants import ChunkerStrategy
from medici.common.utils.factory import Factory, ServiceScope
from medici.ingestion.chunking.Chunker import Chunker
from medici.ingestion.chunking.chunking_config import ChunkingConfig


class ChunkerFactory(Factory[Chunker]):
    """actory for creating Chunker instances."""


chunker_factory = ChunkerFactory()


def register_chunker(
    chunker_type: str,
    chunker_initializer: Callable[..., Chunker],
    scope: ServiceScope = "transient",
):
    chunker_factory.register(chunker_type, chunker_initializer, scope)


def create_chunker(config: ChunkingConfig):
    config_model = config.model_dump(exclude={"type"})

    logfire.debug("CONFIG MODEL: {config_model=}", config_model=config_model)

    chunker_strategy = config.type

    if chunker_strategy not in chunker_factory:
        match chunker_strategy:
            case ChunkerStrategy.FIXED:
                from medici.ingestion.chunking.fixed_window import FixedWindow

                register_chunker(ChunkerStrategy.FIXED, FixedWindow)
            case ChunkerStrategy.RECURSIVE_CHARACTER:
                from medici.ingestion.chunking.recursive_character import (
                    RecursiveCharacterChunker,
                )

                register_chunker(ChunkerStrategy.RECURSIVE_CHARACTER, RecursiveCharacterChunker)
            case ChunkerStrategy.SENTENCE_BOUNDARY:
                from medici.ingestion.chunking.sentence_boundary import (
                    SentenceBoundaryChunker,
                )

                register_chunker(ChunkerStrategy.SENTENCE_BOUNDARY, SentenceBoundaryChunker)
            case ChunkerStrategy.SEMANTIC:
                from medici.ingestion.chunking.semantic import SemanticChunker

                register_chunker(ChunkerStrategy.SEMANTIC, SemanticChunker)
            case _:
                msg = f"ChunkingConfig.strategy '{chunker_strategy}' is not registered in the ChunkerFactory. Registered types: {', '.join(chunker_factory.keys())}."
                raise ValueError(msg)

    return chunker_factory.create(chunker_strategy, init_args=config_model)
