from enum import Enum
from dataclasses import dataclass, field

from src.utils.config import config


class RetrievalDecision(Enum):
    SUFFICIENT = "sufficient"
    REFINE_QUERY = "refine_query"
    EXPAND_SEARCH = "expand_search"
    EXHAUSTED = "exhausted"


@dataclass
class RetrievalRound:
    query_used: str
    chunk_retrieved: list[dict]
    relevance_score: list[float]
    decision: RetrievalDecision
    reasoning: str
    refined_query: str | None = None


@dataclass()
class AgentState:
    original_question: str
    doc_id_filter: str | None = None

    sub_questions: list[str] = field(default_factory=list)

    retrieval_rounds: list[RetrievalRound] = field(default_factory=list)
    accepted_chunks: list[dict] = field(default_factory=list)

    max_retrieval_rounds: int = config.MAX_RETRIEVAL_ROUND
    current_round: int = 0
    is_complete: bool = False

    final_answer: str = ""
    source_used: list[str] = field(default_factory=list)

    @property
    def all_retrieved_context(self) -> str:
        """Concatenate all accepted chunks for the synthesizer."""

        return "\n\n---\n\n".join(
            f"[Source: {c['source']} | Section: {c['section']}]\n{c['text']}"
            for c in self.accepted_chunks
        )
