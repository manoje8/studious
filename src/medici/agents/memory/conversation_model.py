from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class ConversationTurn:
    role: str
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = field(default_factory=dict)


@dataclass
class ConversationSession:
    session_id: str
    user_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def add_turn(self, role: str, content: str, metadata: dict = None):
        self.turns.append(ConversationTurn(role=role, content=content, metadata=metadata or {}))

    def get_recent_turns(self, n: int = 4):
        return self.turns[-n:]

    def to_prompt_format(self, n: int = 6) -> str:
        """Return a plain-text string of recent turns (used by QueryRewriter)."""
        recent = self.get_recent_turns(n)

        lines = []

        for turn in recent:
            prefix = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.content}")

        return "\n".join(lines)

    def to_history_dicts(self, n: int = 6) -> list[dict]:
        """
        Return recent turns as list[dict] with 'user'/'assistant' keys.

        This is the format expected by the graph State (``conversational_history``)
        and consumed by ``SynthesizerAgent._format_history()``.
        Turns are paired so each dict contains one user message and the
        assistant reply that immediately followed it.
        """
        recent = self.get_recent_turns(n)
        pairs: list[dict] = []
        i = 0
        while i < len(recent):
            turn = recent[i]
            if turn.role == "user":
                user_msg = turn.content
                asst_msg = ""
                if i + 1 < len(recent) and recent[i + 1].role == "assistant":
                    asst_msg = recent[i + 1].content
                    i += 2
                else:
                    i += 1
                pairs.append({"user": user_msg, "assistant": asst_msg})
            else:
                i += 1
        return pairs


@dataclass
class EpisodicSummary:
    """LLM-compressed summary of a past conversation session."""

    user_id: str
    session_id: str
    summary: str
    topic_tags: list[str] = field(default_factory=list)
    turn_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
