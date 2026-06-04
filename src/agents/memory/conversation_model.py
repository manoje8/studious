from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConversationTurn:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow())
    metadata: dict = field(default_factory=dict)


@dataclass()
class ConversationSession:
    session_id: str
    user_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow())

    def add_turn(self, role: str, content: str, metadata: dict = None):
        self.turns.append(
            ConversationTurn(role=role, content=content, metadata=metadata or {})
        )

    def get_recent_turns(self, n: int = 4):
        return self.turns[-n:]

    def to_prompt_format(self, n: int = 6):
        recent = self.get_recent_turns(n)

        lines = []

        for turn in recent:
            prefix = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.content}")

        return "\n".join(lines)
