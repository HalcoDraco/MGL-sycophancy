from dataclasses import dataclass
from typing import List, Literal

TurnRole = Literal["user", "model"]


@dataclass(frozen=True)
class ConversationTurn:
    role: TurnRole
    content: str


@dataclass(frozen=True)
class Conversation:
    system_prompt: str
    turns: List[ConversationTurn]

    def append_user(self, content: str) -> "Conversation":
        return Conversation(
            system_prompt=self.system_prompt,
            turns=[*self.turns, ConversationTurn(role="user", content=content)],
        )

    def append_model(self, content: str) -> "Conversation":
        return Conversation(
            system_prompt=self.system_prompt,
            turns=[*self.turns, ConversationTurn(role="model", content=content)],
        )