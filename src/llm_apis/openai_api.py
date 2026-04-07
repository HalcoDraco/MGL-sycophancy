import logging
import os
from typing import Dict, List

from openai import OpenAI

from src.llm_apis.conversation import Conversation

logger = logging.getLogger(__name__)


def _conversation_to_messages(conversation: Conversation) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if conversation.system_prompt.strip():
        # OpenAI uses the developer role for instruction-level guidance.
        messages.append({"role": "developer", "content": conversation.system_prompt})

    for turn in conversation.turns:
        provider_role = "assistant" if turn.role == "model" else "user"
        messages.append({"role": provider_role, "content": turn.content})

    return messages


def openai_chat(
    model: str,
    temperature: float,
    conversation: Conversation,
    max_tokens: int | None = None,
    timeout: int | None = None,
) -> str:
    token = os.getenv("OPENAI_API_KEY")
    if not token:
        raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY.")

    client = OpenAI(api_key=token, timeout=timeout)

    response = client.chat.completions.create(
        model=model,
        messages=_conversation_to_messages(conversation),
        max_completion_tokens=max_tokens,
        reasoning_effort="medium",
    )

    content = response.choices[0].message.content
    if not content or not content.strip():
        finish_reason = getattr(response.choices[0], "finish_reason", "unknown")
        diagnostic = (
            f"OpenAI returned blank content | model={model} | finish_reason={finish_reason}"
        )
        print(diagnostic)
        logger.error(diagnostic)

    return (content or "").strip()