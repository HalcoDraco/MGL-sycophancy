import os
from typing import Dict, List

from groq import Groq

from src.llm_apis.conversation import Conversation


def _conversation_to_messages(conversation: Conversation) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if conversation.system_prompt.strip():
        messages.append({"role": "system", "content": conversation.system_prompt})

    for turn in conversation.turns:
        provider_role = "assistant" if turn.role == "model" else "user"
        messages.append({"role": provider_role, "content": turn.content})

    return messages


def groq_chat(
    model: str,
    temperature: float,
    max_tokens: int,
    conversation: Conversation,
    timeout: int = 120,
) -> str:
    token = os.getenv("GROQ_API_KEY")
    if not token:
        raise ValueError("Missing Groq API key. Set GROQ_API_KEY.")

    client = Groq(api_key=token, timeout=timeout)

    chat_completion = client.chat.completions.create(
        messages=_conversation_to_messages(conversation),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = chat_completion.choices[0].message.content
    return (content or "").strip()
