import os
from typing import Dict, List

from huggingface_hub import InferenceClient

from src.llm_apis.conversation import Conversation


def _conversation_to_messages(conversation: Conversation) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if conversation.system_prompt.strip():
        messages.append({"role": "system", "content": conversation.system_prompt})

    for turn in conversation.turns:
        provider_role = "assistant" if turn.role == "model" else "user"
        messages.append({"role": provider_role, "content": turn.content})

    return messages


def huggingface_chat(
    model: str,
    temperature: float,
    max_tokens: int,
    conversation: Conversation,
    timeout: int = 120,
) -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Missing Hugging Face token. Set HF_TOKEN.")

    client = InferenceClient(api_key=token, timeout=timeout)
    response = client.chat_completion(
        model=model,
        messages=_conversation_to_messages(conversation),
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = response.choices[0].message.content
    return (content or "").strip()