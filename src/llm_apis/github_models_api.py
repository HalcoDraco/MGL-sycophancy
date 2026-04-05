import os
from typing import List, Union

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import AssistantMessage, ChatRequestMessage, SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

from src.llm_apis.conversation import Conversation

API_ENDPOINT = "https://models.github.ai/inference"


def _conversation_to_messages(conversation: Conversation) -> List[ChatRequestMessage]:
    messages: List[ChatRequestMessage] = []
    if conversation.system_prompt.strip():
        messages.append(SystemMessage(conversation.system_prompt))

    for turn in conversation.turns:
        provider_message: Union[UserMessage, AssistantMessage]
        if turn.role == "model":
            provider_message = AssistantMessage(turn.content)
        else:
            provider_message = UserMessage(turn.content)
        messages.append(provider_message)

    return messages


def github_models_chat(
    model: str,
    temperature: float,
    conversation: Conversation,
    max_tokens: int | None = None,
    timeout: int | None = None,
) -> str:
    """Call GitHub Models Chat Completions and return assistant text."""
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PAT")
    if not token:
        raise ValueError("Missing GitHub token. Set GITHUB_TOKEN or GITHUB_PAT.")

    client = ChatCompletionsClient(
        endpoint=API_ENDPOINT,
        credential=AzureKeyCredential(token),
    )
    response = client.complete(
        model=model,
        messages=_conversation_to_messages(conversation),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    return (content or "").strip()
