import logging
import os
from typing import Any, Dict, List

import requests

from src.llm_apis.conversation import Conversation

API_URL = "https://models.github.ai/inference/chat/completions"
logger = logging.getLogger(__name__)


def _conversation_to_messages(conversation: Conversation) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if conversation.system_prompt.strip():
        messages.append({"role": "system", "content": conversation.system_prompt})

    for turn in conversation.turns:
        provider_role = "assistant" if turn.role == "model" else "user"
        messages.append({"role": provider_role, "content": turn.content})

    return messages


def _post_chat_completion(headers: Dict[str, str], payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    response = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
    if not response.ok:
        logger.error(
            "GitHub Models API request failed | status_code=%s | response_body=%s",
            response.status_code,
            response.text,
        )
    response.raise_for_status()
    return response.json()


def _build_headers(github_pat: str) -> Dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_pat}",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }


def github_models_chat(
    model: str,
    temperature: float,
    max_tokens: int,
    conversation: Conversation,
    timeout: int = 120,
) -> str:
    """Call GitHub Models Chat Completions and return assistant text."""
    token = os.getenv("GITHUB_PAT")
    if not token:
        raise ValueError("Missing GitHub token. Set GITHUB_PAT.")

    payload = {
        "model": model,
        "messages": _conversation_to_messages(conversation),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response_json = _post_chat_completion(_build_headers(token), payload, timeout)
    return response_json["choices"][0]["message"]["content"].strip()
