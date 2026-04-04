import os
from typing import Any, Dict, List, Optional

import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

API_URL = "https://models.github.ai/inference/chat/completions"


def _is_rate_limit_error(exception: BaseException) -> bool:
    if not isinstance(exception, requests.exceptions.HTTPError):
        return False

    response = exception.response
    if response is None:
        return False

    if response.status_code == 429:
        return True

    if response.status_code == 403 and "rate limit" in response.text.lower():
        return True

    return False


@retry(
    retry=retry_if_exception(_is_rate_limit_error),
    wait=wait_exponential(min=1, max=60),
    stop=stop_after_attempt(7),
    reraise=True,
)
def _post_with_rate_limit_retry(headers: Dict[str, str], payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    response = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
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
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    github_pat: Optional[str] = None,
    timeout: int = 60,
) -> str:
    """Call GitHub Models Chat Completions and return assistant text."""
    token = github_pat or os.getenv("GITHUB_PAT")
    if not token:
        raise ValueError("Missing GitHub token. Set GITHUB_PAT or pass github_pat.")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response_json = _post_with_rate_limit_retry(_build_headers(token), payload, timeout)
    return response_json["choices"][0]["message"]["content"].strip()
