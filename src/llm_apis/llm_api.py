import logging
from typing import Dict, List, TypedDict

import requests
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from src.llm_apis.conversation import Conversation
from src.llm_apis.github_models_api import github_models_chat
from src.llm_apis.google_ai_studio_api import google_ai_studio_chat
from src.llm_apis.groq_api import groq_chat
from src.llm_apis.huggingface_api import huggingface_chat

logger = logging.getLogger(__name__)


class ProviderModelConfig(TypedDict):
    provider: str
    model: str


MODEL_PROVIDER_REGISTRY: Dict[str, List[ProviderModelConfig]] = {
    "llama-3.3-70b-instruct": [
        {"provider": "groq", "model": "llama-3.3-70b-versatile"},
        {"provider": "github", "model": "meta/Llama-3.3-70B-Instruct"},
    ],
    "gemini-3.1-flash-lite": [
        {"provider": "google_ai_studio", "model": "gemini-3.1-flash-lite-preview"},
    ],
    "Qwen3-8B": [
        {"provider": "huggingface", "model": "Qwen/Qwen3-8B:nscale"},
    ],
}


def _select_provider_and_model(model: str) -> ProviderModelConfig:
    """Resolve a canonical model id to the first provider/model mapping in priority order."""
    provider_options = MODEL_PROVIDER_REGISTRY.get(model)
    if not provider_options:
        supported_models = ", ".join(sorted(MODEL_PROVIDER_REGISTRY.keys()))
        raise ValueError(f"Unsupported model '{model}'. Supported models: {supported_models}")

    # For now, always use the highest-priority provider (first entry).
    return provider_options[0]


def _is_retryable_error(exception: BaseException) -> bool:
    if isinstance(exception, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True

    if not isinstance(exception, requests.exceptions.HTTPError):
        return False

    response = exception.response
    if response is None:
        return False

    if response.status_code in (429, 500, 502, 503, 504):
        return True

    if response.status_code == 403 and "rate limit" in response.text.lower():
        return True

    return False


def _dispatch_chat(
    provider: str,
    model: str,
    conversation: Conversation,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    if provider == "github":
        return github_models_chat(
            model=model,
            conversation=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    if provider == "groq":
        return groq_chat(
            model=model,
            conversation=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    if provider == "google_ai_studio":
        return google_ai_studio_chat(
            model=model,
            conversation=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    if provider == "huggingface":
        return huggingface_chat(
            model=model,
            conversation=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    raise ValueError(
        f"Unsupported provider '{provider}'. Supported providers are: github, groq, google_ai_studio, huggingface."
    )


def llm_chat(
    model: str,
    conversation: Conversation,
    temperature: float,
    max_tokens: int,
    timeout: int = 120,
    retries: int = 9,
) -> str:
    """Route a chat request to a provider client with centralized retry/error handling."""
    selected_config = _select_provider_and_model(model)
    provider = selected_config["provider"]
    provider_model = selected_config["model"]

    logger.debug(
        "Dispatching LLM chat request | provider=%s | model=%s | retries=%s | timeout=%s",
        provider,
        provider_model,
        retries,
        timeout,
    )

    retrying = Retrying(
        retry=retry_if_exception(_is_retryable_error),
        wait=wait_exponential(min=1, max=120),
        stop=stop_after_attempt(retries),
        reraise=True,
    )

    try:
        for attempt in retrying:
            with attempt:
                return _dispatch_chat(
                    provider=provider,
                    model=provider_model,
                    conversation=conversation,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
    except requests.exceptions.RequestException:
        logger.exception(
            "LLM API request failed after retries | provider=%s | model=%s",
            provider,
            provider_model,
        )
        raise

    raise RuntimeError("Unexpected retry loop termination without result.")
