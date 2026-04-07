import json
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


class InvalidLLMResponseError(requests.exceptions.RequestException):
    """Raised when a provider returns an empty or malformed text response."""

class RetryableLLMError(requests.exceptions.RequestException):
    """Raised when an error occurs that may be transient and worth retrying."""

class ProviderModelConfig(TypedDict):
    provider: str
    model: str


MODEL_PROVIDER_REGISTRY: Dict[str, List[ProviderModelConfig]] = {
    "llama-3.3-70b-instruct": [
        # {"provider": "groq", "model": "llama-3.3-70b-versatile"},
        {"provider": "huggingface", "model": "meta-llama/Llama-3.3-70B-Instruct:novita"},
        {"provider": "github", "model": "meta/Llama-3.3-70B-Instruct"},
    ],
    "llama-3.1-8b": [
        # {"provider": "groq", "model": "llama-3.1-8b-instant"},
        {"provider": "huggingface", "model": "meta-llama/Llama-3.1-8B-Instruct:novita"},
    ],
    "Hermes-3-Llama-3.1-8B": [
        {"provider": "huggingface", "model": "NousResearch/Hermes-3-Llama-3.1-8B:featherless-ai"},
    ],
    "gemini-3.1-flash-lite": [
        {"provider": "google_ai_studio", "model": "gemini-3.1-flash-lite-preview"},
    ],
    "DeepSeek-V3": [
        {"provider": "huggingface", "model": "deepseek-ai/DeepSeek-V3:novita"},
    ],
    "DeepSeek-R1": [
        {"provider": "huggingface", "model": "deepseek-ai/DeepSeek-R1:novita"},
    ],
    "Qwen3-8B": [
        {"provider": "huggingface", "model": "Qwen/Qwen3-8B:nscale"},
    ],
    "Qwen2.5-1.5B-Instruct": [
        {"provider": "huggingface", "model": "Qwen/Qwen2.5-1.5B-Instruct:featherless-ai"},
    ],
    "Qwen2.5-7B-Instruct": [
        {"provider": "huggingface", "model": "Qwen/Qwen2.5-7B-Instruct:featherless-ai"},
    ],
    "Qwen2.5-14B-Instruct": [
        {"provider": "huggingface", "model": "Qwen/Qwen2.5-14B-Instruct:featherless-ai"},
    ],
    "Qwen2.5-32B-Instruct": [
        {"provider": "huggingface", "model": "Qwen/Qwen2.5-32B-Instruct:featherless-ai"},
    ],
    "Qwen2.5-72B-Instruct": [
        {"provider": "huggingface", "model": "Qwen/Qwen2.5-72B-Instruct:novita"},
    ],
    "Mistral-7B-Instruct-v0.2": [
        {"provider": "huggingface", "model": "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai"},
    ],
    "gemma-2-9b-it": [
        {"provider": "huggingface", "model": "google/gemma-2-9b-it:featherless-ai"},
    ],
    "mistral-7b-sft-beta": [
        {"provider": "huggingface", "model": "HuggingFaceH4/mistral-7b-sft-beta:featherless-ai"},
    ],
    "zephyr-7b-beta": [
        {"provider": "huggingface", "model": "HuggingFaceH4/zephyr-7b-beta:featherless-ai"},
    ],
}


def _select_provider_and_model(model: str, provider: str | None = None) -> ProviderModelConfig:
    """Resolve a canonical model id to the first provider/model mapping in priority order."""
    provider_options = MODEL_PROVIDER_REGISTRY.get(model)
    if not provider_options:
        supported_models = ", ".join(sorted(MODEL_PROVIDER_REGISTRY.keys()))
        raise ValueError(f"Unsupported model '{model}'. Supported models: {supported_models}")

    if provider:
        for option in provider_options:
            if option["provider"] == provider:
                return option
        supported_providers = ", ".join(option["provider"] for option in provider_options)
        raise ValueError(
            f"Model '{model}' is not available from provider '{provider}'. "
            f"Supported providers for this model are: {supported_providers}"
        )
    
    return provider_options[0]


def _is_retryable_error(exception: BaseException) -> bool:
    error_text = str(exception).lower()

    # Fallback textual detection for SDK-wrapped HTTP errors that may not
    # expose a requests.Response object.
    if "429" in error_text or "too many requests" in error_text:
        return True

    if isinstance(exception, InvalidLLMResponseError):
        return True
    
    if isinstance(exception, RetryableLLMError):
        return True

    # Some SDKs (notably Hugging Face) may occasionally return malformed JSON.
    # Treat parser failures as transient provider errors and retry.
    if isinstance(exception, json.JSONDecodeError):
        return True

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
    max_tokens: int | None = None,
    timeout: int | None = None,
) -> str:
    if provider == "github":
        ai_response = github_models_chat(
            model=model,
            conversation=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    elif provider == "groq":
        ai_response = groq_chat(
            model=model,
            conversation=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    elif provider == "google_ai_studio":
        ai_response = google_ai_studio_chat(
            model=model,
            conversation=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    elif provider == "huggingface":
        ai_response = huggingface_chat(
            model=model,
            conversation=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    else:
        raise ValueError(
            f"Unsupported provider '{provider}'. Supported providers are: github, groq, google_ai_studio, huggingface."
        )

    if not ai_response or not isinstance(ai_response, str) or not ai_response.strip():
        raise InvalidLLMResponseError(
            f"Provider '{provider}' returned an invalid response for model '{model}'."
        )

    return ai_response.strip()

def llm_chat(
    model: str,
    conversation: Conversation,
    temperature: float,
    max_tokens: int | None = None,
    timeout: int | None = None,
    provider: str | None = None,
    retries: int = 9,
) -> str:
    """Route a chat request to a provider client with centralized retry/error handling."""
    selected_config = _select_provider_and_model(model, provider)
    provider = selected_config["provider"]
    provider_model = selected_config["model"]

    logger.debug(
        "Dispatching LLM chat request | provider=%s | model=%s | retries=%s | timeout=%s",
        provider,
        provider_model,
        retries,
        timeout,
    )

    def _log_retry_attempt(retry_state: object) -> None:
        attempt_number = getattr(retry_state, "attempt_number", 0)
        next_retry_number = attempt_number + 1
        next_action = getattr(retry_state, "next_action", None)
        wait_seconds = getattr(next_action, "sleep", None)

        outcome = getattr(retry_state, "outcome", None)
        error_text = "unknown"
        if outcome is not None and getattr(outcome, "failed", False):
            error = outcome.exception()
            if error is not None:
                error_text = str(error)

        logger.info(
            "Retrying LLM chat request | provider=%s | model=%s | retry=%s/%s | wait=%.2fs | error=%s",
            provider,
            provider_model,
            next_retry_number,
            retries,
            wait_seconds if wait_seconds is not None else 0.0,
            error_text,
        )

    retrying = Retrying(
        retry=retry_if_exception(_is_retryable_error),
        wait=wait_exponential(min=1, max=120),
        stop=stop_after_attempt(retries),
        before_sleep=_log_retry_attempt,
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
    except (requests.exceptions.RequestException, json.JSONDecodeError):
        logger.exception(
            "LLM API request failed after retries | provider=%s | model=%s",
            provider,
            provider_model,
        )
        raise

    raise RuntimeError("Unexpected retry loop termination without result.")
