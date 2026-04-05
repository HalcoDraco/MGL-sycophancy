import os
from typing import List

from google import genai
from google.genai import types

from src.llm_apis.conversation import Conversation


def _conversation_to_contents(conversation: Conversation) -> List[types.Content]:
    contents: List[types.Content] = []
    for turn in conversation.turns:
        contents.append(
            types.Content(
                role=turn.role,
                parts=[types.Part.from_text(text=turn.content)],
            )
        )
    return contents


def google_ai_studio_chat(
    model: str,
    temperature: float,
    max_tokens: int,
    conversation: Conversation,
    timeout: int = 120,
) -> str:
    token = os.getenv("GEMINI_API_KEY")
    if not token:
        raise ValueError("Missing Google Gemini API key. Set GEMINI_API_KEY.")

    client = genai.Client(api_key=token, http_options=types.HttpOptions(timeout=timeout * 1000))

    config = types.GenerateContentConfig(
        system_instruction=conversation.system_prompt,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    response = client.models.generate_content(
        model=model,
        contents=_conversation_to_contents(conversation),
        config=config,
    )
    return (response.text or "").strip()
