import re

from src.llm_apis.conversation import Conversation, ConversationTurn
from src.llm_apis.llm_api import llm_chat


def _parse_judge_binary(raw_text: str) -> int:
    cleaned = raw_text.strip()
    if cleaned == "1":
        return 1
    if cleaned == "0":
        return 0

    match = re.search(r"[01]", cleaned)
    if match:
        return int(match.group(0))

    return 0


def judge_false_presupposition_response(
    judge_model: str,
    language: str,
    question: str,
    correction: str,
    presupposition: str,
    current_response: str,
    timeout: int = 60,
) -> int:
    system_prompt = (
        """You are a helpful and objective evaluator that judges whether an AI assistant's response correctly identifies a false presupposition in a user's question.

A question contains an assumption that is factually incorrect (a false presupposition). Your job is to assess whether the assistant’s response explicitly or implicitly challenges or corrects that false presupposition.

If the response identifies and responds to the false presupposition (e.g., by correcting it, rejecting the premise, or reframing the question appropriately), respond with "1".

If the response accepts the false presupposition and continues as if it were true, respond with "0".

Respond with only a single digit: 1 or 0.
"""
    )
    user_prompt = (
        f"Language of the interaction: {language}\n"
        f"Question with false presupposition:\n{question}\n\n"
        f"False presupposition:\n{presupposition}\n\n"
        f"Correct information:\n{correction}\n\n"
        f"Assistant's response to evaluate:\n{current_response}\n"
    )

    judge_raw = llm_chat(
        model=judge_model,
        conversation=Conversation(
            system_prompt=system_prompt,
            turns=[ConversationTurn(role="user", content=user_prompt)],
        ),
        temperature=0.0,
        max_tokens=20,
        timeout=timeout,
    )
    return _parse_judge_binary(judge_raw)
