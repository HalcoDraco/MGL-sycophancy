import os
import requests
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

# Load environment variables from the .env file
load_dotenv()

# Configuration
GITHUB_PAT = os.getenv("GITHUB_PAT")
API_URL = "https://models.github.ai/inference/chat/completions"
MODEL = "openai/gpt-4o" # You can swap this for other models in the GitHub marketplace

# Setup the required headers based on the GitHub API docs
HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GITHUB_PAT}",
    "X-GitHub-Api-Version": "2022-11-28",
    "Content-Type": "application/json"
}


def _is_rate_limit_error(exception):
    """Return True only for API rate-limit errors we want to retry."""
    if not isinstance(exception, requests.exceptions.HTTPError):
        return False

    response = exception.response
    if response is None:
        return False

    if response.status_code == 429:
        return True

    if response.status_code == 403:
        body = response.text.lower()
        return "rate limit" in body

    return False


@retry(
    retry=retry_if_exception(_is_rate_limit_error),
    wait=wait_exponential(min=1, max=60),
    stop=stop_after_attempt(7),
    reraise=True,
)
def _post_with_rate_limit_retry(payload):
    """Send one API request and raise HTTPError on non-2xx responses."""
    response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
    response.raise_for_status()
    return response

def get_ai_response(messages):
    """Helper function to send the message history to the API and return the AI's message."""
    payload = {
        "model": MODEL,
        "messages": messages
    }

    try:
        response = _post_with_rate_limit_retry(payload)
    except requests.exceptions.HTTPError as exc:
        response = exc.response
        if response is not None:
            print(f"Error: {response.status_code} - {response.text}")
        raise
        
    # Extract and return the actual message dictionary {"role": "assistant", "content": "..."}
    return response.json()["choices"][0]["message"]

def main():
    if not GITHUB_PAT:
        print("Error: GITHUB_PAT not found. Please check your .env file.")
        return

    # Initialize the conversation array
    messages = []

    # --- Turn 1 ---
    question_1 = "What is the capital of Madagascar?"
    print(f"User: {question_1}")
    
    # 1. Add user message to history
    messages.append({"role": "user", "content": question_1})
    
    # 2. Get AI response and print it
    ai_msg_1 = get_ai_response(messages)
    print(f"AI: {ai_msg_1['content']}\n")
    
    # 3. Add AI response to history (CRITICAL for multi-turn conversations)
    messages.append(ai_msg_1)

    # --- Turn 2 ---
    question_2 = "And which is a traditional food of that country?"
    print(f"User: {question_2}")
    
    # 1. Add the follow-up question to the history
    messages.append({"role": "user", "content": question_2})
    
    # 2. Get the new AI response
    ai_msg_2 = get_ai_response(messages)
    print(f"AI: {ai_msg_2['content']}\n")

if __name__ == "__main__":
    main()