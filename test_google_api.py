import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present

# Initialize the client. 
# Ensure your API key is set as an environment variable named GEMINI_API_KEY
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Missing Google Gemini API key. Set GEMINI_API_KEY environment variable.")
client = genai.Client(api_key=api_key)

# 1. Define the conversation history explicitly.
# The order should logically alternate between "user" and "model".
conversation_history = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="Hello! My name is Alex.")]
    ),
    types.Content(
        role="model",
        parts=[types.Part.from_text(text="Hi Alex! How can I help you today?")]
    ),
    # The new user query that relies on the context above
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="Do you remember my name?")] 
    )
]

# 2. Set your system prompt and any other generation parameters
config = types.GenerateContentConfig(
    system_instruction="You are a highly concise and helpful AI assistant.",
    temperature=0.3 # Optional: lower temperature for more deterministic answers
)

# 3. Make a single, stateless request passing the full history
response = client.models.generate_content(
    model='gemini-3.1-flash-lite-preview', # Or your preferred model (e.g., gemini-2.5-pro)
    contents=conversation_history,
    config=config
)

# Output the agent's response
print(response.text)