import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present

# Initialize the client. 
# Ensure your Hugging Face token is set as an environment variable named HF_TOKEN
# You can generate a token in your Hugging Face account settings (Read access is enough for inference)
client = InferenceClient(api_key=os.environ.get("HF_TOKEN"))

# Define the full context: system prompt, history, and the new query
messages = [
    # System instruction
    {"role": "system", "content": "You are a highly concise and helpful AI assistant."},
    
    # Historical conversation
    {"role": "user", "content": "Hello! My name is Alex."},
    {"role": "assistant", "content": "Hi Alex! How can I help you today?"},
    
    # The new user query
    {"role": "user", "content": "Do you remember my name?"}
]

# Make a single, stateless request using the chat_completion method
response = client.chat_completion(
    model="HuggingFaceH4/zephyr-7b-beta:featherless-ai", # Replace with any chat-supported model ID from the Hub
    messages=messages,
    max_tokens=500,
    temperature=0.3 # Optional: lower temperature for more deterministic output
)

# Output the agent's response
print(response.choices[0].message.content)