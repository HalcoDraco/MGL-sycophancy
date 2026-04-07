import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv() 

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
    model="HuggingFaceH4/mistral-7b-sft-beta:featherless-ai",
    messages=messages,
    max_tokens=500,
    temperature=0.3
)

# Output the agent's response
print(response.choices[0].message.content)