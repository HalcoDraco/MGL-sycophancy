import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() 

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

messages = [
    # System instruction
    {"role": "developer", "content": "You are a highly concise and helpful AI assistant."},
    
    # Historical conversation
    {"role": "user", "content": "Hello! My name is Alex."},
    {"role": "assistant", "content": "Hi Alex! How can I help you today?"},
    
    # The new user query
    {"role": "user", "content": "Do you remember my name?"}
]

# Make a single, stateless request using the chat completions endpoint
response = client.chat.completions.create(
    model="gpt-5.4",
    messages=messages,
    max_completion_tokens=500,
    # temperature=0.3, # Temperature not supported
    reasoning_effort="medium",
)

# Output the agent's response
print(response.choices[0].message.content)