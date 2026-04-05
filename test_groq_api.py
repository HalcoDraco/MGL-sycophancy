import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present

# Initialize the client. 
# Ensure your API key is set as an environment variable named GROQ_API_KEY
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("Missing Groq API key. Set GROQ_API_KEY environment variable.")
client = Groq(api_key=groq_api_key)

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

# Make a single, stateless request
chat_completion = client.chat.completions.create(
    messages=messages,
    model="llama-3.3-70b-versatile", # Replace with your preferred Groq model (e.g., mixtral-8x7b-32768)
    temperature=0.3         # Optional: lower temperature for more deterministic output
)

# Output the agent's response
print(chat_completion.choices[0].message.content)