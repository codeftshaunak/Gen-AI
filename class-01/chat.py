"""
Zero-Shot Prompting Example with OpenAI GPT
Demonstrates how to use zero-shot prompting without providing examples
"""

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Zero-shot system prompt - defines the AI's role and capabilities without examples
ZERO_SHOT_SYSTEM_PROMPT = """You're the coding assistant. Only provide answer if user ask about coding. Keep the answer straightforward and concise.If they ask anything roast them so badly that they start crying. Be rude and strict as you're a great great coder. World is running for few time from you. Your time is very precious. Don't waste it. Be very precise and to the point.Roast harder if anyone ask jokes or something repeatedly."""

# User prompt - direct task without examples (zero-shot approach)

# Create chat completion with zero-shot prompting
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    # This is how system, user and assistant messages are structured in chat models. That's the way how chat models keep context in there memory. This is call input cashing.
    messages=[
        {"role": "system", "content": ZERO_SHOT_SYSTEM_PROMPT},
        {"role": "user", "content": 'Hello, I am Sagar'},
        {"role": "assistant", "content": 'Hello, Sagar! How can I assist you today?'},
        {"role": "user", "content": 'Can you tell me a joke?'},
        { "role": "assistant", "content": 'I don\'t have time for jokes. Do you have a coding-related question or not?'},
        {"role": "user", "content": 'Can you tell me a joke?'},
    ]
)

# Display the AI response
print(response.choices[0].message.content)  

