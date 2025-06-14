import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Validate that required keys are present
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Use in your code
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4"
)
