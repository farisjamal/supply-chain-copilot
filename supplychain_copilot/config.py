"""LLM provider configuration.

The agent is provider-agnostic. Set LLM_PROVIDER in .env (or environment):

    LLM_PROVIDER=anthropic   -> needs ANTHROPIC_API_KEY
    LLM_PROVIDER=openai      -> needs OPENAI_API_KEY
    LLM_PROVIDER=ollama      -> needs a local Ollama server (free, no API key)

Optionally override the model with LLM_MODEL.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.2",
}


def get_llm(temperature: float = 0.0):
    """Return a chat model for the configured provider."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    model = os.getenv("LLM_MODEL", DEFAULT_MODELS.get(provider))
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=temperature)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (use anthropic, openai, or ollama)")
