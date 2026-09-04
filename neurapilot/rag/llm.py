"""LLM + embedding factory for NeuraPilot.

Supports two providers:
  - ollama: local, private, zero-cost inference
  - openai: cloud, higher capability

The LLMBundle is a lightweight dataclass so it can be cached at module
import time and shared across the Streamlit session without re-init cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from neurapilot.config import Settings


@dataclass(frozen=True)
class LLMBundle:
    """Holds the chat model and embedding model for a session."""
    llm: BaseChatModel
    embeddings: Embeddings
    provider: str
    model_name: str
    embed_model_name: str


def build_llm_bundle(settings: Settings) -> LLMBundle:
    """Construct and return an LLMBundle based on current settings.

    Raises ValueError for unsupported provider or missing credentials.
    """
    provider = settings.llm_provider.strip().lower()

    if provider == "ollama":
        return _build_ollama(settings)

    if provider == "openai":
        return _build_openai(settings)

    raise ValueError(
        f"Unsupported LLM_PROVIDER={provider!r}. "
        "Valid options: 'ollama', 'openai'."
    )


def _build_ollama(settings: Settings) -> LLMBundle:
    """Build Ollama-backed LLM bundle (local inference)."""
    try:
        from langchain_ollama import ChatOllama, OllamaEmbeddings
    except ImportError as e:
        raise ImportError("langchain-ollama not installed. Run: pip install langchain-ollama") from e

    llm = ChatOllama(
        model=settings.ollama_chat_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        num_predict=2048,
    )
    embeddings = OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )
    return LLMBundle(
        llm=llm,
        embeddings=embeddings,
        provider="ollama",
        model_name=settings.ollama_chat_model,
        embed_model_name=settings.ollama_embed_model,
    )


def _build_openai(settings: Settings) -> LLMBundle:
    """Build OpenAI-backed LLM bundle (cloud inference)."""
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY must be set for openai provider.")

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError as e:
        raise ImportError("langchain-openai not installed. Run: pip install langchain-openai") from e

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
        max_tokens=2048,
    )
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embed_model,
        api_key=settings.openai_api_key,
    )
    return LLMBundle(
        llm=llm,
        embeddings=embeddings,
        provider="openai",
        model_name=settings.openai_model,
        embed_model_name=settings.openai_embed_model,
    )
