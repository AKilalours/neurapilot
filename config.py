"""NeuraPilot configuration — all settings driven by environment variables.

Design: Pydantic-settings with strict validation so misconfiguration is caught
at startup, not buried in a runtime traceback.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration object. Reads from .env and environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── LLM Provider ──────────────────────────────────────────────────────────
    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_chat_model: str = Field(default="llama3.1:8b", alias="OLLAMA_CHAT_MODEL")
    ollama_embed_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBED_MODEL")

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_embed_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBED_MODEL")

    # ── Vector DB ─────────────────────────────────────────────────────────────
    chroma_dir: str = Field(default=".chroma", alias="CHROMA_DIR")
    base_collection: str = Field(default="neurapilot", alias="BASE_COLLECTION")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = Field(default=6, ge=1, le=30, alias="TOP_K")
    candidate_k: int = Field(default=16, ge=4, le=60, alias="CANDIDATE_K")
    chunk_size: int = Field(default=900, ge=200, le=4000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=180, ge=0, alias="CHUNK_OVERLAP")
    mmr_lambda: float = Field(default=0.6, ge=0.0, le=1.0, alias="MMR_LAMBDA")

    # ── Grounding ─────────────────────────────────────────────────────────────
    strict_grounding: bool = Field(default=True, alias="STRICT_GROUNDING")
    hallucination_guard: bool = Field(default=True, alias="HALLUCINATION_GUARD")

    # ── Storage ───────────────────────────────────────────────────────────────
    data_dir: str = Field(default=".data", alias="DATA_DIR")
    upload_dir: str = Field(default=".data/uploads", alias="UPLOAD_DIR")
    db_path: str = Field(default=".data/neurapilot.db", alias="DB_PATH")

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_enabled: bool = Field(default=True, alias="EVAL_ENABLED")

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("llm_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"ollama", "openai"}
        v = v.strip().lower()
        if v not in allowed:
            raise ValueError(f"LLM_PROVIDER must be one of {allowed}, got {v!r}")
        return v

    @model_validator(mode="after")
    def validate_openai_config(self) -> "Settings":
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if self.candidate_k < self.top_k:
            raise ValueError("CANDIDATE_K must be >= TOP_K")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
