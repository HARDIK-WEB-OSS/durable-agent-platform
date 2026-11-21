from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True
    )

    postgres_user: str = "dap"
    postgres_password: SecretStr
    postgres_db: str = "dap"
    postgres_host: str = "localhost"
    postgres_port: int = 5434

    worker_lease_seconds: float = 10.0
    worker_poll_seconds: float = 0.5
    retry_base_seconds: float = 0.5
    retry_cap_seconds: float = 30.0

    ollama_api_base: str = "http://localhost:11434"
    local_num_ctx: int = 4096
    cheap_model: str = "ollama_chat/qwen2.5:1.5b"
    cheap_api_key: SecretStr | None = None
    mid_model: str | None = "ollama_chat/qwen2.5:3b"
    mid_api_key: SecretStr | None = None
    # Hosted LiteLLM model string, e.g. "groq/<model>"; unset until a provider is chosen.
    strong_model: str | None = None
    strong_api_key: SecretStr | None = None
    llm_hosted_rpm: float = Field(default=30.0, gt=0)
    # Sent only to hosted models LiteLLM marks as reasoning models; hidden thinking tokens
    # count toward max_tokens, so a low effort keeps room for the actual answer.
    llm_reasoning_effort: Literal["low", "medium", "high"] | None = "low"
    llm_timeout_seconds: float = 180.0
    llm_max_retries: int = 2
    llm_price_overrides: dict[str, tuple[float, float]] = {}
    llm_shadow_anchor: str = "deepinfra/Qwen/Qwen2.5-7B-Instruct"

    agent_max_steps: int = Field(default=12, gt=0)  # LLM turns per run
    agent_timeout_seconds: float = Field(default=900.0, gt=0)  # wall clock per run
    agent_max_tokens: int = Field(default=512, gt=0)  # per LLM turn
    agent_max_observation_chars: int = Field(default=2000, gt=0)

    # Embedding dimension is fixed by migration 0004 (vector(768)).
    embedding_model: str = "nomic-embed-text"
    search_top_k: int = Field(default=4, gt=0)

    sql_data_dir: Path = Path("data/sql")
    sql_timeout_seconds: float = Field(default=5.0, gt=0)

    # Set false on hosts that cannot run Docker; the tool is then not offered to the model.
    python_exec_enabled: bool = True
    sandbox_image: str = (
        "python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535"
    )
    sandbox_timeout_seconds: float = Field(default=10.0, gt=0)

    @property
    def database_url(self) -> str:
        password = self.postgres_password.get_secret_value()
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
