"""Application settings loaded from environment variables and a local .env file.

Secrets are never hardcoded. Copy .env.example to .env and set required values.
"""

from functools import lru_cache

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_NAMES = {
    "openai_api_key": "OPENAI_API_KEY",
    "openai_model_interviewer": "OPENAI_MODEL_INTERVIEWER",
    "openai_model_scorer": "OPENAI_MODEL_SCORER",
    "openai_embedding_model": "OPENAI_EMBEDDING_MODEL",
    "openai_embedding_dimensions": "OPENAI_EMBEDDING_DIMENSIONS",
    "pinecone_api_key": "PINECONE_API_KEY",
    "pinecone_index": "PINECONE_INDEX",
    "pinecone_cloud": "PINECONE_CLOUD",
    "pinecone_region": "PINECONE_REGION",
    "sqlite_path": "SQLITE_PATH",
    "interview_question_count": "INTERVIEW_QUESTION_COUNT",
}

_REQUIRED_NONEMPTY = (
    "openai_api_key",
    "openai_model_interviewer",
    "openai_model_scorer",
    "openai_embedding_model",
    "pinecone_api_key",
    "pinecone_index",
    "pinecone_cloud",
    "pinecone_region",
    "sqlite_path",
)


class Settings(BaseSettings):
    """Typed configuration. Instantiation fails with ValidationError if required values are missing."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    openai_api_key: str
    openai_model_interviewer: str = "gpt-4o-mini"
    openai_model_scorer: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int | None = Field(
        default=None,
        description="Optional output size for text-embedding-3-* (must match the Pinecone index).",
    )

    pinecone_api_key: str
    pinecone_index: str = "roleready-questions"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    sqlite_path: str = "data/roleready.db"
    interview_question_count: int = Field(default=10, ge=1)

    @field_validator(*_REQUIRED_NONEMPTY, mode="after")
    @classmethod
    def reject_blank_values(cls, value: str, info: ValidationInfo) -> str:
        env_name = _ENV_NAMES[info.field_name]
        if not value.strip():
            raise ValueError(
                f"{env_name} is missing or empty. "
                "Copy .env.example to .env and set this value. "
                "Do not commit secrets."
            )
        return value.strip()


@lru_cache
def get_settings() -> Settings:
    """Load settings once from the environment / .env file."""
    return Settings()


def clear_settings_cache() -> None:
    """Drop the cached Settings instance (useful in tests)."""
    get_settings.cache_clear()
