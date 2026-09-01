"""Application settings loaded from environment variables and a local .env file.

Secrets are never hardcoded. Copy .env.example to .env and set required values.
The .env path is the repository root so Streamlit still finds it when the
working directory is the app file folder. Streamlit Cloud should set the same
names in App settings → Secrets (there is no .env on Cloud).
"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, ValidationError, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/roleready/config/settings.py -> repository root
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"

_ENV_NAMES = {
    "openai_api_key": "OPENAI_API_KEY",
    "openai_model_interviewer": "OPENAI_MODEL_INTERVIEWER",
    "openai_model_scorer": "OPENAI_MODEL_SCORER",
    "openai_model_question_generator": "OPENAI_MODEL_QUESTION_GENERATOR",
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
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    openai_api_key: str
    openai_model_interviewer: str = "gpt-4o-mini"
    openai_model_scorer: str = "gpt-4o-mini"
    openai_model_question_generator: str | None = Field(
        default=None,
        description="Model id for the offline question-generation pipeline. Set via OPENAI_MODEL_QUESTION_GENERATOR.",
    )
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

    @field_validator("openai_embedding_dimensions", mode="before")
    @classmethod
    def blank_embedding_dimensions_is_unset(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("openai_model_question_generator", mode="after")
    @classmethod
    def blank_question_generator_is_unset(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


def apply_env_overrides(values: dict[str, str]) -> None:
    """Copy known config names into the process environment if they are not already set."""
    known = set(_ENV_NAMES.values())
    for key, value in values.items():
        if key not in known:
            continue
        text = str(value).strip()
        if not text:
            continue
        existing = os.environ.get(key, "").strip()
        if existing:
            continue
        os.environ[key] = text


def missing_env_names(exc: ValidationError) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for error in exc.errors():
        loc = error.get("loc") or ()
        if not loc:
            continue
        field = str(loc[0])
        env_name = _ENV_NAMES.get(field, field)
        if env_name not in seen:
            seen.add(env_name)
            names.append(env_name)
    return names


@lru_cache
def get_settings() -> Settings:
    """Load settings once from the environment / .env file."""
    load_dotenv(DEFAULT_ENV_FILE, encoding="utf-8", override=False)
    return Settings(_env_file=DEFAULT_ENV_FILE if DEFAULT_ENV_FILE.exists() else None)


def clear_settings_cache() -> None:
    """Drop the cached Settings instance (useful in tests)."""
    get_settings.cache_clear()
