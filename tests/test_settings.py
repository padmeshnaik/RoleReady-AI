import os

import pytest
from pydantic import ValidationError

from roleready.config.settings import Settings


@pytest.fixture
def no_secret_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)


def _valid_kwargs(**overrides: object) -> dict:
    values: dict = {
        "openai_api_key": "test-openai-key",
        "pinecone_api_key": "test-pinecone-key",
    }
    values.update(overrides)
    return values


def test_loads_required_keys_without_env_file() -> None:
    settings = Settings(_env_file=None, **_valid_kwargs())
    assert settings.openai_api_key == "test-openai-key"
    assert settings.pinecone_api_key == "test-pinecone-key"


def test_interview_question_count_defaults_to_10() -> None:
    settings = Settings(_env_file=None, **_valid_kwargs())
    assert settings.interview_question_count == 10


def test_missing_openai_api_key_raises_clear_error(no_secret_env: None) -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None, pinecone_api_key="test-pinecone-key")
    message = str(exc_info.value)
    assert "openai_api_key" in message
    assert "Field required" in message


def test_blank_openai_api_key_raises_clear_error() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None, **_valid_kwargs(openai_api_key="   "))
    assert "OPENAI_API_KEY" in str(exc_info.value)


def test_missing_pinecone_api_key_raises_clear_error(no_secret_env: None) -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None, openai_api_key="test-openai-key")
    message = str(exc_info.value)
    assert "pinecone_api_key" in message
    assert "Field required" in message


def test_interview_question_count_rejects_zero() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None, **_valid_kwargs(interview_question_count=0))
    assert "interview_question_count" in str(exc_info.value)


def test_blank_embedding_dimensions_are_optional() -> None:
    settings = Settings(_env_file=None, **_valid_kwargs(openai_embedding_dimensions=""))
    assert settings.openai_embedding_dimensions is None


def test_apply_env_overrides_fills_missing_known_names(monkeypatch: pytest.MonkeyPatch) -> None:
    from roleready.config.settings import apply_env_overrides

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    apply_env_overrides({"OPENAI_API_KEY": "from-secrets", "UNKNOWN": "ignore"})
    assert os.environ["OPENAI_API_KEY"] == "from-secrets"
    assert "UNKNOWN" not in os.environ


def test_missing_env_names_maps_validation_fields(no_secret_env: None) -> None:
    from roleready.config.settings import missing_env_names

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)
    names = missing_env_names(exc_info.value)
    assert "OPENAI_API_KEY" in names
    assert "PINECONE_API_KEY" in names


