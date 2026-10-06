from pathlib import Path

import pytest
from pydantic import ValidationError

from labor_code_rag.config import load_settings


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # No stray .env or LOGGING__* variables from the developer machine.
    monkeypatch.chdir(tmp_path)
    for name in ("LOGGING__LEVEL", "LOGGING__FORMAT", "LOGGING__LOG_PAYLOADS"):
        monkeypatch.delenv(name, raising=False)
    for name in ("DATA__SOURCE_URL", "DATA__UI_URL", "DATA__RAW_DIR", "DATA__DOWNLOAD_TIMEOUT_S"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _write_yaml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_defaults_when_yaml_missing(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "missing.yaml")
    assert settings.logging.level == "INFO"
    assert settings.logging.format == "json"
    assert settings.logging.log_payloads is False
    assert settings.data is None


def test_yaml_overrides_defaults(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "logging:\n  level: DEBUG\n  format: console\n")
    settings = load_settings(path)
    assert settings.logging.level == "DEBUG"
    assert settings.logging.format == "console"


def test_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_yaml(tmp_path, "logging:\n  level: DEBUG\n  log_payloads: false\n")
    monkeypatch.setenv("LOGGING__LEVEL", "ERROR")
    monkeypatch.setenv("LOGGING__LOG_PAYLOADS", "true")

    settings = load_settings(path)

    assert settings.logging.level == "ERROR"
    assert settings.logging.log_payloads is True


def test_data_section_from_yaml(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        "data:\n  source_url: https://example.test/f_1.html\n  ui_url: https://example.test/1\n",
    )
    settings = load_settings(path)
    assert settings.data is not None
    assert settings.data.source_url == "https://example.test/f_1.html"
    assert settings.data.ui_url == "https://example.test/1"
    assert settings.data.raw_dir == Path("data/raw")
    assert settings.data.download_timeout_s == 60.0


def test_repo_config_yaml_has_data_section() -> None:
    settings = load_settings(Path(__file__).parents[1] / "config.yaml")
    assert settings.data is not None
    assert settings.data.source_url.endswith("/f_46943.html")


def test_repo_config_yaml_pins_models_and_prices() -> None:
    settings = load_settings(Path(__file__).parents[1] / "config.yaml")
    assert settings.llm is not None
    assert settings.llm.embedding_model == "text-embedding-3-small"
    assert settings.llm.answer_model == "gpt-4.1-mini-2025-04-14"
    assert settings.llm.judge_model == "gpt-4.1-2025-04-14"
    assert settings.retrieval.k == 8
    assert settings.index.chroma_dir == Path("data/chroma")


_LLM_YAML = """\
llm:
  embedding_model: emb
  answer_model: ans
  judge_model: judge
  prices_usd_per_1m:
    emb: {input: 0.02}
    ans: {input: 0.4, output: 1.6}
"""


def test_missing_price_fails_at_startup(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="no price.*judge"):
        load_settings(_write_yaml(tmp_path, _LLM_YAML))


def test_api_key_from_env_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    settings = load_settings(tmp_path / "missing.yaml")
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-from-env"
    assert "sk-from-env" not in repr(settings)

    with pytest.raises(ValueError, match="secrets belong in the environment"):
        load_settings(_write_yaml(tmp_path, "openai_api_key: sk-in-yaml\n"))
