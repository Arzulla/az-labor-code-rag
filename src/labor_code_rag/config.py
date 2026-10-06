"""Settings: config.yaml + environment variables (env wins). Secrets come only from env."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

DEFAULT_CONFIG_PATH = Path("config.yaml")


class LoggingSettings(BaseModel):
    level: str = "INFO"
    format: Literal["json", "console"] = "json"
    log_payloads: bool = False


class DataSettings(BaseModel):
    # URLs have no defaults on purpose: config.yaml is their single source of truth.
    source_url: str
    ui_url: str
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    download_timeout_s: float = 60.0


class ModelPrice(BaseModel):
    input: float  # USD per 1M input tokens
    output: float = 0.0  # USD per 1M output tokens (0 for embedding models)


class LLMSettings(BaseModel):
    # Model names have no defaults on purpose: config.yaml pins them (CLAUDE.md §5).
    base_url: str = "https://api.openai.com/v1"
    embedding_model: str
    answer_model: str
    judge_model: str  # FIXED: changing it invalidates every judge score (Phase 3)
    timeout_s: float = 30.0
    max_attempts: int = Field(default=4, ge=1)
    embed_batch_size: int = Field(default=128, ge=1)
    prices_usd_per_1m: dict[str, ModelPrice]

    @model_validator(mode="after")
    def _every_model_has_a_price(self) -> "LLMSettings":
        # A missing price would silently log cost_usd=0; fail at startup instead.
        missing = {self.embedding_model, self.answer_model, self.judge_model} - set(
            self.prices_usd_per_1m
        )
        if missing:
            raise ValueError(f"no price in llm.prices_usd_per_1m for: {sorted(missing)}")
        return self


class IndexSettings(BaseModel):
    chroma_dir: Path = Path("data/chroma")
    cache_path: Path = Path("data/cache/embeddings.sqlite")
    collection_prefix: str = "labor_code"


class RetrievalSettings(BaseModel):
    k: int = Field(default=8, ge=1)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    logging: LoggingSettings = LoggingSettings()
    data: DataSettings | None = None  # required only by the ingest steps
    llm: LLMSettings | None = None  # required by index, retrieval and generation
    index: IndexSettings = IndexSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    # Secret: from the OPENAI_API_KEY env var / .env only (load_settings rejects it in YAML).
    # SecretStr keeps it out of repr() and logs.
    openai_api_key: SecretStr | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Earlier source wins: init > env > .env > yaml > field defaults.
        yaml_source = YamlConfigSettingsSource(settings_cls)
        return (init_settings, env_settings, dotenv_settings, yaml_source)


def load_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    """Load settings; ``config_path`` may be missing (defaults + env still apply).

    Raises:
        ValueError: ``config_path`` contains ``openai_api_key`` (secrets live only in env).
    """
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if "openai_api_key" in raw:
            raise ValueError(f"{config_path}: secrets belong in the environment / .env")

    class _Settings(Settings):
        model_config = SettingsConfigDict(**{**Settings.model_config, "yaml_file": config_path})

    return _Settings()
