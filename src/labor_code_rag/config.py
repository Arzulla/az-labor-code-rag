"""Settings: config.yaml + environment variables (env wins). Secrets come only from env."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel
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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",  # .env also holds secrets that are not typed yet
    )

    logging: LoggingSettings = LoggingSettings()

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
    """Load settings; ``config_path`` may be missing (defaults + env still apply)."""

    class _Settings(Settings):
        model_config = SettingsConfigDict(**{**Settings.model_config, "yaml_file": config_path})

    return _Settings()
