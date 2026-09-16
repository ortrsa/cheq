from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Section(BaseSettings):
    model_config = SettingsConfigDict(frozen=True, extra="forbid")
