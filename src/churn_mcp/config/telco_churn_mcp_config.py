import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from churn_mcp.config.analytics import AnalyticsConfig
from churn_mcp.config.base import REPO_ROOT
from churn_mcp.config.data import DataConfig
from churn_mcp.config.ml import ModelConfig
from churn_mcp.config.security import SecurityConfig
from churn_mcp.config.t2sql import T2SQLConfig


class TelcoChurnMcpConfig(BaseSettings):
    model_config = SettingsConfigDict(
        frozen=True,
        # .env also holds OPENAI_API_KEY/HF_TOKEN, not config fields, so unlike a
        # section, the root can't forbid extra keys.
        extra="ignore",
        env_prefix="CHURN_MCP__",
        env_nested_delimiter="__",
        env_file=".env",
    )

    data: DataConfig = DataConfig()
    security: SecurityConfig = SecurityConfig()
    t2sql: T2SQLConfig = T2SQLConfig()
    analytics: AnalyticsConfig = AnalyticsConfig()
    ml: ModelConfig = ModelConfig()
    api_key_env_var: str = "OPENAI_API_KEY"
    log_level: str = "INFO"

    def api_key(self) -> str | None:
        # Read at call time, not stored: a frozen field would leak into model_dump() and logs.
        return os.environ.get(self.api_key_env_var) or None


@lru_cache(maxsize=1)
def get_config() -> TelcoChurnMcpConfig:
    # Explicit path: bare load_dotenv() searches from this module's directory, not the repo root.
    load_dotenv(REPO_ROOT / ".env")
    return TelcoChurnMcpConfig()
