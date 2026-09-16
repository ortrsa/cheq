import json

import pytest
from pydantic import ValidationError

from churn_mcp.config import ModelTier, SecurityConfig, TelcoChurnMcpConfig, get_config


@pytest.fixture(autouse=True)
def _fresh_cache():
    get_config.cache_clear()
    yield
    get_config.cache_clear()


def test_get_config_is_cached():
    assert get_config() is get_config()


def test_frozen():
    with pytest.raises(ValidationError):
        get_config().security.max_rows = 1  # type: ignore[misc]


def test_nested_env_override(monkeypatch):
    monkeypatch.setenv("CHURN_MCP__T2SQL__GENERATOR_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("CHURN_MCP__SECURITY__MAX_ROWS", "500")
    cfg = get_config()
    assert cfg.t2sql.generator_model is ModelTier.SOL
    assert cfg.security.max_rows == 500


@pytest.mark.parametrize(
    ("var", "bad"),
    [
        ("CHURN_MCP__SECURITY__MAX_ROWS", "0"),
        ("CHURN_MCP__T2SQL__GENERATOR_MODEL", "gpt-9"),
        ("CHURN_MCP__ANALYTICS__CONFIDENCE_LEVEL", "1.5"),
    ],
)
def test_invalid_values_fail_fast(monkeypatch, var, bad):
    monkeypatch.setenv(var, bad)
    with pytest.raises(ValidationError):
        get_config()


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        SecurityConfig(typo=1)  # type: ignore[call-arg]


def test_api_key_never_in_dump(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    cfg = TelcoChurnMcpConfig()
    assert cfg.api_key() == "sk-secret"
    assert "sk-secret" not in json.dumps(cfg.model_dump(mode="json"))


def test_missing_key_is_none_not_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert TelcoChurnMcpConfig().api_key() is None


def test_env_file_may_hold_unrelated_keys(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-x\nHF_TOKEN=hf_y\nCHURN_MCP__SECURITY__MAX_ROWS=42\n")
    monkeypatch.chdir(tmp_path)
    cfg = TelcoChurnMcpConfig()
    assert cfg.security.max_rows == 42


def test_api_key_is_read_from_the_process_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-environment")
    get_config.cache_clear()
    assert get_config().api_key() == "sk-from-environment"
