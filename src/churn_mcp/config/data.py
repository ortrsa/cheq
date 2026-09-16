from pathlib import Path

from churn_mcp.config.base import REPO_ROOT, Section


class DataConfig(Section):
    dataset_id: str = "aai510-group1/telco-customer-churn"
    revision: str = "c18fe6295a6ca80ca26627a6627c6f11ccd21d86"
    expected_rows: int = 7043
    artifacts_dir: Path = REPO_ROOT / ".artifacts"
    semantic_path: Path = REPO_ROOT / "config" / "semantic.yaml"
