import re
from pathlib import Path

import duckdb
import pandas as pd
from huggingface_hub import hf_hub_download

from churn_mcp.config import TelcoChurnMcpConfig

SPLITS = ("train", "validation", "test")

# Single-valued or duplicated by lat/long; kept in the traps, not the schema.
DROPPED = ("country", "state", "quarter", "lat_long")

ADDONS = (
    "online_security",
    "online_backup",
    "device_protection_plan",
    "premium_tech_support",
    "streaming_tv",
    "streaming_movies",
    "streaming_music",
)

TENURE_EDGES = (0, 6, 12, 24, 48, 1_000)
TENURE_LABELS = ("0-6m", "7-12m", "13-24m", "25-48m", "49m+")


def _snake(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def download(config: TelcoChurnMcpConfig) -> pd.DataFrame:
    frames = [
        pd.read_csv(
            hf_hub_download(
                config.data.dataset_id,
                f"{split}.csv",
                repo_type="dataset",
                revision=config.data.revision,
            )
        )
        for split in SPLITS
    ]
    return pd.concat(frames, ignore_index=True)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns={c: _snake(c) for c in raw.columns})
    df = df.drop(columns=[c for c in DROPPED if c in df.columns])

    df["offer"] = df["offer"].fillna("None")
    df["internet_type"] = df["internet_type"].fillna("No Internet")
    df["churn_category"] = df["churn_category"].fillna("Not Churned")
    df["churn_reason"] = df["churn_reason"].fillna("Not Churned")

    df["is_active"] = (df["customer_status"] != "Churned").astype(int)
    df["arr"] = (df["monthly_charge"] * 12).round(2)
    df["num_addons"] = df[list(ADDONS)].sum(axis=1)
    df["tenure_bucket"] = pd.cut(
        df["tenure_in_months"], bins=TENURE_EDGES, labels=TENURE_LABELS, right=True
    ).astype(str)
    return df


def build(config: TelcoChurnMcpConfig) -> Path:
    df = clean(download(config))
    if len(df) != config.data.expected_rows:
        raise ValueError(f"expected {config.data.expected_rows} rows, got {len(df)}")

    path = config.data.artifacts_dir / "churn.duckdb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)

    con = duckdb.connect(path)
    con.register("cleaned", df)
    con.execute("CREATE TABLE customers AS SELECT * FROM cleaned")
    con.close()
    return path


def connect(config: TelcoChurnMcpConfig) -> duckdb.DuckDBPyConnection:
    path = config.data.artifacts_dir / "churn.duckdb"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing, run `churn-mcp prepare` first")

    con = duckdb.connect(path, read_only=True)
    # Lock after disabling access, or a query could SET it back on.
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")
    return con
