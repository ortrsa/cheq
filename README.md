# Telco Churn MCP

An MCP server that answers natural-language questions about telco customer churn with numbers
you can check: every answer comes with the SQL that produced it, the result table, the
assumptions, and the caveats.

Dataset: [`aai510-group1/telco-customer-churn`](https://huggingface.co/datasets/aai510-group1/telco-customer-churn),
7,043 customers, pinned to a commit so results are reproducible.

Design details (pipeline, model, evaluation) are in `docs/design.pdf`.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key (only `ask_data` needs it; the other tools work without one)

Dependencies are declared in `pyproject.toml` and installed by `uv sync`: `mcp`, `duckdb`,
`sqlglot`, `openai`, `pydantic-settings`, `pandas`, `scikit-learn`, `scipy`, `huggingface-hub`,
`typer`, `pyyaml`, `python-dotenv`.

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | for `ask_data` | LLM calls (routing, SQL generation, synthesis) |
| `PYTHONPATH` | when registering the server | Point at `<repo>/src` so clients can import `churn_mcp` (see Troubleshooting) |
| `CHURN_MCP__<SECTION>__<KEY>` | no | Override any setting, e.g. `CHURN_MCP__T2SQL__GENERATOR_MODEL`, `CHURN_MCP__SECURITY__MAX_ROWS` |

The key is read from `.env` at the repo root, so MCP clients don't need it passed in. An
already-exported `OPENAI_API_KEY` (from `~/.zshrc`, for example) wins over `.env`, and then no
`.env` is needed at all — but only if the client inherits your shell environment. A client
launched from the Dock or Finder (Claude Desktop) does not read `~/.zshrc`; give it the key in
the `env` block of the MCP config, or keep it in `.env`.

## LLM models

`ask_data` calls OpenAI GPT-5.6 through the Responses API, with one model per stage (defaults in
`src/churn_mcp/config/t2sql.py`):

| Stage | Model | Reasoning effort | Override |
|---|---|---|---|
| Router | `gpt-5.6-luna` | `none` | `CHURN_MCP__T2SQL__ROUTER_MODEL` |
| SQL generation and repair | `gpt-5.6-terra` | `medium` | `CHURN_MCP__T2SQL__GENERATOR_MODEL` |
| Answer synthesis | `gpt-5.6-luna` | `low` | `CHURN_MCP__T2SQL__SYNTHESIZER_MODEL` |

Each model can be set to `gpt-5.6-luna`, `gpt-5.6-terra` or `gpt-5.6-sol`; effort is set the same
way with `..._EFFORT` (`none`, `low`, `medium`, `high`).

## Setup

```bash
git clone https://github.com/ortrsa/cheq.git && cd cheq
uv sync                         # install dependencies
cp .env.example .env            # then set OPENAI_API_KEY
uv run churn-mcp prepare        # download data, build DuckDB, train model, write model card
uv run pytest -q                # offline, no API key needed
```

`prepare` must run once before the server is useful.

The eval suite calls the live API, so it needs `OPENAI_API_KEY`. It writes `evals/report.md` and
`evals/results.json`:

```bash
uv run python evals/run_evals.py
```

## Run

```bash
uv run churn-mcp serve
```

The server speaks MCP over stdio, so normally you don't start it by hand — the client below
launches it for you.

## Connect to Claude Code

CLI:

```bash
claude mcp add telco-churn -s user -e PYTHONPATH=/absolute/path/to/cheq/src \
  -- uv --directory /absolute/path/to/cheq run churn-mcp serve
```

Or by hand, in `.mcp.json` at your project root (or `~/.claude.json` for user scope). The same
JSON works in Claude Desktop's `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "telco-churn": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/cheq", "run", "churn-mcp", "serve"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/cheq/src"
      }
    }
  }
}
```

## Connect to Codex

```bash
codex mcp add telco-churn --env PYTHONPATH=/absolute/path/to/cheq/src \
  -- uv --directory /absolute/path/to/cheq run churn-mcp serve
```

Or in `~/.codex/config.toml`:

```toml
[mcp_servers.telco-churn]
command = "uv"
args = ["--directory", "/absolute/path/to/cheq", "run", "churn-mcp", "serve"]
env = { PYTHONPATH = "/absolute/path/to/cheq/src" }
```

## Verify

Ask the client: *"What is the overall churn rate?"* The answer should be **26.5%** across
7,043 customers.

## MCP surface

| Tool | Purpose |
|---|---|
| `ask_data(question, synthesize=True)` | Natural-language question → SQL, table, grounded answer |
| `run_sql(sql, max_rows?)` | Your own SELECT, through the same guard and sandbox |
| `describe_dataset(columns?)` | Columns, roles, exact values, metrics, traps |
| `segment_churn(group_by, filters?, top_n=20)` | Churn by segment with CI, lift, q-values; caveat when grouping or filtering by a leaky column |
| `at_risk_customers(top_n=20)` | Active customers ranked by expected revenue loss |
| `model_card()` | The model's AUC and features. Errors clearly before `prepare` |

| Resource | Content |
|---|---|
| `churn://schema` | Full schema and metric definitions |
| `churn://model-card` | Model AUC, features used. Returns `available: false` before `prepare` |

Every tool returns `{result, caveats, meta}`. Errors come back as `result: null` with
`meta.error`, never as a crash.

## Troubleshooting

- **"churn.duckdb missing"**: run `uv run churn-mcp prepare`.
- **`No module named 'churn_mcp'` / client shows CONNECTION_CLOSED on macOS**: Python 3.12 skips
  `.pth` files carrying the hidden flag, and some setups keep re-applying it. Register the server
  with `PYTHONPATH` pointing at `src`, as shown above. For a one-off fix in a shell:
  `chflags nohidden .venv/lib/python3.12/site-packages/*.pth`.
- **`ask_data` says LLM disabled**: `OPENAI_API_KEY` is not in `.env` or the environment.
- **Client can't connect**: the path in `--directory` must be absolute. Nothing may write to
  stdout except the MCP protocol; the test suite checks this.
