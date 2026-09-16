# Telco Churn MCP

An MCP server that answers natural-language questions about telco customer churn with
numbers you can check. Every answer comes with the SQL that produced it, the result
table, the assumptions made, and caveats.

Dataset: [`aai510-group1/telco-customer-churn`](https://huggingface.co/datasets/aai510-group1/telco-customer-churn),
7,043 customers, pinned to a commit so results are reproducible.

## What you can ask

- "What is the overall churn rate?"
- "Churn rate for fiber customers on month-to-month contracts"
- "How much monthly revenue did we lose to churn?"
- "Which contract type has the worst churn?"
- "Does satisfaction score explain churn?" *(it explains why it can't)*
- "Which segments churn significantly more than average?" → `segment_churn`
- "Which active customers are most likely to leave, and why?" → `at_risk_customers`
- "Delete all churned customers" *(refused before any SQL is written)*

## How it works

Tabular data needs counting and aggregation, which retrieval over rows can't do, so the
core is **governed text-to-SQL** rather than RAG:

```
question → route → generate SQL → guard (AST) → execute (sandbox) → repair → synthesize → ground
```

- **Route**: a cheap model rejects unsafe or off-topic questions before any SQL exists, and sends
  questions about the churn model itself (score, AUC, features) to the model card instead of SQL.
- **Generate**: the prompt carries a semantic layer (`config/semantic.yaml`): every column
  with its role and exact values, metric definitions, known data traps, 15 verified examples.
- **Guard** (`sql_guard.py`): parses the SQL with sqlglot and blocks anything that is not a
  single SELECT, unknown tables/columns, file-reading functions, and literals outside a
  column's value domain. Adds `LIMIT` and a row count to grouped rates.
- **Execute**: DuckDB opened read-only, external access disabled, configuration locked.
- **Repair**: a blocked or failed query goes back to the model with a specific hint
  ("Did you mean `monthly_charge`?", "Valid values: Cable, DSL, Fiber Optic..."), up to 2 times.
- **Ground**: every number in the narrative must match a result cell, or the narrative is
  withheld and only the table is returned.

Two questions need more than SQL, so they get deterministic tools:

- **`segment_churn`**: churn by 1-3 dimensions with Wilson confidence intervals, lift, and
  Benjamini-Hochberg q-values, so "significantly worse" means something. Small segments are flagged.
- **`at_risk_customers`**: logistic regression scored out-of-fold, ranked by probability x ARR,
  with the top three reasons per customer. A leakage firewall blocks outcome columns, exit-survey
  fields and protected attributes from training.

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo> && cd cheq
make setup                      # uv sync
cp .env.example .env            # then set OPENAI_API_KEY
make prepare                    # download data, build DuckDB, train model, write model card
make test                       # offline, no API key needed
```

### Connect to Claude Code

```bash
claude mcp add telco-churn -s user -e PYTHONPATH=/absolute/path/to/cheq/src \
  -- uv --directory /absolute/path/to/cheq run churn-mcp serve
```

### Connect to Codex

In `~/.codex/config.toml`:

```toml
[mcp_servers.telco-churn]
command = "uv"
args = ["--directory", "/absolute/path/to/cheq", "run", "churn-mcp", "serve"]
env = { PYTHONPATH = "/absolute/path/to/cheq/src" }
```

Or `codex mcp add telco-churn --env PYTHONPATH=/absolute/path/to/cheq/src -- uv --directory /absolute/path/to/cheq run churn-mcp serve`.

`PYTHONPATH` isn't strictly required, but it keeps the server importable on macOS setups that hide
the virtualenv's `.pth` files (see Troubleshooting).

The key is read from `.env` at the repo root, so neither client needs it passed in.
Verify with: *"What is the overall churn rate?"* The answer should be **26.5%** across 7,043 customers.

## MCP surface

| Tool | Purpose |
|---|---|
| `ask_data(question, synthesize=True)` | Natural-language question → SQL, table, grounded answer |
| `run_sql(sql, max_rows?)` | Your own SELECT, through the same guard and sandbox |
| `describe_dataset(columns?)` | Columns, roles, exact values, metrics, traps |
| `segment_churn(group_by, filters?, top_n=20)` | Churn by segment with CI, lift, q-values |
| `at_risk_customers(top_n=20)` | Active customers ranked by expected revenue loss |
| `model_card()` | The model's AUC, leaky-feature AUC and features. Errors clearly before `prepare` |

| Resource | Content |
|---|---|
| `churn://schema` | Full schema and metric definitions |
| `churn://model-card` | Model AUC, leaky-feature AUC, features used. Returns `available: false` before `prepare` |

Every tool returns `{result, caveats, meta}`. Errors come back as `result: null` with
`meta.error`, never as a crash.

## Models and configuration

| Stage | Model | Reasoning effort | Why |
|---|---|---|---|
| Router | `gpt-5.6-luna` | none | Classification only; cheapest and fastest |
| SQL generation and repair | `gpt-5.6-terra` | medium | Needs structured reasoning at a reasonable cost |
| Answer synthesis | `gpt-5.6-luna` | low | Short summaries; grounding catches wrong numbers |

All calls use the OpenAI Responses API with strict JSON-schema output. The key comes from
`OPENAI_API_KEY`. Without a key the server still starts: `ask_data` returns a clear
"LLM disabled" error and the other four tools work normally.

Settings live in `src/churn_mcp/config/` as typed, frozen classes. Override any of them
with environment variables, `CHURN_MCP__<SECTION>__<KEY>`:

```bash
CHURN_MCP__T2SQL__GENERATOR_MODEL=gpt-5.6-sol
CHURN_MCP__SECURITY__MAX_ROWS=500
```

The LLM provider sits behind an abstract class (`llm/base.py`), so swapping to another
provider is one new subclass.

## Evaluation

30 held-out questions (disjoint from the verified examples): 10 simple, 8 segmented,
6 data traps, 4 adversarial, 2 off-topic. Run with `make eval` (needs a key).

Each config adds one component, so the table shows what each part contributes:

| Config | Adds | Accuracy | Traps | Adversarial blocked | Avg latency | Cache hit |
|---|---|---|---|---|---|---|
| A | Column names only | 93% | 67% | 100% | 6.2s | 0% |
| B | + value domains, traps | 100% | 100% | 100% | 7.0s | 63% |
| C | + verified examples | 100% | 100% | 100% | 5.8s | 79% |
| D | + repair loop (default) | 100% | 100% | 100% | 5.7s | 78% |

Simple, segmented, adversarial and off-topic questions pass under every config. The semantic
layer earns its place on traps: without it the model reports Offer E's churn as if the offer
caused it, and misreads "customers who joined this quarter" because it doesn't know `Joined`
is a status. Examples and repair add no accuracy on this set, which is already at its ceiling,
but examples make answers faster.

Read this as a smoke test, not a benchmark: 30 questions, one run, written by me. An earlier
run scored config D at 90%. That run exposed an inconsistency, now fixed (the same question
could include or exclude Joined customers), and one answer withheld by the grounding check.
Full results: `evals/report.md` and `evals/results.json`.

## Model

Logistic regression, 5-fold out-of-fold scoring. The leakage ablation shows why the firewall matters:

| Features | AUC |
|---|---|
| Allowed features only | 0.894 |
| Plus the leaky columns declared in `semantic.yaml` (`satisfaction_score`, `churn_score`) | 0.998 |

## Project layout

```
config/semantic.yaml        semantic layer: columns, values, metrics, traps, examples
src/churn_mcp/
  config/                   typed settings, one file per section
  models/                   dataclasses
  exceptions/               error types
  llm/                      LLM abstraction, OpenAI client, mock
  data.py                   download, clean, build sealed DuckDB
  semantic.py               load the semantic layer
  sql_guard.py              AST validation and rewrites
  t2sql.py                  the ask_data pipeline
  analytics.py              segment_churn and its statistics
  model.py                  churn model, firewall, at_risk_customers
  server.py                 MCP tools and resources
  cli.py                    prepare, serve
evals/                      questions, runner, report
tests/                      offline tests, LLM mocked
```

## Limitations and roadmap

- Churn is one quarter of one state, so there is no trend or geographic comparison.
- `at_risk_customers` is correlational. Validate with a holdout before acting on it.
- Cut from the full design (`SPEC_telco_churn_mcp.md`): driver analysis, retention campaign
  planning with budget optimisation, grammar-constrained SQL, OpenTelemetry, result caching,
  remote HTTP transport with auth.

## Troubleshooting

- **"churn.duckdb missing"**: run `make prepare`.
- **`No module named 'churn_mcp'` / client shows CONNECTION_CLOSED on macOS**: Python 3.12 skips
  `.pth` files carrying the hidden flag, and some setups keep re-applying it. Register the server with
  `PYTHONPATH` pointing at `src`, as shown above. For a one-off fix in a shell:
  `chflags nohidden .venv/lib/python3.12/site-packages/*.pth`.
- **`ask_data` says LLM disabled**: `OPENAI_API_KEY` is not in `.env` or the environment.
- **Client can't connect**: the path in `--directory` must be absolute. Nothing may write to
  stdout except the MCP protocol; the test suite checks this.
