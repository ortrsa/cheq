# Telco Churn MCP — Task Board (compact scope, v3)

Interview-sized build of `SPEC_telco_churn_mcp.md`. Rule of thumb: **every file fits on
one screen-ish and can be explained in two minutes.** Source is ≈ 1,700 lines.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done

## File tree

```
cheq/
├── README.md  CLAUDE.md  pyproject.toml  .env.example
├── SPEC_telco_churn_mcp.md  TASKS.md
├── config/
│   └── semantic.yaml            # columns, roles, value domains, 5 metrics, 8 traps, defaults, 15 verified examples
├── src/churn_mcp/
│   ├── config/                  # base.py enums.py data.py security.py t2sql.py analytics.py ml.py
│   │   └── telco_churn_mcp_config.py   # TelcoChurnMcpConfig root + @lru_cache get_config()
│   ├── models/                  # dataclasses per module + Pydantic LLM output models (Route, Draft, Narrative, Intent)
│   ├── exceptions/              # UnknownColumn, UnknownValue, LeakyFeature, LLMRefused, ModelCardMissing
│   ├── llm/                     # base.py (LLM ABC) · openai_client.py (Responses, strict JSON schema) · mock.py
│   ├── data.py                  # HF download → clean → derived columns → .artifacts/churn.duckdb; read-only locked connect
│   ├── semantic.py              # load semantic.yaml; render the prompt context block; value-domain lookup
│   ├── sql_guard.py             # validate(sql) → Verdict(sql, error, warnings, rewrites) via sqlglot AST
│   ├── prompts.py               # router / generator / synthesizer prompts; INTENTS descriptions keyed by Intent
│   ├── t2sql.py                 # ask_data: route → generate → guard → execute → repair → synthesize → ground
│   ├── analytics.py             # segment_churn: Wilson CI, lift, z-test, Benjamini–Hochberg, min-n
│   ├── model.py                 # at_risk_customers: leakage firewall, LR, out-of-fold scores, reason codes
│   ├── server.py                # MCPServer: 6 tools + churn://schema and churn://model-card resources
│   └── cli.py                   # churn-mcp prepare | serve
├── evals/
│   ├── questions.yaml           # 30 held-out questions with gold SQL, by category
│   ├── run_evals.py             # execution accuracy per category; ablation A–D; writes report.md + results.json
│   ├── report.md  results.json
├── docs/
│   ├── design.html              # one-page design doc source
│   └── design.pdf
├── pdf_to_print/                # Hebrew printed study guides (guide, code_book, tools); render.sh rebuilds the PDFs
└── tests/
    ├── conftest.py              # session DuckDB built from the real dataset (builds it if missing); MockLLM
    ├── test_config.py  test_data.py  test_semantic.py  test_sql_guard.py  test_t2sql.py
    ├── test_analytics.py  test_model.py  test_server.py  test_eval_scoring.py
```

## Tasks

### T0 — Project (done when `uv run pytest` is green)
- [x] **T0.1** `config/` package — one `BaseSettings` file per section (`base`, `enums`, `data`, `security`, `t2sql`, `analytics`, `ml`) + `telco_churn_mcp_config.py` root + `get_config()`; 11 tests.
- [x] **T0.2** `pyproject.toml` (uv, py3.12; deps: mcp>=2.2&lt;3, duckdb, sqlglot, pydantic-settings, openai, pandas, scikit-learn, scipy, huggingface-hub, typer, pyyaml; dev: pytest, hypothesis, ruff, mypy), `.env.example`.

### T1 — Data + semantics (done when `run_sql` returns rows from the real 7,043-row table)
- [x] **T1.1** `data.py`: download pinned revision, concat splits, snake_case, null→`None`/`No Internet`, derived `churn`, `is_active`, `arr`, `tenure_bucket`, `referral_bucket`, `num_addons`; write `.artifacts/churn.duckdb`; `connect()` opens read-only with `enable_external_access=false` then `lock_configuration=true`, and is safe to call twice in one process (skips the SETs once locked). Tools take a cursor per call, so parallel MCP calls never share a DuckDB connection.
- [x] **T1.2** `config/semantic.yaml`: every column with role + description + value domain; 5 metric snippets; 8 traps + ambiguous-term defaults; 15 verified examples.
- [x] **T1.3** `semantic.py`: load, validate columns exist in the table, `context_block(cfg)` honouring the ablation toggles.
- [x] **T1.4** `test_data.py` + `test_semantic.py` (39 tests: row count, fills, derived columns, sandbox rejects writes/file reads on the connection and on cursors, double connect, semantic↔table agreement both ways, all 15 examples execute).

### T2 — SQL guard (done when property test proves no DDL/DML ever passes)
- [x] **T2.1** `sql_guard.py`: single SELECT/WITH · tables ⊆ allow-list · columns exist (+ nearest-match hint; `c.*` allowed) · no `read_*`/`pragma`/file functions, including sqlglot-typed ones like `read_csv`, and no `current_setting`/`duckdb_*` introspection · string literal ∈ value domain for `=`/`IN` (+ valid-values hint) · `LIKE`/`ILIKE` pattern must match some domain value · `LIMIT` must be a positive integer, then enforced ≤ `max_rows` · add `COUNT(*) AS n` to grouped rates · leaky-column warning, also for `SELECT *`.
- [x] **T2.2** `test_sql_guard.py`: 41 tests, one case per rule + hypothesis property test that no mutation ever passes.

### T3 — LLM + ask_data pipeline (done when cassette tests pass and a live call answers "overall churn rate")
- [x] **T3.1** `llm/` package: `LLM.complete(model, effort, developer, user, schema, schema_name, max_output_tokens) → (parsed, usage)`; OpenAI Responses client with strict JSON schema, refusal → `LLMRefused`; `MockLLM(replies)`.
- [x] **T3.2** `t2sql.py` stage functions: `route`, `generate`, `execute`, repair loop (guard error or DuckDB error fed back, ≤ `max_repairs`), `synthesize`, `ungrounded` (every number in the answer must match a result cell; one retry, else narrative withheld). Output schemas are Pydantic models `Route`, `Draft`, `Narrative` in `models/t2sql.py` (`extra="forbid"`, all fields required, so `model_json_schema()` is strict-mode ready); prompts live in `prompts.py`. `ask()` composes the stages and returns `Result` with `sql`, `table`, `answer`, `assumptions`, `caveats`, `attempts`, `grounding`, `usage`.
- [x] **T3.3** Degraded mode: no key → structured "LLM disabled" error, server still serves the other 5 tools.
- [x] **T3.5** Router intents are the `Intent` enum in `models/t2sql.py`; `prompts.INTENTS` maps each to its description and generates the router prompt, and `Route.model_json_schema()` carries the enum. `ask()` dispatches with an explicit `match`, and tests check every intent has a handler and appears in both the prompt and the schema. `model_info` answers from a typed `ModelCard` whose summary, table and caveats come from the trained model and semantic layer, grounding-checked like any answer. `churn_score` column questions still go to SQL.
- [x] **T3.4** `test_t2sql.py` (32 tests): happy path, repair after bad column, repair after bad literal, grounding failure fallback, blocked injection, no-key mode, refusal, every intent.

### T4 — Analytics + model (done when tool output equals independent SQL on the fixture)
- [x] **T4.1** `analytics.py`: `segment_churn(group_by, filters, top_n)` with 1–3 `group_by` columns and filters bound as query parameters (no literal splicing); pure stats helpers `wilson_ci`, `two_prop_z`, `bh_qvalues`; flag `n < min_segment_size` as suppressed.
- [x] **T4.2** `model.py`: firewall rejects `outcome|leaky` roles (and `protected` by default); redundant columns excluded (`ALWAYS_EXCLUDED`; `referral_bucket` replaces `number_of_referrals`/`referred_a_friend`); LR in a sklearn pipeline; 5-fold out-of-fold probabilities; up to 3 risk-raising reason codes (centred contributions summed per source column); `at_risk_customers(top_n)` ranked by p × ARR, model trained once per server process on first call; AUC + features written to `model_card.json` at `prepare`.
- [x] **T4.3** `test_analytics.py` (24 tests: Wilson/BH vs statsmodels, dual-path SQL, filter quoting and injection, dimension bounds), `test_model.py` (40 tests: firewall, AUC floor, OOF ≠ in-sample, model card).

### T5 — MCP server (done when Claude Code connects and answers)
- [x] **T5.1** `server.py`: `ask_data`, `run_sql`, `describe_dataset`, `segment_churn`, `at_risk_customers`, `model_card` (same fallback as the resource); every tool and resource has a description; resources `churn://schema` and `churn://model-card` (`{available: false, message}` fallback before `prepare` has run); every result wrapped in `{result, caveats, meta}`. Bad arguments and DuckDB runtime errors in `run_sql`/`segment_churn` come back as `result: null` + `meta.error`, never as a tool crash. `segment_churn` adds the leaky-column caveat when grouping or filtering by a leaky column.
- [x] **T5.2** `cli.py`: `prepare` (data + model artifacts), `serve` (stdio). Evals run via `uv run python evals/run_evals.py`, not a CLI command.
- [x] **T5.3** `test_server.py` (27 tests, incl. the model card tool and resource and their fallbacks, runtime-error envelopes, argument validation, 15 concurrent calls): in-process MCP client lists 6 tools, calls each, nothing on stdout.

### T6 — Evals (done when `report.md` has the ablation table)
- [x] **T6.1** `evals/questions.yaml`: 30 items — simple 10, segmented 8, trap 6, adversarial 4, out_of_scope 2 (no analytic-tool routing exists to test, so "routing" became "out_of_scope") — disjoint from verified examples.
- [x] **T6.2** `evals/run_evals.py`: execution accuracy (numeric-value subset match, so extra columns/rows and label-vs-boolean phrasing don't cause false failures), safety pass rate, cost/latency; configs A→D built via `config.model_copy()`, not env vars; scorer covered by `test_eval_scoring.py` (7 tests).
- [x] **T6.3** Ran live 3 times; the first two exposed scorer bugs and a Joined-customer inconsistency, both fixed. Final: A 93%, B/C/D 100%, traps 67%→100% with the semantic layer. Table in `evals/report.md`.

### T7 — README (done when quickstart works from a clean clone)
- [x] **T7.1** Pitch · requirements · env vars · LLM models per stage · setup + evals · quickstart (Claude Code + Codex) · verify · MCP surface · troubleshooting. Design detail lives in `docs/design.pdf` and `SPEC_telco_churn_mcp.md`, not the README.

### T8 — One-page PDF (required deliverable, carries most of the scoring weight)
- [x] **T8.1** `docs/design.html` → `docs/design.pdf` via headless Chrome, exactly one page, five headings the brief names: Design, Why (incl. what was deliberately avoided), Production, Risks, Business Impact at CHEQ.
- [~] **T8.2** Committed on `main` with `origin` set to the public GitHub repo; push not yet confirmed from this machine.

## Explicitly cut from the full spec (mention in README roadmap)
Hexagonal layers · OpenTelemetry · multi-layer cache · CFG grammar strategies · self-consistency · `churn_drivers` · retention economics / campaign plan · LLM-as-judge · ADRs.
