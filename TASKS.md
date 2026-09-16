# Telco Churn MCP — Task Board (compact scope, v3)

Interview-sized build of `SPEC_telco_churn_mcp.md`. Rule of thumb: **every file fits on
one screen-ish and can be explained in two minutes.** Target ≈ 1,200 lines of source.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done

## File tree

```
cheq/
├── README.md  CLAUDE.md  pyproject.toml  Makefile  .env.example
├── SPEC_telco_churn_mcp.md  TASKS.md
├── config/
│   └── semantic.yaml            # columns, roles, value domains, metrics, traps, ~15 verified examples
├── src/churn_mcp/
│   ├── config/                  # base.py enums.py data.py security.py t2sql.py analytics.py ml.py
│   │   └── telco_churn_mcp_config.py   # TelcoChurnMcpConfig root + @lru_cache get_config()
│   ├── data.py                  # HF download → clean → derived columns → DuckDB read-only `customers` view
│   ├── semantic.py              # load semantic.yaml; render the prompt context block; value-domain lookup
│   ├── sql_guard.py             # validate(sql) → Verdict(status, sql, hints, warnings) via sqlglot AST
│   ├── llm.py                   # OpenAI Responses wrapper (structured output, usage); MockLLM for tests
│   ├── t2sql.py                # ask_data: route → generate → guard → execute → repair → synthesize → ground
│   ├── analytics.py             # segment_churn: Wilson CI, lift, z-test, Benjamini–Hochberg, min-n
│   ├── model.py                 # at_risk_customers: leakage firewall, LR, out-of-fold scores, reason codes
│   ├── server.py                # FastMCP: 6 tools + churn://schema and churn://model-card resources
│   └── cli.py                   # churn-mcp prepare | serve | eval
├── evals/
│   ├── questions.yaml           # ~30 held-out questions with gold SQL, by category
│   ├── run_evals.py             # execution accuracy per category; ablation A–D; writes report.md
│   └── report.md
├── docs/
│   ├── design.html              # one-page design doc source
│   └── design.pdf
└── tests/
    ├── conftest.py              # in-memory DuckDB fixture from a 200-row golden CSV; MockLLM
    ├── test_config.py           # done
    ├── test_sql_guard.py  test_t2sql.py  test_analytics.py  test_model.py  test_data.py  test_server.py
```

## Tasks

### T0 — Project (done when `make test` is green)
- [x] **T0.1** `config/` package — one `BaseSettings` file per section (`base`, `enums`, `data`, `security`, `t2sql`, `analytics`, `ml`) + `telco_churn_mcp_config.py` root + `get_config()`; 9 tests.
- [x] **T0.2** `pyproject.toml` (uv, py3.12; deps: mcp>=2.2&lt;3, duckdb, sqlglot, pydantic-settings, openai, pandas, scikit-learn, scipy, huggingface-hub, typer, pyyaml; dev: pytest, hypothesis, ruff, mypy), `Makefile` (setup/prepare/serve/test/eval/lint), `.env.example`.

### T1 — Data + semantics (done when `run_sql` returns rows from the real 7,043-row view)
- [x] **T1.1** `data.py`: download pinned revision, concat splits, snake_case, null→`None`/`No Internet`, derived `churn`, `is_active`, `arr`, `tenure_bucket`, `num_addons`; save parquet to `.artifacts/`; open DuckDB read-only with `enable_external_access=false`.
- [x] **T1.2** `config/semantic.yaml`: every column with role + description + value domain; 4 metric snippets; 6 traps + ambiguous-term defaults; ~15 verified examples.
- [x] **T1.3** `semantic.py`: load, validate columns exist in the view, `context_block(cfg)` honouring the ablation toggles.
- [x] **T1.4** `test_data.py` + `test_semantic.py` (46 tests: row count, fills, derived columns, sandbox rejects writes/file reads, semantic↔view agreement both ways, all 15 examples execute).

### T2 — SQL guard (done when property test proves no DDL/DML ever passes)
- [x] **T2.1** `sql_guard.py`: single SELECT/WITH · tables ⊆ allow-list · columns exist (+ nearest-match hint) · no `read_*`/`pragma`/file functions · string literal ∈ value domain (+ valid-values hint) · enforce LIMIT · add `COUNT(*) AS n` to grouped rates · leaky-column warning.
- [x] **T2.2** `test_sql_guard.py`: 25 tests, one case per rule + hypothesis property test that no mutation ever passes.

### T3 — LLM + ask_data pipeline (done when cassette tests pass and a live call answers "overall churn rate")
- [x] **T3.1** `llm.py`: `complete(model, effort, system, user, schema) → (parsed, usage)`; refusal → structured error; `MockLLM(responses)`.
- [x] **T3.2** `t2sql.py` stage functions: `route`, `generate`, `execute`, `repair` (error-class-specific hint, ≤ `max_repairs`), `synthesize`, `ground` (every number in the answer must match a result cell; else table-only). `ask_data()` composes them and returns the envelope with `sql`, `table`, `answer`, `assumptions`, `caveats`, `attempts`, `grounding`, `usage`.
- [x] **T3.3** Degraded mode: no key → structured "LLM disabled" error, server still serves the other 4 tools.
- [x] **T3.5** Router intents live in one table, `INTENTS` in `t2sql.py`, which generates the schema enum and the router prompt; `ask()` dispatches with an explicit `match`, and a test sends every intent through it. `model_info` answers from a typed `ModelCard` whose summary, table and caveats come from the trained model and semantic layer, grounding-checked like any answer. `churn_score` column questions still go to SQL.
- [x] **T3.4** `test_t2sql.py` (20 tests): happy path, repair after bad column, repair after bad literal, grounding failure fallback, blocked injection, no-key mode.

### T4 — Analytics + model (done when tool output equals independent SQL on the fixture)
- [x] **T4.1** `analytics.py`: `segment_churn(group_by, filters, top_n)`; pure stats helpers `wilson_ci`, `two_prop_z`, `bh_qvalues`; suppress `n < min_segment_size`.
- [x] **T4.2** `model.py`: firewall rejects `outcome|leaky` roles (and `protected` by default); LR in a sklearn pipeline; 5-fold out-of-fold probabilities; top-3 signed coefficient reason codes; `at_risk_customers(top_n)` ranked by p × ARR; AUC + leaky-ablation AUC written to `model_card.json` at `prepare`.
- [x] **T4.3** `test_analytics.py` (20 tests) (Wilson/BH vs reference values, dual-path), `test_model.py` (firewall, AUC floor, OOF ≠ in-sample).

### T5 — MCP server (done when Claude Code connects and answers)
- [x] **T5.1** `server.py`: `ask_data`, `run_sql`, `describe_dataset`, `segment_churn`, `at_risk_customers`, `model_card` (same fallback as the resource); every tool and resource has a description; resources `churn://schema` and `churn://model-card` (`{available: false, message}` fallback before `prepare` has run); every result wrapped in `{result, caveats, meta}`.
- [x] **T5.2** `cli.py`: `prepare` (data + model artifacts), `serve` (stdio). `eval` deferred to T6, where the runner it wraps is actually built.
- [x] **T5.3** `test_server.py` (20 tests, incl. the model card tool and resource and their fallbacks): in-process MCP client lists 6 tools, calls each, nothing on stdout.

### T6 — Evals (done when `report.md` has the ablation table)
- [x] **T6.1** `evals/questions.yaml`: 30 items — simple 10, segmented 8, trap 6, adversarial 4, out_of_scope 2 (no analytic-tool routing exists to test, so "routing" became "out_of_scope") — disjoint from verified examples.
- [x] **T6.2** `evals/run_evals.py`: execution accuracy (numeric-value subset match, so extra columns/rows and label-vs-boolean phrasing don't cause false failures), safety pass rate, cost/latency; configs A→D built via `config.model_copy()`, not env vars.
- [x] **T6.3** Ran live 3 times; the first two exposed scorer bugs and a Joined-customer inconsistency, both fixed. Final: A 93%, B/C/D 100%, traps 67%→100% with the semantic layer. Table in README.

### T7 — README (done when quickstart works from a clean clone)
- [x] **T7.1** Pitch · 8 example questions · architecture sketch · quickstart (Claude Code + Codex) · model choices & why · validator rules · ablation table · limitations & roadmap (what was cut from the full spec, and why).

### T8 — One-page PDF (required deliverable, carries most of the scoring weight)
- [x] **T8.1** `docs/design.html` → `docs/design.pdf` via headless Chrome, exactly one page, five headings the brief names: Design, Why (incl. what was deliberately avoided), Production, Risks, Business Impact at CHEQ.
- [~] **T8.2** `git init` done; commit and push to a **public** GitHub repo pending your go-ahead.

## Explicitly cut from the full spec (mention in README roadmap)
Hexagonal layers · OpenTelemetry · multi-layer cache · CFG grammar strategies · self-consistency · `churn_drivers` · retention economics / campaign plan · LLM-as-judge · ADRs.
