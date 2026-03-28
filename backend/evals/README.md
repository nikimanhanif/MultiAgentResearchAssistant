# Evaluation Harness

Benchmarks the production research workflow against [DeepResearchBench](https://github.com/Ayanami0730/deep_research_bench) using **DeepSeek as the judge backend**.

> **Note:** The official DRB baseline uses Gemini as the judge. Using DeepSeek means results are not directly comparable to official numbers. The task set, evaluation pipeline (RACE + FACT), and scoring methodology are identical.

## Quick Reference

| Command | Purpose |
|---------|---------|
| `inspect` | Preview loaded DRB queries |
| `generate` | Run the pipeline, export DRB-format output |
| `evaluate` | Invoke RACE + FACT evaluation (DeepSeek judge) |
| `run` | Internal harness with preliminary scorers |
| `--case-id <ID>` | Target a specific case (available for all commands) |
| `--range <START-END>` | Run a specific range of cases by ID |
| `--shuffle` | Randomly sample cases |

## Prerequisites

```bash
# 1. Clone the official DRB repo
git clone https://github.com/Ayanami0730/deep_research_bench.git
cd deep_research_bench && pip install -r requirements.txt

# 2. Set required env vars
export DRB_REPO_PATH="/path/to/deep_research_bench"
export DEEPSEEK_API_KEY="your_deepseek_api_key"   # LLM judge
export JINA_API_KEY="your_jina_api_key"             # Web scraping for FACT

# 3. Optional: enable LangSmith experiment tracking
export LANGSMITH_API_KEY="your_langsmith_api_key"
```

## Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `DEEPSEEK_API_KEY` | Yes | — | DeepSeek API key (judge LLM) |
| `JINA_API_KEY` | Yes | — | Jina web scraping (FACT pipeline) |
| `DRB_REPO_PATH` | Yes | — | Path to cloned DRB repo |
| `LANGSMITH_API_KEY` | For LangSmith | — | Enables experiment tracking |
| `DRB_JUDGE_PROVIDER` | No | `deepseek` | Judge backend: `deepseek` or `gemini` |
| `DRB_RACE_MODEL` | No | `deepseek-chat` | Model for RACE evaluation |
| `DRB_FACT_MODEL` | No | `deepseek-chat` | Model for FACT extraction/validation |
| `DEEPSEEK_BASE_URL` | No | `https://api.deepseek.com` | DeepSeek API endpoint |
| `GEMINI_API_KEY` | If using gemini | — | Only needed if `DRB_JUDGE_PROVIDER=gemini` |

---

## Standard Workflow (No LangSmith)

### Step 1 — Inspect the query set

Check what you're about to run before committing API credits.

```bash
cd backend
PYTHONPATH=. uv run -m evals.cli inspect \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl
```

### Step 2 — Dry-run (validates without running the graph)

```bash
PYTHONPATH=. uv run -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher \
  --dry-run
```

### Step 3 — Generate DRB-format outputs

```bash
# Smoke run — 3 cases to sanity-check the output format
PYTHONPATH=. uv run -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher \
  --limit 3

# Full run — all 100 cases
PYTHONPATH=. uv run -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher
```

Output: `eval_results/multi_agent_researcher.jsonl` — each line is `{id, prompt, article}`.

### Step 4 — Copy output to the DRB repo

```bash
cp eval_results/multi_agent_researcher.jsonl \
   $DRB_REPO_PATH/data/test_data/raw_data/
```

### Step 5 — Run evaluation

```bash
# Both RACE and FACT (default)
PYTHONPATH=. uv run -m evals.cli evaluate \
  --model-name multi_agent_researcher

# Individual phases
PYTHONPATH=. uv run -m evals.cli evaluate \
  --model-name multi_agent_researcher --phase race

PYTHONPATH=. uv run -m evals.cli evaluate \
  --model-name multi_agent_researcher --phase fact
```

Results land in:
- `$DRB_REPO_PATH/results/race/multi_agent_researcher/race_result.txt`
- `$DRB_REPO_PATH/results/fact/multi_agent_researcher/fact_result.txt`

---

## LangSmith Experiment Tracking

When `LANGSMITH_API_KEY` is set, you can add `--langsmith-experiment` to `generate`. This:

1. Uploads the benchmark cases as a LangSmith dataset (idempotent — safe to run repeatedly)
2. Runs each case through `aevaluate()`, creating a LangSmith experiment
3. Links all agent and tool traces as children of each experiment run
4. Logs internal scores (success, latency, citations, etc.) as LangSmith feedback

After running `evaluate`, you can push the official RACE/FACT scores back to the experiment so everything is in one place.

### LangSmith flags

| Flag | Command | Purpose |
|------|---------|---------|
| `--langsmith-experiment` | `generate` | Enable experiment tracking |
| `--langsmith-dataset <name>` | `generate`, `evaluate` | Dataset name (default: `drb-{model-name}`) |
| `--experiment-prefix <prefix>` | `generate` | Experiment name prefix (default: `{model-name}`) |
| `--langsmith-max-concurrency <n>` | `generate` | Concurrent runs in aevaluate (default: 0 = sequential) |
| `--langsmith-experiment <name>` | `evaluate` | Experiment name to upload DRB scores to |

### Full LangSmith workflow

**Step 1 — Generate with experiment tracking**

```bash
PYTHONPATH=. uv run -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher \
  --langsmith-experiment \
  --langsmith-dataset drb-multi-agent \
  --experiment-prefix "multi-agent-v1"
```

The command logs the full experiment name at the end, e.g.:
```
LangSmith experiment: https://smith.langchain.com (experiment: multi-agent-v1-20260328T102300)
```

Copy that name — you'll need it in step 4.

**Step 2 — Copy output to DRB repo** (same as standard workflow)

```bash
cp eval_results/multi_agent_researcher.jsonl \
   $DRB_REPO_PATH/data/test_data/raw_data/
```

**Step 3 — Run DRB evaluation**

```bash
PYTHONPATH=. uv run -m evals.cli evaluate \
  --model-name multi_agent_researcher
```

**Step 4 — Push RACE/FACT scores back to LangSmith**

```bash
PYTHONPATH=. uv run -m evals.cli evaluate \
  --model-name multi_agent_researcher \
  --langsmith-experiment "multi-agent-v1-20260328T102300" \
  --langsmith-dataset drb-multi-agent
```

This attaches `drb_race_score` and `drb_fact_score` as feedback on each individual run in the experiment.

### Incremental batch runs with LangSmith

`--range` and `--case-id` work as normal. Each `generate` call creates a separate experiment, but they all write to the same dataset. Use a consistent `--experiment-prefix` so the batches group together visually in LangSmith.

```bash
# Batch 1
PYTHONPATH=. uv run -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher \
  --range 1-10 \
  --langsmith-experiment \
  --langsmith-dataset drb-multi-agent \
  --experiment-prefix "multi-agent-v1"

# Batch 2 — appends to the same output file and dataset
PYTHONPATH=. uv run -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher \
  --range 11-20 \
  --langsmith-experiment \
  --langsmith-dataset drb-multi-agent \
  --experiment-prefix "multi-agent-v1"
```

`--case-id` is useful for debugging a single failing case — it creates a one-run experiment with the full trace tree visible in LangSmith.

```bash
PYTHONPATH=. uv run -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher \
  --case-id 42 \
  --langsmith-experiment \
  --experiment-prefix "debug-case-42"
```

---

## Targeted and Incremental Runs (Standard)

### Target a single case

```bash
# Generate only case 50
PYTHONPATH=. uv run -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher \
  --case-id 50

cp eval_results/multi_agent_researcher.jsonl $DRB_REPO_PATH/data/test_data/raw_data/

# Evaluate only case 50
PYTHONPATH=. uv run -m evals.cli evaluate \
  --model-name multi_agent_researcher --case-id 50
```

> When `--case-id` is used with `evaluate`, a temporary filtered query file is created automatically so the DRB scripts only process that case.

### Evaluating a partial dataset

`evaluate` has no `--range` flag. It always evaluates **whatever cases are currently in the output JSONL file** — it passes that file directly to the DRB RACE and FACT scripts unchanged.

This means:

- If you only generated cases 51–100, `evaluate` scores those 50 cases. The RACE/FACT result files will reflect a 50-case subset, **not** a full-100 benchmark run.
- Scores from a partial dataset are **not comparable** to official DRB baselines (which require all 100 cases).
- You can still run `evaluate` mid-batch to get a directional signal, but treat those numbers as intermediate diagnostics only.

To evaluate after a partial run:

```bash
# 1. Copy the partial output to the DRB repo (same as always)
cp eval_results/multi_agent_researcher.jsonl \
   $DRB_REPO_PATH/data/test_data/raw_data/

# 2. Run evaluate normally — no range flag needed
PYTHONPATH=. uv run -m evals.cli evaluate \
  --model-name multi_agent_researcher
```

The DRB scripts will process only the IDs present in the JSONL. If you later add more cases (via another `generate --range` batch), re-copy the file and re-run `evaluate`. Use `--force` to overwrite existing intermediate result files:

```bash
PYTHONPATH=. uv run -m evals.cli evaluate \
  --model-name multi_agent_researcher --force
```

For a meaningful final score, wait until `status` confirms all 100 cases are complete before treating the evaluate output as a real benchmark result.

### Incremental batch testing

Run cases in batches and verify progressively. The harness automatically runs in append mode when `--range` is used — existing results for the same IDs are overwritten, new ones are merged in, and the file is kept sorted by ID.

```bash
# Run cases 1-10
PYTHONPATH=. uv run -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher \
  --range 1-10

# Run cases 11-20 (results merged into the same file)
PYTHONPATH=. uv run -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher \
  --range 11-20
```

### Check completion status

```bash
PYTHONPATH=. uv run -m evals.cli status \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher
```

This shows how many of the 100 cases have been completed and which IDs are still missing.

### Budget overrides (shorter runs)

```bash
PYTHONPATH=. uv run -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher \
  --max-iterations 5 --max-sub-agents 3
```

---

## Judge Backend

The evaluator deploys a provider-aware `utils/api.py` into the DRB repo before running:

```
evals/drb_patches/api.py  →  deep_research_bench/utils/api.py
                               (original backed up as api.py.gemini_original)
```

The patched file keeps the same interface (`AIClient`, `call_model`, `scrape_url`) while routing calls through DeepSeek or Gemini depending on `DRB_JUDGE_PROVIDER`.

| Purpose | DeepSeek default | Gemini default | Env var |
|---------|-----------------|----------------|---------|
| RACE (quality judge) | `deepseek-chat` | `gemini-2.5-pro` | `DRB_RACE_MODEL` |
| FACT (extraction/validation) | `deepseek-chat` | `gemini-2.5-flash` | `DRB_FACT_MODEL` |

To switch to Gemini:

```bash
export DRB_JUDGE_PROVIDER=gemini
export GEMINI_API_KEY="your_gemini_key"
PYTHONPATH=. uv run -m evals.cli evaluate \
  --model-name multi_agent_researcher
```

---

## Scoring

| Type | Framework | Source |
|------|-----------|--------|
| **DRB RACE** | Reference-based quality evaluation | DRB repo (DeepSeek judge) |
| **DRB FACT** | Citation accuracy and claim support | DRB repo (Jina scraping + DeepSeek) |
| **Internal** | Success, latency, findings, citations, completeness | `evals/scorers.py` |

Internal scorers are diagnostics only — use `evaluate` for results that mean anything.

---

## Architecture

```
evals/
├── cli.py               # Commands: inspect, generate, evaluate, run, status
├── runner.py             # EvalRunner: wraps the production graph
├── langsmith_eval.py     # LangSmith experiment tracking integration
├── drb_evaluator.py      # RACE + FACT subprocess wrappers + judge setup
├── drb_patches/
│   └── api.py            # Provider-aware drop-in for DRB utils/api.py
├── policies.py           # Deterministic interrupt handling
├── scorers.py            # Internal diagnostic scorers
├── reporters.py          # JSONL, CSV, Markdown, DRB output format
├── models.py             # BenchmarkCase, EvalResult, RunMetadata
├── adapters/
│   ├── base.py           # Abstract BenchmarkAdapter
│   ├── jsonl_adapter.py  # Generic JSONL loader
│   └── deep_research_bench.py  # Official DRB schema adapter
└── fixtures/
    ├── sample_cases.jsonl       # Internal test cases
    └── sample_drb_queries.jsonl # DRB-format test queries
```

## Running Tests

```bash
cd backend
PYTHONPATH=. uv run pytest tests/unit/test_evals/ -v
```
