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

## Incremental Generation

Running all 100 cases in a single command is impractical — each case takes minutes. The recommended approach is to generate in batches using `--range` or target individual cases with `--case-id`, then evaluate once the full dataset is complete.

### How batching works

`--range` and `--case-id` both run in **append mode**: new results are merged into the existing output JSONL and the file is kept sorted by ID. Existing IDs are skipped automatically (use `--force` to overwrite). This means you can stop and resume at any point without losing prior work.

### Check completion status

After each batch, check how many of the 100 cases are done and which IDs are still missing:

```bash
PYTHONPATH=. uv run -m evals.cli status \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher
```

### Generate in batches

```bash
# Batch 1 — cases 1 to 25
PYTHONPATH=. uv run -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher \
  --range 1-25

# Batch 2 — appends to the same output file
PYTHONPATH=. uv run -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher \
  --range 26-50

# ... continue until status shows 100/100
```

### Re-run a single case

Use `--case-id` to re-run a specific case (e.g., one that timed out or returned a partial result). Combined with `--force` to overwrite the existing entry:

```bash
PYTHONPATH=. uv run -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher \
  --case-id 42 --force
```

### Budget overrides (faster smoke runs)

Reduce iterations to get a quick directional signal without spending full API credits:

```bash
PYTHONPATH=. uv run -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher \
  --range 1-5 \
  --max-iterations 5 --max-sub-agents 3
```

---

## Intermediate Evaluation (Local Only)

You can run `evaluate` at any point during batching to get a directional signal on the cases generated so far. `evaluate` always scores **whatever is currently in the output JSONL** — it passes that file directly to the DRB RACE and FACT scripts unchanged.

```bash
# Copy current output to DRB repo
cp eval_results/multi_agent_researcher.jsonl \
   $DRB_REPO_PATH/data/test_data/raw_data/

# Score what you have so far (no LangSmith upload)
PYTHONPATH=. uv run -m evals.cli evaluate \
  --model-name multi_agent_researcher
```

After adding more batches, re-copy and re-run with `--force` to overwrite cached intermediate results:

```bash
cp eval_results/multi_agent_researcher.jsonl \
   $DRB_REPO_PATH/data/test_data/raw_data/

PYTHONPATH=. uv run -m evals.cli evaluate \
  --model-name multi_agent_researcher --force
```

> Intermediate scores are **not comparable** to official DRB baselines. A partial dataset will produce artificially high or low RACE/FACT scores depending on which cases happen to be included. Treat them as diagnostics only.

> When `--case-id` is used with `evaluate`, a temporary filtered query file is created automatically so the DRB scripts only process that case.

---

## LangSmith Experiment Tracking

LangSmith serves two separate purposes here — **tracing** and **score visibility** — and it helps to keep them distinct.

- **Tracing**: every agent node, LLM call, and tool invocation recorded as a linked trace tree. This happens automatically during `generate` when `--langsmith-experiment` is set. No extra steps needed.
- **Score visibility**: the RACE/FACT scores are aggregate numbers produced by the local `evaluate` command. They live in local result files. The `--langsmith-experiment` flag on `evaluate` is an optional step to attach those scores to LangSmith — but it is not required.

For most purposes: use `--langsmith-experiment` on `generate` to get traces, run `evaluate` locally to get scores, and check the result files. That's it.

### LangSmith flags

| Flag | Command | Purpose |
|------|---------|---------|
| `--langsmith-experiment` | `generate` | Enable tracing — records all runs in a LangSmith experiment |
| `--langsmith-dataset <name>` | `generate` | Dataset name to accumulate cases across batches (default: `drb-{model-name}`) |
| `--experiment-prefix <prefix>` | `generate` | Prefix for the experiment name (default: `{model-name}`) |
| `--langsmith-max-concurrency <n>` | `generate` | Concurrent runs in aevaluate (default: 0 = sequential) |

### Recommended workflow

**Step 1 — Generate all batches with tracing enabled**

Use a consistent `--langsmith-dataset` name across all batches. This is the shared dataset that accumulates all 100 cases — it is the durable artifact in LangSmith that persists across batch runs. Each batch creates its own experiment (one per `generate` call), all pointing at the same dataset.

```bash
# Batch 1
PYTHONPATH=. uv run -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher \
  --range 1-25 \
  --langsmith-experiment \
  --langsmith-dataset drb-multi-agent \
  --experiment-prefix "multi-agent-v1"

# Batch 2 — same dataset, new experiment for this batch
PYTHONPATH=. uv run -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher \
  --range 26-50 \
  --langsmith-experiment \
  --langsmith-dataset drb-multi-agent \
  --experiment-prefix "multi-agent-v1"

# ... continue until all 100 cases are done
```

**Step 2 — Check completion**

```bash
PYTHONPATH=. uv run -m evals.cli status \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher
# Should show: Completed: 100/100
```

**Step 3 — Run final evaluation locally**

```bash
cp eval_results/multi_agent_researcher.jsonl \
   $DRB_REPO_PATH/data/test_data/raw_data/

PYTHONPATH=. uv run -m evals.cli evaluate \
  --model-name multi_agent_researcher --force
```

The definitive aggregate scores are now in:
- `$DRB_REPO_PATH/results/race/multi_agent_researcher/race_result.txt`
- `$DRB_REPO_PATH/results/fact/multi_agent_researcher/fact_result.txt`

If `LANGSMITH_API_KEY` is set, `evaluate` automatically writes the aggregate scores as metadata on the LangSmith dataset (e.g. `drb-multi-agent`). The dataset is shared across all batch experiments so this gives you one place to see the overall benchmark result without caring which experiment it came from. Re-running `evaluate` overwrites the previous metadata with the latest scores.

The dataset metadata will contain:
```
race_overall_score, race_comprehensiveness, race_insight,
race_instruction_following, race_readability,
fact_valid_rate, fact_total_citations, fact_total_valid_citations,
model_name, evaluated_at
```

The traces for all runs are already in LangSmith from step 1. This is the complete workflow.

### Optional: attaching per-run DRB scores to traces

If you also want the RACE/FACT scores visible alongside individual run traces (e.g. to filter or sort runs by score in the experiment view), pass `--langsmith-experiment` with the name logged during `generate`:

```bash
# The experiment name is printed at the end of each generate call:
# "LangSmith experiment name: multi-agent-v1-20260328T102300"

PYTHONPATH=. uv run -m evals.cli evaluate \
  --model-name multi_agent_researcher \
  --langsmith-experiment "multi-agent-v1-20260328T102300" \
  --langsmith-dataset drb-multi-agent
```

This attaches `drb_race_score` and `drb_fact_score` as feedback on each individual run. If you generated in batches, repeat this for each batch experiment name. Only do this after the final `evaluate --force` with all 100 cases — uploading against partial result files and then re-running later creates duplicate feedback entries.

This step is purely additive and separate from the dataset metadata update above.

### Debugging a single failing case

`--case-id` creates a one-run experiment with the full agent trace visible in LangSmith. Useful for inspecting why a specific case returned a partial result or timed out:

```bash
PYTHONPATH=. uv run -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher \
  --case-id 42 --force \
  --langsmith-experiment \
  --experiment-prefix "debug-case-42"
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
