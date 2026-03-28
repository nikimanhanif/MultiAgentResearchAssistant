# Evaluation Harness

Benchmark the production research workflow using the [DeepResearchBench](https://github.com/Ayanami0730/deep_research_bench) framework with **DeepSeek as the judge backend**.

> **Note:** This is a custom DRB evaluation setup. The official DRB default uses Gemini as the judge LLM. Using DeepSeek means results are **not directly comparable** to official Gemini-judged baselines. The DRB task set, evaluation pipeline (RACE + FACT), and scoring methodology remain identical.

## Quick Reference

| Command | Purpose |
|---------|---------|
| `inspect` | Preview loaded DRB queries |
| `generate` | Run the pipeline, export DRB-format output |
| `evaluate` | Invoke RACE + FACT evaluation (DeepSeek judge) |
| `run` | Internal harness with preliminary scorers |
| `--case-id <ID>` | Target a specific case (available for all commands) |
| `--shuffle` | Randomly sample cases (for `inspect`, `generate`, `run`) |

## Prerequisites

```bash
# 1. Clone the official DRB repo
git clone https://github.com/Ayanami0730/deep_research_bench.git
cd deep_research_bench && pip install -r requirements.txt

# 2. Set required env vars
export DRB_REPO_PATH="/Users/bukuo/Documents/FYP/deep_research_bench"
export DEEPSEEK_API_KEY="your_deepseek_api_key"   # LLM judge (default)
export JINA_API_KEY="your_jina_api_key"             # Web scraping for FACT
```

## Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `DEEPSEEK_API_KEY` | **Yes** | — | DeepSeek API key (judge LLM) |
| `JINA_API_KEY` | **Yes** | — | Jina web scraping (FACT) |
| `DRB_REPO_PATH` | **Yes** | — | Path to cloned DRB repo |
| `DRB_JUDGE_PROVIDER` | No | `deepseek` | Judge backend: `deepseek` or `gemini` |
| `DRB_RACE_MODEL` | No | `deepseek-chat` | Model for RACE evaluation |
| `DRB_FACT_MODEL` | No | `deepseek-chat` | Model for FACT extraction/validation |
| `DEEPSEEK_BASE_URL` | No | `https://api.deepseek.com` | DeepSeek API endpoint |
| `GEMINI_API_KEY` | If `gemini` | — | Only needed if `DRB_JUDGE_PROVIDER=gemini` |

## End-to-End Workflow

### Step 1: Inspect the DRB query set

```bash
cd backend
PYTHONPATH=. .venv/bin/python -m evals.cli inspect \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl
```

### Step 2: Dry-run generate (validates without running the graph)

```bash
PYTHONPATH=. .venv/bin/python -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher \
  --dry-run
```

### Step 3: Generate DRB-format outputs

```bash
# Smoke run (3 cases)
PYTHONPATH=. .venv/bin/python -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher \
  --limit 3

# Full run (all 100 cases)
PYTHONPATH=. .venv/bin/python -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher
```

Output: `eval_results/multi_agent_researcher.jsonl` with `{id, prompt, article}` records.

### Step 4: Copy output to DRB repo

```bash
cp eval_results/multi_agent_researcher.jsonl \
   $DRB_REPO_PATH/data/test_data/raw_data/
```

### Step 5: Run evaluation (DeepSeek judge)

```bash
# Both RACE and FACT
PYTHONPATH=. .venv/bin/python -m evals.cli evaluate \
  --model-name multi_agent_researcher

# Individual phases
PYTHONPATH=. .venv/bin/python -m evals.cli evaluate \
  --model-name multi_agent_researcher --phase race

PYTHONPATH=. .venv/bin/python -m evals.cli evaluate \
  --model-name multi_agent_researcher --phase fact
```

### Step 6: Targeted Evaluation (Optional)

If you only want to investigate a specific case (e.g., ID 50), you can skip the full run:

```bash
# 1. Generate only case 50
PYTHONPATH=. .venv/bin/python -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher \
  --case-id 50

# 2. Copy the results (as usual)
cp eval_results/multi_agent_researcher.jsonl $DRB_REPO_PATH/data/test_data/raw_data/

# 3. Evaluate only case 50
PYTHONPATH=. .venv/bin/python -m evals.cli evaluate \
  --model-name multi_agent_researcher --case-id 50
```

> [!NOTE]
> When using `--case-id` with `evaluate`, the system automatically handles creating a temporary filtered version of `query.jsonl` so that the official DRB scripts only process that specific case.

### Step 7: Incremental Batch Testing (Optional)

If you want to run queries in sequential batches (e.g., 1-10, then 11-20) and verify them progressively, use the `--range` flag. This flag filters by the numeric `id` in the benchmark dataset.

When using `--range`, the harness automatically runs in **append mode**. It will load existing results from `multi_agent_researcher.jsonl`, merge in the new results (overwriting any matching IDs), and write the file sorted by ID.

```bash
# 1. Run the first 10 queries
PYTHONPATH=. .venv/bin/python -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher \
  --range 1-10

# 2. (Verify output manually if desired)

# 3. Run the next 10 queries (results safely appended and sorted)
PYTHONPATH=. .venv/bin/python -m evals.cli generate \
  --queries $DRB_REPO_PATH/data/prompt_data/query.jsonl \
  --output eval_results/ \
  --model-name multi_agent_researcher \
  --range 11-20
```

Results saved to:
- `$DRB_REPO_PATH/results/race/multi_agent_researcher/race_result.txt`
- `$DRB_REPO_PATH/results/fact/multi_agent_researcher/fact_result.txt`

### Optional: Switch back to Gemini judge

```bash
export DRB_JUDGE_PROVIDER=gemini
export GEMINI_API_KEY="your_gemini_key"
PYTHONPATH=. .venv/bin/python -m evals.cli evaluate \
  --model-name multi_agent_researcher
```

## Judge Backend Architecture

The evaluator automatically deploys a provider-aware `utils/api.py` into the DRB repo:

```
evals/drb_patches/api.py  →  deep_research_bench/utils/api.py
                               (original backed up as api.py.gemini_original)
```

The patched `api.py` preserves the identical interface (`AIClient`, `call_model`, `scrape_url`) while routing LLM calls through DeepSeek (default) or Gemini (optional).

### Model Defaults

| Purpose | DeepSeek Default | Gemini Default | Env Var |
|---------|-----------------|---------------|---------|
| RACE (quality judge) | `deepseek-chat` | `gemini-2.5-pro` | `DRB_RACE_MODEL` |
| FACT (extraction/validation) | `deepseek-chat` | `gemini-2.5-flash` | `DRB_FACT_MODEL` |

## Architecture

```
evals/
├── cli.py               # CLI: inspect, generate, evaluate, run
├── runner.py             # EvalRunner: wraps production graph
├── drb_evaluator.py      # RACE + FACT subprocess wrappers + judge setup
├── drb_patches/
│   └── api.py            # Provider-aware drop-in for DRB utils/api.py
├── policies.py           # Deterministic interrupt handling
├── scorers.py            # Internal diagnostic scorers (not DRB-official)
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

## Scoring

| Type | Framework | Source |
|------|-----------|--------|
| **DRB Evaluation** | RACE (quality) + FACT (citations) | DeepResearch Bench repo (DeepSeek judge) |
| **Internal** | Success, latency, findings, citations, completeness | `evals/scorers.py` |

> Internal scorers are preliminary diagnostics. For proper evaluation, use `python -m evals.cli evaluate`.

## Running Tests

```bash
cd backend
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_evals/ -v
```

## Budget Overrides

For shorter smoke runs:
```bash
--max-iterations 5 --max-sub-agents 3
```
