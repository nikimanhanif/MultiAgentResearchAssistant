"""
LangSmith experiment tracking integration for the evaluation harness.

Provides:
  - Dataset management: upload/sync BenchmarkCase objects to a LangSmith dataset
  - Experiment runner: wraps aevaluate() around run_case() so every case
    execution is a traced LangSmith run linked to the experiment
  - DRB score upload: push official RACE/FACT scores back to individual runs

Usage:
  from evals.langsmith_eval import run_with_langsmith, upload_drb_scores_to_experiment

The existing JSONL/CSV/Markdown/DRB reporters continue to work unchanged —
run_with_langsmith() returns the same List[EvalResult] as run_batch() does.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from evals.models import BenchmarkCase, EvalResult, RunStatus

if TYPE_CHECKING:
    from evals.runner import EvalRunner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def is_langsmith_available() -> bool:
    """Return True if LANGSMITH_API_KEY is set and langsmith can be imported."""
    if not os.environ.get("LANGSMITH_API_KEY"):
        logger.warning(
            "LANGSMITH_API_KEY is not set. LangSmith experiment tracking is disabled."
        )
        return False
    try:
        import langsmith  # noqa: F401
        return True
    except ImportError:
        logger.warning(
            "langsmith package not importable. Install it with: uv add langsmith"
        )
        return False


# ---------------------------------------------------------------------------
# Example conversion helpers
# ---------------------------------------------------------------------------

def _case_to_example(case: BenchmarkCase) -> dict:
    """
    Convert a BenchmarkCase to a LangSmith example dict.

    inputs  — everything needed to reconstruct the case inside target()
    outputs — reference answer (if available), used by evaluators
    metadata — external_id is the stable upsert key matching case_id
    """
    inputs: dict[str, Any] = {
        "case_id": str(case.case_id),
        "query": case.query,
    }
    if case.context is not None:
        inputs["context"] = case.context
    if case.constraints is not None:
        inputs["constraints"] = case.constraints
    if case.expected_topics is not None:
        inputs["expected_topics"] = case.expected_topics
    # Preserve all adapter-specific metadata (drb_prompt, topic, language, etc.)
    for k, v in case.metadata.items():
        if k not in inputs:
            inputs[k] = v

    outputs: dict[str, Any] = {}
    if case.expected_answer is not None:
        outputs["expected_answer"] = case.expected_answer

    metadata = {
        "external_id": str(case.case_id),
        "topic": case.metadata.get("topic", ""),
        "language": case.metadata.get("language", ""),
        "adapter": case.metadata.get("adapter", ""),
    }

    return {"inputs": inputs, "outputs": outputs, "metadata": metadata}


def case_from_inputs(inputs: dict) -> BenchmarkCase:
    """
    Reconstruct a BenchmarkCase from a LangSmith dataset example's inputs dict.

    Called inside the target() function to recover the full case object
    from the flattened inputs that LangSmith passes through.
    """
    known_fields = {"case_id", "query", "context", "constraints", "expected_topics"}
    metadata = {k: v for k, v in inputs.items() if k not in known_fields}
    return BenchmarkCase(
        case_id=inputs["case_id"],
        query=inputs["query"],
        context=inputs.get("context"),
        constraints=inputs.get("constraints"),
        expected_topics=inputs.get("expected_topics"),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Dataset management
# ---------------------------------------------------------------------------

def ensure_dataset(
    client: Any,
    dataset_name: str,
    cases: list[BenchmarkCase],
) -> Any:
    """
    Create or update a LangSmith dataset with the given benchmark cases.

    - If the dataset does not exist, it is created and all cases are added.
    - If it exists, only cases whose case_id is not already present are added.
    - Uses external_id in example metadata as the stable deduplication key.

    Idempotent: safe to call for the same cases across multiple partial runs.
    """
    # Check if dataset exists
    dataset = None
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        logger.info("Found existing LangSmith dataset: '%s'", dataset_name)
    except Exception:
        pass

    if dataset is None:
        dataset = client.create_dataset(
            dataset_name,
            description="DeepResearch Bench evaluation cases for multi-agent-researcher",
        )
        logger.info("Created new LangSmith dataset: '%s'", dataset_name)

    # Determine which case_ids are already present
    existing_ids: set[str] = set()
    try:
        for example in client.list_examples(dataset_id=dataset.id):
            ext_id = (example.metadata or {}).get("external_id")
            if ext_id:
                existing_ids.add(str(ext_id))
    except Exception as exc:
        logger.warning(
            "Could not list existing examples (will create all): %s", exc
        )

    new_cases = [c for c in cases if str(c.case_id) not in existing_ids]

    if not new_cases:
        logger.info(
            "Dataset '%s' already contains all %d cases.", dataset_name, len(cases)
        )
        return dataset

    examples = [_case_to_example(c) for c in new_cases]
    client.create_examples(
        inputs=[e["inputs"] for e in examples],
        outputs=[e["outputs"] for e in examples],
        metadata=[e["metadata"] for e in examples],
        dataset_id=dataset.id,
    )
    logger.info(
        "Added %d new examples to dataset '%s' (%d already existed).",
        len(new_cases), dataset_name, len(existing_ids),
    )
    return dataset


# ---------------------------------------------------------------------------
# Target function and evaluators
# ---------------------------------------------------------------------------

def build_target(
    runner: "EvalRunner",
    result_cache: dict[str, EvalResult],
) -> Callable:
    """
    Build a @traceable async target function for aevaluate().

    The @traceable decorator makes this span the parent of all agent/tool
    traces that fire inside run_case(), automatically linking them to the
    LangSmith experiment. The full EvalResult is stored in result_cache so
    the CLI can pass it to the existing downstream reporters.
    """
    from langsmith import traceable

    @traceable(name="eval_run_case")
    async def target(inputs: dict) -> dict:
        case = case_from_inputs(inputs)
        result = await runner.run_case(case)
        result_cache[str(case.case_id)] = result
        return {
            "report": result.report_content or "",
            "status": result.status.value,
            "findings_count": result.findings_count,
            "duration_seconds": result.metadata.duration_seconds or 0.0,
            # Flat scores dict so evaluators can read them without re-running scorers
            "scores": {s.metric: s.value for s in result.scores},
            # case_id in outputs lets upload_drb_scores_to_experiment map runs without N+1 lookups
            "case_id": str(result.case_id),
        }

    return target


def make_langsmith_evaluators() -> list[Callable]:
    """
    Return a list of LangSmith-compatible evaluator functions.

    A single evaluator reads all pre-computed scores from run.outputs["scores"]
    and returns them as a list of {key, score} dicts. This avoids re-running
    scorer logic (which requires live graph state) inside the evaluator.
    """
    def all_scores_evaluator(run: Any, example: Any) -> list[dict]:
        scores: dict = (run.outputs or {}).get("scores", {})
        return [
            {"key": metric, "score": value}
            for metric, value in scores.items()
            if value is not None
        ]

    return [all_scores_evaluator]


def make_summary_evaluator() -> Callable:
    """Return a summary evaluator computing success_rate across all experiment runs."""
    def summary_evaluator(runs: list[Any], examples: list[Any]) -> dict:
        statuses = [(r.outputs or {}).get("status") for r in runs]
        success_count = sum(1 for s in statuses if s == RunStatus.SUCCESS.value)
        total = len(runs)
        return {
            "key": "success_rate",
            "score": success_count / total if total else 0.0,
            "comment": f"{success_count}/{total} runs succeeded",
        }

    return summary_evaluator


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

async def run_with_langsmith(
    cases: list[BenchmarkCase],
    runner: "EvalRunner",
    *,
    dataset_name: str,
    experiment_prefix: str,
    experiment_metadata: dict | None = None,
    max_concurrency: int = 0,
) -> tuple[list[EvalResult], str | None]:
    """
    Run eval via LangSmith aevaluate() and return (results, experiment_url).

    - Uploads benchmark cases to the named dataset (idempotent upsert).
    - Only runs the examples matching the current batch (supports --range and --case-id).
    - Returns results in the same order as input cases so downstream reporters work unchanged.
    - experiment_url may be None if it cannot be determined from the API response.
    """
    import langsmith

    try:
        from langsmith.evaluation import aevaluate
    except ImportError:
        raise RuntimeError(
            "langsmith.evaluation.aevaluate not available. "
            "Upgrade langsmith: uv add 'langsmith>=0.2.0'"
        )

    client = langsmith.Client()

    # 1. Ensure dataset is populated with current cases
    logger.info("Syncing LangSmith dataset '%s' (%d cases)...", dataset_name, len(cases))
    dataset = ensure_dataset(client, dataset_name, cases)

    # 2. Fetch only the examples for the current batch from the dataset.
    #    This is critical for --range / --case-id: the dataset may have more
    #    cases from previous runs, but this experiment only runs the current batch.
    case_ids = {str(c.case_id) for c in cases}
    batch_examples = [
        ex for ex in client.list_examples(dataset_id=dataset.id)
        if (ex.metadata or {}).get("external_id") in case_ids
    ]

    if not batch_examples:
        logger.error("No dataset examples found for the current batch. Cannot run experiment.")
        return [], None

    logger.info(
        "Found %d/%d dataset examples for this batch.", len(batch_examples), len(cases)
    )

    # 3. Build target and evaluators
    result_cache: dict[str, EvalResult] = {}
    target = build_target(runner, result_cache)
    evaluators = make_langsmith_evaluators()
    summary_evaluators = [make_summary_evaluator()]

    # 4. Run the experiment (only against the batch examples)
    logger.info(
        "Starting LangSmith experiment (prefix='%s', %d cases, max_concurrency=%d)...",
        experiment_prefix, len(batch_examples), max_concurrency,
    )
    experiment_results = await aevaluate(
        target,
        data=batch_examples,
        evaluators=evaluators,
        summary_evaluators=summary_evaluators,
        experiment_prefix=experiment_prefix,
        metadata=experiment_metadata or {},
        max_concurrency=max_concurrency,
    )

    # 5. Resolve experiment URL / name for the user to reference
    experiment_url: str | None = None
    experiment_name: str | None = None
    try:
        experiment_name = experiment_results.experiment_name  # type: ignore[attr-defined]
        try:
            project = client.read_project(project_name=experiment_name)
            experiment_url = getattr(project, "url", None)
        except Exception:
            pass
        if not experiment_url and experiment_name:
            experiment_url = f"https://smith.langchain.com (experiment: {experiment_name})"
    except Exception:
        experiment_url = f"https://smith.langchain.com (prefix: {experiment_prefix})"

    if experiment_name:
        logger.info("LangSmith experiment name: %s", experiment_name)

    # 6. Return results in original case order
    ordered_results: list[EvalResult] = []
    for case in cases:
        result = result_cache.get(str(case.case_id))
        if result:
            ordered_results.append(result)
        else:
            logger.warning("No result cached for case %s — it may have been skipped.", case.case_id)

    logger.info(
        "Experiment complete. %d/%d results captured.",
        len(ordered_results), len(cases),
    )
    return ordered_results, experiment_url


# ---------------------------------------------------------------------------
# DRB score upload
# ---------------------------------------------------------------------------

def _parse_race_result_file(path: str) -> dict[str, float]:
    """
    Parse a DRB race_result.txt file and return {case_id: score}.

    DRB RACE output is typically an aggregate file. If per-case scores are
    present, they are extracted. Otherwise returns an empty dict and logs a
    warning so the caller can handle it gracefully.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as exc:
        logger.warning("Could not read RACE result file %s: %s", path, exc)
        return {}

    scores: dict[str, float] = {}

    # Pattern: lines like  "1: 0.85"  or  "id=1 score=0.85"  or  "case_id=1, score=0.85"
    patterns = [
        re.compile(r'"?(\d+)"?\s*:\s*([0-9]*\.?[0-9]+)'),
        re.compile(r'id[=\s]+(\d+)[,\s]+score[=\s]+([0-9]*\.?[0-9]+)', re.IGNORECASE),
        re.compile(r'case[_\s]?id[=\s]+(\d+)[,\s]+([0-9]*\.?[0-9]+)', re.IGNORECASE),
    ]

    for pattern in patterns:
        for match in pattern.finditer(content):
            case_id, score = match.group(1), match.group(2)
            scores[case_id] = float(score)
        if scores:
            break

    if not scores:
        logger.warning(
            "Could not extract per-case scores from RACE result file: %s\n"
            "The file may use an aggregate format. DRB scores will not be "
            "uploaded per-run to LangSmith.",
            path,
        )

    return scores


def _parse_fact_result_file(path: str) -> dict[str, float]:
    """
    Parse a DRB fact_result.txt file and return {case_id: score}.

    Same strategy as _parse_race_result_file. Returns empty dict if the
    file only contains aggregate statistics.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as exc:
        logger.warning("Could not read FACT result file %s: %s", path, exc)
        return {}

    scores: dict[str, float] = {}

    patterns = [
        re.compile(r'"?(\d+)"?\s*:\s*([0-9]*\.?[0-9]+)'),
        re.compile(r'id[=\s]+(\d+)[,\s]+score[=\s]+([0-9]*\.?[0-9]+)', re.IGNORECASE),
        re.compile(r'case[_\s]?id[=\s]+(\d+)[,\s]+([0-9]*\.?[0-9]+)', re.IGNORECASE),
    ]

    for pattern in patterns:
        for match in pattern.finditer(content):
            case_id, score = match.group(1), match.group(2)
            scores[case_id] = float(score)
        if scores:
            break

    if not scores:
        logger.warning(
            "Could not extract per-case scores from FACT result file: %s\n"
            "The file may use an aggregate format. DRB scores will not be "
            "uploaded per-run to LangSmith.",
            path,
        )

    return scores


def upload_drb_scores_to_experiment(
    experiment_name: str,
    race_result_path: str | None,
    fact_result_path: str | None,
    *,
    model_name: str,
) -> None:
    """
    Parse DRB RACE/FACT result files and log scores as LangSmith feedback
    on individual runs within the named experiment.

    Links scores via case_id stored in run.outputs["case_id"], avoiding
    N+1 API lookups per run.
    """
    if not is_langsmith_available():
        return

    import langsmith
    client = langsmith.Client()

    # 1. Find the experiment project
    project = None
    try:
        project = client.read_project(project_name=experiment_name)
    except Exception:
        # Try searching by name_contains
        try:
            matches = list(client.list_projects(name_contains=experiment_name))
            if matches:
                project = matches[0]
                if len(matches) > 1:
                    logger.warning(
                        "Multiple experiments match '%s'. Using: %s",
                        experiment_name, project.name,
                    )
        except Exception as exc:
            logger.error(
                "Could not find LangSmith experiment '%s': %s\n"
                "Tip: copy the exact experiment name logged during 'generate'.",
                experiment_name, exc,
            )
            return

    if project is None:
        logger.error(
            "LangSmith experiment '%s' not found. "
            "Tip: copy the exact name logged during 'generate --langsmith-experiment'.",
            experiment_name,
        )
        return

    # 2. List root runs and build {case_id: run_id} map from run.outputs["case_id"]
    case_to_run: dict[str, str] = {}
    try:
        for run in client.list_runs(project_id=str(project.id), is_root=True):
            case_id = (run.outputs or {}).get("case_id")
            if case_id:
                case_to_run[str(case_id)] = str(run.id)
    except Exception as exc:
        logger.error("Could not list runs for experiment '%s': %s", experiment_name, exc)
        return

    if not case_to_run:
        logger.warning(
            "No runs with case_id in outputs found for experiment '%s'. "
            "Ensure you used --langsmith-experiment during generate.",
            experiment_name,
        )
        return

    logger.info(
        "Found %d runs in experiment '%s'. Uploading DRB scores...",
        len(case_to_run), experiment_name,
    )

    # 3. Parse result files
    race_scores = _parse_race_result_file(race_result_path) if race_result_path else {}
    fact_scores = _parse_fact_result_file(fact_result_path) if fact_result_path else {}

    # 4. Upload per-run feedback
    race_uploaded = fact_uploaded = 0
    for case_id, run_id in case_to_run.items():
        if case_id in race_scores:
            try:
                client.create_feedback(
                    run_id=run_id,
                    key="drb_race_score",
                    score=race_scores[case_id],
                    comment=f"Official DRB RACE evaluation ({model_name})",
                )
                race_uploaded += 1
            except Exception as exc:
                logger.warning("Could not upload RACE score for case %s: %s", case_id, exc)

        if case_id in fact_scores:
            try:
                client.create_feedback(
                    run_id=run_id,
                    key="drb_fact_score",
                    score=fact_scores[case_id],
                    comment=f"Official DRB FACT evaluation ({model_name})",
                )
                fact_uploaded += 1
            except Exception as exc:
                logger.warning("Could not upload FACT score for case %s: %s", case_id, exc)

    logger.info(
        "DRB score upload complete: %d RACE scores, %d FACT scores uploaded to experiment '%s'.",
        race_uploaded, fact_uploaded, experiment_name,
    )

    if race_uploaded == 0 and fact_uploaded == 0:
        logger.info(
            "No per-case scores were found in the result files. "
            "DRB may report only aggregate statistics. "
            "Check the result files manually:\n  RACE: %s\n  FACT: %s",
            race_result_path, fact_result_path,
        )
