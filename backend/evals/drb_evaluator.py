"""
DRB evaluation integration via subprocess.

Wraps the official DeepResearchBench evaluation scripts with a custom
judge backend. By default, uses DeepSeek as the LLM judge instead of
the official Gemini default.

IMPORTANT: Using DeepSeek as the judge means results are NOT directly
comparable to official Gemini-judged DRB baselines. This is a custom
evaluation setup using the DRB framework with an alternative judge.

Judge setup:
  1. Deploys a patched utils/api.py into the DRB repo (backs up original)
  2. Ensures the `openai` dependency is available in the DRB environment
  3. Passes judge configuration via environment variables

Evaluation pipeline:
  - RACE: deepresearch_bench_race.py
  - FACT: utils.extract → utils.deduplicate → utils.scrape → utils.validate → utils.stat

Required env vars:
  - DEEPSEEK_API_KEY  (LLM judge, default provider)
  - JINA_API_KEY      (web scraping for FACT)

Optional env vars:
  - DRB_JUDGE_PROVIDER  ("deepseek" default, or "gemini")
  - DRB_RACE_MODEL      (default: deepseek-chat)
  - DRB_FACT_MODEL      (default: deepseek-chat)
  - DEEPSEEK_BASE_URL   (default: https://api.deepseek.com)
  - DRB_REPO_PATH       (path to DRB repo clone)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from app.config import settings

logger = logging.getLogger(__name__)

# Path to our patched api.py
_PATCHES_DIR = Path(__file__).parent / "drb_patches"
_PATCHED_API = _PATCHES_DIR / "api.py"


@dataclass
class DRBEvalConfig:
    """Configuration for running DRB evaluation."""
    drb_repo_path: str
    model_name: str
    raw_output_path: Optional[str] = None
    max_workers: int = 10
    limit: Optional[int] = None
    case_id: Optional[str] = None
    only_language: Optional[str] = None  # "en" or "zh"
    skip_cleaning: bool = False
    force: bool = False
    query_file_override: Optional[str] = None  # Internal use for temporary files


@dataclass
class DRBEvalResult:
    """Result of a DRB evaluation run."""
    model_name: str
    judge_provider: str = "deepseek"
    race_result_path: Optional[str] = None
    fact_result_path: Optional[str] = None
    race_return_code: Optional[int] = None
    fact_return_code: Optional[int] = None
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return (
            self.race_return_code == 0
            and self.fact_return_code == 0
            and not self.errors
        )


def _validate_drb_repo(repo_path: str) -> List[str]:
    """Check that the DRB repo has the expected structure."""
    errors: List[str] = []
    p = Path(repo_path)
    if not p.is_dir():
        errors.append(f"DRB repo path does not exist: {repo_path}")
        return errors

    expected = [
        "deepresearch_bench_race.py",
        "data/prompt_data/query.jsonl",
        "utils",
    ]
    for rel in expected:
        if not (p / rel).exists():
            errors.append(f"Missing expected DRB path: {p / rel}")

    return errors


def _get_judge_provider() -> str:
    """Get the active judge provider from env or settings."""
    return (os.environ.get("DRB_JUDGE_PROVIDER") or settings.DRB_JUDGE_PROVIDER).lower()


def _validate_env_vars() -> List[str]:
    """Check required environment variables based on judge provider."""
    errors: List[str] = []
    provider = _get_judge_provider()

    if provider == "deepseek":
        if not os.environ.get("DEEPSEEK_API_KEY"):
            errors.append(
                "DEEPSEEK_API_KEY is not set. Required for DeepSeek judge backend.\n"
                "Set it in .env or export it: export DEEPSEEK_API_KEY='your_key'"
            )
    elif provider == "gemini":
        if not os.environ.get("GEMINI_API_KEY"):
            errors.append(
                "GEMINI_API_KEY is not set. Required when DRB_JUDGE_PROVIDER=gemini."
            )
    else:
        errors.append(
            f"Unknown DRB_JUDGE_PROVIDER: '{provider}'. Supported: 'deepseek', 'gemini'."
        )

    if not os.environ.get("JINA_API_KEY"):
        errors.append(
            "JINA_API_KEY is not set. Required for FACT evaluation (web scraping)."
        )

    return errors


def _deploy_patched_api(repo_path: str) -> None:
    """
    Deploy our provider-aware api.py into the DRB repo.

    Backs up the original utils/api.py as utils/api.py.gemini_original
    if a backup doesn't already exist.
    """
    target = Path(repo_path) / "utils" / "api.py"
    backup = Path(repo_path) / "utils" / "api.py.gemini_original"

    if not _PATCHED_API.is_file():
        raise FileNotFoundError(
            f"Patched api.py not found at {_PATCHED_API}. "
            "Ensure evals/drb_patches/api.py exists."
        )

    # Back up original if not already backed up
    if target.is_file() and not backup.is_file():
        shutil.copy2(str(target), str(backup))
        logger.info("Backed up original api.py → api.py.gemini_original")

    # Deploy patched version
    shutil.copy2(str(_PATCHED_API), str(target))
    logger.info("Deployed patched api.py to %s", target)


def _ensure_openai_dependency(repo_path: str) -> None:
    """
    Ensure the `openai` package is available for the DeepSeek provider.

    Checks if openai is importable in the DRB repo's Python environment.
    If not, installs it.
    """
    provider = _get_judge_provider()
    if provider != "deepseek":
        return

    # Check if openai is already available
    result = subprocess.run(
        [sys.executable, "-c", "import openai"],
        cwd=repo_path,
        capture_output=True,
    )
    if result.returncode == 0:
        return

    logger.info("Installing 'openai' package for DeepSeek provider...")
    install_result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "openai"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if install_result.returncode != 0:
        logger.error("Failed to install openai: %s", install_result.stderr)
        raise RuntimeError(
            "Could not install 'openai' package. Install it manually:\n"
            f"  cd {repo_path} && pip install openai"
        )
    logger.info("Successfully installed 'openai' package")


def _build_judge_env() -> dict[str, str]:
    """
    Build environment variables dict for subprocess calls.

    Passes through all judge-related env vars so the patched api.py
    can read them in the subprocess.
    """
    env = os.environ.copy()
    
    # Map from our config names to what the patched api.py expects
    config_map = {
        "DRB_JUDGE_PROVIDER": settings.DRB_JUDGE_PROVIDER,
        "DEEPSEEK_API_KEY": settings.DEEPSEEK_API_KEY,
        "DEEPSEEK_BASE_URL": settings.DEEPSEEK_BASE_URL,
        "DRB_RACE_MODEL": settings.DRB_RACE_MODEL,
        "DRB_FACT_MODEL": settings.DRB_FACT_MODEL,
        "GEMINI_API_KEY": settings.GOOGLE_GEMINI_API_KEY,
        "JINA_API_KEY": settings.JINA_API_KEY,
    }

    for var, val in config_map.items():
        # Subprocess environment should prioritize os.environ if set, 
        # otherwise use the value from our settings (which loaded .env)
        if var not in env and val:
            env[var] = str(val)
            
    return env


def _run_subprocess(
    cmd: List[str], cwd: str, label: str
) -> int:
    """Run a subprocess with live output, passing judge env vars."""
    logger.info("[%s] Running: %s", label, " ".join(cmd))
    env = _build_judge_env()
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=False,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            logger.error("[%s] Exited with code %d", label, result.returncode)
        return result.returncode
    except FileNotFoundError as e:
        logger.error("[%s] Command not found: %s", label, e)
        return -1


def resolve_drb_repo_path(cli_path: Optional[str] = None) -> str:
    """
    Resolve DRB repo path from CLI arg or env var.

    Raises ValueError if none are set.
    """
    path = cli_path or os.environ.get("DRB_REPO_PATH")
    if not path:
        raise ValueError(
            "DRB repo path not specified. Use --drb-repo-path, set DRB_REPO_PATH env var, or update .env.\n"
            "Clone the repo: git clone https://github.com/Ayanami0730/deep_research_bench"
        )
    return str(Path(path).resolve())


def setup_drb_judge(repo_path: str) -> None:
    """
    One-time setup: deploy patched api.py and ensure dependencies.

    Call this before running any evaluation. It is idempotent.
    """
    _deploy_patched_api(repo_path)
    _ensure_openai_dependency(repo_path)
    provider = _get_judge_provider()
    race_model = os.environ.get("DRB_RACE_MODEL", "deepseek-reasoner" if provider == "deepseek" else "gemini-2.5-pro-preview-06-05")
    fact_model = os.environ.get("DRB_FACT_MODEL", "deepseek-reasoner" if provider == "deepseek" else "gemini-2.5-flash-preview-05-20")
    logger.info(
        "Judge setup complete: provider=%s, race_model=%s, fact_model=%s",
        provider, race_model, fact_model,
    )


def run_race(config: DRBEvalConfig) -> int:
    """Run the RACE evaluation script."""
    repo = Path(config.drb_repo_path)
    output_dir = repo / "results" / "race" / config.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-u", "deepresearch_bench_race.py",
        config.model_name,
        "--raw_data_dir", str(repo / "data" / "test_data" / "raw_data"),
        "--max_workers", str(config.max_workers),
        "--query_file", config.query_file_override or str(repo / "data" / "prompt_data" / "query.jsonl"),
        "--output_dir", str(output_dir),
    ]

    if config.limit is not None:
        cmd.extend(["--limit", str(config.limit)])
    if config.skip_cleaning:
        cmd.append("--skip_cleaning")
    if config.only_language == "zh":
        cmd.append("--only_zh")
    elif config.only_language == "en":
        cmd.append("--only_en")
    if config.force:
        cmd.append("--force")

    return _run_subprocess(cmd, cwd=str(repo), label="RACE")


def run_fact(config: DRBEvalConfig) -> int:
    """
    Run the FACT evaluation pipeline (5 steps).

    Steps: extract → deduplicate → scrape → validate → stat
    """
    repo = Path(config.drb_repo_path)
    raw_data_path = config.raw_output_path or str(
        repo / "data" / "test_data" / "raw_data" / f"{config.model_name}.jsonl"
    )
    output_dir = repo / "results" / "fact" / config.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    query_data_path = config.query_file_override or str(repo / "data" / "prompt_data" / "query.jsonl")
    n_workers = str(config.max_workers)

    steps = [
        ("extract", [
            sys.executable, "-u", "-m", "utils.extract",
            "--raw_data_path", raw_data_path,
            "--output_path", str(output_dir / "extracted.jsonl"),
            "--query_data_path", query_data_path,
            "--n_total_process", n_workers,
        ]),
        ("deduplicate", [
            sys.executable, "-u", "-m", "utils.deduplicate",
            "--raw_data_path", str(output_dir / "extracted.jsonl"),
            "--output_path", str(output_dir / "deduplicated.jsonl"),
            "--query_data_path", query_data_path,
            "--n_total_process", n_workers,
        ]),
        ("scrape", [
            sys.executable, "-u", "-m", "utils.scrape",
            "--raw_data_path", str(output_dir / "deduplicated.jsonl"),
            "--output_path", str(output_dir / "scraped.jsonl"),
            "--n_total_process", n_workers,
        ]),
        ("validate", [
            sys.executable, "-u", "-m", "utils.validate",
            "--raw_data_path", str(output_dir / "scraped.jsonl"),
            "--output_path", str(output_dir / "validated.jsonl"),
            "--query_data_path", query_data_path,
            "--n_total_process", n_workers,
        ]),
        ("stat", [
            sys.executable, "-u", "-m", "utils.stat",
            "--input_path", str(output_dir / "validated.jsonl"),
            "--output_path", str(output_dir / "fact_result.txt"),
        ]),
    ]

    for step_name, cmd in steps:
        logger.info("FACT step: %s", step_name)
        rc = _run_subprocess(cmd, cwd=str(repo), label=f"FACT-{step_name}")
        if rc != 0:
            logger.error("FACT pipeline failed at step: %s", step_name)
            return rc

    return 0


def run_full_evaluation(config: DRBEvalConfig) -> DRBEvalResult:
    """
    Run both RACE and FACT evaluation against generated model outputs.

    Validates prerequisites, deploys patched judge, and runs evaluation.
    """
    import json
    import tempfile

    provider = _get_judge_provider()
    result = DRBEvalResult(model_name=config.model_name, judge_provider=provider)
    repo = Path(config.drb_repo_path)

    # Handle case_id filtering by creating a temporary query file
    temp_dir = None
    if config.case_id:
        try:
            from evals.adapters.deep_research_bench import DeepResearchBenchAdapter
            original_query_file = repo / "data" / "prompt_data" / "query.jsonl"
            
            # Load and filter
            with open(original_query_file, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            
            filtered = [l for l in lines if str(l.get("id")) == config.case_id]
            if not filtered:
                msg = f"Case ID '{config.case_id}' not found in {original_query_file}"
                logger.error(msg)
                result.errors.append(msg)
                return result
            
            # Create temp file
            temp_dir = tempfile.mkdtemp(prefix="drb_eval_")
            temp_query = Path(temp_dir) / "query.jsonl"
            with open(temp_query, "w", encoding="utf-8") as f:
                for item in filtered:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            config.query_file_override = str(temp_query)
            logger.info("Targeted evaluation for case ID %s (temp query: %s)", config.case_id, temp_query)
        except Exception as e:
            msg = f"Failed to create temporary query file: {e}"
            logger.error(msg)
            result.errors.append(msg)
            return result

    try:
        # Prerequisite checks
        repo_errors = _validate_drb_repo(config.drb_repo_path)
        env_errors = _validate_env_vars()
        all_errors = repo_errors + env_errors

        raw_data_path = config.raw_output_path or str(
            repo / "data" / "test_data" / "raw_data" / f"{config.model_name}.jsonl"
        )
        if not Path(raw_data_path).is_file():
            all_errors.append(
                f"Model output file not found: {raw_data_path}\n"
                f"Generate it first with: python -m evals.cli generate --queries ... --model-name {config.model_name}"
            )

        if all_errors:
            result.errors = all_errors
            for err in all_errors:
                logger.error(err)
            return result

        # Deploy patched judge
        try:
            setup_drb_judge(config.drb_repo_path)
        except Exception as e:
            result.errors.append(f"Judge setup failed: {e}")
            logger.error("Judge setup failed: %s", e)
            return result

        # RACE
        logger.info("=== Phase 1: RACE Evaluation for %s (judge: %s) ===", config.model_name, provider)
        result.race_return_code = run_race(config)
        race_result_path = repo / "results" / "race" / config.model_name / "race_result.txt"
        if race_result_path.is_file():
            result.race_result_path = str(race_result_path)

        # FACT
        logger.info("=== Phase 2: FACT Evaluation for %s (judge: %s) ===", config.model_name, provider)
        result.fact_return_code = run_fact(config)
        fact_result_path = repo / "results" / "fact" / config.model_name / "fact_result.txt"
        if fact_result_path.is_file():
            result.fact_result_path = str(fact_result_path)

        return result
    finally:
        # Cleanup temp file
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.debug("Cleaned up temporary query directory: %s", temp_dir)
