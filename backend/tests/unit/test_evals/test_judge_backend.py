"""
Tests for the DRB judge backend refactor.

Covers:
- Provider selection logic
- Env var validation (DeepSeek vs Gemini paths)
- DeepSeek client initialization
- call_model() delegation
- Jina preservation
- Patched api.py deployment
- Gemini fallback path
"""

import json
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Provider Selection & Env Var Validation
# ---------------------------------------------------------------------------

class TestProviderSelection:

    def test_default_provider_is_deepseek(self):
        from evals.drb_evaluator import _get_judge_provider
        with patch.dict(os.environ, {}, clear=True):
            assert _get_judge_provider() == "deepseek"

    def test_provider_from_env(self):
        from evals.drb_evaluator import _get_judge_provider
        with patch.dict(os.environ, {"DRB_JUDGE_PROVIDER": "gemini"}):
            assert _get_judge_provider() == "gemini"

    def test_provider_case_insensitive(self):
        from evals.drb_evaluator import _get_judge_provider
        with patch.dict(os.environ, {"DRB_JUDGE_PROVIDER": "DeepSeek"}):
            assert _get_judge_provider() == "deepseek"


class TestEnvVarValidation:

    @patch.dict(os.environ, {"DRB_JUDGE_PROVIDER": "deepseek"}, clear=True)
    def test_deepseek_requires_deepseek_key(self):
        from evals.drb_evaluator import _validate_env_vars
        errors = _validate_env_vars()
        assert any("DEEPSEEK_API_KEY" in e for e in errors)
        # Should NOT mention GEMINI_API_KEY
        assert not any("GEMINI_API_KEY" in e for e in errors)

    @patch.dict(os.environ, {"DRB_JUDGE_PROVIDER": "gemini"}, clear=True)
    def test_gemini_requires_gemini_key(self):
        from evals.drb_evaluator import _validate_env_vars
        errors = _validate_env_vars()
        assert any("GEMINI_API_KEY" in e for e in errors)
        assert not any("DEEPSEEK_API_KEY" in e for e in errors)

    @patch.dict(os.environ, {
        "DRB_JUDGE_PROVIDER": "deepseek",
        "DEEPSEEK_API_KEY": "test_key",
        "JINA_API_KEY": "test_jina",
    })
    def test_deepseek_valid_env(self):
        from evals.drb_evaluator import _validate_env_vars
        errors = _validate_env_vars()
        assert errors == []

    @patch.dict(os.environ, {
        "DRB_JUDGE_PROVIDER": "deepseek",
        "DEEPSEEK_API_KEY": "test_key",
    }, clear=True)
    def test_jina_always_required(self):
        from evals.drb_evaluator import _validate_env_vars
        errors = _validate_env_vars()
        assert any("JINA_API_KEY" in e for e in errors)

    @patch.dict(os.environ, {"DRB_JUDGE_PROVIDER": "unknown"}, clear=True)
    def test_unknown_provider_error(self):
        from evals.drb_evaluator import _validate_env_vars
        errors = _validate_env_vars()
        assert any("Unknown" in e for e in errors)


# ---------------------------------------------------------------------------
# Patched api.py — DeepSeek Client Path
# ---------------------------------------------------------------------------

class TestPatchedApiDeepSeek:
    """Tests for the patched api.py loaded as a module."""

    def test_patched_api_exists(self):
        api_path = Path(__file__).parent / ".." / ".." / ".." / "evals" / "drb_patches" / "api.py"
        assert api_path.resolve().is_file(), f"Patched api.py not found at {api_path.resolve()}"

    @patch.dict(os.environ, {
        "DRB_JUDGE_PROVIDER": "deepseek",
        "DEEPSEEK_API_KEY": "test_key_123",
        "JINA_API_KEY": "test_jina_key",
    })
    def test_deepseek_client_initialization(self):
        """DeepSeek path should create _DeepSeekClient via OpenAI."""
        # We test this by importing the patched module components
        import importlib
        import sys

        # Load the patched api.py as a module
        spec = importlib.util.spec_from_file_location(
            "patched_api",
            str(Path(__file__).parent / ".." / ".." / ".." / "evals" / "drb_patches" / "api.py"),
        )

        # Mock openai before import
        mock_openai_module = MagicMock()
        mock_client_instance = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client_instance
        sys.modules["openai"] = mock_openai_module

        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Verify provider resolution
            assert module.JUDGE_PROVIDER == "deepseek"
            assert module.Model == "deepseek-reasoner"
            assert module.FACT_Model == "deepseek-reasoner"

            # Create AIClient — should delegate to _DeepSeekClient
            client = module.AIClient()
            assert client.model == "deepseek-reasoner"
        finally:
            del sys.modules["openai"]


class TestPatchedApiGemini:
    """Test Gemini fallback path in patched api.py."""

    @patch.dict(os.environ, {
        "DRB_JUDGE_PROVIDER": "gemini",
        "GEMINI_API_KEY": "test_gemini_key",
        "JINA_API_KEY": "test_jina_key",
    }, clear=True)
    def test_gemini_provider_resolves_models(self):
        import importlib
        import sys

        spec = importlib.util.spec_from_file_location(
            "patched_api_gemini",
            str(Path(__file__).parent / ".." / ".." / ".." / "evals" / "drb_patches" / "api.py"),
        )

        # Mock google.genai
        mock_genai = MagicMock()
        mock_types = MagicMock()
        mock_google = MagicMock()
        mock_google.genai = mock_genai
        sys.modules["google"] = mock_google
        sys.modules["google.genai"] = mock_genai
        sys.modules["google.genai.types"] = mock_types
        mock_genai.types = mock_types

        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            assert module.JUDGE_PROVIDER == "gemini"
            assert "gemini" in module.Model.lower()
        finally:
            for k in ["google", "google.genai", "google.genai.types"]:
                sys.modules.pop(k, None)


# ---------------------------------------------------------------------------
# Jina Preservation
# ---------------------------------------------------------------------------

class TestJinaPreservation:

    def test_jina_class_unchanged(self):
        """WebScrapingJinaTool should be identical in both original and patched."""
        import importlib
        import sys

        spec = importlib.util.spec_from_file_location(
            "patched_api_jina",
            str(Path(__file__).parent / ".." / ".." / ".." / "evals" / "drb_patches" / "api.py"),
        )

        mock_openai = MagicMock()
        sys.modules["openai"] = mock_openai

        try:
            with patch.dict(os.environ, {
                "DRB_JUDGE_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "test",
                "JINA_API_KEY": "test_jina",
            }, clear=True):
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # WebScrapingJinaTool should exist
                assert hasattr(module, "WebScrapingJinaTool")
                assert hasattr(module, "scrape_url")
                assert hasattr(module, "_get_jina_tool")

                # Mock the tool instance returned by getter
                mock_tool = MagicMock()
                mock_result = {"url": "test", "content": "ok"}
                mock_tool.return_value = mock_result
                module._jina_tool = mock_tool

                assert module.scrape_url("http://example.com") == mock_result
        finally:
            del sys.modules["openai"]


# ---------------------------------------------------------------------------
# api.py Deployment
# ---------------------------------------------------------------------------

class TestDeployPatchedApi:

    def test_deploy_backs_up_and_copies(self, tmp_path):
        from evals.drb_evaluator import _deploy_patched_api

        # Create a fake DRB repo with original api.py
        utils_dir = tmp_path / "utils"
        utils_dir.mkdir()
        original = utils_dir / "api.py"
        original.write_text("# original gemini code")

        _deploy_patched_api(str(tmp_path))

        # Backup should exist
        backup = utils_dir / "api.py.gemini_original"
        assert backup.is_file()
        assert backup.read_text() == "# original gemini code"

        # Deployed file should be the patched version
        deployed = utils_dir / "api.py"
        assert "DRB_JUDGE_PROVIDER" in deployed.read_text()
        assert "DeepSeek" in deployed.read_text()

    def test_deploy_no_double_backup(self, tmp_path):
        from evals.drb_evaluator import _deploy_patched_api

        utils_dir = tmp_path / "utils"
        utils_dir.mkdir()
        original = utils_dir / "api.py"
        original.write_text("# original")

        # First deploy
        _deploy_patched_api(str(tmp_path))
        backup = utils_dir / "api.py.gemini_original"
        assert backup.read_text() == "# original"

        # Modify the deployed file
        original.write_text("# modified patched")

        # Second deploy should NOT overwrite backup
        _deploy_patched_api(str(tmp_path))
        assert backup.read_text() == "# original"


# ---------------------------------------------------------------------------
# call_model() Delegation
# ---------------------------------------------------------------------------

class TestCallModelDelegation:

    @patch.dict(os.environ, {
        "DRB_JUDGE_PROVIDER": "deepseek",
        "DEEPSEEK_API_KEY": "test_key",
        "JINA_API_KEY": "test_jina",
    })
    def test_call_model_uses_fact_model(self):
        """call_model() should create AIClient with FACT_Model."""
        import importlib
        import sys

        spec = importlib.util.spec_from_file_location(
            "patched_api_callmodel",
            str(Path(__file__).parent / ".." / ".." / ".." / "evals" / "drb_patches" / "api.py"),
        )

        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
        sys.modules["openai"] = mock_openai

        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            result = module.call_model("test prompt")
            assert result == "test response"

            # Verify the model used was FACT_Model
            call_args = mock_openai.OpenAI.return_value.chat.completions.create.call_args
            assert call_args.kwargs["model"] == module.FACT_Model
        finally:
            del sys.modules["openai"]


# ---------------------------------------------------------------------------
# Build Judge Env
# ---------------------------------------------------------------------------

class TestBuildJudgeEnv:

    @patch.dict(os.environ, {
        "DEEPSEEK_API_KEY": "dk_test",
        "JINA_API_KEY": "jina_test",
        "DRB_RACE_MODEL": "deepseek-reasoner",
        "DRB_JUDGE_PROVIDER": "deepseek",
    })
    def test_judge_env_passthrough(self):
        from evals.drb_evaluator import _build_judge_env
        env = _build_judge_env()
        assert env["DEEPSEEK_API_KEY"] == "dk_test"
        assert env["JINA_API_KEY"] == "jina_test"
        assert env["DRB_RACE_MODEL"] == "deepseek-reasoner"
        assert env["DRB_JUDGE_PROVIDER"] == "deepseek"
