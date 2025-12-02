"""Unit tests for app.prompts module initialization and exports."""

import pytest


class TestPromptsModuleExports:
    """Test cases for app.prompts module exports."""

    def test_module_imports_successfully(self):
        """Test that app.prompts module can be imported."""
        import app.prompts
        assert app.prompts is not None

    def test_scope_question_generation_template_exported(self):
        """Test that SCOPE_QUESTION_GENERATION_TEMPLATE is exported."""
        from app.prompts import SCOPE_QUESTION_GENERATION_TEMPLATE
        assert SCOPE_QUESTION_GENERATION_TEMPLATE is not None

    def test_scope_completion_detection_template_exported(self):
        """Test that SCOPE_COMPLETION_DETECTION_TEMPLATE is exported."""
        from app.prompts import SCOPE_COMPLETION_DETECTION_TEMPLATE
        assert SCOPE_COMPLETION_DETECTION_TEMPLATE is not None

    def test_scope_brief_generation_template_exported(self):
        """Test that SCOPE_BRIEF_GENERATION_TEMPLATE is exported."""
        from app.prompts import SCOPE_BRIEF_GENERATION_TEMPLATE
        assert SCOPE_BRIEF_GENERATION_TEMPLATE is not None

    def test_all_exports_in_dunder_all(self):
        """Test that __all__ contains all exported templates."""
        from app.prompts import __all__
        
        expected_exports = [
            "SCOPE_QUESTION_GENERATION_TEMPLATE",
            "SCOPE_COMPLETION_DETECTION_TEMPLATE",
            "SCOPE_BRIEF_GENERATION_TEMPLATE",
        ]
        
        for export in expected_exports:
            assert export in __all__

    def test_no_extra_exports_in_dunder_all(self):
        """Test that __all__ only contains expected exports."""
        from app.prompts import __all__
        
        expected_exports = [
            # Scope prompts
            "SCOPE_QUESTION_GENERATION_TEMPLATE",
            "SCOPE_COMPLETION_DETECTION_TEMPLATE",
            "SCOPE_BRIEF_GENERATION_TEMPLATE",
            # Report prompts
            "get_report_generation_prompt",
            # Research prompts (Phase 8)
            "CREDIBILITY_HEURISTICS",
            "RESEARCH_STRATEGY_SELECTION_TEMPLATE",
            "RESEARCH_TASK_DECOMPOSITION_TEMPLATE",
            "RESEARCH_ERROR_RE_DELEGATION_TEMPLATE",
            "RESEARCH_FINDINGS_COMPRESSION_TEMPLATE",
            "SUPERVISOR_GAP_ANALYSIS_TEMPLATE",
            "SUPERVISOR_FINDINGS_AGGREGATION_TEMPLATE",
            "SUB_AGENT_RESEARCH_TEMPLATE",
            "SUB_AGENT_CITATION_EXTRACTION_TEMPLATE",
        ]
        
        assert len(__all__) == len(expected_exports)
        for export in expected_exports:
            assert export in __all__


class TestScopePromptsSubmodule:
    """Test cases for app.prompts.scope_prompts submodule."""

    def test_scope_prompts_module_imports(self):
        """Test that scope_prompts submodule can be imported."""
        from app.prompts import scope_prompts
        assert scope_prompts is not None

    def test_direct_import_from_scope_prompts(self):
        """Test that templates can be imported directly from scope_prompts."""
        from app.prompts.scope_prompts import (
            SCOPE_QUESTION_GENERATION_TEMPLATE,
            SCOPE_COMPLETION_DETECTION_TEMPLATE,
            SCOPE_BRIEF_GENERATION_TEMPLATE,
        )
        
        assert SCOPE_QUESTION_GENERATION_TEMPLATE is not None
        assert SCOPE_COMPLETION_DETECTION_TEMPLATE is not None
        assert SCOPE_BRIEF_GENERATION_TEMPLATE is not None

    def test_imports_are_consistent(self):
        """Test that imports from app.prompts match imports from scope_prompts."""
        from app.prompts import (
            SCOPE_QUESTION_GENERATION_TEMPLATE as template1_v1,
            SCOPE_COMPLETION_DETECTION_TEMPLATE as template2_v1,
            SCOPE_BRIEF_GENERATION_TEMPLATE as template3_v1,
        )
        from app.prompts.scope_prompts import (
            SCOPE_QUESTION_GENERATION_TEMPLATE as template1_v2,
            SCOPE_COMPLETION_DETECTION_TEMPLATE as template2_v2,
            SCOPE_BRIEF_GENERATION_TEMPLATE as template3_v2,
        )
        
        # Should be the same objects
        assert template1_v1 is template1_v2
        assert template2_v1 is template2_v2
        assert template3_v1 is template3_v2


class TestResearchPromptsSubmodule:
    """Test cases for app.prompts.research_prompts submodule."""

    def test_research_prompts_module_imports(self):
        """Test that research_prompts submodule can be imported."""
        from app.prompts import research_prompts
        assert research_prompts is not None

    def test_research_strategy_template_exists(self):
        """Test that RESEARCH_STRATEGY_SELECTION_TEMPLATE exists."""
        from app.prompts.research_prompts import RESEARCH_STRATEGY_SELECTION_TEMPLATE
        assert RESEARCH_STRATEGY_SELECTION_TEMPLATE is not None

    def test_task_decomposition_template_exists(self):
        """Test that RESEARCH_TASK_DECOMPOSITION_TEMPLATE exists."""
        from app.prompts.research_prompts import RESEARCH_TASK_DECOMPOSITION_TEMPLATE
        assert RESEARCH_TASK_DECOMPOSITION_TEMPLATE is not None

    def test_error_re_delegation_template_exists(self):
        """Test that RESEARCH_ERROR_RE_DELEGATION_TEMPLATE exists."""
        from app.prompts.research_prompts import RESEARCH_ERROR_RE_DELEGATION_TEMPLATE
        assert RESEARCH_ERROR_RE_DELEGATION_TEMPLATE is not None

    def test_findings_compression_template_exists(self):
        """Test that RESEARCH_FINDINGS_COMPRESSION_TEMPLATE exists."""
        from app.prompts.research_prompts import RESEARCH_FINDINGS_COMPRESSION_TEMPLATE
        assert RESEARCH_FINDINGS_COMPRESSION_TEMPLATE is not None


class TestReportPromptsSubmodule:
    """Test cases for app.prompts.report_prompts submodule."""

    def test_report_prompts_module_imports(self):
        """Test that report_prompts submodule can be imported."""
        from app.prompts import report_prompts
        assert report_prompts is not None

    def test_report_generation_template_exists(self):
        """Test that report generation prompt function exists."""
        from app.prompts.report_prompts import get_report_generation_prompt
        assert callable(get_report_generation_prompt)

    def test_all_format_instruction_functions_exist(self):
        """Test that all format instruction functions are defined."""
        from app.prompts.report_prompts import (
            get_summary_format_instructions,
            get_comparison_format_instructions,
            get_literature_review_instructions,
            get_gap_analysis_instructions,
            get_fact_validation_instructions,
            get_ranking_format_instructions,
        )
        
        # All should be callable
        assert callable(get_summary_format_instructions)
        assert callable(get_comparison_format_instructions)
        assert callable(get_literature_review_instructions)
        assert callable(get_gap_analysis_instructions)
        assert callable(get_fact_validation_instructions)
        assert callable(get_ranking_format_instructions)


class TestPromptsModuleStructure:
    """Test cases for overall prompts module structure."""

    def test_no_circular_imports(self):
        """Test that there are no circular import issues."""
        # This test passes if imports don't raise ImportError
        from app.prompts import scope_prompts
        from app.prompts import research_prompts
        from app.prompts import report_prompts
        from app.prompts import (
            SCOPE_QUESTION_GENERATION_TEMPLATE,
            SCOPE_COMPLETION_DETECTION_TEMPLATE,
            SCOPE_BRIEF_GENERATION_TEMPLATE,
        )
        
        assert True  # If we get here, no circular imports

    def test_module_has_docstring(self):
        """Test that prompts module has a docstring."""
        import app.prompts
        assert app.prompts.__doc__ is not None
        assert len(app.prompts.__doc__) > 0

    def test_submodules_have_docstrings(self):
        """Test that all submodules have docstrings."""
        from app.prompts import scope_prompts, research_prompts, report_prompts
        
        assert scope_prompts.__doc__ is not None
        assert len(scope_prompts.__doc__) > 0
        
        assert research_prompts.__doc__ is not None
        assert len(research_prompts.__doc__) > 0
        
        assert report_prompts.__doc__ is not None
        assert len(report_prompts.__doc__) > 0

