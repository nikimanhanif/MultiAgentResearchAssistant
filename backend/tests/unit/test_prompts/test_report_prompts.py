"""Unit tests for Report Agent prompt templates and format instruction functions.

Tests ensure that report prompts are correctly structured and format instruction
functions return properly formatted markdown templates.
"""

import pytest
from langchain_core.prompts import ChatPromptTemplate

from app.prompts.report_prompts import (
    get_report_generation_prompt,
    format_findings_for_prompt,
    get_summary_format_instructions,
    get_comparison_format_instructions,
    get_literature_review_instructions,
    get_gap_analysis_instructions,
    get_fact_validation_instructions,
    get_ranking_format_instructions,
)


class TestReportGenerationPrompt:
    """Test cases for get_report_generation_prompt function."""

    def test_function_returns_chat_prompt_template(self):
        """Test that function returns a ChatPromptTemplate instance."""
        template = get_report_generation_prompt()
        assert isinstance(template, ChatPromptTemplate)

    def test_template_has_required_input_variables(self):
        """Test that template contains required input variables."""
        template = get_report_generation_prompt()
        input_vars = template.input_variables
        assert "brief_scope" in input_vars
        assert "findings_context" in input_vars
        assert "format_instructions" in input_vars
        assert "reviewer_feedback" in input_vars

    def test_template_formats_with_valid_inputs(self):
        """Test that template formats correctly with valid inputs."""
        template = get_report_generation_prompt()
        formatted = template.format_messages(
            brief_scope="Research scope",
            brief_subtopics="- Topic 1\n- Topic 2",
            brief_constraints="Time: 2020-2024",
            brief_format="summary",
            findings_context="Finding 1\nFinding 2",
            format_instructions="Summary format instructions",
            reviewer_feedback="No feedback"
        )
        assert len(formatted) == 2  # System + Human messages
        assert formatted[0].type == "system"
        assert formatted[1].type == "human"

    def test_template_system_message_contains_guidelines(self):
        """Test that system message contains report generation guidelines."""
        template = get_report_generation_prompt()
        formatted = template.format_messages(
            brief_scope="Test",
            brief_subtopics="Topic",
            brief_constraints="None",
            brief_format="summary",
            findings_context="Test findings",
            format_instructions="Test instructions",
            reviewer_feedback=""
        )
        system_msg = formatted[0].content
        assert "report" in system_msg.lower()
        assert "markdown" in system_msg.lower()
        assert "citation" in system_msg.lower()

    def test_template_human_message_contains_all_inputs(self):
        """Test that human message contains all input data."""
        template = get_report_generation_prompt()
        test_scope = "Research machine learning applications"
        test_findings = "Finding: ML is widely used"
        test_format = "summary"
        
        formatted = template.format_messages(
            brief_scope=test_scope,
            brief_subtopics="AI, ML",
            brief_constraints="2020-2024",
            brief_format=test_format,
            findings_context=test_findings,
            format_instructions="Summary instructions",
            reviewer_feedback=""
        )
        human_msg = formatted[1].content
        assert test_scope in human_msg
        assert test_findings in human_msg
        assert test_format in human_msg

    def test_template_with_empty_findings(self):
        """Test that template handles empty findings."""
        template = get_report_generation_prompt()
        formatted = template.format_messages(
            brief_scope="Brief",
            brief_subtopics="Topic",
            brief_constraints="None",
            brief_format="summary",
            findings_context="",
            format_instructions="Instructions",
            reviewer_feedback=""
        )
        assert len(formatted) == 2

    def test_template_with_reviewer_feedback(self):
        """Test that template includes reviewer feedback when provided."""
        template = get_report_generation_prompt()
        feedback = "Please add more details about recent developments"
        formatted = template.format_messages(
            brief_scope="Test scope",
            brief_subtopics="Topics",
            brief_constraints="Constraints",
            brief_format="summary",
            findings_context="Findings",
            format_instructions="Instructions",
            reviewer_feedback=feedback
        )
        human_msg = formatted[1].content
        assert feedback in human_msg or "feedback" in human_msg.lower()


class TestSummaryFormatInstructions:
    """Test cases for get_summary_format_instructions function."""

    def test_function_returns_string(self):
        """Test that function returns a string."""
        result = get_summary_format_instructions()
        assert isinstance(result, str)

    def test_function_returns_non_empty_string(self):
        """Test that function returns non-empty string."""
        result = get_summary_format_instructions()
        assert len(result) > 0

    def test_instructions_contain_summary_keywords(self):
        """Test that instructions contain summary-related keywords."""
        result = get_summary_format_instructions()
        assert "summary" in result.lower()
        assert "executive" in result.lower() or "overview" in result.lower()

    def test_instructions_contain_markdown_formatting(self):
        """Test that instructions show markdown formatting examples."""
        result = get_summary_format_instructions()
        assert "#" in result  # Markdown headers
        assert "##" in result or "###" in result

    def test_instructions_mention_references(self):
        """Test that instructions mention references section."""
        result = get_summary_format_instructions()
        assert "reference" in result.lower() or "citation" in result.lower()

    def test_instructions_are_consistent_across_calls(self):
        """Test that function returns same result on multiple calls."""
        result1 = get_summary_format_instructions()
        result2 = get_summary_format_instructions()
        assert result1 == result2


class TestComparisonFormatInstructions:
    """Test cases for get_comparison_format_instructions function."""

    def test_function_returns_string(self):
        """Test that function returns a string."""
        result = get_comparison_format_instructions()
        assert isinstance(result, str)

    def test_function_returns_non_empty_string(self):
        """Test that function returns non-empty string."""
        result = get_comparison_format_instructions()
        assert len(result) > 0

    def test_instructions_contain_comparison_keywords(self):
        """Test that instructions contain comparison-related keywords."""
        result = get_comparison_format_instructions()
        assert "comparison" in result.lower() or "compare" in result.lower()

    def test_instructions_contain_table_format(self):
        """Test that instructions show table formatting."""
        result = get_comparison_format_instructions()
        assert "|" in result  # Markdown table syntax

    def test_instructions_mention_criteria(self):
        """Test that instructions mention comparison criteria."""
        result = get_comparison_format_instructions()
        assert "criteria" in result.lower() or "criterion" in result.lower()


class TestLiteratureReviewInstructions:
    """Test cases for get_literature_review_instructions function."""

    def test_function_returns_string(self):
        """Test that function returns a string."""
        result = get_literature_review_instructions()
        assert isinstance(result, str)

    def test_function_returns_non_empty_string(self):
        """Test that function returns non-empty string."""
        result = get_literature_review_instructions()
        assert len(result) > 0

    def test_instructions_contain_literature_review_keywords(self):
        """Test that instructions contain literature review keywords."""
        result = get_literature_review_instructions()
        assert "literature" in result.lower()
        assert "review" in result.lower()

    def test_instructions_mention_thematic_sections(self):
        """Test that instructions mention thematic organization."""
        result = get_literature_review_instructions()
        assert "theme" in result.lower() or "topic" in result.lower()

    def test_instructions_mention_research_gaps(self):
        """Test that instructions mention research gaps section."""
        result = get_literature_review_instructions()
        assert "gap" in result.lower()

    def test_instructions_mention_credibility(self):
        """Test that instructions mention credibility indicators."""
        result = get_literature_review_instructions()
        assert "credibility" in result.lower() or "score" in result.lower()


class TestGapAnalysisInstructions:
    """Test cases for get_gap_analysis_instructions function."""

    def test_function_returns_string(self):
        """Test that function returns a string."""
        result = get_gap_analysis_instructions()
        assert isinstance(result, str)

    def test_function_returns_non_empty_string(self):
        """Test that function returns non-empty string."""
        result = get_gap_analysis_instructions()
        assert len(result) > 0

    def test_instructions_contain_gap_analysis_keywords(self):
        """Test that instructions contain gap analysis keywords."""
        result = get_gap_analysis_instructions()
        assert "gap" in result.lower()
        assert "analysis" in result.lower()

    def test_instructions_mention_coverage_analysis(self):
        """Test that instructions mention coverage analysis."""
        result = get_gap_analysis_instructions()
        assert "coverage" in result.lower()

    def test_instructions_mention_recommendations(self):
        """Test that instructions mention recommendations section."""
        result = get_gap_analysis_instructions()
        assert "recommendation" in result.lower()

    def test_instructions_mention_gap_categories(self):
        """Test that instructions mention different gap categories."""
        result = get_gap_analysis_instructions()
        result_lower = result.lower()
        # Should mention at least some gap types
        gap_types = ["coverage", "depth", "temporal", "perspective"]
        assert any(gap_type in result_lower for gap_type in gap_types)


class TestFactValidationInstructions:
    """Test cases for get_fact_validation_instructions function."""

    def test_function_returns_string(self):
        """Test that function returns a string."""
        result = get_fact_validation_instructions()
        assert isinstance(result, str)

    def test_function_returns_non_empty_string(self):
        """Test that function returns non-empty string."""
        result = get_fact_validation_instructions()
        assert len(result) > 0

    def test_instructions_contain_validation_keywords(self):
        """Test that instructions contain validation keywords."""
        result = get_fact_validation_instructions()
        assert "validation" in result.lower() or "validate" in result.lower()

    def test_instructions_mention_claims(self):
        """Test that instructions mention claims or statements."""
        result = get_fact_validation_instructions()
        assert "claim" in result.lower() or "statement" in result.lower()

    def test_instructions_mention_credibility_scores(self):
        """Test that instructions mention credibility scoring."""
        result = get_fact_validation_instructions()
        assert "credibility" in result.lower() and "score" in result.lower()

    def test_instructions_show_validation_indicators(self):
        """Test that instructions show validation result indicators."""
        result = get_fact_validation_instructions()
        # Should show some kind of validation indicators (checkmarks, warnings, etc.)
        assert any(indicator in result for indicator in ["✅", "⚠️", "❌", "Supported"])


class TestRankingFormatInstructions:
    """Test cases for get_ranking_format_instructions function."""

    def test_function_returns_string(self):
        """Test that function returns a string."""
        result = get_ranking_format_instructions()
        assert isinstance(result, str)

    def test_function_returns_non_empty_string(self):
        """Test that function returns non-empty string."""
        result = get_ranking_format_instructions()
        assert len(result) > 0

    def test_instructions_contain_ranking_keywords(self):
        """Test that instructions contain ranking keywords."""
        result = get_ranking_format_instructions()
        assert "ranking" in result.lower() or "rank" in result.lower()

    def test_instructions_show_numbered_structure(self):
        """Test that instructions show numbered ranking structure."""
        result = get_ranking_format_instructions()
        # Should show numbered rankings
        assert any(str(i) in result for i in range(1, 4))

    def test_instructions_mention_justification(self):
        """Test that instructions mention justification for rankings."""
        result = get_ranking_format_instructions()
        assert "justification" in result.lower() or "reason" in result.lower()


class TestFormatInstructionsIntegration:
    """Integration tests for format instruction functions."""

    def test_all_functions_return_distinct_results(self):
        """Test that each format function returns different instructions."""
        summary = get_summary_format_instructions()
        comparison = get_comparison_format_instructions()
        literature = get_literature_review_instructions()
        gap = get_gap_analysis_instructions()
        fact = get_fact_validation_instructions()
        ranking = get_ranking_format_instructions()
        
        results = [summary, comparison, literature, gap, fact, ranking]
        # All should be different
        assert len(set(results)) == len(results)

    def test_all_functions_return_markdown_formatted_strings(self):
        """Test that all format functions return markdown-formatted strings."""
        functions = [
            get_summary_format_instructions,
            get_comparison_format_instructions,
            get_literature_review_instructions,
            get_gap_analysis_instructions,
            get_fact_validation_instructions,
            get_ranking_format_instructions,
        ]
        
        for func in functions:
            result = func()
            # All should contain markdown headers
            assert "#" in result

    def test_all_functions_mention_references_or_citations(self):
        """Test that all format functions mention references or citations."""
        functions = [
            get_summary_format_instructions,
            get_comparison_format_instructions,
            get_literature_review_instructions,
            get_gap_analysis_instructions,
            get_fact_validation_instructions,
            get_ranking_format_instructions,
        ]
        
        for func in functions:
            result = func().lower()
            # All should mention references, citations, or bibliography
            assert any(term in result for term in ["reference", "citation", "bibliography"])

    def test_all_functions_are_deterministic(self):
        """Test that all functions return consistent results."""
        functions = [
            get_summary_format_instructions,
            get_comparison_format_instructions,
            get_literature_review_instructions,
            get_gap_analysis_instructions,
            get_fact_validation_instructions,
            get_ranking_format_instructions,
        ]
        
        for func in functions:
            result1 = func()
            result2 = func()
            assert result1 == result2

