"""Unit tests for Report Agent prompt templates and format instruction functions.

Tests ensure that report prompts are correctly structured and format instruction
functions return properly formatted markdown templates.
"""

import pytest
from langchain_core.prompts import ChatPromptTemplate

from app.prompts.report_prompts import (
    REPORT_GENERATION_TEMPLATE,
    get_summary_format_instructions,
    get_comparison_format_instructions,
    get_literature_review_instructions,
    get_gap_analysis_instructions,
    get_fact_validation_instructions,
    get_ranking_format_instructions,
)


class TestReportGenerationTemplate:
    """Test cases for REPORT_GENERATION_TEMPLATE."""

    def test_template_is_chat_prompt_template(self):
        """Test that template is a ChatPromptTemplate instance."""
        assert isinstance(REPORT_GENERATION_TEMPLATE, ChatPromptTemplate)

    def test_template_has_required_input_variables(self):
        """Test that template contains required input variables."""
        input_vars = REPORT_GENERATION_TEMPLATE.input_variables
        assert "research_brief" in input_vars
        assert "summarized_findings" in input_vars
        assert "format_type" in input_vars

    def test_template_has_exactly_three_input_variables(self):
        """Test that template has exactly three input variables."""
        input_vars = REPORT_GENERATION_TEMPLATE.input_variables
        assert len(input_vars) == 3

    def test_template_formats_with_valid_inputs(self):
        """Test that template formats correctly with valid inputs."""
        formatted = REPORT_GENERATION_TEMPLATE.format_messages(
            research_brief="Brief description",
            summarized_findings="Finding 1, Finding 2",
            format_type="summary"
        )
        assert len(formatted) == 2  # System + Human messages
        assert formatted[0].type == "system"
        assert formatted[1].type == "human"

    def test_template_system_message_contains_guidelines(self):
        """Test that system message contains report generation guidelines."""
        formatted = REPORT_GENERATION_TEMPLATE.format_messages(
            research_brief="Test",
            summarized_findings="Test",
            format_type="summary"
        )
        system_msg = formatted[0].content
        assert "report generator" in system_msg.lower()
        assert "markdown" in system_msg.lower()
        assert "citation" in system_msg.lower()

    def test_template_human_message_contains_all_inputs(self):
        """Test that human message contains all input data."""
        test_brief = "Research machine learning applications"
        test_findings = "Finding: ML is widely used"
        test_format = "summary"
        
        formatted = REPORT_GENERATION_TEMPLATE.format_messages(
            research_brief=test_brief,
            summarized_findings=test_findings,
            format_type=test_format
        )
        human_msg = formatted[1].content
        assert test_brief in human_msg
        assert test_findings in human_msg
        assert test_format in human_msg

    def test_template_with_empty_findings(self):
        """Test that template handles empty findings."""
        formatted = REPORT_GENERATION_TEMPLATE.format_messages(
            research_brief="Brief",
            summarized_findings="",
            format_type="summary"
        )
        assert len(formatted) == 2
        # Should not raise errors

    def test_template_with_long_inputs(self):
        """Test that template handles long inputs."""
        long_brief = "Research " * 1000
        long_findings = "Finding " * 1000
        
        formatted = REPORT_GENERATION_TEMPLATE.format_messages(
            research_brief=long_brief,
            summarized_findings=long_findings,
            format_type="summary"
        )
        assert len(formatted) == 2


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

