"""Structure tests for evaluation infrastructure.

These tests verify that the evaluation infrastructure is properly set up
without requiring API keys or making external calls.
"""

import pytest
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval


class TestEvaluationInfrastructure:
    """Test suite for evaluation infrastructure setup."""
    
    def test_deepeval_imports(self):
        """Test that DeepEval imports work correctly."""
        # If this test passes, DeepEval is installed correctly
        assert AnswerRelevancyMetric is not None
        assert FaithfulnessMetric is not None
        assert GEval is not None
    
    def test_llm_test_case_creation(self):
        """Test that LLMTestCase can be created."""
        test_case = LLMTestCase(
            input="Test input",
            actual_output="Test output",
        )
        
        assert test_case.input == "Test input"
        assert test_case.actual_output == "Test output"
    
    def test_llm_test_case_with_context(self):
        """Test that LLMTestCase with context can be created."""
        test_case = LLMTestCase(
            input="Test input",
            actual_output="Test output",
            retrieval_context=["Context 1", "Context 2"],
        )
        
        assert test_case.input == "Test input"
        assert test_case.actual_output == "Test output"
        assert test_case.retrieval_context == ["Context 1", "Context 2"]
    
    def test_llm_test_case_with_expected_output(self):
        """Test that LLMTestCase with expected output can be created."""
        test_case = LLMTestCase(
            input="Test input",
            actual_output="Test output",
            expected_output="Expected output",
        )
        
        assert test_case.input == "Test input"
        assert test_case.actual_output == "Test output"
        assert test_case.expected_output == "Expected output"


class TestFixtures:
    """Test suite for evaluation fixtures."""
    
    def test_sample_llm_test_case_fixture(self, sample_llm_test_case: LLMTestCase):
        """Test that sample_llm_test_case fixture works."""
        assert sample_llm_test_case.input is not None
        assert sample_llm_test_case.actual_output is not None
    
    def test_sample_llm_test_case_with_context_fixture(
        self, sample_llm_test_case_with_context: LLMTestCase
    ):
        """Test that sample_llm_test_case_with_context fixture works."""
        assert sample_llm_test_case_with_context.input is not None
        assert sample_llm_test_case_with_context.actual_output is not None
        assert sample_llm_test_case_with_context.retrieval_context is not None
        assert len(sample_llm_test_case_with_context.retrieval_context) > 0
    
    def test_sample_research_brief_fixture(self, sample_research_brief):
        """Test that sample_research_brief fixture works."""
        assert sample_research_brief.scope is not None
        assert len(sample_research_brief.sub_topics) > 0
        assert sample_research_brief.constraints is not None
    
    def test_sample_citation_fixture(self, sample_citation):
        """Test that sample_citation fixture works."""
        assert sample_citation.title is not None
        assert sample_citation.author is not None
        assert len(sample_citation.author) > 0
    
    def test_evaluation_thresholds_fixture(self, evaluation_thresholds):
        """Test that evaluation_thresholds fixture works."""
        assert "answer_relevance" in evaluation_thresholds
        assert "faithfulness" in evaluation_thresholds
        assert "context_recall" in evaluation_thresholds
        assert "format_compliance" in evaluation_thresholds
        
        # Verify thresholds are reasonable
        assert 0.0 <= evaluation_thresholds["answer_relevance"] <= 1.0
        assert 0.0 <= evaluation_thresholds["faithfulness"] <= 1.0
        assert 0.0 <= evaluation_thresholds["context_recall"] <= 1.0
        assert 0.0 <= evaluation_thresholds["format_compliance"] <= 1.0


class TestHelperFunctions:
    """Test suite for helper functions."""
    
    def test_create_test_case_helper(self):
        """Test that create_test_case helper function works."""
        from tests.evaluation.conftest import create_test_case
        
        test_case = create_test_case(
            input_text="Test input",
            output_text="Test output",
        )
        
        assert test_case.input == "Test input"
        assert test_case.actual_output == "Test output"
    
    def test_create_test_case_with_context(self):
        """Test create_test_case with context."""
        from tests.evaluation.conftest import create_test_case
        
        test_case = create_test_case(
            input_text="Test input",
            output_text="Test output",
            context=["Context 1", "Context 2"],
        )
        
        assert test_case.input == "Test input"
        assert test_case.actual_output == "Test output"
        assert test_case.retrieval_context == ["Context 1", "Context 2"]
    
    def test_create_test_case_with_expected_output(self):
        """Test create_test_case with expected output."""
        from tests.evaluation.conftest import create_test_case
        
        test_case = create_test_case(
            input_text="Test input",
            output_text="Test output",
            expected_output="Expected output",
        )
        
        assert test_case.input == "Test input"
        assert test_case.actual_output == "Test output"
        assert test_case.expected_output == "Expected output"

