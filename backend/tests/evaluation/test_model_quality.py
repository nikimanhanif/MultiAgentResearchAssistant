"""Model quality evaluation tests using DeepEval metrics with Gemini.

This module tests the quality of LLM outputs using DeepEval's built-in metrics:
- AnswerRelevance: Measures if outputs are relevant to inputs
- Faithfulness: Detects hallucinations (claims not in source material)
- ContextRecall: Measures if relevant information was retrieved

These tests establish baseline quality metrics for the system.

**Important**: These tests use REAL LLM API calls (Google Gemini 2.5 Flash)
to evaluate quality. They will incur API costs but are essential for
proper evaluation of your system's outputs.
"""

import pytest
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
)


class TestAnswerRelevance:
    """Test suite for AnswerRelevance metric.
    
    AnswerRelevance measures if the model's output is relevant to the input query.
    Threshold: > 0.7 (70% relevance required)
    
    Uses Gemini 2.5 Flash as the evaluation judge.
    """
    
    def test_answer_relevance_basic(
        self, sample_llm_test_case: LLMTestCase, gemini_evaluation_model
    ):
        """Test that basic LLM responses are relevant to the input query.
        
        Uses the AnswerRelevancyMetric with Gemini to ensure the output addresses the input.
        """
        metric = AnswerRelevancyMetric(
            model=gemini_evaluation_model,
            threshold=0.7
        )
        
        # Measure the test case
        metric.measure(sample_llm_test_case)
        
        # Assert the metric passes the threshold
        assert metric.score >= 0.7, (
            f"Answer relevance score {metric.score} is below threshold 0.7. "
            f"Reason: {metric.reason}"
        )
    
    def test_answer_relevance_clarification_question(self, gemini_evaluation_model):
        """Test that clarification questions are relevant to vague user queries.
        
        This tests the Scope Agent's ability to ask relevant clarifying questions.
        """
        test_case = LLMTestCase(
            input="I need to research AI",
            actual_output=(
                "What specific aspect of AI are you interested in? "
                "(e.g., machine learning, natural language processing, computer vision)"
            ),
        )
        
        metric = AnswerRelevancyMetric(
            model=gemini_evaluation_model,
            threshold=0.7
        )
        metric.measure(test_case)
        
        assert metric.score >= 0.7, (
            f"Clarification question relevance score {metric.score} is below threshold. "
            f"Reason: {metric.reason}"
        )
    
    def test_answer_relevance_with_context(
        self, sample_llm_test_case_with_context: LLMTestCase, gemini_evaluation_model
    ):
        """Test answer relevance when retrieval context is provided.
        
        Ensures the model uses retrieved context to provide relevant answers.
        """
        metric = AnswerRelevancyMetric(
            model=gemini_evaluation_model,
            threshold=0.7
        )
        metric.measure(sample_llm_test_case_with_context)
        
        assert metric.score >= 0.7, (
            f"Answer relevance with context score {metric.score} is below threshold. "
            f"Reason: {metric.reason}"
        )


class TestFaithfulness:
    """Test suite for Faithfulness metric.
    
    Faithfulness detects hallucinations by ensuring outputs only contain
    information present in the source material (retrieval context).
    Threshold: > 0.9 (90% faithfulness required - strict to prevent hallucinations)
    
    Uses Gemini 2.5 Flash as the evaluation judge.
    """
    
    def test_faithfulness_with_context(
        self, sample_llm_test_case_with_context: LLMTestCase, gemini_evaluation_model
    ):
        """Test that outputs are faithful to the retrieval context.
        
        Ensures the model doesn't hallucinate information not in the context.
        """
        metric = FaithfulnessMetric(
            model=gemini_evaluation_model,
            threshold=0.9
        )
        metric.measure(sample_llm_test_case_with_context)
        
        assert metric.score >= 0.9, (
            f"Faithfulness score {metric.score} is below threshold 0.9. "
            f"Reason: {metric.reason}. "
            "This indicates potential hallucination."
        )
    
    def test_faithfulness_research_brief_generation(self, gemini_evaluation_model):
        """Test that research briefs are faithful to user inputs.
        
        Ensures the Scope Agent doesn't add information not provided by the user.
        """
        test_case = LLMTestCase(
            input=(
                "User query: AI in healthcare\n"
                "User clarifications:\n"
                "- Aspect: Medical diagnostics\n"
                "- Time period: Last 5 years\n"
                "- Format: Literature review"
            ),
            actual_output=(
                "Research Brief:\n"
                "Scope: AI applications in medical diagnostics\n"
                "Time range: 2019-2024\n"
                "Format: Literature review\n"
                "Sub-topics: Deep learning for medical imaging, "
                "NLP for clinical notes, Predictive models for diagnosis"
            ),
            retrieval_context=[
                "User query: AI in healthcare",
                "User clarifications: Aspect: Medical diagnostics",
                "User clarifications: Time period: Last 5 years",
                "User clarifications: Format: Literature review",
            ],
        )
        
        metric = FaithfulnessMetric(
            model=gemini_evaluation_model,
            threshold=0.9
        )
        metric.measure(test_case)
        
        assert metric.score >= 0.9, (
            f"Research brief faithfulness score {metric.score} is below threshold. "
            f"Reason: {metric.reason}"
        )
    
    def test_faithfulness_report_generation(self, gemini_evaluation_model):
        """Test that reports are faithful to research findings.
        
        Ensures the Report Agent doesn't hallucinate beyond provided findings.
        """
        test_case = LLMTestCase(
            input="Generate a summary report from the research findings.",
            actual_output=(
                "Key Findings:\n"
                "1. Deep learning models achieve 95% accuracy in detecting lung cancer\n"
                "2. NLP models extract clinical information with 88% precision\n"
                "3. Predictive models show promise in early disease detection"
            ),
            retrieval_context=[
                "Deep learning models achieve 95% accuracy in detecting lung cancer from CT scans",
                "NLP models can extract clinical information from unstructured notes with 88% precision",
                "Predictive models show promise in early disease detection",
            ],
        )
        
        metric = FaithfulnessMetric(
            model=gemini_evaluation_model,
            threshold=0.9
        )
        metric.measure(test_case)
        
        assert metric.score >= 0.9, (
            f"Report faithfulness score {metric.score} is below threshold. "
            f"Reason: {metric.reason}"
        )


class TestContextRecall:
    """Test suite for ContextualRecall metric.
    
    ContextualRecall measures if the retrieval system found relevant information.
    Threshold: > 0.7 (70% of relevant information should be retrieved)
    
    Uses Gemini 2.5 Flash as the evaluation judge.
    """
    
    def test_context_recall_basic(
        self, sample_llm_test_case_with_context: LLMTestCase, gemini_evaluation_model
    ):
        """Test that retrieval context contains relevant information.
        
        Ensures the Research Agent retrieves pertinent information.
        """
        metric = ContextualRecallMetric(
            model=gemini_evaluation_model,
            threshold=0.7
        )
        metric.measure(sample_llm_test_case_with_context)
        
        assert metric.score >= 0.7, (
            f"Context recall score {metric.score} is below threshold 0.7. "
            f"Reason: {metric.reason}. "
            "This indicates insufficient retrieval coverage."
        )
    
    def test_context_recall_research_findings(self, gemini_evaluation_model):
        """Test that research findings cover the research brief requirements.
        
        Ensures the Research Agent retrieves information for all sub-topics.
        """
        test_case = LLMTestCase(
            input=(
                "Research Brief: AI in medical diagnostics\n"
                "Sub-topics: Deep learning for medical imaging, "
                "NLP for clinical notes, Predictive models"
            ),
            actual_output=(
                "Found research covering:\n"
                "1. Deep learning applications in medical imaging (15 papers)\n"
                "2. NLP techniques for clinical notes (12 papers)\n"
                "3. Predictive models for disease diagnosis (10 papers)"
            ),
            retrieval_context=[
                "Deep learning models achieve 95% accuracy in detecting lung cancer",
                "Convolutional neural networks for medical image segmentation",
                "NLP models extract clinical information with 88% precision",
                "Transformer models for clinical text understanding",
                "Predictive models show promise in early disease detection",
                "Machine learning for risk prediction in healthcare",
            ],
            expected_output=(
                "Research should cover all three sub-topics: "
                "deep learning for medical imaging, NLP for clinical notes, "
                "and predictive models for diagnosis"
            ),
        )
        
        metric = ContextualRecallMetric(
            model=gemini_evaluation_model,
            threshold=0.7
        )
        metric.measure(test_case)
        
        assert metric.score >= 0.7, (
            f"Research findings context recall score {metric.score} is below threshold. "
            f"Reason: {metric.reason}"
        )


class TestModelQualityIntegration:
    """Integration tests combining multiple DeepEval metrics.
    
    These tests evaluate multiple quality dimensions simultaneously using Gemini.
    """
    
    def test_complete_pipeline_quality(self, gemini_evaluation_model):
        """Test quality metrics across a complete research pipeline.
        
        Evaluates AnswerRelevance, Faithfulness, and ContextRecall together.
        """
        test_case = LLMTestCase(
            input="What are the main applications of deep learning in medical imaging?",
            actual_output=(
                "Deep learning is primarily used in medical imaging for:\n"
                "1. Disease detection with 95% accuracy in lung cancer screening\n"
                "2. Image segmentation for tumor boundary identification\n"
                "3. Reducing false positives by 30% compared to traditional methods"
            ),
            retrieval_context=[
                "Deep learning models achieve 95% accuracy in detecting lung cancer from CT scans",
                "Convolutional neural networks enable precise tumor segmentation",
                "AI-assisted diagnosis reduces false positives by 30%",
            ],
            expected_output=(
                "Deep learning applications in medical imaging include disease detection, "
                "image segmentation, and reducing false positives"
            ),
        )
        
        # Create metrics with Gemini
        answer_relevance = AnswerRelevancyMetric(
            model=gemini_evaluation_model,
            threshold=0.7
        )
        faithfulness = FaithfulnessMetric(
            model=gemini_evaluation_model,
            threshold=0.9
        )
        context_recall = ContextualRecallMetric(
            model=gemini_evaluation_model,
            threshold=0.7
        )
        
        # Evaluate all metrics
        evaluate(
            test_cases=[test_case],
            metrics=[answer_relevance, faithfulness, context_recall],
        )
        
        # Assert all metrics pass
        assert answer_relevance.score >= 0.7, (
            f"Answer relevance failed: {answer_relevance.reason}"
        )
        assert faithfulness.score >= 0.9, (
            f"Faithfulness failed: {faithfulness.reason}"
        )
        assert context_recall.score >= 0.7, (
            f"Context recall failed: {context_recall.reason}"
        )
    
    def test_batch_evaluation(self, gemini_evaluation_model):
        """Test batch evaluation of multiple test cases.
        
        Demonstrates how to evaluate multiple scenarios efficiently.
        """
        test_cases = [
            LLMTestCase(
                input="What is AI?",
                actual_output="AI is artificial intelligence, the simulation of human intelligence by machines.",
            ),
            LLMTestCase(
                input="What are the benefits of AI in healthcare?",
                actual_output=(
                    "AI in healthcare improves diagnostic accuracy, "
                    "reduces costs, and enables personalized treatment."
                ),
            ),
            LLMTestCase(
                input="How does deep learning work?",
                actual_output=(
                    "Deep learning uses neural networks with multiple layers "
                    "to learn hierarchical representations of data."
                ),
            ),
        ]
        
        # Evaluate all test cases with AnswerRelevancyMetric using Gemini
        metric = AnswerRelevancyMetric(
            model=gemini_evaluation_model,
            threshold=0.7
        )
        evaluate(test_cases=test_cases, metrics=[metric])
        
        # All test cases should pass the threshold
        assert metric.score >= 0.7, (
            f"Batch evaluation failed with score {metric.score}: {metric.reason}"
        )


# Standalone Metric Measurement Examples
def test_standalone_answer_relevance_measurement(
    sample_llm_test_case: LLMTestCase,
    gemini_evaluation_model,
):
    """Example of using metrics standalone for debugging.
    
    This pattern is useful for debugging specific test cases without
    running the full evaluate() function.
    """
    metric = AnswerRelevancyMetric(
        model=gemini_evaluation_model,
        threshold=0.7
    )
    
    # Measure standalone
    metric.measure(sample_llm_test_case)
    
    # Print results for debugging
    print(f"Score: {metric.score}")
    print(f"Reason: {metric.reason}")
    
    # Assert
    assert metric.score >= 0.7


def test_standalone_faithfulness_measurement(
    sample_llm_test_case_with_context: LLMTestCase,
    gemini_evaluation_model,
):
    """Example of standalone faithfulness measurement for debugging."""
    metric = FaithfulnessMetric(
        model=gemini_evaluation_model,
        threshold=0.9
    )
    
    # Measure standalone
    metric.measure(sample_llm_test_case_with_context)
    
    # Print results for debugging
    print(f"Score: {metric.score}")
    print(f"Reason: {metric.reason}")
    
    # Assert
    assert metric.score >= 0.9
