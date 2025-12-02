"""Pytest fixtures for evaluation tests.

This module provides shared fixtures for DeepEval and LangSmith evaluation tests,
including mock LLM clients, sample test data, and evaluation configuration.
"""

import os
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock
import pytest
from deepeval.test_case import LLMTestCase
from langchain_core.messages import AIMessage, HumanMessage

from app.models.schemas import (
    ResearchBrief,
    Citation,
    Finding,
)


# Environment Configuration
@pytest.fixture(scope="session", autouse=True)
def configure_langsmith():
    """Configure LangSmith tracing for evaluation tests.
    
    Traces will be visible at https://smith.langchain.com/
    """
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
        os.environ["LANGSMITH_PROJECT"] = os.getenv(
            "LANGSMITH_PROJECT", "multi-agent-research-assistant-eval"
        )
        os.environ["LANGSMITH_ENDPOINT"] = os.getenv(
            "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
        )
        os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    yield
    # Cleanup is automatic


@pytest.fixture(scope="session", autouse=True)
def configure_deepeval():
    """Configure DeepEval for OpenAI gpt-5-nano evaluation.
    
    Note: This configuration uses REAL LLM calls for evaluation metrics.
    Do NOT mock the evaluation LLM - it needs to make real API calls to judge quality.
    """
    # Opt out of telemetry by default (can be overridden in .env)
    if "DEEPEVAL_TELEMETRY_OPT_OUT" not in os.environ:
        os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "true"
    
    # Validate OpenAI configuration is available
    from app.config import settings
    
    if not settings.OPENAI_API_KEY:
        import warnings
        warnings.warn(
            "OPENAI_API_KEY not configured. "
            "Evaluation metrics will fail without an API key. "
            "Add OPENAI_API_KEY to your .env file."
        )
    
    yield


# OpenAI Model Fixture for Evaluation
@pytest.fixture
def evaluation_model():
    """Get OpenAI gpt-5-nano for DeepEval evaluation metrics.
    
    This model is used by evaluation metrics (AnswerRelevancyMetric, GEval, etc.)
    to judge the quality of outputs. It makes REAL API calls.
    
    Uses DeepEval's built-in GPTModel class for OpenAI gpt-5-nano.
    
    Returns:
        GPTModel: Configured OpenAI gpt-5-nano model for evaluation
        
    Note:
        Do NOT use this for mocking! Evaluation metrics need real LLM calls.
    """
    from tests.evaluation.deepeval_config import get_evaluation_model
    from app.config import settings
    
    # Verify API key is available
    if not settings.OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY not configured")
    
    return get_evaluation_model()


@pytest.fixture
def evaluation_model_with_temperature():
    """Get OpenAI gpt-5-nano model with custom temperature for evaluation.
    
    Returns:
        callable: Function that takes temperature and returns GPTModel
    """
    from tests.evaluation.deepeval_config import get_evaluation_model
    
    def _get_model(temperature: float = 0.0):
        return get_evaluation_model(temperature=temperature)
    
    return _get_model


# Sample Test Data Fixtures
@pytest.fixture
def sample_user_query() -> str:
    """Sample vague user query for scope agent testing."""
    return "I need to research AI in healthcare"


@pytest.fixture
def sample_clarification_questions() -> List[str]:
    """Sample clarification questions from scope agent."""
    return [
        "What specific aspect of AI in healthcare are you interested in? (e.g., diagnostics, treatment planning, drug discovery)",
        "What time period should the research cover? (e.g., last 5 years, all time)",
        "Are you looking for a specific type of output? (e.g., literature review, comparison, gap analysis)",
    ]


@pytest.fixture
def sample_research_brief() -> ResearchBrief:
    """Sample research brief for testing."""
    return ResearchBrief(
        scope="AI applications in medical diagnostics",
        sub_topics=[
            "Deep learning for medical imaging",
            "Natural language processing for clinical notes",
            "Predictive models for disease diagnosis",
        ],
        constraints={
            "time_range": "2019-2024",
            "source_types": ["peer-reviewed", "academic"],
            "min_credibility": 0.7,
        },
        deliverables="Literature review with comprehensive analysis of AI in medical diagnostics",
        format=None,  # Optional field
        metadata=None,  # Optional field
    )


@pytest.fixture
def sample_citation() -> Citation:
    """Sample citation for testing."""
    from app.models.schemas import SourceType
    
    return Citation(
        source="Nature Medicine",
        title="Deep Learning for Medical Image Analysis",
        author="Smith, J., Doe, A.",
        year=2023,
        url="https://example.com/paper",
        doi="10.1234/example.doi",
        source_type=SourceType.PEER_REVIEWED,
        credibility_score=0.92,
        venue="Nature Medicine",
    )




# DeepEval Test Case Fixtures
@pytest.fixture
def sample_llm_test_case() -> LLMTestCase:
    """Sample LLMTestCase for DeepEval metrics testing.
    
    Returns:
        LLMTestCase: Basic test case with input and output
    """
    return LLMTestCase(
        input="What are the main applications of AI in healthcare?",
        actual_output=(
            "AI is used in healthcare for medical imaging analysis, "
            "clinical decision support, drug discovery, and personalized treatment planning."
        ),
    )


@pytest.fixture
def sample_llm_test_case_with_context() -> LLMTestCase:
    """Sample LLMTestCase with retrieval context for testing.
    
    Returns:
        LLMTestCase: Test case with input, output, and retrieval context
    """
    return LLMTestCase(
        input="What are the benefits of deep learning in medical imaging?",
        actual_output=(
            "Deep learning improves medical imaging by achieving higher accuracy "
            "in detecting diseases, reducing false positives, and enabling faster diagnosis."
        ),
        retrieval_context=[
            "Deep learning models achieve 95% accuracy in detecting lung cancer from CT scans.",
            "Convolutional neural networks reduce false positives by 30% compared to traditional methods.",
            "AI-assisted diagnosis reduces radiologist workload by 40%.",
        ],
    )


@pytest.fixture
def sample_llm_test_case_with_expected_output() -> LLMTestCase:
    """Sample LLMTestCase with expected output for faithfulness testing.
    
    Returns:
        LLMTestCase: Test case with input, actual output, and expected output
    """
    return LLMTestCase(
        input="Summarize the key findings from the research brief.",
        actual_output=(
            "The research shows that deep learning models achieve 95% accuracy "
            "in medical imaging, NLP models extract clinical information with 88% precision, "
            "and predictive models show promise in early disease detection."
        ),
        expected_output=(
            "Key findings include high accuracy in medical imaging (95%), "
            "effective clinical information extraction (88% precision), "
            "and promising results in early disease detection."
        ),
    )


# Scope Agent Test Fixtures
@pytest.fixture
def sample_conversation_history() -> List[Dict[str, str]]:
    """Sample conversation history for scope agent testing."""
    return [
        {
            "role": "user",
            "content": "I need to research AI in healthcare",
        },
        {
            "role": "assistant",
            "content": "What specific aspect of AI in healthcare are you interested in?",
        },
        {
            "role": "user",
            "content": "Medical diagnostics using deep learning",
        },
        {
            "role": "assistant",
            "content": "What time period should the research cover?",
        },
        {
            "role": "user",
            "content": "Last 5 years",
        },
    ]


@pytest.fixture
def sample_scope_agent_input() -> Dict[str, Any]:
    """Sample input for scope agent evaluation."""
    return {
        "user_query": "I need to research AI in healthcare",
        "conversation_history": [],
    }


# Evaluation Configuration Fixtures
@pytest.fixture
def evaluation_thresholds() -> Dict[str, float]:
    """Standard evaluation thresholds for metrics.
    
    Returns:
        Dict[str, float]: Metric name to threshold mapping
    """
    return {
        "answer_relevance": 0.7,
        "faithfulness": 0.9,
        "context_recall": 0.7,
        "format_compliance": 0.8,
    }


@pytest.fixture
def mock_scope_agent():
    """Mock scope agent for testing without real LLM calls."""
    from app.agents.scope_agent import ScopeAgent
    
    mock_agent = MagicMock(spec=ScopeAgent)
    mock_agent.generate_clarification_questions.return_value = [
        "What specific aspect of AI in healthcare are you interested in?",
        "What time period should the research cover?",
    ]
    mock_agent.check_scope_completion.return_value = False
    mock_agent.generate_research_brief.return_value = ResearchBrief(
        scope="AI in medical diagnostics",
        sub_topics=["Deep learning", "Medical imaging"],
        constraints={"time_range": "2019-2024"},
        format_type="literature_review",
        depth_level="detailed",
    )
    
    return mock_agent


# Helper Functions
def create_test_case(
    input_text: str,
    output_text: str,
    context: List[str] = None,
    expected_output: str = None,
) -> LLMTestCase:
    """Helper function to create LLMTestCase instances.
    
    Args:
        input_text: Input query or prompt
        output_text: Actual output from the model
        context: Optional retrieval context
        expected_output: Optional expected output for comparison
        
    Returns:
        LLMTestCase: Configured test case
    """
    return LLMTestCase(
        input=input_text,
        actual_output=output_text,
        retrieval_context=context,
        expected_output=expected_output,
    )


# Export helper function for use in tests
__all__ = ["create_test_case"]

