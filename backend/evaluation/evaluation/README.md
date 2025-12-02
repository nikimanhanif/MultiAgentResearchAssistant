# Evaluation Framework Guide

Quick guide for running evaluations on the Multi-Agent Research Assistant using DeepEval and LangSmith.

---

## Quick Setup

### 1. Install Dependencies

```bash
cd backend
uv add deepeval
```

### 2. Configure Environment Variables

Add to your `.env` file:

```bash
# Google Gemini (Evaluation Judge)
GOOGLE_GEMINI_API_KEY=your_google_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# LangSmith (Trace Observability)
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=multi-agent-research-assistant-eval
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

# Optional
DEEPEVAL_TELEMETRY_OPT_OUT=true
```

**Get API Keys:**
- Gemini: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
- LangSmith: [https://smith.langchain.com/](https://smith.langchain.com/) → Settings → API Keys

### 3. Verify Setup

```bash
uv run pytest tests/evaluation/test_structure.py -v
```

All 12 tests should pass.

---

## Running Evaluations

### Run All Evaluation Tests

```bash
uv run pytest tests/evaluation/ -v
```

### Run Specific Test Suites

```bash
# Model quality tests (AnswerRelevance, Faithfulness, ContextRecall)
uv run pytest tests/evaluation/test_model_quality.py -v

# Agent behavior tests (Format compliance, Tool selection, Reasoning)
uv run pytest tests/evaluation/test_agent_behavior.py -v

# Scope agent baseline evaluation
uv run pytest tests/evaluation/test_scope_agent_evaluation.py -v
```

### Run Individual Test Classes

```bash
uv run pytest tests/evaluation/test_model_quality.py::TestAnswerRelevance -v
```

### Run Individual Tests

```bash
uv run pytest tests/evaluation/test_model_quality.py::TestAnswerRelevance::test_answer_relevance_basic -v
```

### Run with Detailed Output

```bash
uv run pytest tests/evaluation/ -v -s
```

The `-s` flag shows print statements and metric scores/reasons.

---

## Available Metrics

### 1. AnswerRelevancyMetric (Threshold: 0.7)

Measures if the output is relevant to the input.

**Use for:** Clarification questions, research findings, Q&A tasks

```python
from deepeval.metrics import AnswerRelevancyMetric

def test_relevance(gemini_evaluation_model):
    metric = AnswerRelevancyMetric(
        model=gemini_evaluation_model,
        threshold=0.7
    )
    # Use metric...
```

### 2. FaithfulnessMetric (Threshold: 0.9)

Detects hallucinations - checks if output is faithful to source material.

**Use for:** Research briefs, reports, summaries

```python
from deepeval.metrics import FaithfulnessMetric

def test_faithfulness(gemini_evaluation_model):
    metric = FaithfulnessMetric(
        model=gemini_evaluation_model,
        threshold=0.9
    )
    # Use metric...
```

### 3. ContextualRecallMetric (Threshold: 0.7)

Measures if relevant information was retrieved.

**Use for:** Research agent retrieval, citation quality

```python
from deepeval.metrics import ContextualRecallMetric

def test_recall(gemini_evaluation_model):
    metric = ContextualRecallMetric(
        model=gemini_evaluation_model,
        threshold=0.7
    )
    # Use metric...
```

### 4. GEval (Custom Threshold)

Custom evaluation criteria for behavioral testing.

**Use for:** Format compliance, tool selection, reasoning quality

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

def test_format(gemini_evaluation_model):
    metric = GEval(
        name="Format Compliance",
        model=gemini_evaluation_model,
        criteria="Does the output follow the required format?",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.8,
    )
    # Use metric...
```

---

## Writing Tests

### Basic Test Structure

```python
"""Test module docstring."""

import pytest
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase


def test_my_feature(gemini_evaluation_model):
    """Test docstring."""
    # 1. Create metric with evaluation model
    metric = AnswerRelevancyMetric(
        model=gemini_evaluation_model,
        threshold=0.7
    )
    
    # 2. Create test case
    test_case = LLMTestCase(
        input="Your input query",
        actual_output="Output from your system"
    )
    
    # 3. Measure and assert
    metric.measure(test_case)
    
    assert metric.score >= 0.7, (
        f"Score {metric.score} below threshold. "
        f"Reason: {metric.reason}"
    )
```

### Using Fixtures

Available fixtures in `conftest.py`:

```python
def test_with_fixtures(
    gemini_evaluation_model,  # Gemini model for evaluation
    sample_research_brief,    # Sample ResearchBrief instance
    sample_citation,          # Sample Citation instance
    evaluation_thresholds,    # Dict of standard thresholds
):
    """Test using predefined fixtures."""
    # Use fixtures in your test
    ...
```

---

## Debugging Failed Tests

### 1. Check the Metric Reason

```python
metric.measure(test_case)
print(f"Score: {metric.score}")
print(f"Reason: {metric.reason}")  # Why it passed/failed
```

### 2. View LangSmith Traces

1. Go to [https://smith.langchain.com/](https://smith.langchain.com/)
2. Select project: `multi-agent-research-assistant-eval`
3. Find the failing test run
4. Inspect:
   - Input query
   - Agent reasoning
   - Tool calls
   - Output generation
   - Evaluation LLM reasoning

---

## Switching Evaluation Models

### Using a Different Model

By default, the evaluation framework uses **Gemini 2.5 Flash** as configured in `app/config.py`.

#### Option 1: Change Model in Settings

Update `backend/app/config.py`:

```python
class Settings(BaseSettings):
    # Change the model used for evaluation
    GEMINI_MODEL: str = "gemini-1.5-pro"  # or "gemini-1.5-flash"
```

#### Option 2: Use DeepSeek (or Other Models)

To use DeepSeek or another model for evaluation:

**Step 1:** Update `conftest.py`:

```python
@pytest.fixture
def evaluation_model():
    """Get configured model for DeepEval evaluation metrics."""
    from deepeval.models import OpenAIModel  # or other model class
    from app.config import settings
    
    # For DeepSeek (using OpenAI-compatible API)
    return OpenAIModel(
        model="deepseek-chat",
        api_key=settings.DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com/v1"
    )
    
    # For OpenAI
    # return OpenAIModel(model="gpt-4")
    
    # For other models, see DeepEval docs:
    # https://docs.confident-ai.com/integrations/models
```

**Step 2:** Update test files to use `evaluation_model` instead of `gemini_evaluation_model`.

**Supported Models:**
- **Gemini** (current): `GeminiModel`
- **OpenAI**: `OpenAIModel` (GPT-4, GPT-3.5)
- **Anthropic**: `AnthropicModel` (Claude)
- **Azure OpenAI**: `AzureOpenAIModel`
- **Custom**: Extend `DeepEvalBaseLLM`

See DeepEval docs for more: [https://docs.confident-ai.com/integrations/models](https://docs.confident-ai.com/integrations/models)

#### Option 3: Use Multiple Models

Create multiple fixtures for different models:

```python
@pytest.fixture
def gemini_evaluation_model():
    """Gemini 2.5 Flash for evaluation."""
    from deepeval.models import GeminiModel
    return GeminiModel(model="gemini-2.5-flash")

@pytest.fixture
def deepseek_evaluation_model():
    """DeepSeek for evaluation."""
    from deepeval.models import OpenAIModel
    return OpenAIModel(
        model="deepseek-chat",
        api_key=settings.DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com/v1"
    )
```

Then use whichever fixture you need:

```python
def test_with_gemini(gemini_evaluation_model):
    metric = AnswerRelevancyMetric(model=gemini_evaluation_model)
    # ...

def test_with_deepseek(deepseek_evaluation_model):
    metric = AnswerRelevancyMetric(model=deepseek_evaluation_model)
    # ...
```

---


## Troubleshooting

### API Key Not Configured

```
ValueError: Google API key is required
```

**Solution:** Add `GOOGLE_GEMINI_API_KEY` to your `.env` file.

### Tests Are Slow

Evaluation uses real LLM calls, which takes time.

**Solutions:**
- Run specific tests: `pytest tests/evaluation/test_model_quality.py::TestAnswerRelevance -v`
- Mock your application's LLM (but not the evaluation LLM)
- Use parallel execution: `uv add pytest-xdist` then `pytest -n auto`

### Metric Score Below Threshold

**Steps:**
1. Check metric reason: `print(metric.reason)`
2. View LangSmith trace at [smith.langchain.com](https://smith.langchain.com)
3. Fix your source code (per `backend-test-fixes.mdc`)
4. Re-run evaluation

### API Rate Limits

**Solutions:**
- Use Gemini 2.5 Flash (higher rate limits)
- Add delays between tests
- Request higher limits from Google

---

## Summary

### Quick Start

```bash
# 1. Setup
cd backend
uv add deepeval

# 2. Configure .env
echo "GOOGLE_GEMINI_API_KEY=your_key" >> .env
echo "LANGSMITH_API_KEY=your_key" >> .env

# 3. Verify
uv run pytest tests/evaluation/test_structure.py -v

# 4. Run evaluations
uv run pytest tests/evaluation/ -v
```

### Key Commands

```bash
# All tests
uv run pytest tests/evaluation/ -v

# Specific suite
uv run pytest tests/evaluation/test_model_quality.py -v

# Individual test
uv run pytest tests/evaluation/test_model_quality.py::TestAnswerRelevance::test_answer_relevance_basic -v

# With output
uv run pytest tests/evaluation/ -v -s
```

### View Traces

[https://smith.langchain.com/](https://smith.langchain.com/) → `multi-agent-research-assistant-eval`

---

## References

- **DeepEval Docs**: [https://docs.confident-ai.com/](https://docs.confident-ai.com/)
- **DeepEval Models**: [https://docs.confident-ai.com/integrations/models](https://docs.confident-ai.com/integrations/models)
- **LangSmith**: [https://docs.smith.langchain.com/](https://docs.smith.langchain.com/)
- **Gemini Pricing**: [https://ai.google.dev/pricing](https://ai.google.dev/pricing)
