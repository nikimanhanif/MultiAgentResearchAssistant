"""Unit tests for scope agent.

This module contains comprehensive tests for the scope agent module,
including LLM initialization, conversation handling, and research brief generation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from app.agents.scope_agent import (
    clarify_scope,
    generate_clarification_questions,
    generate_research_brief,
    check_scope_completion,
    _get_llm,
    _format_conversation_history,
    _parse_json_response,
    _invoke_llm_with_retry,
)
from app.models.schemas import ClarificationQuestions, ResearchBrief


class TestGetLlm:
    """Test cases for LLM initialization (_get_llm function)."""

    @patch("app.agents.scope_agent.settings")
    def test_get_llm_with_gemini_returns_configured_instance(self, mock_settings):
        """Test that _get_llm with Gemini model returns configured LLM instance."""
        # Arrange
        mock_settings.DEFAULT_MODEL = "gemini"
        mock_settings.GOOGLE_GEMINI_API_KEY = "test_gemini_key"
        mock_settings.GEMINI_MODEL = "gemini-2.5-flash"
        
        # Act
        llm = _get_llm(temperature=0.7)
        
        # Assert
        assert llm is not None
        assert llm.temperature == 0.7
        assert hasattr(llm, 'model')

    @patch("app.agents.scope_agent.settings")
    def test_get_llm_with_deepseek_returns_configured_instance(self, mock_settings):
        """Test that _get_llm with DeepSeek model returns configured LLM instance."""
        # Arrange
        mock_settings.DEFAULT_MODEL = "deepseek"
        mock_settings.DEEPSEEK_API_KEY = "test_deepseek_key"
        mock_settings.DEEPSEEK_MODEL = "deepseek-chat"
        
        # Act
        llm = _get_llm(temperature=0.5)
        
        # Assert
        assert llm is not None
        assert llm.temperature == 0.5
        assert hasattr(llm, 'model_name')

    @patch("app.agents.scope_agent.settings")
    def test_get_llm_with_custom_temperature_returns_correct_value(self, mock_settings):
        """Test that _get_llm respects custom temperature parameter."""
        # Arrange
        mock_settings.DEFAULT_MODEL = "gemini"
        mock_settings.GOOGLE_GEMINI_API_KEY = "test_key"
        mock_settings.GEMINI_MODEL = "gemini-2.5-flash"
        custom_temp = 0.3
        
        # Act
        llm = _get_llm(temperature=custom_temp)
        
        # Assert
        assert llm.temperature == custom_temp

    @patch("app.agents.scope_agent.settings")
    def test_get_llm_with_missing_gemini_key_raises_value_error(self, mock_settings):
        """Test that _get_llm raises ValueError when Gemini API key is missing."""
        # Arrange
        mock_settings.DEFAULT_MODEL = "gemini"
        mock_settings.GOOGLE_GEMINI_API_KEY = ""
        
        # Act & Assert
        with pytest.raises(ValueError, match="GOOGLE_GEMINI_API_KEY not configured"):
            _get_llm()

    @patch("app.agents.scope_agent.settings")
    def test_get_llm_with_missing_deepseek_key_raises_value_error(self, mock_settings):
        """Test that _get_llm raises ValueError when DeepSeek API key is missing."""
        # Arrange
        mock_settings.DEFAULT_MODEL = "deepseek"
        mock_settings.DEEPSEEK_API_KEY = ""
        
        # Act & Assert
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY not configured"):
            _get_llm()

    @patch("app.agents.scope_agent.settings")
    def test_get_llm_with_invalid_model_raises_value_error(self, mock_settings):
        """Test that _get_llm raises ValueError for unsupported model name."""
        # Arrange
        mock_settings.DEFAULT_MODEL = "unsupported_model"
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid model"):
            _get_llm()


class TestFormatConversationHistory:
    """Test cases for conversation history formatting (_format_conversation_history function)."""

    def test_format_conversation_history_with_none_returns_default_message(self):
        """Test that _format_conversation_history with None returns default message."""
        # Arrange & Act
        result = _format_conversation_history(None)
        
        # Assert
        assert result == "No previous conversation."

    def test_format_conversation_history_with_empty_list_returns_default_message(self):
        """Test that _format_conversation_history with empty list returns default message."""
        # Arrange & Act
        result = _format_conversation_history([])
        
        # Assert
        assert result == "No previous conversation."

    def test_format_conversation_history_with_single_turn_returns_formatted_string(self):
        """Test that _format_conversation_history with single turn formats correctly."""
        # Arrange
        history = [{"role": "user", "content": "What is AI?"}]
        
        # Act
        result = _format_conversation_history(history)
        
        # Assert
        assert "USER: What is AI?" in result
        assert isinstance(result, str)

    def test_format_conversation_history_with_multiple_turns_preserves_order(self):
        """Test that _format_conversation_history preserves conversation order."""
        # Arrange
        history = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "Can you specify the aspect?"},
            {"role": "user", "content": "Machine learning aspect"},
        ]
        
        # Act
        result = _format_conversation_history(history)
        
        # Assert
        assert "USER: What is AI?" in result
        assert "ASSISTANT: Can you specify the aspect?" in result
        assert "USER: Machine learning aspect" in result
        # Verify order is preserved
        first_user_pos = result.find("USER: What is AI?")
        assistant_pos = result.find("ASSISTANT: Can you specify")
        second_user_pos = result.find("USER: Machine learning")
        assert first_user_pos < assistant_pos < second_user_pos

    def test_format_conversation_history_with_missing_role_handles_gracefully(self):
        """Test that _format_conversation_history handles missing role field."""
        # Arrange
        history = [{"content": "Test message"}]
        
        # Act
        result = _format_conversation_history(history)
        
        # Assert
        assert "UNKNOWN: Test message" in result

    def test_format_conversation_history_with_missing_content_handles_gracefully(self):
        """Test that _format_conversation_history handles missing content field."""
        # Arrange
        history = [{"role": "user"}]
        
        # Act
        result = _format_conversation_history(history)
        
        # Assert
        assert "USER:" in result


class TestParseJsonResponse:
    """Test cases for JSON response parsing (_parse_json_response function)."""

    def test_parse_json_response_with_valid_json_returns_dict(self):
        """Test that _parse_json_response with valid JSON returns parsed dictionary."""
        # Arrange
        response = '{"key": "value", "number": 42}'
        
        # Act
        result = _parse_json_response(response)
        
        # Assert
        assert result == {"key": "value", "number": 42}
        assert isinstance(result, dict)

    def test_parse_json_response_with_markdown_json_block_strips_markers(self):
        """Test that _parse_json_response strips markdown ```json code blocks."""
        # Arrange
        response = '```json\n{"key": "value"}\n```'
        
        # Act
        result = _parse_json_response(response)
        
        # Assert
        assert result == {"key": "value"}

    def test_parse_json_response_with_markdown_plain_block_strips_markers(self):
        """Test that _parse_json_response strips markdown ``` code blocks."""
        # Arrange
        response = '```\n{"key": "value"}\n```'
        
        # Act
        result = _parse_json_response(response)
        
        # Assert
        assert result == {"key": "value"}

    def test_parse_json_response_with_whitespace_strips_correctly(self):
        """Test that _parse_json_response strips leading and trailing whitespace."""
        # Arrange
        response = '  \n{"key": "value"}\n  '
        
        # Act
        result = _parse_json_response(response)
        
        # Assert
        assert result == {"key": "value"}

    def test_parse_json_response_with_nested_objects_returns_complete_structure(self):
        """Test that _parse_json_response handles nested JSON objects."""
        # Arrange
        response = '{"outer": {"inner": "value"}, "list": [1, 2, 3]}'
        
        # Act
        result = _parse_json_response(response)
        
        # Assert
        assert result["outer"]["inner"] == "value"
        assert result["list"] == [1, 2, 3]

    def test_parse_json_response_with_invalid_json_raises_value_error(self):
        """Test that _parse_json_response raises ValueError for invalid JSON."""
        # Arrange
        response = "not valid json"
        
        # Act & Assert
        with pytest.raises(ValueError, match="Failed to parse JSON response"):
            _parse_json_response(response)

    def test_parse_json_response_with_empty_string_raises_value_error(self):
        """Test that _parse_json_response raises ValueError for empty string."""
        # Arrange
        response = ""
        
        # Act & Assert
        with pytest.raises(ValueError, match="Failed to parse JSON response"):
            _parse_json_response(response)

    def test_parse_json_response_with_incomplete_json_raises_value_error(self):
        """Test that _parse_json_response raises ValueError for incomplete JSON."""
        # Arrange
        response = '{"key": "value"'  # Missing closing brace
        
        # Act & Assert
        with pytest.raises(ValueError, match="Failed to parse JSON response"):
            _parse_json_response(response)


class TestInvokeLlmWithRetry:
    """Test cases for LLM invocation with retry logic (_invoke_llm_with_retry function)."""

    @pytest.mark.asyncio
    async def test_invoke_llm_with_retry_on_first_attempt_returns_content(self):
        """Test that _invoke_llm_with_retry succeeds on first attempt."""
        # Arrange
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_llm.ainvoke.return_value = mock_response
        
        # Act
        result = await _invoke_llm_with_retry(mock_llm, "Test prompt")
        
        # Assert
        assert result == "Test response"
        assert mock_llm.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_invoke_llm_with_retry_after_failures_succeeds_eventually(self):
        """Test that _invoke_llm_with_retry succeeds after transient failures."""
        # Arrange
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Test response"
        
        # Fail twice, then succeed
        mock_llm.ainvoke.side_effect = [
            Exception("Temporary error 1"),
            Exception("Temporary error 2"),
            mock_response,
        ]
        
        # Act
        result = await _invoke_llm_with_retry(mock_llm, "Test prompt", max_retries=3)
        
        # Assert
        assert result == "Test response"
        assert mock_llm.ainvoke.call_count == 3

    @pytest.mark.asyncio
    async def test_invoke_llm_with_retry_with_persistent_failure_raises_exception(self):
        """Test that _invoke_llm_with_retry raises exception after max retries."""
        # Arrange
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = Exception("Persistent error")
        
        # Act & Assert
        with pytest.raises(Exception, match="LLM invocation failed after .* attempts"):
            await _invoke_llm_with_retry(mock_llm, "Test prompt", max_retries=2)
        
        assert mock_llm.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_invoke_llm_with_retry_with_single_retry_attempts_twice(self):
        """Test that _invoke_llm_with_retry with max_retries=1 attempts once."""
        # Arrange
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = Exception("Error")
        
        # Act & Assert
        with pytest.raises(Exception):
            await _invoke_llm_with_retry(mock_llm, "Test prompt", max_retries=1)
        
        assert mock_llm.ainvoke.call_count == 1


class TestGenerateClarificationQuestions:
    """Test cases for clarification question generation (generate_clarification_questions function)."""

    @pytest.mark.asyncio
    @patch("app.agents.scope_agent._get_llm")
    @patch("app.agents.scope_agent._invoke_llm_with_retry")
    async def test_generate_clarification_questions_with_simple_query_returns_questions(
        self, mock_invoke, mock_get_llm
    ):
        """Test that generate_clarification_questions for simple query returns appropriate questions."""
        # Arrange
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_response = json.dumps({
            "questions": [
                "What specific aspect of AI are you interested in?",
                "What is your intended use for this research?"
            ],
            "context": "Need to understand scope and depth"
        })
        mock_invoke.return_value = mock_response
        
        # Act
        result = await generate_clarification_questions("What is AI?")
        
        # Assert
        assert isinstance(result, ClarificationQuestions)
        assert len(result.questions) == 2
        assert "What specific aspect" in result.questions[0]
        assert result.context == "Need to understand scope and depth"
        mock_get_llm.assert_called_once()
        mock_invoke.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.agents.scope_agent._get_llm")
    @patch("app.agents.scope_agent._invoke_llm_with_retry")
    async def test_generate_clarification_questions_with_history_considers_context(
        self, mock_invoke, mock_get_llm
    ):
        """Test that generate_clarification_questions considers conversation history."""
        # Arrange
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_response = json.dumps({
            "questions": ["What time period are you interested in?"],
            "context": "Need temporal constraints"
        })
        mock_invoke.return_value = mock_response
        
        history = [
            {"role": "user", "content": "Tell me about quantum computing"},
            {"role": "assistant", "content": "Can you specify the aspect?"},
            {"role": "user", "content": "Hardware developments"}
        ]
        
        # Act
        result = await generate_clarification_questions(
            "Tell me about quantum computing", history
        )
        
        # Assert
        assert isinstance(result, ClarificationQuestions)
        assert len(result.questions) == 1
        assert "time period" in result.questions[0]

    @pytest.mark.asyncio
    @patch("app.agents.scope_agent._get_llm")
    @patch("app.agents.scope_agent._invoke_llm_with_retry")
    async def test_generate_clarification_questions_with_clear_scope_returns_empty_list(
        self, mock_invoke, mock_get_llm
    ):
        """Test that generate_clarification_questions returns empty list when scope is clear."""
        # Arrange
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_response = json.dumps({
            "questions": [],
            "context": "Scope is clear"
        })
        mock_invoke.return_value = mock_response
        
        # Act
        result = await generate_clarification_questions("Detailed query with context")
        
        # Assert
        assert isinstance(result, ClarificationQuestions)
        assert len(result.questions) == 0
        assert result.context == "Scope is clear"

    @pytest.mark.asyncio
    @patch("app.agents.scope_agent._get_llm")
    @patch("app.agents.scope_agent._invoke_llm_with_retry")
    async def test_generate_clarification_questions_with_no_history_uses_default(
        self, mock_invoke, mock_get_llm
    ):
        """Test that generate_clarification_questions handles None history correctly."""
        # Arrange
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_response = json.dumps({
            "questions": ["Sample question"],
            "context": "Initial clarification"
        })
        mock_invoke.return_value = mock_response
        
        # Act
        result = await generate_clarification_questions("Test query", None)
        
        # Assert
        assert isinstance(result, ClarificationQuestions)
        assert len(result.questions) == 1


class TestCheckScopeCompletion:
    """Test cases for scope completion detection (check_scope_completion function)."""

    @pytest.mark.asyncio
    @patch("app.agents.scope_agent._get_llm")
    @patch("app.agents.scope_agent._invoke_llm_with_retry")
    async def test_check_scope_completion_with_sufficient_info_returns_complete(
        self, mock_invoke, mock_get_llm
    ):
        """Test that check_scope_completion returns complete when information is sufficient."""
        # Arrange
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_response = json.dumps({
            "is_complete": True,
            "reason": "All necessary information provided",
            "missing_info": []
        })
        mock_invoke.return_value = mock_response
        
        # Act
        result = await check_scope_completion("Detailed research query")
        
        # Assert
        assert result["is_complete"] is True
        assert "necessary information" in result["reason"]
        assert len(result["missing_info"]) == 0
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    @patch("app.agents.scope_agent._get_llm")
    @patch("app.agents.scope_agent._invoke_llm_with_retry")
    async def test_check_scope_completion_with_insufficient_info_returns_incomplete(
        self, mock_invoke, mock_get_llm
    ):
        """Test that check_scope_completion returns incomplete when more information needed."""
        # Arrange
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_response = json.dumps({
            "is_complete": False,
            "reason": "Need more details on scope",
            "missing_info": ["time period", "geographic focus"]
        })
        mock_invoke.return_value = mock_response
        
        # Act
        result = await check_scope_completion("Vague query")
        
        # Assert
        assert result["is_complete"] is False
        assert "more details" in result["reason"]
        assert len(result["missing_info"]) == 2
        assert "time period" in result["missing_info"]
        assert "geographic focus" in result["missing_info"]

    @pytest.mark.asyncio
    @patch("app.agents.scope_agent._get_llm")
    @patch("app.agents.scope_agent._invoke_llm_with_retry")
    async def test_check_scope_completion_with_history_considers_context(
        self, mock_invoke, mock_get_llm
    ):
        """Test that check_scope_completion considers conversation history."""
        # Arrange
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_response = json.dumps({
            "is_complete": True,
            "reason": "All details clarified through conversation",
            "missing_info": []
        })
        mock_invoke.return_value = mock_response
        
        history = [
            {"role": "user", "content": "Research AI"},
            {"role": "assistant", "content": "What aspect?"},
            {"role": "user", "content": "Machine learning"}
        ]
        
        # Act
        result = await check_scope_completion("Research AI", history)
        
        # Assert
        assert result["is_complete"] is True


class TestGenerateResearchBrief:
    """Test cases for research brief generation (generate_research_brief function)."""

    @pytest.mark.asyncio
    @patch("app.agents.scope_agent._get_llm")
    @patch("app.agents.scope_agent._invoke_llm_with_retry")
    async def test_generate_research_brief_with_complete_history_returns_detailed_brief(
        self, mock_invoke, mock_get_llm
    ):
        """Test that generate_research_brief with conversation history returns detailed brief."""
        # Arrange
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_response = json.dumps({
            "scope": "Research latest developments in quantum computing hardware",
            "sub_topics": [
                "Qubit technologies",
                "Error correction",
                "Scalability challenges"
            ],
            "constraints": {
                "time_period": "2020-2024",
                "depth": "detailed technical analysis"
            },
            "deliverables": "Comprehensive technical report with citations",
            "format": "detailed",
            "metadata": {}
        })
        mock_invoke.return_value = mock_response
        
        history = [
            {"role": "user", "content": "Quantum computing research"},
            {"role": "assistant", "content": "What aspect?"},
            {"role": "user", "content": "Hardware developments"}
        ]
        
        # Act
        result = await generate_research_brief("Quantum computing", history)
        
        # Assert
        assert isinstance(result, ResearchBrief)
        assert "quantum computing hardware" in result.scope.lower()
        assert len(result.sub_topics) == 3
        assert "Qubit" in result.sub_topics[0]
        assert result.constraints["time_period"] == "2020-2024"
        assert result.format == "detailed"
        assert result.metadata["clarification_turns"] == 1
        assert result.metadata["original_query"] == "Quantum computing"

    @pytest.mark.asyncio
    @patch("app.agents.scope_agent._get_llm")
    @patch("app.agents.scope_agent._invoke_llm_with_retry")
    async def test_generate_research_brief_without_history_returns_basic_brief(
        self, mock_invoke, mock_get_llm
    ):
        """Test that generate_research_brief without history returns basic brief."""
        # Arrange
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_response = json.dumps({
            "scope": "Overview of machine learning",
            "sub_topics": ["Supervised learning", "Unsupervised learning"],
            "constraints": {},
            "deliverables": "Summary report",
            "format": "summary",
            "metadata": {}
        })
        mock_invoke.return_value = mock_response
        
        # Act
        result = await generate_research_brief("Machine learning overview")
        
        # Assert
        assert isinstance(result, ResearchBrief)
        assert "machine learning" in result.scope.lower()
        assert len(result.sub_topics) == 2
        assert result.deliverables == "Summary report"

    @pytest.mark.asyncio
    @patch("app.agents.scope_agent._get_llm")
    @patch("app.agents.scope_agent._invoke_llm_with_retry")
    async def test_generate_research_brief_counts_clarification_turns_correctly(
        self, mock_invoke, mock_get_llm
    ):
        """Test that generate_research_brief counts assistant turns correctly in metadata."""
        # Arrange
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_response = json.dumps({
            "scope": "Test scope",
            "sub_topics": ["topic1"],
            "constraints": {},
            "deliverables": "Test deliverables",
            "format": "summary",
            "metadata": {}
        })
        mock_invoke.return_value = mock_response
        
        history = [
            {"role": "user", "content": "Query 1"},
            {"role": "assistant", "content": "Question 1"},
            {"role": "user", "content": "Answer 1"},
            {"role": "assistant", "content": "Question 2"},
            {"role": "user", "content": "Answer 2"}
        ]
        
        # Act
        result = await generate_research_brief("Test query", history)
        
        # Assert
        assert result.metadata["clarification_turns"] == 2


class TestClarifyScope:
    """Test cases for main clarify_scope orchestration function."""

    @pytest.mark.asyncio
    @patch("app.agents.scope_agent.check_scope_completion")
    @patch("app.agents.scope_agent.generate_research_brief")
    async def test_clarify_scope_with_complete_scope_returns_research_brief(
        self, mock_generate_brief, mock_check_completion
    ):
        """Test that clarify_scope returns ResearchBrief when scope is complete."""
        # Arrange
        mock_check_completion.return_value = {
            "is_complete": True,
            "reason": "Sufficient information",
            "missing_info": []
        }
        
        expected_brief = ResearchBrief(
            scope="Test scope",
            sub_topics=["topic1", "topic2"],
            constraints={},
            deliverables="Test deliverables",
            format="summary"
        )
        mock_generate_brief.return_value = expected_brief
        
        # Act
        result = await clarify_scope("Test query")
        
        # Assert
        assert isinstance(result, ResearchBrief)
        assert result.scope == "Test scope"
        assert len(result.sub_topics) == 2
        mock_check_completion.assert_called_once()
        mock_generate_brief.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.agents.scope_agent.check_scope_completion")
    @patch("app.agents.scope_agent.generate_clarification_questions")
    async def test_clarify_scope_with_incomplete_scope_returns_questions(
        self, mock_generate_questions, mock_check_completion
    ):
        """Test that clarify_scope returns ClarificationQuestions when scope incomplete."""
        # Arrange
        mock_check_completion.return_value = {
            "is_complete": False,
            "reason": "Need more info",
            "missing_info": ["aspect", "depth"]
        }
        
        expected_questions = ClarificationQuestions(
            questions=["What aspect?", "What depth?"],
            context="Need more details"
        )
        mock_generate_questions.return_value = expected_questions
        
        # Act
        result = await clarify_scope("Vague query")
        
        # Assert
        assert isinstance(result, ClarificationQuestions)
        assert len(result.questions) == 2
        mock_check_completion.assert_called_once()
        mock_generate_questions.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.agents.scope_agent.check_scope_completion")
    @patch("app.agents.scope_agent.generate_clarification_questions")
    async def test_clarify_scope_with_history_passes_history_to_subfunctions(
        self, mock_generate_questions, mock_check_completion
    ):
        """Test that clarify_scope passes conversation history to subfunctions correctly."""
        # Arrange
        mock_check_completion.return_value = {
            "is_complete": False,
            "reason": "Need format preference",
            "missing_info": ["format"]
        }
        
        expected_questions = ClarificationQuestions(
            questions=["What format would you like?"],
            context="Need output format"
        )
        mock_generate_questions.return_value = expected_questions
        
        history = [
            {"role": "user", "content": "Research AI"},
            {"role": "assistant", "content": "What aspect?"},
            {"role": "user", "content": "Machine learning"}
        ]
        
        # Act
        result = await clarify_scope("Research AI", history)
        
        # Assert
        assert isinstance(result, ClarificationQuestions)
        mock_check_completion.assert_called_once_with("Research AI", history)
        mock_generate_questions.assert_called_once_with("Research AI", history)

    @pytest.mark.asyncio
    @patch("app.agents.scope_agent.check_scope_completion")
    @patch("app.agents.scope_agent.generate_research_brief")
    async def test_clarify_scope_without_history_uses_none_default(
        self, mock_generate_brief, mock_check_completion
    ):
        """Test that clarify_scope handles None history parameter correctly."""
        # Arrange
        mock_check_completion.return_value = {
            "is_complete": True,
            "reason": "Query is detailed enough",
            "missing_info": []
        }
        
        expected_brief = ResearchBrief(
            scope="Direct scope",
            sub_topics=["topic1"],
            constraints={},
            deliverables="Deliverables",
            format="summary"
        )
        mock_generate_brief.return_value = expected_brief
        
        # Act
        result = await clarify_scope("Detailed query", None)
        
        # Assert
        assert isinstance(result, ResearchBrief)
        mock_check_completion.assert_called_once_with("Detailed query", None)
