"""Unit tests for error handling utilities.

This module contains tests for the @safe_node decorator and error handling
functions for LangGraph node error recovery.
"""

import pytest
from unittest.mock import AsyncMock, patch
import asyncio

from app.utils.error_handling import (
    safe_node,
    log_node_entry,
    log_node_exit,
    create_error_state,
    should_retry_error,
)


class TestSafeNodeDecorator:
    """Test cases for @safe_node decorator."""

    @pytest.mark.asyncio
    async def test_safe_node_with_successful_async_function_returns_result(self):
        """Test that @safe_node with successful async function returns result."""
        # Arrange
        @safe_node()
        async def successful_node(state: dict) -> dict:
            return {"result": "success", "data": state.get("data", "")}
        
        # Act
        result = await successful_node({"data": "test"})
        
        # Assert
        assert result["result"] == "success"
        assert result["data"] == "test"
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_safe_node_with_failing_async_function_returns_error_state(self):
        """Test that @safe_node with failing async function returns error state."""
        # Arrange
        @safe_node()
        async def failing_node(state: dict) -> dict:
            raise ValueError("Test error")
        
        # Act
        result = await failing_node({"data": "test"})
        
        # Assert
        assert "error" in result
        assert "Test error" in result["error"]
        assert "failing_node" in result["error"]

    @pytest.mark.asyncio
    async def test_safe_node_with_custom_error_field_uses_custom_field(self):
        """Test that @safe_node with custom error field uses it."""
        # Arrange
        @safe_node(error_field="custom_error")
        async def failing_node(state: dict) -> dict:
            raise ValueError("Test error")
        
        # Act
        result = await failing_node({"data": "test"})
        
        # Assert
        assert "custom_error" in result
        assert "error" not in result
        assert "Test error" in result["custom_error"]

    @pytest.mark.asyncio
    async def test_safe_node_with_default_return_merges_error_into_default(self):
        """Test that @safe_node with default_return merges error."""
        # Arrange
        default_state = {"is_complete": False, "findings": []}
        
        @safe_node(default_return=default_state)
        async def failing_node(state: dict) -> dict:
            raise ValueError("Test error")
        
        # Act
        result = await failing_node({"data": "test"})
        
        # Assert
        assert result["is_complete"] is False
        assert result["findings"] == []
        assert "error" in result
        assert "Test error" in result["error"]

    @pytest.mark.asyncio
    async def test_safe_node_with_max_retries_retries_on_failure(self):
        """Test that @safe_node retries function on transient failures."""
        # Arrange
        call_count = 0
        
        @safe_node(max_retries=2, retry_delay=0.01)
        async def flaky_node(state: dict) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Transient error")
            return {"result": "success"}
        
        # Act
        result = await flaky_node({"data": "test"})
        
        # Assert
        assert call_count == 3  # Failed twice, succeeded on third attempt
        assert result["result"] == "success"
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_safe_node_with_max_retries_exhausted_returns_error(self):
        """Test that @safe_node returns error when retries exhausted."""
        # Arrange
        call_count = 0
        
        @safe_node(max_retries=2, retry_delay=0.01)
        async def failing_node(state: dict) -> dict:
            nonlocal call_count
            call_count += 1
            raise ValueError("Persistent error")
        
        # Act
        result = await failing_node({"data": "test"})
        
        # Assert
        assert call_count == 3  # Initial + 2 retries
        assert "error" in result
        assert "Persistent error" in result["error"]

    def test_safe_node_with_sync_function_handles_errors(self):
        """Test that @safe_node works with synchronous functions."""
        # Arrange
        @safe_node()
        def failing_sync_node(state: dict) -> dict:
            raise ValueError("Sync error")
        
        # Act
        result = failing_sync_node({"data": "test"})
        
        # Assert
        assert "error" in result
        assert "Sync error" in result["error"]

    def test_safe_node_with_successful_sync_function_returns_result(self):
        """Test that @safe_node with successful sync function returns result."""
        # Arrange
        @safe_node()
        def successful_sync_node(state: dict) -> dict:
            return {"result": "success", "data": state.get("data", "")}
        
        # Act
        result = successful_sync_node({"data": "test"})
        
        # Assert
        assert result["result"] == "success"
        assert result["data"] == "test"
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_safe_node_preserves_function_name_and_docstring(self):
        """Test that @safe_node preserves function metadata."""
        # Arrange
        @safe_node()
        async def documented_node(state: dict) -> dict:
            """This is a documented node."""
            return {"result": "success"}
        
        # Act & Assert
        assert documented_node.__name__ == "documented_node"
        assert documented_node.__doc__ == "This is a documented node."


class TestLogNodeFunctions:
    """Test cases for log_node_entry and log_node_exit."""

    @patch("app.utils.error_handling.logger")
    def test_log_node_entry_logs_node_name_and_state_keys(self, mock_logger):
        """Test that log_node_entry logs node name and state keys."""
        # Arrange
        state = {"research_brief": {}, "findings": [], "error": None}
        
        # Act
        log_node_entry("test_node", state)
        
        # Assert
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "test_node" in call_args
        mock_logger.debug.assert_called_once()

    @patch("app.utils.error_handling.logger")
    def test_log_node_exit_logs_node_name_and_result_keys(self, mock_logger):
        """Test that log_node_exit logs node name and result keys."""
        # Arrange
        result = {"findings": [], "is_complete": True}
        
        # Act
        log_node_exit("test_node", result)
        
        # Assert
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "test_node" in call_args
        mock_logger.debug.assert_called_once()


class TestCreateErrorState:
    """Test cases for create_error_state function."""

    def test_create_error_state_with_message_returns_error_dict(self):
        """Test that create_error_state returns dict with error message."""
        # Arrange & Act
        result = create_error_state("Test error occurred")
        
        # Assert
        assert result["error"] == "Test error occurred"
        assert result["error_type"] == "node_error"
        assert result["is_complete"] is False

    def test_create_error_state_with_custom_error_type_uses_it(self):
        """Test that create_error_state uses custom error type."""
        # Arrange & Act
        result = create_error_state("Test error", error_type="validation_error")
        
        # Assert
        assert result["error"] == "Test error"
        assert result["error_type"] == "validation_error"

    def test_create_error_state_with_additional_fields_includes_them(self):
        """Test that create_error_state includes additional fields."""
        # Arrange & Act
        result = create_error_state(
            "Test error",
            additional_fields={"task_id": "task_123", "retry_count": 3}
        )
        
        # Assert
        assert result["error"] == "Test error"
        assert result["task_id"] == "task_123"
        assert result["retry_count"] == 3

    def test_create_error_state_always_sets_is_complete_to_false(self):
        """Test that create_error_state always sets is_complete to False."""
        # Arrange & Act
        result = create_error_state("Test error")
        
        # Assert
        assert result["is_complete"] is False


class TestShouldRetryError:
    """Test cases for should_retry_error function."""

    def test_should_retry_error_with_timeout_error_returns_true(self):
        """Test that timeout errors are marked as retryable."""
        # Arrange
        error = Exception("Request timeout occurred")
        
        # Act
        result = should_retry_error(error)
        
        # Assert
        assert result is True

    def test_should_retry_error_with_connection_error_returns_true(self):
        """Test that connection errors are marked as retryable."""
        # Arrange
        error = Exception("Connection refused by server")
        
        # Act
        result = should_retry_error(error)
        
        # Assert
        assert result is True

    def test_should_retry_error_with_rate_limit_error_returns_true(self):
        """Test that rate limit errors are marked as retryable."""
        # Arrange
        error = Exception("Rate limit exceeded")
        
        # Act
        result = should_retry_error(error)
        
        # Assert
        assert result is True

    def test_should_retry_error_with_503_error_returns_true(self):
        """Test that 503 errors are marked as retryable."""
        # Arrange
        error = Exception("HTTP 503 Service Unavailable")
        
        # Act
        result = should_retry_error(error)
        
        # Assert
        assert result is True

    def test_should_retry_error_with_validation_error_returns_false(self):
        """Test that validation errors are not marked as retryable."""
        # Arrange
        error = ValueError("Invalid input format")
        
        # Act
        result = should_retry_error(error)
        
        # Assert
        assert result is False

    def test_should_retry_error_with_not_found_error_returns_false(self):
        """Test that not found errors are not marked as retryable."""
        # Arrange
        error = Exception("Resource not found")
        
        # Act
        result = should_retry_error(error)
        
        # Assert
        assert result is False

    def test_should_retry_error_is_case_insensitive(self):
        """Test that should_retry_error is case insensitive."""
        # Arrange
        error = Exception("CONNECTION TIMEOUT")
        
        # Act
        result = should_retry_error(error)
        
        # Assert
        assert result is True


class TestSafeNodeIntegration:
    """Integration tests for @safe_node decorator in realistic scenarios."""

    @pytest.mark.asyncio
    async def test_safe_node_with_langgraph_node_structure(self):
        """Test @safe_node with typical LangGraph node structure."""
        # Arrange
        from app.graphs.state import ResearchState
        
        @safe_node(max_retries=2, retry_delay=0.01)
        async def research_node(state: ResearchState) -> dict:
            # Simulate research logic that might fail
            if state.get("error"):
                raise ValueError("Previous error detected")
            
            return {
                "findings": [{"topic": "test", "summary": "results"}],
                "is_complete": True
            }
        
        initial_state: ResearchState = {
            "research_brief": None,
            "strategy": None,
            "tasks": [],
            "findings": [],
            "summarized_findings": None,
            "gaps": None,
            "extraction_budget": {"used": 0, "max": 5},
            "is_complete": False,
            "error": None,
            "messages": []
        }
        
        # Act
        result = await research_node(initial_state)
        
        # Assert
        assert result["is_complete"] is True
        assert len(result["findings"]) == 1
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_safe_node_with_error_recovery_pattern(self):
        """Test @safe_node with error recovery pattern."""
        # Arrange
        @safe_node(
            default_return={"is_complete": False, "findings": []},
            max_retries=1,
            retry_delay=0.01
        )
        async def risky_node(state: dict) -> dict:
            # Simulate persistent failure
            raise RuntimeError("Persistent failure")
        
        # Act
        result = await risky_node({"data": "test"})
        
        # Assert
        assert "error" in result
        assert result["is_complete"] is False
        assert result["findings"] == []
        assert "Persistent failure" in result["error"]


    @pytest.mark.asyncio
    async def test_safe_node_without_traceback_logging(self):
        """Test that @safe_node respects log_traceback=False."""
        # Arrange
        @safe_node(log_traceback=False)
        async def failing_node(state: dict) -> dict:
            raise ValueError("Test error")
        
        # Act
        with patch("app.utils.error_handling.logger") as mock_logger:
            await failing_node({})
            
            # Assert
            # Should log the error message but NOT the traceback
            assert mock_logger.error.call_count == 1
            assert "Traceback" not in mock_logger.error.call_args[0][0]

    def test_safe_node_sync_with_default_return(self):
        """Test that @safe_node sync wrapper uses default_return."""
        # Arrange
        default_state = {"is_complete": False, "findings": []}
        
        @safe_node(default_return=default_state)
        def failing_sync_node(state: dict) -> dict:
            raise ValueError("Sync error")
        
        # Act
        result = failing_sync_node({})
        
        # Assert
        assert result["is_complete"] is False
        assert result["findings"] == []
        assert "error" in result
        assert "Sync error" in result["error"]

    def test_safe_node_sync_without_traceback_logging(self):
        """Test that @safe_node sync wrapper respects log_traceback=False."""
        # Arrange
        @safe_node(log_traceback=False)
        def failing_sync_node(state: dict) -> dict:
            raise ValueError("Sync error")
        
        # Act
        with patch("app.utils.error_handling.logger") as mock_logger:
            failing_sync_node({})
            
            # Assert
            assert mock_logger.error.call_count == 1
            assert "Traceback" not in mock_logger.error.call_args[0][0]


class TestExampleUsage:
    """Test cases for example usage functions (for coverage)."""

    @pytest.mark.asyncio
    async def test_example_safe_node_usage_runs(self):
        """Test that example_safe_node_usage runs without error."""
        from app.utils.error_handling import example_safe_node_usage
        
        # Just ensure it runs, as it prints to stdout
        await example_safe_node_usage()
