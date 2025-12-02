"""Additional test cases for content_and_artifact tools."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import ValidationError

from app.tools.tool_registry import _safe_tool_execute


class TestContentAndArtifactTools:
    """Test suite for tools with response_format='content_and_artifact'."""
    
    def test_preserves_valid_tuple_result(self):
        """Test that valid tuple results are preserved."""
        def tuple_func(x):
            return ("This is content", {"key": "artifact"})
        
        tool = StructuredTool.from_function(
            func=tuple_func,
            name="mcp_tool",
            description="MCP tool with tuple return"
        )
        # Simulate MCP tool response format
        tool.response_format = 'content_and_artifact'
        
        wrapped_tool = _safe_tool_execute(tool)
        # Use _run directly for content_and_artifact tools
        result = wrapped_tool._run("input")
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == "This is content"
        assert result[1] == {"key": "artifact"}
    
    def test_handles_empty_content_preserves_artifact(self):
        """Test that empty content is replaced but artifact is preserved."""
        def empty_content_func(x):
            return ("", {"key": "artifact"})
        
        tool = StructuredTool.from_function(
            func=empty_content_func,
            name="empty_mcp_tool",
            description="MCP tool with empty content"
        )
        tool.response_format = 'content_and_artifact'
        
        wrapped_tool = _safe_tool_execute(tool)
        result = wrapped_tool._run("input")
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert "No results found" in result[0]
        assert result[1] == {"key": "artifact"}
    
    def test_handles_none_content_preserves_artifact(self):
        """Test that None content is replaced but artifact is preserved."""
        def none_content_func(x):
            return (None, {"data": "preserved"})
        
        tool = StructuredTool.from_function(
            func=none_content_func,
            name="none_mcp_tool",
            description="MCP tool with None content"
        )
        tool.response_format = 'content_and_artifact'
        
        wrapped_tool = _safe_tool_execute(tool)
        result = wrapped_tool._run("input")
        
        assert isinstance(result, tuple)
        assert "No results found" in result[0]
        assert result[1] == {"data": "preserved"}
    
    def test_handles_malformed_tuple_single_element(self):
        """Test that malformed single-element tuple is wrapped properly."""
        def malformed_func(x):
            return ("only one element",)
        
        tool = StructuredTool.from_function(
            func=malformed_func,
            name="malformed_mcp_tool",
            description="MCP tool with malformed tuple"
        )
        tool.response_format = 'content_and_artifact'
        
        wrapped_tool = _safe_tool_execute(tool)
        result = wrapped_tool._run("input")
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert "Tool returned malformed response" in result[0]
    
    def test_handles_non_tuple_return(self):
        """Test that non-tuple return from content_and_artifact tool is wrapped."""
        def non_tuple_func(x):
            return "This should be a tuple but isn't"
        
        tool = StructuredTool.from_function(
            func=non_tuple_func,
            name="broken_mcp_tool",
            description="MCP tool returning string instead of tuple"
        )
        tool.response_format = 'content_and_artifact'
        
        wrapped_tool = _safe_tool_execute(tool)
        result = wrapped_tool._run("input")
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert "Tool returned malformed response" in result[0]
        assert result[1] == "This should be a tuple but isn't"
    
    def test_wraps_validation_error_in_tuple(self):
        """Test that ValidationError is wrapped in tuple for content_and_artifact tools."""
        def raise_validation_error(x):
            raise ValidationError.from_exception_data("Invalid input", [])
        
        tool = StructuredTool.from_function(
            func=raise_validation_error,
            name="validation_mcp_tool",
            description="MCP tool that raises ValidationError"
        )
        tool.response_format = 'content_and_artifact'
        
        wrapped_tool = _safe_tool_execute(tool)
        result = wrapped_tool._run("input")
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert "Invalid argument" in result[0]
        assert result[1] is None
    
    def test_wraps_tool_exception_in_tuple(self):
        """Test that ToolException is wrapped in tuple for content_and_artifact tools."""
        def raise_tool_exception(x):
            raise ToolException("MCP API failed")
        
        tool = StructuredTool.from_function(
            func=raise_tool_exception,
            name="exception_mcp_tool",
            description="MCP tool that raises ToolException"
        )
        tool.response_format = 'content_and_artifact'
        
        wrapped_tool = _safe_tool_execute(tool)
        result = wrapped_tool._run("input")
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert "Tool execution failed: MCP API failed" in result[0]
        assert result[1] is None
    
    def test_wraps_generic_exception_in_tuple(self):
        """Test that generic exceptions are wrapped in tuple for content_and_artifact tools."""
        def raise_generic_exception(x):
            raise RuntimeError("Unexpected MCP error")
        
        tool = StructuredTool.from_function(
            func=raise_generic_exception,
            name="runtime_error_mcp_tool",
            description="MCP tool that raises RuntimeError"
        )
        tool.response_format = 'content_and_artifact'
        
        wrapped_tool = _safe_tool_execute(tool)
        result = wrapped_tool._run("input")
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert "Unexpected error: Unexpected MCP error" in result[0]
        assert result[1] is None
    
    @pytest.mark.asyncio
    async def test_async_preserves_tuple(self):
        """Test async execution preserves tuple structure."""
        async def async_tuple_func(x):
            return ("Async content", {"async": "artifact"})
        
        tool = StructuredTool.from_function(
            coroutine=async_tuple_func,
            name="async_mcp_tool",
            description="Async MCP tool"
        )
        tool.response_format = 'content_and_artifact'
        
        wrapped_tool = _safe_tool_execute(tool)
        result = await wrapped_tool._arun("input")
        
        assert isinstance(result, tuple)
        assert result[0] == "Async content"
        assert result[1] == {"async": "artifact"}
    
    @pytest.mark.asyncio
    async def test_async_wraps_error_in_tuple(self):
        """Test async execution wraps errors in tuple."""
        async def async_fail(x):
            raise ValueError("Async MCP error")
        
        tool = StructuredTool.from_function(
            coroutine=async_fail,
            name="async_fail_mcp_tool",
            description="Async failing MCP tool"
        )
        tool.response_format = 'content_and_artifact'
        
        wrapped_tool = _safe_tool_execute(tool)
        result = await wrapped_tool._arun("input")
        
        assert isinstance(result, tuple)
        assert "Unexpected error: Async MCP error" in result[0]
        assert result[1] is None
    
    def test_mixed_tool_types_handled_correctly(self):
        """Test that wrapper correctly handles both regular and content_and_artifact tools."""
        # Regular tool
        regular_func = lambda x: "Regular result"
        regular_tool = StructuredTool.from_function(
            func=regular_func,
            name="regular_tool",
            description="Regular tool"
        )
        
        # MCP tool
        mcp_func = lambda x: ("MCP content", {"mcp": "artifact"})
        mcp_tool = StructuredTool.from_function(
            func=mcp_func,
            name="mcp_tool",
            description="MCP tool"
        )
        mcp_tool.response_format = 'content_and_artifact'
        
        # Wrap both
        wrapped_regular = _safe_tool_execute(regular_tool)
        wrapped_mcp = _safe_tool_execute(mcp_tool)
        
        regular_result = wrapped_regular._run("input")
        mcp_result = wrapped_mcp._run("input")
        
        # Regular tool returns string
        assert isinstance(regular_result, str)
        assert regular_result == "Regular result"
        
        # MCP tool returns tuple
        assert isinstance(mcp_result, tuple)
        assert mcp_result[0] == "MCP content"
        assert mcp_result[1] == {"mcp": "artifact"}
    
    def test_handles_empty_list_content_in_tuple(self):
        """Test that empty list content is detected and replaced."""
        def empty_list_func(x):
            return ([], {"artifact": "data"})
        
        tool = StructuredTool.from_function(
            func=empty_list_func,
            name="empty_list_mcp",
            description="MCP with empty list"
        )
        tool.response_format = 'content_and_artifact'
        
        wrapped_tool = _safe_tool_execute(tool)
        result = wrapped_tool._run("input")
        
        assert isinstance(result, tuple)
        assert "No results found" in result[0]
        assert result[1] == {"artifact": "data"}
    
    def test_handles_empty_dict_content_in_tuple(self):
        """Test that empty dict content is detected and replaced."""
        def empty_dict_func(x):
            return ({}, {"artifact": "preserved"})
        
        tool = StructuredTool.from_function(
            func=empty_dict_func,
            name="empty_dict_mcp",
            description="MCP with empty dict"
        )
        tool.response_format = 'content_and_artifact'
        
        wrapped_tool = _safe_tool_execute(tool)
        result = wrapped_tool._run("input")
        
        assert isinstance(result, tuple)
        assert "No results found" in result[0]
        assert result[1] == {"artifact": "preserved"}
