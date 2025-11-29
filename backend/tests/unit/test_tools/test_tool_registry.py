"""Unit tests for tool registry and safety wrappers.

Tests for tool combination and execution safety logic.

"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import ValidationError

from app.tools.tool_registry import get_research_tools, _safe_tool_execute


class TestSafeToolExecute:
    """Test suite for _safe_tool_execute wrapper."""
    
    def test_wraps_tool_successfully(self):
        """Test that wrapper returns a tool with modified run methods."""
        tool = StructuredTool.from_function(
            func=lambda x: x,
            name="test_tool",
            description="Test tool"
        )
        
        wrapped_tool = _safe_tool_execute(tool)
        
        assert wrapped_tool.handle_tool_error is True
        assert wrapped_tool.handle_validation_error is True
        assert wrapped_tool._run.__name__ == "safe_run"
        assert wrapped_tool._arun.__name__ == "safe_arun"

    def test_handles_empty_results(self):
        """Test that empty results are replaced with helpful message."""
        tool = StructuredTool.from_function(
            func=lambda x: "",
            name="empty_tool",
            description="Returns empty string"
        )
        wrapped_tool = _safe_tool_execute(tool)
        
        result = wrapped_tool.invoke("input")
        assert "No results found" in result

    def test_handles_none_results(self):
        """Test that None results are replaced with helpful message."""
        tool = StructuredTool.from_function(
            func=lambda x: None,
            name="none_tool",
            description="Returns None"
        )
        wrapped_tool = _safe_tool_execute(tool)
        
        result = wrapped_tool.invoke("input")
        assert "No results found" in result

    def test_handles_validation_error(self):
        """Test that ValidationError is caught and formatted."""
        def raise_validation_error(x):
            raise ValidationError.from_exception_data("Invalid input", [])
            
        tool = StructuredTool.from_function(
            func=raise_validation_error,
            name="validation_error_tool",
            description="Raises ValidationError"
        )
        wrapped_tool = _safe_tool_execute(tool)
        
        # We need to mock the _run method directly because invoke catches exceptions differently
        # depending on handle_tool_error settings
        result = wrapped_tool._run("input")
        assert "Invalid argument" in result

    def test_handles_tool_exception(self):
        """Test that ToolException is caught and formatted."""
        def raise_tool_exception(x):
            raise ToolException("API failed")
            
        tool = StructuredTool.from_function(
            func=raise_tool_exception,
            name="tool_exception_tool",
            description="Raises ToolException"
        )
        wrapped_tool = _safe_tool_execute(tool)
        
        result = wrapped_tool._run("input")
        assert "Tool execution failed: API failed" in result

    @pytest.mark.asyncio
    async def test_async_safe_run(self):
        """Test async execution safety wrapper."""
        async def async_func(x):
            return ""
            
        tool = StructuredTool.from_function(
            coroutine=async_func,
            name="async_tool",
            description="Async tool"
        )
        wrapped_tool = _safe_tool_execute(tool)
        
        result = await wrapped_tool.ainvoke("input")
        assert "No results found" in result


class TestGetResearchTools:
    """Test suite for get_research_tools function."""
    
    @pytest.mark.asyncio
    async def test_combines_tavily_and_mcp_tools(self):
        """Test that tools from both sources are combined."""
        tavily_tool = StructuredTool.from_function(lambda x: x, name="tavily", description="tavily")
        mcp_tool = StructuredTool.from_function(lambda x: x, name="mcp", description="mcp")
        
        with patch("app.tools.tool_registry.get_tavily_tools", return_value=[tavily_tool]):
            with patch("app.tools.tool_registry.load_mcp_tools", new_callable=AsyncMock) as mock_load:
                mock_load.return_value = [mcp_tool]
                
                tools = await get_research_tools(["server"])
                
                assert len(tools) == 2
                assert tools[0].name == "tavily"
                assert tools[1].name == "mcp"

    @pytest.mark.asyncio
    async def test_handles_tavily_failure_gracefully(self):
        """Test that Tavily loading failure doesn't crash execution."""
        mcp_tool = StructuredTool.from_function(lambda x: x, name="mcp", description="mcp")
        
        with patch("app.tools.tool_registry.get_tavily_tools", side_effect=Exception("Tavily failed")):
            with patch("app.tools.tool_registry.load_mcp_tools", new_callable=AsyncMock) as mock_load:
                mock_load.return_value = [mcp_tool]
                
                tools = await get_research_tools(["server"])
                
                assert len(tools) == 1
                assert tools[0].name == "mcp"

    @pytest.mark.asyncio
    async def test_handles_mcp_failure_gracefully(self):
        """Test that MCP loading failure doesn't crash execution."""
        tavily_tool = StructuredTool.from_function(lambda x: x, name="tavily", description="tavily")
        
        with patch("app.tools.tool_registry.get_tavily_tools", return_value=[tavily_tool]):
            with patch("app.tools.tool_registry.load_mcp_tools", new_callable=AsyncMock) as mock_load:
                mock_load.side_effect = Exception("MCP failed")
                
                tools = await get_research_tools(["server"])
                
                assert len(tools) == 1
                assert tools[0].name == "tavily"

    @pytest.mark.asyncio
    async def test_wraps_all_tools(self):
        """Test that all returned tools are wrapped with safety logic."""
        tavily_tool = StructuredTool.from_function(lambda x: x, name="tavily", description="tavily")
        
        with patch("app.tools.tool_registry.get_tavily_tools", return_value=[tavily_tool]):
            tools = await get_research_tools()
            
            assert tools[0]._run.__name__ == "safe_run"
