"""Unit tests for tool registry."""

import pytest
from unittest.mock import patch
from langchain_core.tools import StructuredTool

from app.tools.tool_registry import get_research_tools


class TestGetResearchTools:
    """Test suite for get_research_tools function."""
    
    @pytest.mark.asyncio
    async def test_combines_tavily_and_academic_tools(self):
        """Test that tools from both sources are combined."""
        tavily_tool = StructuredTool.from_function(lambda x: x, name="tavily", description="tavily")
        academic_tool = StructuredTool.from_function(lambda x: x, name="search_papers", description="search")
        
        with patch("app.tools.tool_registry.get_tavily_tools", return_value=[tavily_tool]):
            with patch("app.tools.tool_registry.get_academic_tools", return_value=[academic_tool]):
                async with get_research_tools() as tools:
                    assert len(tools) == 2
                    assert tools[0].name == "tavily"
                    assert tools[1].name == "search_papers"

    @pytest.mark.asyncio
    async def test_handles_tavily_failure_gracefully(self):
        """Test that Tavily loading failure doesn't crash execution."""
        academic_tool = StructuredTool.from_function(lambda x: x, name="search_papers", description="search")
        
        with patch("app.tools.tool_registry.get_tavily_tools", side_effect=Exception("Tavily failed")):
            with patch("app.tools.tool_registry.get_academic_tools", return_value=[academic_tool]):
                async with get_research_tools() as tools:
                    assert len(tools) == 1
                    assert tools[0].name == "search_papers"

    @pytest.mark.asyncio
    async def test_handles_academic_failure_gracefully(self):
        """Test that academic tools loading failure doesn't crash execution."""
        tavily_tool = StructuredTool.from_function(lambda x: x, name="tavily", description="tavily")
        
        with patch("app.tools.tool_registry.get_tavily_tools", return_value=[tavily_tool]):
            with patch("app.tools.tool_registry.get_academic_tools", side_effect=Exception("Academic failed")):
                async with get_research_tools() as tools:
                    assert len(tools) == 1
                    assert tools[0].name == "tavily"

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_all_fail(self):
        """Test that empty list is returned when all tool sources fail."""
        with patch("app.tools.tool_registry.get_tavily_tools", side_effect=Exception("Tavily failed")):
            with patch("app.tools.tool_registry.get_academic_tools", side_effect=Exception("Academic failed")):
                async with get_research_tools() as tools:
                    assert len(tools) == 0

    @pytest.mark.asyncio
    async def test_disabled_tavily_via_env(self):
        """Test that Tavily can be disabled via environment variable."""
        academic_tool = StructuredTool.from_function(lambda x: x, name="search_papers", description="search")
        
        with patch.dict("os.environ", {"DISABLE_TAVILY": "true"}):
            with patch("app.tools.tool_registry.get_tavily_tools") as mock_tavily:
                with patch("app.tools.tool_registry.get_academic_tools", return_value=[academic_tool]):
                    async with get_research_tools() as tools:
                        mock_tavily.assert_not_called()
                        assert len(tools) == 1
