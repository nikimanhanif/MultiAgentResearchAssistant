"""Unit tests for Tavily tools integration.

Tests for langchain-tavily integration and credibility scoring.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch, TavilyExtract

from app.tools.tavily_tools import get_tavily_tools


class TestGetTavilyTools:
    """Test suite for get_tavily_tools function."""
    
    def test_returns_list_of_tools(self):
        """Test that get_tavily_tools returns a list of BaseTool instances."""
        with patch("app.tools.tavily_tools.settings") as mock_settings:
            mock_settings.TAVILY_API_KEY = "test-api-key"
            
            tools = get_tavily_tools()
            
            assert isinstance(tools, list)
            assert len(tools) == 2
            assert all(isinstance(tool, BaseTool) for tool in tools)
    
    def test_returns_tavily_search_and_extract(self):
        """Test that get_tavily_tools returns TavilySearch and TavilyExtract."""
        with patch("app.tools.tavily_tools.settings") as mock_settings:
            mock_settings.TAVILY_API_KEY = "test-api-key"
            
            tools = get_tavily_tools()
            
            assert isinstance(tools[0], TavilySearch)
            assert isinstance(tools[1], TavilyExtract)
    
    def test_tavily_search_configured_correctly(self):
        """Test that TavilySearch is configured with correct parameters."""
        with patch("app.tools.tavily_tools.settings") as mock_settings:
            mock_settings.TAVILY_API_KEY = "test-api-key"
            
            tools = get_tavily_tools()
            tavily_search = tools[0]
            
            assert tavily_search.max_results == 5
            assert tavily_search.search_depth == "advanced"
    
    def test_empty_api_key_raises_error(self):
        """Test that empty TAVILY_API_KEY raises ValueError."""
        with patch("app.tools.tavily_tools.settings") as mock_settings:
            mock_settings.TAVILY_API_KEY = ""
            
            with pytest.raises(ValueError) as exc:
                get_tavily_tools()
            
            assert "TAVILY_API_KEY not configured" in str(exc.value)
    
    def test_none_api_key_raises_error(self):
        """Test that None TAVILY_API_KEY raises ValueError."""
        with patch("app.tools.tavily_tools.settings") as mock_settings:
            mock_settings.TAVILY_API_KEY = None
            
            with pytest.raises(ValueError) as exc:
                get_tavily_tools()
            
            assert "TAVILY_API_KEY not configured" in str(exc.value)
    
    def test_initialization_error_raises_runtime_error(self):
        """Test that tool initialization errors are wrapped in RuntimeError."""
        with patch("app.tools.tavily_tools.settings") as mock_settings:
            mock_settings.TAVILY_API_KEY = "test-api-key"
            
            with patch("app.tools.tavily_tools.TavilySearch") as mock_search:
                mock_search.side_effect = Exception("API connection failed")
                
                with pytest.raises(RuntimeError) as exc:
                    get_tavily_tools()
                
                assert "Failed to initialize Tavily tools" in str(exc.value)
    
    def test_logging_on_success(self):
        """Test that successful initialization logs info message."""
        with patch("app.tools.tavily_tools.settings") as mock_settings:
            mock_settings.TAVILY_API_KEY = "test-api-key"
            
            with patch("app.tools.tavily_tools.logger") as mock_logger:
                get_tavily_tools()
                
                mock_logger.info.assert_called_once()
                assert "Successfully initialized 2 Tavily tools" in mock_logger.info.call_args[0][0]
    
    def test_logging_on_api_key_error(self):
        """Test that missing API key logs error message."""
        with patch("app.tools.tavily_tools.settings") as mock_settings:
            mock_settings.TAVILY_API_KEY = ""
            
            with patch("app.tools.tavily_tools.logger") as mock_logger:
                with pytest.raises(ValueError):
                    get_tavily_tools()
                
                mock_logger.error.assert_called_once()
                assert "TAVILY_API_KEY not configured" in mock_logger.error.call_args[0][0]
    
    def test_logging_on_initialization_error(self):
        """Test that initialization errors are logged."""
        with patch("app.tools.tavily_tools.settings") as mock_settings:
            mock_settings.TAVILY_API_KEY = "test-api-key"
            
            with patch("app.tools.tavily_tools.TavilySearch") as mock_search:
                mock_search.side_effect = Exception("Connection timeout")
                
                with patch("app.tools.tavily_tools.logger") as mock_logger:
                    with pytest.raises(RuntimeError):
                        get_tavily_tools()
                    
                    mock_logger.error.assert_called_once()
                    assert "Failed to initialize Tavily tools" in mock_logger.error.call_args[0][0]
