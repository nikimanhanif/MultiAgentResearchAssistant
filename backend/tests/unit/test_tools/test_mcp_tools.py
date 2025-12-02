"""Unit tests for MCP tools integration.

Tests for langchain-mcp-adapters integration and tool discovery.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.tools import BaseTool, StructuredTool

from app.tools.mcp_tools import server_configs, get_mcp_client, load_mcp_tools


class TestServerConfigs:
    """Test suite for server_configs dictionary."""
    
    def test_server_configs_is_dict(self):
        """Test that server_configs is a dictionary."""
        assert isinstance(server_configs, dict)
    
    def test_scientific_papers_config_exists(self):
        """Test that scientific-papers server is configured."""
        assert "scientific-papers" in server_configs
    
    def test_scientific_papers_config_structure(self):
        """Test that scientific-papers config has required fields."""
        config = server_configs["scientific-papers"]
        
        assert config["transport"] == "stdio"
        assert config["command"] == "npx"
        assert config["args"] == ["-y", "@futurelab-studio/latest-science-mcp@latest"]
        assert "env" in config
        assert "CORE_API_KEY" in config["env"]


class TestGetMcpClient:
    """Test suite for get_mcp_client function."""
    
    def test_returns_mcp_client_with_valid_servers(self):
        """Test that get_mcp_client returns MultiServerMCPClient instance."""
        from langchain_mcp_adapters.client import MultiServerMCPClient
        
        client = get_mcp_client(["scientific-papers"])
        
        assert isinstance(client, MultiServerMCPClient)
    
    def test_filters_enabled_servers_correctly(self):
        """Test that only enabled servers are included in client config."""
        with patch("app.tools.mcp_tools.MultiServerMCPClient") as mock_client_class:
            get_mcp_client(["scientific-papers"])
            
            call_args = mock_client_class.call_args[0][0]
            assert "scientific-papers" in call_args
            assert len(call_args) == 1
    
    def test_empty_server_list_returns_empty_client(self):
        """Test that empty enabled_servers returns client with empty config."""
        with patch("app.tools.mcp_tools.MultiServerMCPClient") as mock_client_class:
            get_mcp_client([])
            
            call_args = mock_client_class.call_args[0][0]
            assert call_args == {}
    
    def test_invalid_server_name_raises_error(self):
        """Test that invalid server name raises ValueError."""
        with pytest.raises(ValueError) as exc:
            get_mcp_client(["invalid-server"])
        
        assert "Invalid server names: ['invalid-server']" in str(exc.value)
        assert "Available:" in str(exc.value)
    
    def test_mixed_valid_invalid_servers_raises_error(self):
        """Test that mix of valid and invalid servers raises ValueError."""
        with pytest.raises(ValueError) as exc:
            get_mcp_client(["scientific-papers", "fake-server"])
        
        assert "fake-server" in str(exc.value)
    
    def test_logging_on_success(self):
        """Test that successful initialization logs info message."""
        with patch("app.tools.mcp_tools.logger") as mock_logger:
            get_mcp_client(["scientific-papers"])
            
            mock_logger.info.assert_called_once()
            assert "scientific-papers" in mock_logger.info.call_args[0][0]
    
    def test_logging_on_empty_list(self):
        """Test that empty server list logs warning."""
        with patch("app.tools.mcp_tools.logger") as mock_logger:
            get_mcp_client([])
            
            mock_logger.warning.assert_called_once()
            assert "No MCP servers enabled" in mock_logger.warning.call_args[0][0]
    
    def test_logging_on_invalid_server(self):
        """Test that invalid server logs error."""
        with patch("app.tools.mcp_tools.logger") as mock_logger:
            with pytest.raises(ValueError):
                get_mcp_client(["invalid"])
            
            mock_logger.error.assert_called_once()
            assert "Invalid server names" in mock_logger.error.call_args[0][0]


class TestLoadMcpTools:
    """Test suite for load_mcp_tools async function."""
    
    @pytest.mark.asyncio
    async def test_returns_list_of_tools(self):
        """Test that load_mcp_tools returns list of BaseTool instances."""
        mock_tool = StructuredTool.from_function(
            func=lambda x: x,
            name="test_tool",
            description="Test tool"
        )
        
        with patch("app.tools.mcp_tools.get_mcp_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get_tools = MagicMock(return_value=[mock_tool])
            mock_get_client.return_value = mock_client
            
            tools = await load_mcp_tools(["scientific-papers"])
            
            assert isinstance(tools, list)
            assert len(tools) == 1
            assert isinstance(tools[0], BaseTool)
    
    @pytest.mark.asyncio
    async def test_empty_server_list_returns_empty_list(self):
        """Test that empty enabled_servers returns empty list."""
        tools = await load_mcp_tools([])
        
        assert tools == []
    
    @pytest.mark.asyncio
    async def test_calls_get_mcp_client_with_servers(self):
        """Test that load_mcp_tools calls get_mcp_client correctly."""
        with patch("app.tools.mcp_tools.get_mcp_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get_tools = MagicMock(return_value=[])
            mock_get_client.return_value = mock_client
            
            await load_mcp_tools(["scientific-papers"])
            
            mock_get_client.assert_called_once_with(["scientific-papers"])
    
    @pytest.mark.asyncio
    async def test_graceful_fallback_on_connection_error(self):
        """Test that connection errors return empty list instead of crashing."""
        with patch("app.tools.mcp_tools.get_mcp_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(side_effect=ConnectionError("Server unavailable"))
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_get_client.return_value = mock_client
            
            tools = await load_mcp_tools(["scientific-papers"])
            
            assert tools == []
    
    @pytest.mark.asyncio
    async def test_graceful_fallback_on_generic_error(self):
        """Test that generic errors return empty list with logging."""
        with patch("app.tools.mcp_tools.get_mcp_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(side_effect=Exception("Unknown error"))
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_get_client.return_value = mock_client
            
            with patch("app.tools.mcp_tools.logger") as mock_logger:
                tools = await load_mcp_tools(["scientific-papers"])
                
                assert tools == []
                mock_logger.error.assert_called_once()
                mock_logger.warning.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_value_error_propagates(self):
        """Test that ValueError from invalid servers propagates correctly."""
        with pytest.raises(ValueError) as exc:
            await load_mcp_tools(["invalid-server"])
        
        assert "Invalid server names" in str(exc.value)
    
    @pytest.mark.asyncio
    async def test_logging_on_success(self):
        """Test that successful tool loading logs info message."""
        mock_tool = StructuredTool.from_function(
            func=lambda x: x,
            name="test_tool",
            description="Test tool"
        )
        
        with patch("app.tools.mcp_tools.get_mcp_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get_tools = MagicMock(return_value=[mock_tool])
            mock_get_client.return_value = mock_client
            
            with patch("app.tools.mcp_tools.logger") as mock_logger:
                await load_mcp_tools(["scientific-papers"])
                
                assert any("Successfully loaded" in str(call) for call in mock_logger.info.call_args_list)
    
    @pytest.mark.asyncio
    async def test_logging_on_empty_list(self):
        """Test that empty server list logs info message."""
        with patch("app.tools.mcp_tools.logger") as mock_logger:
            await load_mcp_tools([])
            
            mock_logger.info.assert_called_once()
            assert "No MCP servers enabled" in mock_logger.info.call_args[0][0]
