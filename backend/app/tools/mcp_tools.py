"""
MCP server tool configuration and client management.

Manages MCP server connections using langchain-mcp-adapters and provides
automatic tool discovery from enabled MCP servers.
"""

import logging
import os
from typing import List, Dict

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import settings

logger = logging.getLogger(__name__)

server_configs: Dict[str, Dict] = {
    "scientific-papers": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@futurelab-studio/latest-science-mcp@latest"],
        "env": {"CORE_API_KEY": os.getenv("CORE_API_KEY", settings.CORE_API_KEY)}
    }
}


def get_mcp_client(enabled_servers: List[str]) -> MultiServerMCPClient:
    """
    Get configured MCP client for enabled servers.
    
    Args:
        enabled_servers: List of server names to enable (must exist in server_configs).
        
    Returns:
        MultiServerMCPClient: Client instance with filtered server configurations.
        
    Raises:
        ValueError: If any enabled server name is not in server_configs.
    """
    if not enabled_servers:
        logger.warning("No MCP servers enabled, returning client with empty config")
        return MultiServerMCPClient({})
    
    invalid_servers = [s for s in enabled_servers if s not in server_configs]
    if invalid_servers:
        error_msg = f"Invalid server names: {invalid_servers}. Available: {list(server_configs.keys())}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    filtered_configs = {
        name: config 
        for name, config in server_configs.items() 
        if name in enabled_servers
    }
    
    logger.info(f"Initializing MCP client with servers: {list(filtered_configs.keys())}")
    return MultiServerMCPClient(filtered_configs)


async def load_mcp_tools(enabled_servers: List[str]) -> List[BaseTool]:
    """
    Load tools from enabled MCP servers with automatic discovery.
    
    Args:
        enabled_servers: List of server names to load tools from.
        
    Returns:
        List[BaseTool]: List of BaseTool instances from all enabled servers.
        
    Raises:
        ValueError: If any enabled server name is invalid.
    """
    if not enabled_servers:
        logger.info("No MCP servers enabled, returning empty tool list")
        return []
    
    try:
        logger.info(f"Loading MCP tools from servers: {enabled_servers}")
        client = get_mcp_client(enabled_servers)
        
        async with client:
            logger.info(f"MCP client connected to {len(enabled_servers)} server(s)")
            tools = client.get_tools()  # Synchronous call, but context manager needed for connection
            
            logger.info(f"Successfully loaded {len(tools)} tools from {len(enabled_servers)} MCP server(s)")
            if tools:
                for i, tool in enumerate(tools, 1):
                    logger.info(f"  Tool {i}: {tool.name} - {tool.description[:80] if tool.description else 'No description'}...")
            
            return tools
        
    except ValueError:
        raise
        
    except Exception as e:
        error_msg = f"Failed to load MCP tools from servers {enabled_servers}: {str(e)}"
        logger.error(error_msg)
        logger.warning("Returning empty tool list due to MCP server connection failure")
        return []
