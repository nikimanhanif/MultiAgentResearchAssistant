"""Central tool registry for research pipeline.

This module combines all research tools (Tavily + MCP) into a unified registry.
Handles dynamic tool loading and adds safety wrappers for robust execution.

"""

import logging
from typing import List, Optional, Any
from pydantic import ValidationError

from langchain_core.tools import BaseTool, ToolException

from app.tools.tavily_tools import get_tavily_tools
from app.tools.mcp_tools import load_mcp_tools

logger = logging.getLogger(__name__)


def _safe_tool_execute(tool: BaseTool) -> BaseTool:
    """Wrap a tool with safety logic for empty results and validation errors.
    
    Args:
        tool: The tool to wrap
        
    Returns:
        The wrapped tool with error handling
    """
    # Store original run methods
    original_run = tool._run
    original_arun = tool._arun
    
    def _handle_result(result: Any) -> Any:
        """Handle empty results."""
        if result is None or result == "" or result == [] or result == {}:
            return "No results found. Try broadening your search or using different keywords."
        return result

    def _handle_error(e: Exception) -> str:
        """Handle execution errors."""
        if isinstance(e, ValidationError):
            return f"Invalid argument: {str(e)}. Please check the tool schema and try again."
        if isinstance(e, ToolException):
            return f"Tool execution failed: {str(e)}"
        return f"Unexpected error: {str(e)}"

    def safe_run(*args, config=None, **kwargs):
        try:
            # StructuredTool requires config to be passed explicitly
            result = original_run(*args, config=config, **kwargs)
            return _handle_result(result)
        except Exception as e:
            return _handle_error(e)

    async def safe_arun(*args, config=None, **kwargs):
        try:
            # StructuredTool requires config to be passed explicitly
            result = await original_arun(*args, config=config, **kwargs)
            return _handle_result(result)
        except Exception as e:
            return _handle_error(e)

    # Monkey-patch the tool instance
    # Note: This is a runtime modification of the instance. 
    # For Pydantic models, we need to be careful, but _run/_arun are standard methods.
    tool._run = safe_run
    tool._arun = safe_arun
    
    # Ensure tool handles validation errors gracefully by default
    tool.handle_tool_error = True
    tool.handle_validation_error = True
    
    return tool


async def get_research_tools(enabled_mcp_servers: Optional[List[str]] = None) -> List[BaseTool]:
    """Get all configured research tools (Tavily + MCP).
    
    Args:
        enabled_mcp_servers: List of MCP server names to enable. 
                           If None, no MCP tools will be loaded.
                           
    Returns:
        List of configured and wrapped tools
    """
    tools: List[BaseTool] = []
    
    # 1. Load Tavily Tools
    try:
        tavily_tools = get_tavily_tools()
        tools.extend(tavily_tools)
    except Exception as e:
        logger.error(f"Failed to load Tavily tools: {e}")
        # Continue without Tavily tools rather than crashing
    
    # 2. Load MCP Tools
    if enabled_mcp_servers:
        try:
            mcp_tools = await load_mcp_tools(enabled_mcp_servers)
            tools.extend(mcp_tools)
        except Exception as e:
            logger.error(f"Failed to load MCP tools: {e}")
            # Continue without MCP tools
            
    # 3. Apply Safety Wrapper
    wrapped_tools = [_safe_tool_execute(tool) for tool in tools]
    
    logger.info(f"Loaded {len(wrapped_tools)} research tools (Tavily + {len(enabled_mcp_servers or [])} MCP servers)")
    return wrapped_tools
