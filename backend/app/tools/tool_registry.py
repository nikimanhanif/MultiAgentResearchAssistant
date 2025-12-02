"""
Central tool registry for research pipeline.

Combines all research tools (Tavily + MCP) into a unified registry.
Handles dynamic tool loading and adds safety wrappers for robust execution.
"""

from contextlib import asynccontextmanager
import logging
import os
from typing import List, Optional, Any, AsyncGenerator
from pydantic import ValidationError

from langchain_core.tools import BaseTool, ToolException

from app.tools.tavily_tools import get_tavily_tools
from app.tools.mcp_tools import get_mcp_client

logger = logging.getLogger(__name__)


def _safe_tool_execute(tool: BaseTool) -> BaseTool:
    """
    Wrap a tool with safety logic for empty results and validation errors.
    
    Handles both regular tools (string returns) and MCP tools with 
    response_format='content_and_artifact' (tuple returns).
    
    Args:
        tool: The tool to wrap.
        
    Returns:
        BaseTool: The wrapped tool with error handling.
    """
    original_run = tool._run
    original_arun = tool._arun
    
    uses_content_and_artifact = getattr(tool, 'response_format', None) == 'content_and_artifact'
    
    def _handle_result(result: Any) -> Any:
        """Handle empty results, preserving tuple structure if needed."""
        if uses_content_and_artifact:
            if isinstance(result, tuple) and len(result) == 2:
                content, artifact = result
                if content is None or content == "" or content == [] or content == {}:
                    return ("No results found. Try broadening your search or using different keywords.", artifact)
                return result
            else:
                logger.warning(f"Tool {tool.name} has response_format='content_and_artifact' but returned non-tuple: {type(result)}")
                return ("Tool returned malformed response.", result)
        else:
            if result is None or result == "" or result == [] or result == {}:
                return "No results found. Try broadening your search or using different keywords."
            return result

    def _handle_error(e: Exception, args: tuple, kwargs: dict) -> Any:
        """Handle execution errors, wrapping in tuple if needed."""
        logger.error(f"Error executing tool {tool.name}: {type(e).__name__}: {e}", exc_info=True)
        
        if isinstance(e, ValidationError):
            error_msg = f"Invalid argument: {str(e)}. Please check the tool schema and try again."
        elif isinstance(e, ToolException):
            error_msg = f"Tool execution failed: {str(e)}"
        else:
            error_msg = f"Unexpected error: {str(e)}"
        
        if uses_content_and_artifact:
            return (error_msg, None)
        return error_msg

    def safe_run(*args, config=None, **kwargs):
        try:
            result = original_run(*args, config=config, **kwargs)
            return _handle_result(result)
        except Exception as e:
            return _handle_error(e, args, kwargs)

    async def safe_arun(*args, config=None, **kwargs):
        try:
            result = await original_arun(*args, config=config, **kwargs)
            return _handle_result(result)
        except Exception as e:
            return _handle_error(e, args, kwargs)

    # Monkey-patch the tool instance
    # Note: This is a runtime modification of the instance. 
    # For Pydantic models, we need to be careful, but _run/_arun are standard methods.
    tool._run = safe_run
    tool._arun = safe_arun
    
    # Ensure tool handles validation errors gracefully by default
    tool.handle_tool_error = True
    tool.handle_validation_error = True
    
    return tool


@asynccontextmanager
async def get_research_tools(enabled_mcp_servers: Optional[List[str]] = None) -> AsyncGenerator[List[BaseTool], None]:
    """
    Get all configured research tools (Tavily + MCP) within a context manager.
    
    Must be used as an async context manager to ensure MCP client connections 
    stay alive during tool usage.
    
    Args:
        enabled_mcp_servers: List of MCP server names to enable. 
                           If None, no MCP tools will be loaded.
                           
    Yields:
        List[BaseTool]: List of configured and wrapped tools.
    """
    tools: List[BaseTool] = []
    
    # 1. Load Tavily Tools
    if os.getenv("DISABLE_TAVILY", "").lower() == "true":
        logger.info("Tavily tools disabled via environment variable")
    else:
        try:
            tavily_tools = get_tavily_tools()
            tools.extend(tavily_tools)
        except Exception as e:
            logger.error(f"Failed to load Tavily tools: {e}")
    
    # 2. Load MCP Tools
    if enabled_mcp_servers:
        try:
            client = get_mcp_client(enabled_mcp_servers)
            async with client:
                mcp_tools = client.get_tools()
                tools.extend(mcp_tools)
                
                wrapped_tools = [_safe_tool_execute(tool) for tool in tools]
                yield wrapped_tools
                
        except Exception as e:
            logger.error(f"Failed to load MCP tools: {e}")
            wrapped_tools = [_safe_tool_execute(tool) for tool in tools]
            yield wrapped_tools
    else:
        wrapped_tools = [_safe_tool_execute(tool) for tool in tools]
        yield wrapped_tools
