"""
Central tool registry for research pipeline.

Combines all research tools (Tavily + Academic) into a unified registry.
Handles dynamic tool loading and adds safety wrappers for robust execution.
"""

from contextlib import asynccontextmanager
import logging
import os
from typing import List, Optional, Any, AsyncGenerator
from pydantic import ValidationError

from langchain_core.tools import BaseTool, ToolException

from app.tools.tavily_tools import get_tavily_tools
from app.tools.academic_tools import get_academic_tools, _extract_paper_sections

logger = logging.getLogger(__name__)


def _safe_tool_execute(tool: BaseTool) -> BaseTool:
    """
    Wrap a tool with safety logic for empty results and validation errors.
    
    Handles both regular tools (string returns) and MCP tools with 
    response_format='content_and_artifact' (tuple returns).
    
    For fetch_content tool, extracts only Introduction and Conclusion sections.
    
    Args:
        tool: The tool to wrap.
        
    Returns:
        BaseTool: The wrapped tool with error handling.
    """
    original_run = tool._run
    original_arun = tool._arun
    
    uses_content_and_artifact = getattr(tool, 'response_format', None) == 'content_and_artifact'
    is_fetch_content = 'fetch_content' in tool.name.lower()
    
    MAX_TOOL_OUTPUT_CHARS = 15000
    
    def _truncate_if_needed(text: str) -> str:
        """Truncate text if it exceeds max chars."""
        if len(text) > MAX_TOOL_OUTPUT_CHARS:
            return text[:MAX_TOOL_OUTPUT_CHARS] + "\n\n[OUTPUT TRUNCATED - Use more specific queries]"
        return text
    
    def _handle_result(result: Any) -> Any:
        """Handle empty results, extract sections for fetch_content, truncate others."""
        if uses_content_and_artifact:
            # MCP tools return (list_of_content, artifact) for content_and_artifact format
            # For fetch_content: ([description_str, full_paper_str], None)
            if isinstance(result, (tuple, list)) and len(result) == 2:
                content, artifact = result[0], result[1]
                
                # Handle nested list structure: content = [description, full_paper_text]
                if is_fetch_content and isinstance(content, list) and len(content) == 2:
                    desc, full_text = content[0], content[1]
                    if isinstance(full_text, str) and len(full_text) > 10000:
                        extracted = _extract_paper_sections(full_text)
                        # Return combined content as main result for agent
                        combined = f"{desc}\n\n{extracted}"
                        return (combined, None)
                    # If not long enough, still combine
                    combined = f"{desc}\n\n{full_text}" if full_text else desc
                    return (combined, None)
                
                # For other tools or different formats
                if content is None or content == "" or content == [] or content == {}:
                    return ("No results found. Try broadening your search or using different keywords.", artifact)
                if isinstance(content, str):
                    if is_fetch_content and len(content) > 10000:
                        content = _extract_paper_sections(content)
                    else:
                        content = _truncate_if_needed(content)
                return (content, artifact)
            else:
                # Handle malformed returns from content_and_artifact tools
                logger.warning(f"Tool {tool.name} has response_format='content_and_artifact' but returned unexpected: {type(result)}")
                
                if isinstance(result, tuple) and len(result) == 1:
                    # Single-element tuple - wrap with None artifact
                    return (f"Tool returned malformed response. Content: {result[0]}", None)
                elif isinstance(result, str):
                    # String instead of tuple - wrap it
                    if is_fetch_content and len(result) > 10000:
                        result = _extract_paper_sections(result)
                    return (f"Tool returned malformed response. Content: {result}", result)
                else:
                    # Other types - convert to string and wrap
                    return (f"Tool returned malformed response: {str(result)[:500]}", None)
        else:
            if result is None or result == "" or result == [] or result == {}:
                return "No results found. Try broadening your search or using different keywords."
            if isinstance(result, str):
                if is_fetch_content and len(result) > 10000:
                    result = _extract_paper_sections(result)
                else:
                    result = _truncate_if_needed(result)
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
    tool._run = safe_run
    tool._arun = safe_arun
    
    # Also wrap coroutine if present (StructuredTool.ainvoke calls this directly)
    if hasattr(tool, 'coroutine') and tool.coroutine is not None:
        original_coroutine = tool.coroutine
        
        async def safe_coroutine(*args, **kwargs):
            try:
                result = await original_coroutine(*args, **kwargs)
                return _handle_result(result)
            except Exception as e:
                return _handle_error(e, args, kwargs)
        
        tool.coroutine = safe_coroutine
    
    # Ensure tool handles validation errors gracefully by default
    tool.handle_tool_error = True
    tool.handle_validation_error = True
    
    return tool


@asynccontextmanager
async def get_research_tools(enabled_mcp_servers: Optional[List[str]] = None) -> AsyncGenerator[List[BaseTool], None]:
    """
    Get all configured research tools (Tavily + Academic) within a context manager.
    
    Note: The context manager pattern is preserved for backwards compatibility
    with existing sub_agent.py code, even though academic tools don't require it.
    
    Args:
        enabled_mcp_servers: DEPRECATED - Kept for backwards compatibility.
                           Academic tools are always loaded regardless of this parameter.
                           
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
            logger.info(f"Loaded {len(tavily_tools)} Tavily tool(s)")
        except Exception as e:
            logger.error(f"Failed to load Tavily tools: {e}")
    
    # 2. Load Academic Tools (replaces MCP scientific-papers server)
    try:
        academic_tools = get_academic_tools()
        tools.extend(academic_tools)
        logger.info(f"Loaded {len(academic_tools)} academic tool(s): {[t.name for t in academic_tools]}")
    except Exception as e:
        logger.error(f"Failed to load academic tools: {e}")
    
    # Wrap all tools with safety logic
    wrapped_tools = [_safe_tool_execute(tool) for tool in tools]
    
    logger.info(f"Total research tools available: {len(wrapped_tools)}")
    yield wrapped_tools
