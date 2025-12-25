"""
Middleware for LangChain agents to handle context trimming and tool safety.

This module provides:
1. TrimmingMiddleware: Manages context window by trimming messages before they reach the model.
2. ToolSafetyMiddleware: Wraps tool execution with error handling, truncation, and content extraction.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, cast

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import ToolException
from langgraph.types import Command
from pydantic import ValidationError

from app.tools.academic_tools import _extract_paper_sections

logger = logging.getLogger(__name__)


class TrimmingMiddleware(AgentMiddleware):
    """
    Middleware to trim conversation history before passing it to the model.
    Does NOT modify the persistent graph state, only the model input.
    """

    def __init__(
        self,
        max_tokens: int = 64000,
        strategy: str = "last",
        token_counter: Callable = count_tokens_approximately,
        start_on: str = "human",
        end_on: Union[str, Sequence[str]] = ("human", "tool"),
        include_system: bool = True,
    ):
        super().__init__()
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.token_counter = token_counter
        self.start_on = start_on
        self.end_on = end_on
        self.include_system = include_system

    def wrap_model_call(
        self, 
        request: ModelRequest, 
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelCallResult:
        """Synchronous wrapper for model call."""
        # Trim messages in the request
        request.messages = trim_messages(
            request.messages,
            max_tokens=self.max_tokens,
            strategy=self.strategy,
            token_counter=self.token_counter,
            start_on=self.start_on,
            end_on=self.end_on,
            include_system=self.include_system,
        )
        return handler(request)

    async def awrap_model_call(
        self, 
        request: ModelRequest, 
        handler: Callable[[ModelRequest], Any]
    ) -> ModelCallResult:
        """Asynchronous wrapper for model call."""
        # Trim messages in the request
        # trim_messages is synchronous, safe to call here
        request.messages = trim_messages(
            request.messages,
            max_tokens=self.max_tokens,
            strategy=self.strategy,
            token_counter=self.token_counter,
            start_on=self.start_on,
            end_on=self.end_on,
            include_system=self.include_system,
        )
        return await handler(request)


class ToolSafetyMiddleware(AgentMiddleware):
    """
    Middleware to safely execute tools.
    - Handles exceptions (ValidationError, ToolException).
    - Truncates long outputs.
    - Extracts relevant sections from academic papers (fetch_content).
    - Handles 'content_and_artifact' tool outputs.
    """

    MAX_TOOL_OUTPUT_CHARS = 15000

    def __init__(self):
        super().__init__()

    def _truncate_if_needed(self, text: str) -> str:
        """Truncate text if it exceeds max chars."""
        if len(text) > self.MAX_TOOL_OUTPUT_CHARS:
            return text[:self.MAX_TOOL_OUTPUT_CHARS] + "\n\n[OUTPUT TRUNCATED - Use more specific queries]"
        return text

    def _process_result(self, result: Any, tool_name: str, uses_content_and_artifact: bool) -> Any:
        """Process the raw result from the tool."""
        is_fetch_content = 'fetch_content' in tool_name.lower()

        if uses_content_and_artifact:
            # MCP tools return (content, artifact)
            if isinstance(result, (tuple, list)) and len(result) == 2:
                content, artifact = result[0], result[1]

                # Handle nested list structure from some MCP tools
                if is_fetch_content and isinstance(content, list) and len(content) == 2:
                    desc, full_text = content[0], content[1]
                    if isinstance(full_text, str) and len(full_text) > 10000:
                        extracted = _extract_paper_sections(full_text)
                        combined = f"{desc}\n\n{extracted}"
                        return (combined, None)
                    combined = f"{desc}\n\n{full_text}" if full_text else desc
                    return (combined, None)

                if content is None or content == "" or content == [] or content == {}:
                    return ("No results found. Try broadening your search or using different keywords.", artifact)

                if isinstance(content, str):
                    if is_fetch_content and len(content) > 10000:
                        content = _extract_paper_sections(content)
                    else:
                        content = self._truncate_if_needed(content)
                return (content, artifact)
            else:
                # Malformed content_and_artifact response
                logger.warning(f"Tool {tool_name} expects content_and_artifact but returned: {type(result)}")
                if isinstance(result, tuple) and len(result) == 1:
                    return (f"Tool returned malformed response. Content: {result[0]}", None)
                elif isinstance(result, str):
                     if is_fetch_content and len(result) > 10000:
                        result = _extract_paper_sections(result)
                     return (f"Tool returned malformed response. Content: {result}", result)
                else:
                    return (f"Tool returned malformed response: {str(result)[:500]}", None)
        else:
            # Standard tools
            if result is None or result == "" or result == [] or result == {}:
                return "No results found. Try broadening your search or using different keywords."
            
            if isinstance(result, str):
                if is_fetch_content and len(result) > 10000:
                    result = _extract_paper_sections(result)
                else:
                    result = self._truncate_if_needed(result)
            return result

    def _handle_error(self, e: Exception, tool_name: str, uses_content_and_artifact: bool) -> Any:
        """Handle execution errors."""
        logger.error(f"Error executing tool {tool_name}: {type(e).__name__}: {e}", exc_info=True)
        
        if isinstance(e, ValidationError):
            error_msg = f"Invalid argument: {str(e)}. Please check the tool schema and try again."
        elif isinstance(e, ToolException):
            error_msg = f"Tool execution failed: {str(e)}"
        else:
            error_msg = f"Unexpected error: {str(e)}"
        
        if uses_content_and_artifact:
            return (error_msg, None)
        return error_msg

    def wrap_tool_call(
        self, 
        request: ToolCallRequest, 
        handler: Callable[[ToolCallRequest], Union[ToolMessage, Command]]
    ) -> Union[ToolMessage, Command]:
        """Synchronous wrapper for tool call."""
        tool_name = request.tool.name
        uses_content_and_artifact = getattr(request.tool, 'response_format', None) == 'content_and_artifact'

        try:
            # Execute the tool
            result_msg = handler(request)
            
            if isinstance(result_msg, ToolMessage):
                content = result_msg.content
                artifact = result_msg.artifact
                
                is_fetch_content = 'fetch_content' in tool_name.lower()
                
                if isinstance(content, str):
                    if is_fetch_content and len(content) > 10000:
                        new_content = _extract_paper_sections(content)
                        return ToolMessage(
                            content=new_content, 
                            artifact=artifact, 
                            tool_call_id=result_msg.tool_call_id, 
                            name=result_msg.name, 
                            status=result_msg.status
                        )
                    
                    if len(content) > self.MAX_TOOL_OUTPUT_CHARS:
                         new_content = self._truncate_if_needed(content)
                         return ToolMessage(
                             content=new_content, 
                             artifact=artifact, 
                             tool_call_id=result_msg.tool_call_id, 
                             name=result_msg.name, 
                             status=result_msg.status
                        )
                
                return result_msg
            
            return result_msg
            
        except Exception as e:
            error_msg = self._handle_error(e, tool_name, uses_content_and_artifact)
            content = error_msg[0] if isinstance(error_msg, tuple) else error_msg
            art = error_msg[1] if isinstance(error_msg, tuple) else None
            return ToolMessage(
                content=str(content), 
                artifact=art, 
                tool_call_id=request.tool_call["id"], 
                status="error"
            )

    async def awrap_tool_call(
        self, 
        request: ToolCallRequest, 
        handler: Callable[[ToolCallRequest], Any]
    ) -> Union[ToolMessage, Command]:
        """Asynchronous wrapper for tool call."""
        tool_name = request.tool.name
        uses_content_and_artifact = getattr(request.tool, 'response_format', None) == 'content_and_artifact'
        
        try:
            result_msg = await handler(request)
            
            if isinstance(result_msg, ToolMessage):
                content = result_msg.content
                artifact = result_msg.artifact
                
                is_fetch_content = 'fetch_content' in tool_name.lower()
                
                if isinstance(content, str):
                    if is_fetch_content and len(content) > 10000:
                        new_content = _extract_paper_sections(content)
                        return ToolMessage(
                            content=new_content, 
                            artifact=artifact, 
                            tool_call_id=result_msg.tool_call_id, 
                            name=result_msg.name, 
                            status=result_msg.status
                        )
                    
                    if len(content) > self.MAX_TOOL_OUTPUT_CHARS:
                         new_content = self._truncate_if_needed(content)
                         return ToolMessage(
                             content=new_content, 
                             artifact=artifact, 
                             tool_call_id=result_msg.tool_call_id, 
                             name=result_msg.name, 
                             status=result_msg.status
                        )
                
                return result_msg
                
            return result_msg

        except Exception as e:
             error_msg = self._handle_error(e, tool_name, uses_content_and_artifact)
             content = error_msg[0] if isinstance(error_msg, tuple) else error_msg
             art = error_msg[1] if isinstance(error_msg, tuple) else None
             return ToolMessage(
                 content=str(content), 
                 artifact=art, 
                 tool_call_id=request.tool_call["id"], 
                 status="error"
            )
