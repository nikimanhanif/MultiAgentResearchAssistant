"""Error handling utilities for LangGraph nodes.

This module provides decorators and utilities for graceful error handling in
LangGraph workflows, preventing node failures from crashing the entire graph.

Key Features:
- @safe_node decorator: Catches errors and returns failure state branch
- Error logging with context
- Graceful degradation patterns
- Retry logic support
"""

import functools
import logging
from typing import Callable, Any, Dict, Optional, TypeVar, cast
import traceback
import asyncio

logger = logging.getLogger(__name__)

T = TypeVar('T')


def safe_node(
    *,
    error_field: str = "error",
    log_traceback: bool = True,
    default_return: Optional[Dict[str, Any]] = None,
    max_retries: int = 0,
    retry_delay: float = 1.0
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to catch and handle errors in LangGraph nodes.
    
    Wraps a node function to catch exceptions and return a failure state
    instead of crashing the graph. Supports optional retry logic.
    
    Args:
        error_field: State field name to store error message (default: "error")
        log_traceback: Whether to log full traceback (default: True)
        default_return: Default state to return on error (default: {"error": "..."})
        max_retries: Number of retry attempts for transient failures (default: 0)
        retry_delay: Delay in seconds between retries (default: 1.0)
        
    Returns:
        Decorated function that handles errors gracefully
        
    Example:
        ```python
        @safe_node(max_retries=3, retry_delay=2.0)
        async def research_node(state: ResearchState) -> ResearchState:
            # Node logic that might fail
            result = await external_api_call()
            return {"findings": result}
        ```
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Async wrapper for node functions."""
            last_error: Optional[Exception] = None
            
            for attempt in range(max_retries + 1):
                try:
                    # Execute the node function
                    result = await func(*args, **kwargs)
                    return result
                    
                except Exception as e:
                    last_error = e
                    
                    # Log error with context
                    error_msg = f"Error in node '{func.__name__}' (attempt {attempt + 1}/{max_retries + 1}): {str(e)}"
                    logger.error(error_msg)
                    
                    if log_traceback:
                        logger.error(f"Traceback:\n{traceback.format_exc()}")
                    
                    # Retry if attempts remaining
                    if attempt < max_retries:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        continue
                    
                    # All retries exhausted, return error state
                    if default_return is not None:
                        error_state = default_return.copy()
                        error_state[error_field] = error_msg
                        return error_state
                    else:
                        return {error_field: error_msg}
            
            # Should never reach here, but handle gracefully
            return {error_field: f"Unexpected error in {func.__name__}"}
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Sync wrapper for node functions."""
            try:
                # Execute the node function
                result = func(*args, **kwargs)
                return result
                
            except Exception as e:
                # Log error with context
                error_msg = f"Error in node '{func.__name__}': {str(e)}"
                logger.error(error_msg)
                
                if log_traceback:
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                
                # Return error state
                if default_return is not None:
                    error_state = default_return.copy()
                    error_state[error_field] = error_msg
                    return error_state
                else:
                    return {error_field: error_msg}
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return cast(Callable[..., T], async_wrapper)
        else:
            return cast(Callable[..., T], sync_wrapper)
    
    return decorator


def log_node_entry(node_name: str, state: Dict[str, Any]) -> None:
    """Log node entry with state summary.
    
    Args:
        node_name: Name of the node being entered
        state: Current state dictionary
    """
    logger.info(f"Entering node: {node_name}")
    logger.debug(f"State keys: {list(state.keys())}")


def log_node_exit(node_name: str, result: Dict[str, Any]) -> None:
    """Log node exit with result summary.
    
    Args:
        node_name: Name of the node being exited
        result: Result dictionary returned by node
    """
    logger.info(f"Exiting node: {node_name}")
    logger.debug(f"Result keys: {list(result.keys())}")


def create_error_state(
    error_message: str,
    error_type: str = "node_error",
    additional_fields: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create standardized error state.
    
    Args:
        error_message: Error message to include in state
        error_type: Type/category of error
        additional_fields: Additional fields to include in error state
        
    Returns:
        Error state dictionary
    """
    error_state: Dict[str, Any] = {
        "error": error_message,
        "error_type": error_type,
        "is_complete": False
    }
    
    if additional_fields:
        error_state.update(additional_fields)
    
    return error_state


def should_retry_error(error: Exception) -> bool:
    """Determine if an error is transient and should be retried.
    
    Args:
        error: Exception that occurred
        
    Returns:
        True if error should be retried, False otherwise
    """
    # Common transient error patterns
    transient_errors = [
        "timeout",
        "connection",
        "rate limit",
        "service unavailable",
        "temporarily unavailable",
        "502",
        "503",
        "504"
    ]
    
    error_str = str(error).lower()
    return any(pattern in error_str for pattern in transient_errors)


# Example usage patterns

async def example_safe_node_usage() -> None:
    """Example demonstrating @safe_node decorator usage."""
    
    from app.graphs.state import ResearchState
    
    @safe_node(max_retries=3, retry_delay=2.0)
    async def risky_research_node(state: ResearchState) -> Dict[str, Any]:
        """Example node that might fail."""
        # Simulate risky operation
        if state.get("error"):
            raise ValueError("Previous error detected")
        
        # Normal processing
        return {
            "findings": [{"topic": "test", "summary": "results"}],
            "is_complete": True
        }
    
    # Usage
    initial_state: ResearchState = {"research_brief": None, "error": None}
    result = await risky_research_node(initial_state)
    
    if result.get("error"):
        print(f"Node failed: {result['error']}")
    else:
        print(f"Node succeeded: {result}")

