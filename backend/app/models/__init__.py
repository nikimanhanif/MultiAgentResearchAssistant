"""
Models module containing Pydantic schemas and data structures.

This module exports the core data models used for API communication and
internal state management, including:
- ChatRequest/Response: API interaction models.
- ResearchRequest: Task definition models.
- ErrorResponse: Standardized error handling.
"""

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    ResearchRequest,
)

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ResearchRequest",
    "ErrorResponse",
]
