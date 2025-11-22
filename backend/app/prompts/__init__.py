"""Centralized prompt management for all agents.

This module provides prompt templates for the multi-agent research assistant.
All prompts use LangChain's ChatPromptTemplate for consistent formatting.

Exports:
    Scope Agent Prompts:
        - SCOPE_QUESTION_GENERATION_TEMPLATE
        - SCOPE_COMPLETION_DETECTION_TEMPLATE
        - SCOPE_BRIEF_GENERATION_TEMPLATE
    
    Research Agent Prompts:
        - (To be implemented in Phase 8)
    
    Report Agent Prompts:
        - (To be implemented in Phase 4)
"""

from app.prompts.scope_prompts import (
    SCOPE_QUESTION_GENERATION_TEMPLATE,
    SCOPE_COMPLETION_DETECTION_TEMPLATE,
    SCOPE_BRIEF_GENERATION_TEMPLATE,
)

__all__ = [
    # Scope Agent
    "SCOPE_QUESTION_GENERATION_TEMPLATE",
    "SCOPE_COMPLETION_DETECTION_TEMPLATE",
    "SCOPE_BRIEF_GENERATION_TEMPLATE",
]

