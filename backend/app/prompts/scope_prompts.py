"""Prompt templates for Scope Agent.

This module contains all prompt templates used by the Scope Agent for
multi-turn clarification conversations and research brief generation.

All prompts use LangChain's ChatPromptTemplate for consistent message formatting
and follow best practices from the LangChain documentation.
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# Question Generation Prompt
# Required inputs: user_query (str), conversation_history (str)
SCOPE_QUESTION_GENERATION_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a research scope clarification assistant. Your job is to ask clarifying questions to understand the user's research needs better.

Your goal is to gather information about:
- The specific aspect or angle they want to research
- The depth and breadth of research needed
- Any constraints (time period, geographic focus, etc.)
- The intended use or audience for the research
- Preferred format for the final report (if not already clear)

If the scope seems sufficiently clear, return an empty list of questions.

Return your response as a JSON object with this structure:
{{
    "questions": ["question 1", "question 2", ...],
    "context": "Brief explanation of why you're asking these questions"
}}

Return ONLY the JSON object, no other text."""
    ),
    HumanMessagePromptTemplate.from_template(
        """User's Query: {user_query}

Conversation History:
{conversation_history}

Based on the user's query and conversation so far, generate 1-3 clarifying questions."""
    ),
])


# Completion Detection Prompt
# Required inputs: user_query (str), conversation_history (str)
SCOPE_COMPLETION_DETECTION_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a research scope analyzer. Your job is to determine if we have enough information to proceed with research.

Analyze if we have enough information to:
1. Understand what the user wants to research
2. Know the scope and boundaries of the research
3. Understand any constraints or requirements
4. Know what format the final deliverable should take

Return your response as a JSON object:
{{
    "is_complete": true/false,
    "reason": "Brief explanation of your decision",
    "missing_info": ["list of missing information"] or []
}}

Return ONLY the JSON object, no other text."""
    ),
    HumanMessagePromptTemplate.from_template(
        """User's Original Query: {user_query}

Conversation History:
{conversation_history}

Determine if we have sufficient information to proceed with research."""
    ),
])


# Research Brief Generation Prompt
# Required inputs: user_query (str), conversation_history (str)
SCOPE_BRIEF_GENERATION_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a research brief generator. Your job is to create a comprehensive research brief from the clarification conversation.

Create a detailed research brief that will guide the research process. Include:
- Clear research scope and objectives
- Sub-topics to investigate
- Any constraints (time, geography, depth, etc.)
- Expected deliverables and format
- Any other relevant context

Return your response as a JSON object with this structure:
{{
    "scope": "Clear statement of the research scope",
    "sub_topics": ["subtopic 1", "subtopic 2", ...],
    "constraints": {{
        "time_period": "...",
        "geographic_focus": "...",
        "depth": "...",
        "other": "..."
    }},
    "deliverables": "Description of expected output",
    "format": "summary/comparison/ranking/detailed/literature_review/gap_analysis/other",
    "metadata": {{
        "original_query": "...",
        "clarification_turns": 0
    }}
}}

Return ONLY the JSON object, no other text."""
    ),
    HumanMessagePromptTemplate.from_template(
        """User's Original Query: {user_query}

Conversation History:
{conversation_history}

Generate a comprehensive research brief based on the conversation."""
    ),
])

