"""Scope Agent - Human-in-the-loop clarification agent.

This agent handles multi-turn clarification conversations with users to determine
the research scope. It asks clarifying questions and generates a research brief
once the scope is sufficient.
"""

from typing import Optional, List, Dict, Any, Union
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek

from app.models.schemas import (
    ClarificationQuestions,
    ResearchBrief,
    ScopeStatus,
)
from app.config import settings


# Prompt Templates

QUESTION_GENERATION_PROMPT = """You are a research scope clarification assistant. Your job is to ask clarifying questions to understand the user's research needs better.

User's Query: {user_query}

Conversation History:
{conversation_history}

Based on the user's query and conversation so far, generate 1-3 clarifying questions to understand:
- The specific aspect or angle they want to research
- The depth and breadth of research needed
- Any constraints (time period, geographic focus, etc.)
- The intended use or audience for the research
- Preferred format for the final report (if not already clear)

If the scope seems sufficiently clear, return an empty list.

Return your response as a JSON object with this structure:
{{
    "questions": ["question 1", "question 2", ...],
    "context": "Brief explanation of why you're asking these questions"
}}

Return ONLY the JSON object, no other text."""

COMPLETION_DETECTION_PROMPT = """You are a research scope analyzer. Your job is to determine if we have enough information to proceed with research.

User's Original Query: {user_query}

Conversation History:
{conversation_history}

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

RESEARCH_BRIEF_GENERATION_PROMPT = """You are a research brief generator. Your job is to create a comprehensive research brief from the clarification conversation.

User's Original Query: {user_query}

Conversation History:
{conversation_history}

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
    "format": "summary/comparison/ranking/detailed/other",
    "metadata": {{
        "original_query": "{user_query}",
        "clarification_turns": 0
    }}
}}

Return ONLY the JSON object, no other text."""


# Helper Functions

def _get_llm(temperature: float = 0.7):
    """Get configured LLM based on settings.
    
    Args:
        temperature: Temperature for LLM generation (0.0-1.0)
        
    Returns:
        Configured LLM instance (Gemini or DeepSeek)
        
    Raises:
        ValueError: If no API key is configured or invalid model specified
    """
    if settings.DEFAULT_MODEL == "gemini":
        if not settings.GOOGLE_GEMINI_API_KEY:
            raise ValueError("GOOGLE_GEMINI_API_KEY not configured")
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_GEMINI_API_KEY,
            temperature=temperature,
        )
    elif settings.DEFAULT_MODEL == "deepseek":
        if not settings.DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY not configured")
        return ChatDeepSeek(
            model=settings.DEEPSEEK_MODEL,
            api_key=settings.DEEPSEEK_API_KEY,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Invalid model: {settings.DEFAULT_MODEL}")


def _format_conversation_history(history: Optional[List[Dict[str, str]]]) -> str:
    """Format conversation history for prompts.
    
    Args:
        history: List of conversation turns with 'role' and 'content'
        
    Returns:
        Formatted string representation of conversation history
    """
    if not history:
        return "No previous conversation."
    
    formatted = []
    for turn in history:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        formatted.append(f"{role.upper()}: {content}")
    
    return "\n".join(formatted)


async def _invoke_llm_with_retry(
    llm,
    prompt: str,
    max_retries: int = 3
) -> str:
    """Invoke LLM with retry logic for transient failures.
    
    Args:
        llm: LLM instance to invoke
        prompt: Prompt string to send to LLM
        max_retries: Maximum number of retry attempts
        
    Returns:
        LLM response string
        
    Raises:
        Exception: If all retry attempts fail
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = await llm.ainvoke(prompt)
            return response.content
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                # Wait briefly before retrying (exponential backoff)
                import asyncio
                await asyncio.sleep(2 ** attempt)
            continue
    
    raise Exception(f"LLM invocation failed after {max_retries} attempts: {last_error}")


def _parse_json_response(response: str) -> Dict[str, Any]:
    """Parse JSON response from LLM, handling markdown code blocks.
    
    Args:
        response: Raw response string from LLM
        
    Returns:
        Parsed JSON as dictionary
        
    Raises:
        ValueError: If JSON parsing fails
    """
    # Remove markdown code blocks if present
    cleaned = response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    
    cleaned = cleaned.strip()
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response}")


# Main Functions

async def generate_clarification_questions(
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> ClarificationQuestions:
    """Generate clarifying questions based on user query and conversation.
    
    Args:
        user_query: User's original or current query
        conversation_history: List of previous conversation turns
        
    Returns:
        ClarificationQuestions object with questions and context
        
    Raises:
        ValueError: If LLM configuration is invalid
        Exception: If LLM invocation fails after retries
    """
    llm = _get_llm(temperature=0.7)
    
    formatted_history = _format_conversation_history(conversation_history)
    
    prompt = QUESTION_GENERATION_PROMPT.format(
        user_query=user_query,
        conversation_history=formatted_history
    )
    
    response = await _invoke_llm_with_retry(llm, prompt)
    parsed = _parse_json_response(response)
    
    return ClarificationQuestions(
        questions=parsed.get("questions", []),
        context=parsed.get("context")
    )


async def check_scope_completion(
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """Check if we have enough information to proceed with research.
    
    Args:
        user_query: User's original query
        conversation_history: List of conversation turns
        
    Returns:
        Dictionary with 'is_complete', 'reason', and 'missing_info' keys
        
    Raises:
        ValueError: If LLM configuration is invalid
        Exception: If LLM invocation fails after retries
    """
    llm = _get_llm(temperature=0.3)  # Lower temperature for more consistent decisions
    
    formatted_history = _format_conversation_history(conversation_history)
    
    prompt = COMPLETION_DETECTION_PROMPT.format(
        user_query=user_query,
        conversation_history=formatted_history
    )
    
    response = await _invoke_llm_with_retry(llm, prompt)
    parsed = _parse_json_response(response)
    
    return {
        "is_complete": parsed.get("is_complete", False),
        "reason": parsed.get("reason", ""),
        "missing_info": parsed.get("missing_info", [])
    }


async def generate_research_brief(
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> ResearchBrief:
    """Generate a research brief from the clarification conversation.
    
    Args:
        user_query: User's original query
        conversation_history: List of conversation turns
        
    Returns:
        ResearchBrief object with complete research scope
        
    Raises:
        ValueError: If LLM configuration is invalid
        Exception: If LLM invocation fails after retries
    """
    llm = _get_llm(temperature=0.5)
    
    formatted_history = _format_conversation_history(conversation_history)
    
    prompt = RESEARCH_BRIEF_GENERATION_PROMPT.format(
        user_query=user_query,
        conversation_history=formatted_history
    )
    
    response = await _invoke_llm_with_retry(llm, prompt)
    parsed = _parse_json_response(response)
    
    # Update metadata with actual clarification turn count
    if conversation_history:
        clarification_turns = len([
            turn for turn in conversation_history 
            if turn.get("role") == "assistant"
        ])
        if "metadata" not in parsed:
            parsed["metadata"] = {}
        parsed["metadata"]["clarification_turns"] = clarification_turns
        parsed["metadata"]["original_query"] = user_query
    
    return ResearchBrief(
        scope=parsed.get("scope", ""),
        sub_topics=parsed.get("sub_topics", []),
        constraints=parsed.get("constraints", {}),
        deliverables=parsed.get("deliverables", ""),
        format=parsed.get("format"),
        metadata=parsed.get("metadata", {})
    )


async def clarify_scope(
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Union[ClarificationQuestions, ResearchBrief]:
    """Main function to handle scope clarification workflow.
    
    This function orchestrates the multi-turn clarification process:
    1. Check if scope is complete
    2. If complete, generate research brief
    3. If not complete, generate clarifying questions
    
    Args:
        user_query: User's original query
        conversation_history: List of previous conversation turns
        
    Returns:
        Either ClarificationQuestions (if more info needed) or 
        ResearchBrief (if scope is complete)
        
    Raises:
        ValueError: If LLM configuration is invalid
        Exception: If LLM invocation fails after retries
    """
    # Check if we have enough information
    completion_status = await check_scope_completion(user_query, conversation_history)
    
    # If scope is complete, generate research brief
    if completion_status["is_complete"]:
        return await generate_research_brief(user_query, conversation_history)
    
    # Otherwise, generate more clarifying questions
    return await generate_clarification_questions(user_query, conversation_history)
