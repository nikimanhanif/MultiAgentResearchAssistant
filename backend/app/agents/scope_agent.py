"""Scope Agent - Human-in-the-loop clarification agent.

This agent handles multi-turn clarification conversations with users to determine
the research scope. It asks clarifying questions and generates a research brief
once the scope is sufficient.
"""

from typing import Optional, List, Dict, Any, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek

from app.models.schemas import (
    ClarificationQuestions,
    ResearchBrief,
    ScopeStatus,
    ScopeCompletionCheck,
)
from app.prompts.scope_prompts import (
    SCOPE_QUESTION_GENERATION_TEMPLATE,
    SCOPE_COMPLETION_DETECTION_TEMPLATE,
    SCOPE_BRIEF_GENERATION_TEMPLATE,
)
from app.config import settings


# Helper Functions

def _get_llm(temperature: float = 0.7) -> Union[ChatGoogleGenerativeAI, ChatDeepSeek]:
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


def _update_brief_metadata(
    parsed: Dict[str, Any],
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]]
) -> Dict[str, Any]:
    """Update research brief metadata with actual clarification turn count.
    
    Args:
        parsed: Parsed brief dictionary from LLM
        user_query: Original user query
        conversation_history: List of conversation turns
        
    Returns:
        Updated brief dictionary with metadata
    """
    if conversation_history:
        clarification_turns = len([
            turn for turn in conversation_history 
            if turn.get("role") == "assistant"
        ])
        if "metadata" not in parsed:
            parsed["metadata"] = {}
        parsed["metadata"]["clarification_turns"] = clarification_turns
        parsed["metadata"]["original_query"] = user_query
    
    return parsed


# Chain Building Functions

def _build_question_generation_chain() -> Any:
    """Build chain for generating clarification questions.
    
    Returns:
        Runnable chain: prompt | LLM | parser
    """
    parser = PydanticOutputParser(pydantic_object=ClarificationQuestions)
    llm = _get_llm(temperature=0.7)
    
    # Add format instructions to the prompt
    prompt = SCOPE_QUESTION_GENERATION_TEMPLATE.partial(
        format_instructions=parser.get_format_instructions()
    )
    
    chain = prompt | llm | parser
    return chain


def _build_completion_detection_chain() -> Any:
    """Build chain for detecting scope completion.
    
    Returns:
        Runnable chain: prompt | LLM | parser
    """
    parser = PydanticOutputParser(pydantic_object=ScopeCompletionCheck)
    llm = _get_llm(temperature=0.3)  # Lower temperature for consistent decisions
    
    # Add format instructions to the prompt
    prompt = SCOPE_COMPLETION_DETECTION_TEMPLATE.partial(
        format_instructions=parser.get_format_instructions()
    )
    
    chain = prompt | llm | parser
    return chain


def _build_brief_generation_chain() -> Any:
    """Build chain for generating research brief.
    
    Returns:
        Runnable chain: prompt | LLM | parser | metadata updater
    """
    parser = PydanticOutputParser(pydantic_object=ResearchBrief)
    llm = _get_llm(temperature=0.5)
    
    # Add format instructions to the prompt
    prompt = SCOPE_BRIEF_GENERATION_TEMPLATE.partial(
        format_instructions=parser.get_format_instructions()
    )
    
    chain = prompt | llm | parser
    return chain


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
        Exception: If LLM invocation fails
    """
    chain = _build_question_generation_chain()
    
    formatted_history = _format_conversation_history(conversation_history)
    
    try:
        result = await chain.ainvoke({
            "user_query": user_query,
            "conversation_history": formatted_history
        })
        return result
    except Exception as e:
        raise Exception(f"Failed to generate clarification questions: {e}")


async def check_scope_completion(
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> ScopeCompletionCheck:
    """Check if we have enough information to proceed with research.
    
    Args:
        user_query: User's original query
        conversation_history: List of conversation turns
        
    Returns:
        ScopeCompletionCheck object with completion status and reasoning
        
    Raises:
        ValueError: If LLM configuration is invalid
        Exception: If LLM invocation fails
    """
    chain = _build_completion_detection_chain()
    
    formatted_history = _format_conversation_history(conversation_history)
    
    try:
        result = await chain.ainvoke({
            "user_query": user_query,
            "conversation_history": formatted_history
        })
        return result
    except Exception as e:
        raise Exception(f"Failed to check scope completion: {e}")


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
        Exception: If LLM invocation fails
    """
    chain = _build_brief_generation_chain()
    
    formatted_history = _format_conversation_history(conversation_history)
    
    try:
        result = await chain.ainvoke({
            "user_query": user_query,
            "conversation_history": formatted_history
        })
        
        # Update metadata with actual turn count
        if conversation_history:
            clarification_turns = len([
                turn for turn in conversation_history 
                if turn.get("role") == "assistant"
            ])
            if result.metadata is None:
                result.metadata = {}
            result.metadata["clarification_turns"] = clarification_turns
            result.metadata["original_query"] = user_query
        
        return result
    except Exception as e:
        raise Exception(f"Failed to generate research brief: {e}")


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
        Exception: If LLM invocation fails
    """
    # Check if we have enough information
    completion_check = await check_scope_completion(user_query, conversation_history)
    
    # If scope is complete, generate research brief
    if completion_check.is_complete:
        return await generate_research_brief(user_query, conversation_history)
    
    # Otherwise, generate more clarifying questions
    return await generate_clarification_questions(user_query, conversation_history)
