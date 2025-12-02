"""
Scope Agent - Handles initial research scoping and clarification.

This agent engages in a multi-turn conversation with the user to clarify
the research intent. It generates clarifying questions if the scope is ambiguous
and produces a structured ResearchBrief once the scope is clear.
"""

from typing import Optional, List, Dict, Any, Union
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

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
from app.config import get_deepseek_chat
from app.graphs.state import ResearchState

logger = logging.getLogger(__name__)


def _format_conversation_history(history: Optional[List[Dict[str, str]]]) -> str:
    """
    Format conversation history for prompt context.
    
    Args:
        history: List of conversation turns (role, content).
        
    Returns:
        str: Formatted conversation string.
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
    """
    Update research brief metadata with conversation stats.
    
    Args:
        parsed: Parsed brief dictionary from LLM.
        user_query: Original user query.
        conversation_history: List of conversation turns.
        
    Returns:
        Dict[str, Any]: Updated brief dictionary.
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


def _build_question_generation_chain() -> Any:
    """
    Build chain for generating clarification questions.
    
    Returns:
        Runnable: Chain of prompt | LLM | parser.
    """
    parser = PydanticOutputParser(pydantic_object=ClarificationQuestions)
    llm = get_deepseek_chat(temperature=0.7)
    
    # Add format instructions to the prompt
    prompt = SCOPE_QUESTION_GENERATION_TEMPLATE.partial(
        format_instructions=parser.get_format_instructions()
    )
    
    chain = prompt | llm | parser
    return chain


def _build_completion_detection_chain() -> Any:
    """
    Build chain for detecting scope completion.
    
    Returns:
        Runnable: Chain of prompt | LLM | parser.
    """
    parser = PydanticOutputParser(pydantic_object=ScopeCompletionCheck)
    llm = get_deepseek_chat(temperature=0.3)  # Lower temperature for consistent decisions
    
    # Add format instructions to the prompt
    prompt = SCOPE_COMPLETION_DETECTION_TEMPLATE.partial(
        format_instructions=parser.get_format_instructions()
    )
    
    chain = prompt | llm | parser
    return chain


def _build_brief_generation_chain() -> Any:
    """
    Build chain for generating the research brief.
    
    Returns:
        Runnable: Chain of prompt | LLM | parser.
    """
    parser = PydanticOutputParser(pydantic_object=ResearchBrief)
    llm = get_deepseek_chat(temperature=0.5)
    
    # Add format instructions to the prompt
    prompt = SCOPE_BRIEF_GENERATION_TEMPLATE.partial(
        format_instructions=parser.get_format_instructions()
    )
    
    chain = prompt | llm | parser
    return chain


async def generate_clarification_questions(
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> ClarificationQuestions:
    """
    Generate clarifying questions based on the user's query.
    
    Args:
        user_query: User's original or current query.
        conversation_history: List of previous conversation turns.
        
    Returns:
        ClarificationQuestions: Object containing questions and context.
        
    Raises:
        Exception: If generation fails.
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
    """
    Check if the research scope is sufficiently defined.
    
    Args:
        user_query: User's original query.
        conversation_history: List of conversation turns.
        
    Returns:
        ScopeCompletionCheck: Status and reasoning.
        
    Raises:
        Exception: If check fails.
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
    """
    Generate a formal research brief from the conversation.
    
    Args:
        user_query: User's original query.
        conversation_history: List of conversation turns.
        
    Returns:
        ResearchBrief: The defined research scope.
        
    Raises:
        Exception: If generation fails.
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
    """
    Orchestrate the scope clarification workflow.
    
    1. Check if scope is complete.
    2. If complete, generate ResearchBrief.
    3. If incomplete, generate ClarificationQuestions.
    
    Args:
        user_query: User's original query.
        conversation_history: List of previous conversation turns.
        
    Returns:
        Union[ClarificationQuestions, ResearchBrief]: The next step in the process.
    """
    # Check if we have enough information
    completion_check = await check_scope_completion(user_query, conversation_history)
    
    # If scope is complete, generate research brief
    if completion_check.is_complete:
        return await generate_research_brief(user_query, conversation_history)
    
    # Otherwise, generate more clarifying questions
    return await generate_clarification_questions(user_query, conversation_history)


async def scope_node(state: ResearchState) -> Dict[str, Any]:
    """
    LangGraph node for the Scope Agent.
    
    Manages the clarification loop. Generates either clarification questions
    (if scope is incomplete) or a ResearchBrief (if scope is complete).
    
    Args:
        state: Current research state.
        
    Returns:
        Dict[str, Any]: State update with new messages or research_brief.
    """
    if state.get("research_brief"):
        return {}
    
    messages = state.get("messages", [])
    user_messages = [msg for msg in messages if msg.get("role") == "user"]
    if not user_messages:
        return {
            "messages": [{
                "role": "assistant",
                "content": "Error: No user query provided."
            }]
        }
    
    user_query = user_messages[0].get("content", "")
    conversation_history = [
        {"role": msg.get("role"), "content": msg.get("content")}
        for msg in messages[1:]
    ] if len(messages) > 1 else None
    
    try:
        completion_check = await check_scope_completion(user_query, conversation_history)
        
        if completion_check.is_complete:
            brief = await generate_research_brief(user_query, conversation_history)
            return {
                "research_brief": brief,
                "messages": [{
                    "role": "assistant",
                    "content": f"Research brief created. Proceeding with research on: {brief.scope}"
                }]
            }
        else:
            questions = await generate_clarification_questions(user_query, conversation_history)
            questions_text = "\n".join([
                f"{i+1}. {q.question}" 
                for i, q in enumerate(questions.clarification_questions)
            ])
            message_content = f"{questions.context}\n\n{questions_text}"
            return {
                "messages": [{
                    "role": "assistant",
                    "content": message_content
                }]
            }
    
    except Exception as e:
        logger.error(f"Scope Node: Error processing - {e}")
        return {
            "messages": [{
                "role": "assistant",
                "content": f"Error processing your request: {str(e)}"
            }],
            "error": str(e)
        }
