"""Scope Agent - Human-in-the-loop clarification agent.

This agent handles multi-turn clarification conversations with users to determine
the research scope. It asks clarifying questions and generates a research brief
once the scope is sufficient.
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


# Helper Functions


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
    llm = get_deepseek_chat(temperature=0.7)
    
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
    llm = get_deepseek_chat(temperature=0.3)  # Lower temperature for consistent decisions
    
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
    llm = get_deepseek_chat(temperature=0.5)
    
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


# LangGraph Node Integration

async def scope_node(state: ResearchState) -> Dict[str, Any]:
    """Scope agent node for LangGraph integration.
    
    This node handles the clarification conversation loop within the main graph.
    It uses state["messages"] for conversation history and generates either:
    - Clarification questions (if scope incomplete)
    - ResearchBrief (if scope complete)
    
    Args:
        state: Current research state with messages
        
    Returns:
        State update with new messages or research_brief
    """
    logger.info("Scope Node: Processing user input")
    
    # Extract messages from state
    messages = state.get("messages", [])
    
    # Get the last user message as the current query
    user_messages = [msg for msg in messages if msg.get("role") == "user"]
    if not user_messages:
        logger.error("No user messages found in state")
        return {
            "messages": [{
                "role": "assistant",
                "content": "Error: No user query provided."
            }]
        }
    
    # Get original query (first user message) and conversation history
    user_query = user_messages[0].get("content", "")
    
    # Format conversation history (exclude the initial query from history)
    conversation_history = [
        {"role": msg.get("role"), "content": msg.get("content")}
        for msg in messages[1:]  # Skip first message as it's the user_query
    ] if len(messages) > 1 else None
    
    try:
        # Check if scope is complete
        completion_check = await check_scope_completion(user_query, conversation_history)
        
        if completion_check.is_complete:
            # Generate research brief
            logger.info("Scope Node: Scope complete, generating research brief")
            brief = await generate_research_brief(user_query, conversation_history)
            
            # Return state update with brief and completion message
            return {
                "research_brief": brief,
                "messages": [{
                    "role": "assistant",
                    "content": f"Research brief created. Proceeding with research on: {brief.scope}"
                }]
            }
        else:
            # Generate clarification questions
            logger.info("Scope Node: Scope incomplete, generating clarification questions")
            questions = await generate_clarification_questions(user_query, conversation_history)
            
            # Format questions as a message
            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions.clarification_questions)])
            message_content = f"{questions.context}\n\n{questions_text}"
            
            # Return state update with clarification questions
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
