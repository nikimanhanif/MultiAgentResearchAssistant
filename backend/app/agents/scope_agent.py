"""
Scope Agent - Handles initial research scoping and clarification.

This agent engages in a multi-turn conversation with the user to clarify
the research intent. It generates clarifying questions if the scope is ambiguous
and produces a structured ResearchBrief once the scope is clear.
"""

from typing import Optional, List, Dict, Any, Union
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
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
    parser = StrOutputParser()
    llm = get_deepseek_chat(temperature=0.7)
    
    # Prompt now expects Markdown output, no format instructions needed
    prompt = SCOPE_QUESTION_GENERATION_TEMPLATE
    
    # Add tag for streaming identification
    chain = prompt | llm.with_config(tags=["user_visible"]) | parser
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
) -> str:
    """
    Generate clarifying questions based on the user's query.
    
    Args:
        user_query: User's original or current query.
        conversation_history: List of previous conversation turns.
        
    Returns:
        str: Clarification questions and context as a string.
        
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
) -> Union[str, ResearchBrief]:
    """
    Orchestrate the scope clarification workflow.
    
    1. Check if scope is complete.
    2. If complete, generate ResearchBrief.
    3. If incomplete, generate ClarificationQuestions.
    
    Args:
        user_query: User's original query.
        conversation_history: List of previous conversation turns.
        
    Returns:
        Union[str, ResearchBrief]: The next step in the process.
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
    
    Uses interrupt pattern to pause and wait for user clarification responses.
    This keeps the graph paused (not exited) so checkpoints persist properly.
    
    Implements a max clarification round limit to prevent infinite loops.
    After 3 rounds of clarification, forces brief generation.
    
    Args:
        state: Current research state.
        
    Returns:
        Dict[str, Any]: State update with new messages or research_brief.
    """
    from langgraph.types import interrupt
    
    MAX_CLARIFICATION_ROUNDS = 3
    
    # Skip if we already have a research brief
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
    
    # Get current clarification round count
    current_round = state.get("scope_clarification_rounds", 0)
    
    # Get original query (first user message)
    user_query = user_messages[0].get("content", "")
    
    # Build conversation history from all messages
    conversation_history = [
        {"role": msg.get("role"), "content": msg.get("content")}
        for msg in messages
    ] if messages else None
    
    try:
        # Check if we've exceeded max clarification rounds - force brief generation
        if current_round >= MAX_CLARIFICATION_ROUNDS:
            logger.info(f"Max clarification rounds ({MAX_CLARIFICATION_ROUNDS}) reached. Forcing brief generation.")
            brief = await generate_research_brief(user_query, conversation_history)
            return {
                "research_brief": brief,
                "messages": [{
                    "role": "assistant",
                    "content": f"Research brief created. Proceeding with research on: {brief.scope}"
                }]
            }
        
        # Check if scope is complete
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
            # Generate clarification questions
            message_content = await generate_clarification_questions(user_query, conversation_history)
            
            # Increment clarification round counter
            new_round = current_round + 1
            
            # Use interrupt to pause and wait for user response
            user_response = interrupt(value={
                "type": "clarification_request",
                "questions": message_content
            })
            
            # --- After interrupt resumes ---
            # User has responded, re-check completion immediately
            updated_history = conversation_history.copy() if conversation_history else []
            updated_history.append({"role": "assistant", "content": message_content})
            updated_history.append({"role": "user", "content": user_response})
            
            # Immediately check if scope is now complete
            try:
                post_response_check = await check_scope_completion(user_query, updated_history)
                
                if post_response_check.is_complete:
                    # User's answer was sufficient - generate brief now
                    brief = await generate_research_brief(user_query, updated_history)
                    return {
                        "research_brief": brief,
                        "scope_clarification_rounds": new_round,
                        "messages": [
                            {"role": "assistant", "content": message_content},
                            {"role": "user", "content": user_response},
                            {"role": "assistant", "content": f"Research brief created. Proceeding with research on: {brief.scope}"}
                        ]
                    }
            except Exception as recheck_error:
                logger.warning(f"Scope completion recheck failed: {recheck_error}. Proceeding with next round.")
            
            # Scope still not complete, add messages and let graph loop back
            return {
                "scope_clarification_rounds": new_round,
                "messages": [
                    {"role": "assistant", "content": message_content},
                    {"role": "user", "content": user_response}
                ]
            }
    
    except Exception as e:
        # Check if it's a GraphInterrupt (which should be re-raised)
        if "Interrupt" in type(e).__name__:
            raise e
        
        logger.error(f"Scope Node: Error processing - {e}")
        
        # On persistent errors, force brief generation to prevent infinite loops
        if current_round >= 1:
            logger.warning(f"Error after {current_round} clarification rounds. Forcing brief generation.")
            try:
                brief = await generate_research_brief(user_query, conversation_history)
                return {
                    "research_brief": brief,
                    "messages": [{
                        "role": "assistant",
                        "content": f"Research brief created. Proceeding with research on: {brief.scope}"
                    }]
                }
            except Exception as brief_error:
                logger.error(f"Could not generate brief: {brief_error}")
        
        return {
            "scope_clarification_rounds": current_round + 1,
            "messages": [{
                "role": "assistant",
                "content": f"I encountered an issue processing your request. Let me proceed with what I understand so far."
            }],
            "error": [str(e)]
        }

