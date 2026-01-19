"""
Scope Agent - Handles initial research scoping and clarification.

This agent engages in a multi-turn conversation with the user to clarify
the research intent. It generates clarifying questions if the scope is ambiguous
and produces a structured ResearchBrief once the scope is clear.
"""

from typing import Optional, List, Dict, Any, Union
import logging
from langsmith import traceable
import langsmith as ls
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
    prompt = SCOPE_QUESTION_GENERATION_TEMPLATE
    return prompt | llm.with_config(tags=["user_visible"]) | parser


def _build_completion_detection_chain() -> Any:
    """
    Build chain for detecting scope completion.
    
    Returns:
        Runnable: Chain of prompt | LLM | parser.
    """
    parser = PydanticOutputParser(pydantic_object=ScopeCompletionCheck)
    llm = get_deepseek_chat(temperature=0.3)
    prompt = SCOPE_COMPLETION_DETECTION_TEMPLATE.partial(
        format_instructions=parser.get_format_instructions()
    )
    return prompt | llm | parser


def _build_brief_generation_chain() -> Any:
    """
    Build chain for generating the research brief.
    
    Returns:
        Runnable: Chain of prompt | LLM | parser.
    """
    parser = PydanticOutputParser(pydantic_object=ResearchBrief)
    llm = get_deepseek_chat(temperature=0.5)
    prompt = SCOPE_BRIEF_GENERATION_TEMPLATE.partial(
        format_instructions=parser.get_format_instructions()
    )
    return prompt | llm | parser


@traceable(name="Generate Clarification Questions", metadata={"agent": "scope", "operation": "question_generation"})
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
        return await chain.ainvoke({
            "user_query": user_query,
            "conversation_history": formatted_history
        })
    except Exception as e:
        raise Exception(f"Failed to generate clarification questions: {e}")


@traceable(name="Check Scope Completion", metadata={"agent": "scope", "operation": "completion_check"})
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
        return await chain.ainvoke({
            "user_query": user_query,
            "conversation_history": formatted_history
        })
    except Exception as e:
        raise Exception(f"Failed to check scope completion: {e}")


@traceable(name="Generate Research Brief", metadata={"agent": "scope", "operation": "brief_generation"})
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
    
    if completion_check.is_complete:
        return await generate_research_brief(user_query, conversation_history)
    
    return await generate_clarification_questions(user_query, conversation_history)


@traceable(name="Scope Node", metadata={"agent": "scope", "phase": "scoping"})
async def scope_node(state: ResearchState) -> Dict[str, Any]:
    """
    LangGraph node for the Scope Agent - Generates clarifying questions.
    
    This node checks if scope is complete and either:
    1. Generates a research brief if scope is clear
    2. Generates clarifying questions and stores them in state
    
    Does NOT call interrupt - that's handled by scope_wait_node.
    This ensures questions are persisted to state before the graph pauses.
    
    Args:
        state: Current research state.
        
    Returns:
        Dict[str, Any]: State update with research_brief or pending_clarification_questions.
    """
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
    
    current_round = state.get("scope_clarification_rounds", 0)
    user_query = user_messages[0].get("content", "")
    conversation_history = [
        {"role": msg.get("role"), "content": msg.get("content")}
        for msg in messages
    ] if messages else None
    
    if state.get("pending_clarification_questions"):
        return {}
    
    try:
        if current_round >= MAX_CLARIFICATION_ROUNDS:
            logger.info("Max rounds reached. Forcing brief generation.")
            brief = await generate_research_brief(user_query, conversation_history)
            return {
                "research_brief": brief,
                "pending_clarification_questions": None,
                "messages": [{
                    "role": "assistant",
                    "content": f"Research brief created. Proceeding with research on: {brief.scope}"
                }]
            }
        
        completion_check = await check_scope_completion(user_query, conversation_history)
        
        if completion_check.is_complete:
            brief = await generate_research_brief(user_query, conversation_history)
            return {
                "research_brief": brief,
                "pending_clarification_questions": None,
                "messages": [{
                    "role": "assistant",
                    "content": f"Research brief created. Proceeding with research on: {brief.scope}"
                }]
            }
        
        questions = await generate_clarification_questions(user_query, conversation_history)
        return {
            "scope_clarification_rounds": current_round + 1,
            "pending_clarification_questions": questions
        }
    
    except Exception as e:
        logger.exception(f"Scope Node process failed: {e}")
        
        if current_round >= 1:
            try:
                # Attempt emergency brief generation if we have some history
                brief = await generate_research_brief(user_query, conversation_history)
                return {
                    "research_brief": brief,
                    "pending_clarification_questions": None,
                    "messages": [{
                        "role": "assistant", 
                        "content": f"I encountered an issue, but I've generated a research brief based on our conversation: {brief.scope}"
                    }]
                }
            except Exception as fallback_error:
                logger.error(f"Fallback brief generation also failed: {fallback_error}")
        
        return {
            "scope_clarification_rounds": current_round + 1,
            "pending_clarification_questions": None,
            "messages": [{
                "role": "assistant",
                "content": "I encountered an error while defining the research scope. Let me proceed with what I have so far."
            }],
            "error": [str(e)]
        }


async def scope_wait_node(state: ResearchState) -> Dict[str, Any]:
    """
    LangGraph node that handles waiting for user clarification responses.
    
    This node is called AFTER scope_node has generated and stored questions.
    It calls interrupt to pause the graph and wait for user input.
    When resumed, it processes the user's response and updates state.
    
    Args:
        state: Current research state (should contain pending_clarification_questions).
        
    Returns:
        Dict[str, Any]: State update with messages and cleared pending questions.
    """
    from langgraph.types import interrupt
    
    questions = state.get("pending_clarification_questions")
    if not questions:
        return {}
    
    messages = state.get("messages", [])
    user_messages = [msg for msg in messages if msg.get("role") == "user"]
    user_query = user_messages[0].get("content", "") if user_messages else ""
    conversation_history = [{"role": msg.get("role"), "content": msg.get("content")} for msg in messages] if messages else []
    
    user_response = interrupt(value={"type": "clarification_request", "questions": questions})
    
    updated_history = conversation_history + [
        {"role": "assistant", "content": questions},
        {"role": "user", "content": user_response}
    ]
    
    try:
        check = await check_scope_completion(user_query, updated_history)
        if check.is_complete:
            brief = await generate_research_brief(user_query, updated_history)
            return {
                "research_brief": brief,
                "pending_clarification_questions": None,
                "messages": [
                    {"role": "assistant", "content": questions},
                    {"role": "user", "content": user_response},
                    {"role": "assistant", "content": f"Research brief created: {brief.scope}"}
                ]
            }
    except Exception as e:
        logger.warning(f"Recheck failed: {e}")
    
    return {
        "pending_clarification_questions": None,
        "messages": [
            {"role": "assistant", "content": questions},
            {"role": "user", "content": user_response}
        ]
    }


