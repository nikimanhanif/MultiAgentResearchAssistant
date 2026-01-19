"""
Sub-Agent - Executes specific research tasks.

These agents are responsible for:
- Executing research tasks assigned by the supervisor.
- Using tools (Tavily, MCP) to gather information.
- Extracting and scoring findings with citations.
- Producing summaries for supervisor context.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langsmith import traceable
import langsmith as ls

from app.graphs.state import SubAgentState
from app.models.schemas import Finding, ResearchTask, SubAgentSummary
from app.config import get_deepseek_chat
from app.prompts.research_prompts import (
    SUB_AGENT_CITATION_EXTRACTION_TEMPLATE,
    SUB_AGENT_RESEARCH_TEMPLATE,
    CREDIBILITY_HEURISTICS,
)
from app.tools.tool_registry import get_research_tools
from app.agents.middleware import TrimmingMiddleware, ToolSafetyMiddleware

from langchain.agents import create_agent
from langgraph.errors import GraphRecursionError
from langchain_core.messages import ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

logger = logging.getLogger(__name__)


class CitationExtractionOutput(BaseModel):
    """Structured output for citation extraction with task summary."""
    findings: List[Finding] = Field(
        description="List of extracted findings with citations"
    )
    task_answered: bool = Field(
        default=True,
        description="Whether the research sufficiently answered the task"
    )
    key_insights: List[str] = Field(
        default_factory=list,
        description="3-5 key insights with finding index references"
    )
    gaps_noted: Optional[str] = Field(
        default=None,
        description="Any gaps noticed during research"
    )


def _parse_delegation_request(agent_output: str) -> Optional[ResearchTask]:
    """
    Parse a delegation request from the agent's output.
    
    Format: DELEGATION_REQUEST: topic='[subtopic]', reason='[why needed]'
    
    Args:
        agent_output: Raw text output from the agent.
        
    Returns:
        ResearchTask: A new task if delegation is requested, else None.
    """
    pattern = r"DELEGATION_REQUEST:\s*topic='([^']+)',\s*reason='([^']+)'"
    match = re.search(pattern, agent_output)
    
    if match:
        topic = match.group(1)
        reason = match.group(2)
        
        # Generate task from delegation request
        task_id = f"delegated_{topic.lower().replace(' ', '_')}"
        return ResearchTask(
            task_id=task_id,
            topic=topic,
            query=f"{topic} - {reason}",
            priority=2,  # Delegated tasks get medium priority
            requested_by="sub_agent"
        )
    
    return None


# Max tokens to keep in agent context (leaving room for next LLM response)
MAX_AGENT_CONTEXT_TOKENS = 64000


@traceable(name="Sub Agent Node", metadata={"agent": "sub_agent", "phase": "research"})
async def sub_agent_node(state: SubAgentState) -> Dict[str, Any]:
    """
    LangGraph node for the Sub-Agent.
    
    Executes a research task using available tools, extracts findings,
    scores citations, and handles delegation requests.
    
    Args:
        state: Current sub-agent state.
        
    Returns:
        Dict[str, Any]: State update with findings, budget usage, and new tasks.
    """
    task = state["task"]
    budget = state["budget"]
    
    if task.task_id in state.get("completed_tasks", []):
        return {}
    
    if task.task_id in state.get("failed_tasks", []):
        return {}
    
    # Add dynamic metadata for task-level observability
    rt = ls.get_current_run_tree()
    if rt:
        rt.metadata["task_id"] = task.task_id
        rt.metadata["topic"] = task.topic
        rt.metadata["priority"] = task.priority
    
    max_searches = budget.get("max_searches_per_agent", 2)
    total_searches = budget.get("total_searches", 0)
    budget_remaining = max_searches
    
    brief = state["research_brief"]
    # Provide default for format to avoid validation errors
    brief_format = brief.format if brief.format else "other"
    
    enabled_mcp_servers = (
        brief.metadata.get("enabled_mcp_servers", ["scientific-papers"])
        if brief.metadata else ["scientific-papers"]
    )
    
    # Map ResearchBrief.format to research strategy for prompt
    FORMAT_TO_STRATEGY = {
        "literature_review": "LITERATURE_REVIEW",
        "deep_research": "DEEP_RESEARCH",
        "comparative": "COMPARATIVE",
        "gap_analysis": "GAP_ANALYSIS",
        "other": "DEEP_RESEARCH",  # Default fallback
    }
    format_value = (brief_format.value if hasattr(brief_format, 'value') else str(brief_format)).lower()
    research_goal = FORMAT_TO_STRATEGY.get(format_value, "DEEP_RESEARCH")
    
    async with get_research_tools(enabled_mcp_servers=enabled_mcp_servers) as tools:
        if not tools:
            logger.error("No tools available for research")
            return {
                "failed_tasks": [task.task_id],
                "error": [f"Task {task.task_id} failed: No tools available"]
            }
        
        llm = get_deepseek_chat(temperature=0.7)
        available_tools_str = ", ".join([tool.name for tool in tools])
        
        prompt_inputs = {
            "credibility_heuristics": CREDIBILITY_HEURISTICS,
            "research_goal": research_goal,
            "budget_remaining": budget_remaining,
            "max_searches_per_agent": max_searches,
            "topic": task.topic,
            "query": task.query,
            "priority": task.priority,
            "available_tools": available_tools_str
        }
        
        rendered_messages = SUB_AGENT_RESEARCH_TEMPLATE.format_messages(**prompt_inputs)
        system_message = rendered_messages[0].content + "\n\n" + rendered_messages[1].content
        
        try:
            # Create Agent with new Middleware architecture
            agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=system_message,
                middleware=[
                    TrimmingMiddleware(max_tokens=MAX_AGENT_CONTEXT_TOKENS),
                    ToolSafetyMiddleware()
                ]
            )
            
            try:
                # Use ainvoke with state
                initial_state = {"messages": [{"role": "user", "content": f"Research topic: {task.query}"}]}
                
                result = await agent.ainvoke(
                    initial_state,
                    {"recursion_limit": 25}  # Max 25 steps
                )
            except GraphRecursionError as e:
                logger.warning(f"Sub-agent {task.task_id} hit recursion limit, extracting partial results")
                result = {"messages": getattr(e, 'state', {}).get('messages', [])}
                if not result["messages"]:
                    # If we can't get messages from error, mark as failed
                    return {
                        "failed_tasks": [task.task_id],
                        "budget": {**budget, "total_searches": total_searches},
                        "error": [f"Task {task.task_id} hit recursion limit with no results"]
                    }
            
            # Process results (same logic as before)
            agent_output = result["messages"][-1].content if result.get("messages") else ""
            delegation_task = _parse_delegation_request(agent_output)
            
            tool_results = []
            for msg in result.get("messages", []):
                if isinstance(msg, ToolMessage):
                    tool_name = msg.name or "unknown"
                    if tool_name == "unknown":
                        tool_call_id = msg.tool_call_id
                        for prev_msg in result.get("messages", []):
                            if hasattr(prev_msg, "tool_calls"):
                                for tc in prev_msg.tool_calls:
                                    if tc["id"] == tool_call_id:
                                        tool_name = tc["name"]
                                        break
                                        
                    tool_results.append({
                        "tool": tool_name,
                        "result": msg.content
                    })
            
            searches_used = len([r for r in tool_results if "search" in r["tool"].lower()])
            
            if not tool_results:
                return {
                    "failed_tasks": [task.task_id],
                    "budget": {**budget, "total_searches": total_searches + searches_used}
                }
            
            combined_results = "\n\n".join([
                f"[Source: {r['tool']}]\n{r['result']}"
                for r in tool_results
            ])
            
            extraction_result = await _extract_citations(
                raw_results=combined_results,
                topic=task.topic,
                task_query=task.query,
                source_tools=[r["tool"] for r in tool_results]
            )
            
            # Create summary for supervisor context
            summary = SubAgentSummary(
                task_id=task.task_id,
                task_answered=extraction_result.task_answered,
                key_insights=extraction_result.key_insights,
                gaps_noted=extraction_result.gaps_noted,
                finding_count=len(extraction_result.findings)
            )
            
            state_update: Dict[str, Any] = {
                "findings": extraction_result.findings,
                "sub_agent_summaries": [summary],
                "completed_tasks": [task.task_id],
                "budget": {**budget, "total_searches": total_searches + searches_used}
            }
            
            if delegation_task:
                state_update["task_history"] = [delegation_task]
            
            return state_update
            
        except Exception as e:
            logger.error(f"Sub-agent task {task.task_id} failed: {e}")
            return {
                "failed_tasks": [task.task_id],
                "budget": {**budget, "total_searches": total_searches},
                "error": [f"Task {task.task_id} failed: {str(e)}"]
            }

@traceable(name="Extract Citations", metadata={"agent": "sub_agent", "operation": "citation_extraction"})
async def _extract_citations(
    raw_results: str,
    topic: str,
    task_query: str,
    source_tools: List[str]
) -> CitationExtractionOutput:
    """
    Extract structured findings with citations and task summary from raw tool output.
    
    Args:
        raw_results: Combined text output from tools.
        topic: The research topic.
        task_query: The original task query for context.
        source_tools: List of tools that generated the results.
        
    Returns:
        CitationExtractionOutput: Findings with credibility scores and task summary.
    """
    primary_tool = source_tools[0] if source_tools else "unknown"
    
    prompt_inputs = {
        "credibility_heuristics": CREDIBILITY_HEURISTICS,
        "source_tool": primary_tool,
        "topic": topic,
        "task_query": task_query,
        "raw_results": raw_results[:40000]
    }
    
    llm = get_deepseek_chat(temperature=0.3)
    structured_llm = llm.with_structured_output(CitationExtractionOutput)
    
    chain = SUB_AGENT_CITATION_EXTRACTION_TEMPLATE | structured_llm
    
    try:
        result: CitationExtractionOutput = await chain.ainvoke(prompt_inputs)
        return result
    except Exception as e:
        logger.error(f"Citation extraction failed: {e}")
        return CitationExtractionOutput(
            findings=[],
            task_answered=False,
            key_insights=[],
            gaps_noted=f"Extraction failed: {str(e)}"
        )
