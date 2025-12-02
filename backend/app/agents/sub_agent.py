"""
Sub-Agent - Executes specific research tasks.

These agents are responsible for:
- Executing research tasks assigned by the supervisor.
- Using tools (Tavily, MCP) to gather information.
- Extracting and scoring findings with citations.
- Requesting further research via delegation.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from app.graphs.state import SubAgentState
from app.models.schemas import Finding, ResearchTask, Citation, SourceType
from app.config import get_deepseek_chat
from app.prompts.research_prompts import (
    SUB_AGENT_CITATION_EXTRACTION_TEMPLATE,
    SUB_AGENT_RESEARCH_TEMPLATE,
    CREDIBILITY_HEURISTICS,
)
from app.tools.tool_registry import get_research_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)


class CitationExtractionOutput(BaseModel):
    """Structured output for citation extraction."""
    findings: List[Finding] = Field(
        description="List of extracted findings with citations"
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
    
    max_searches = budget.get("max_searches_per_agent", 2)
    total_searches = budget.get("total_searches", 0)
    budget_remaining = max_searches
    
    brief = state["research_brief"]
    enabled_mcp_servers = (
        brief.metadata.get("enabled_mcp_servers", ["scientific-papers"])
        if brief.metadata else ["scientific-papers"]
    )
    
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
            agent = create_react_agent(
                llm,
                tools=tools,
                prompt=system_message
            )
            
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": f"Research topic: {task.query}"}]
            })
            
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
            
            findings = await _extract_citations(
                raw_results=combined_results,
                topic=task.topic,
                source_tools=[r["tool"] for r in tool_results]
            )
            
            state_update: Dict[str, Any] = {
                "findings": findings,
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


async def _extract_citations(
    raw_results: str,
    topic: str,
    source_tools: List[str]
) -> List[Finding]:
    """
    Extract structured findings with citations from raw tool output.
    
    Args:
        raw_results: Combined text output from tools.
        topic: The research topic.
        source_tools: List of tools that generated the results.
        
    Returns:
        List[Finding]: Extracted findings with credibility scores.
    """
    primary_tool = source_tools[0] if source_tools else "unknown"
    
    prompt_inputs = {
        "credibility_heuristics": CREDIBILITY_HEURISTICS,
        "source_tool": primary_tool,
        "topic": topic,
        "raw_results": raw_results[:4000]
    }
    
    llm = get_deepseek_chat(temperature=0.3)
    structured_llm = llm.with_structured_output(CitationExtractionOutput)
    
    chain = SUB_AGENT_CITATION_EXTRACTION_TEMPLATE | structured_llm
    
    try:
        result: CitationExtractionOutput = await chain.ainvoke(prompt_inputs)
        return result.findings
    except Exception as e:
        logger.error(f"Citation extraction failed: {e}")
        return []
