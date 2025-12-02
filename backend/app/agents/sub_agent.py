"""Sub-Agent - Research execution with delegation capability.

These agents execute specific research tasks assigned by the supervisor:
- Use tools (Tavily, MCP servers) to gather information
- Extract key facts and create Finding objects with embedded citations
- Score citations using LLM with heuristics
- Can request further research via delegation tool

Architecture: Supervisor Loop
- Input: ResearchTask (via LangGraph Send API)
- Output: List[Finding] with embedded citations
- Delegation: Sub-agent can add new tasks to task_history
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
    """Parse delegation request from agent output.
    
    Format: DELEGATION_REQUEST: topic='[subtopic]', reason='[why needed]'
    
    Args:
        agent_output: Raw output from agent
        
    Returns:
        ResearchTask if delegation found, None otherwise
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
    """Sub-agent node for research task execution.
    
    Responsibilities:
    1. Execute research task using tools (Tavily + MCP)
    2. Extract findings with citations from results
    3. Score citations using LLM with credibility heuristics
    4. Handle delegation requests
    5. Track search budget
    
    Args:
        state: SubAgentState with task and shared context
        
    Returns:
        State update with findings, updated budget, and optionally new tasks
    """
    task = state["task"]
    budget = state["budget"]
    
    logger.info(f"Sub-agent: Executing task {task.task_id} - {task.topic}")
    
    # Check if task already completed or failed
    if task.task_id in state.get("completed_tasks", []):
        logger.warning(f"Task {task.task_id} already completed, skipping")
        return {}
    
    if task.task_id in state.get("failed_tasks", []):
        logger.warning(f"Task {task.task_id} already failed, skipping")
        return {}
    
    # Budget check: enforce max_searches_per_agent
    max_searches = budget.get("max_searches_per_agent", 2)
    total_searches = budget.get("total_searches", 0)
    budget_remaining = max_searches  # Each sub-agent gets fresh budget
    
    # Load tools within context manager to keep MCP connection alive
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
        
        # Create ReAct agent for tool execution
        llm = get_deepseek_chat(temperature=0.7)
        
        # Render prompt from template
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
        
        # Render template to string for state_modifier
        rendered_messages = SUB_AGENT_RESEARCH_TEMPLATE.format_messages(**prompt_inputs)
        system_message = rendered_messages[0].content + "\n\n" + rendered_messages[1].content
        
        try:
            # Create agent with tools
            agent = create_react_agent(
                llm,
                tools=tools,
                prompt=system_message
            )
            
            # Execute agent
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": f"Research topic: {task.query}"}]
            })
            
            # Extract final message
            agent_output = result["messages"][-1].content if result.get("messages") else ""
            
            # Check for delegation requests
            delegation_task = _parse_delegation_request(agent_output)
            
            # Extract tool results for citation extraction
            tool_results = []
            for msg in result.get("messages", []):
                if isinstance(msg, ToolMessage):
                    tool_name = msg.name or "unknown"
                    # Fallback if name is missing (though ToolNode usually sets it)
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
            
            # Count searches performed (approximate)
            searches_used = len([r for r in tool_results if "search" in r["tool"].lower()])
            
            # Combine tool results for citation extraction
            if not tool_results:
                logger.warning(f"Task {task.task_id}: No tool results found")
                return {
                    "failed_tasks": [task.task_id],
                    "budget": {**budget, "total_searches": total_searches + searches_used}
                }
            
            combined_results = "\n\n".join([
                f"[Source: {r['tool']}]\n{r['result']}"
                for r in tool_results
            ])
            
            # Extract citations using LLM
            findings = await _extract_citations(
                raw_results=combined_results,
                topic=task.topic,
                source_tools=[r["tool"] for r in tool_results]
            )
            
            logger.info(f"Sub-agent: Extracted {len(findings)} findings from task {task.task_id}")
            
            # Build state update
            state_update: Dict[str, Any] = {
                "findings": findings,
                "completed_tasks": [task.task_id],
                "budget": {**budget, "total_searches": total_searches + searches_used}
            }
            
            # Add delegation task if requested
            if delegation_task:
                state_update["task_history"] = [delegation_task]
                logger.info(f"Sub-agent: Delegated new task: {delegation_task.task_id}")
            
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
    """Extract structured findings with citations using LLM.
    
    Args:
        raw_results: Combined tool results with source tags
        topic: Research topic
        source_tools: List of tools used
        
    Returns:
        List of Finding objects with scored citations
    """
    # Determine primary source tool
    primary_tool = source_tools[0] if source_tools else "unknown"
    
    # Prepare prompt inputs
    prompt_inputs = {
        "credibility_heuristics": CREDIBILITY_HEURISTICS,
        "source_tool": primary_tool,
        "topic": topic,
        "raw_results": raw_results[:4000]  # Limit to prevent token overflow
    }
    
    # Get LLM for structured output
    llm = get_deepseek_chat(temperature=0.3)  # Lower temp for extraction
    structured_llm = llm.with_structured_output(CitationExtractionOutput)
    
    # Create chain
    chain = SUB_AGENT_CITATION_EXTRACTION_TEMPLATE | structured_llm
    
    try:
        result: CitationExtractionOutput = await chain.ainvoke(prompt_inputs)
        return result.findings
    except Exception as e:
        logger.error(f"Citation extraction failed: {e}")
        # Return empty list on failure
        return []
