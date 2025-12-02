"""Supervisor Agent - LLM-based gap analysis and task generation.

This agent coordinates the research phase in the Supervisor Loop:
- Analyzes gaps in current findings vs research brief (LLM-driven)
- Generates new research tasks for missing topics
- Checks completion criteria or budget limits
- Filters and aggregates final findings

Architecture: Supervisor Loop
- Flat task queue
- Prompt-driven strategy selection
- LLM-based gap analysis 
"""

import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict
from pydantic import BaseModel, Field

from app.graphs.state import ResearchState
from app.models.schemas import ResearchTask, Finding
from app.config import get_deepseek_reasoner
from app.prompts.research_prompts import (
    SUPERVISOR_GAP_ANALYSIS_TEMPLATE,
    SUPERVISOR_FINDINGS_AGGREGATION_TEMPLATE,
)

logger = logging.getLogger(__name__)


class GapAnalysisOutput(BaseModel):
    """Structured output from supervisor gap analysis."""
    has_gaps: bool = Field(description="Whether significant gaps remain")
    is_complete: bool = Field(description="Whether research is complete")
    gaps_identified: List[str] = Field(
        default_factory=list,
        description="List of identified gaps"
    )
    new_tasks: List[ResearchTask] = Field(
        default_factory=list,
        description="New tasks to address gaps"
    )
    reasoning: str = Field(description="Reasoning for the decision")


def _format_findings_for_supervisor(findings: List[Finding], max_findings: int = 50) -> str:
    """Format findings for supervisor prompt with content visibility
    
    Groups findings by topic and formats them with key claims and scores,
    truncated to fit context window.
    
    Args:
        findings: List of Finding objects
        max_findings: Maximum number of findings to include
        
    Returns:
        Formatted string with findings grouped by topic
    """
    if not findings:
        return "No findings yet."
    
    # Group findings by topic
    findings_by_topic = defaultdict(list)
    for finding in findings:
        findings_by_topic[finding.topic].append(finding)
    
    # Build formatted output
    formatted_parts = []
    total_included = 0
    
    for topic, topic_findings in sorted(findings_by_topic.items()):
        # Sort findings by credibility within topic
        sorted_findings = sorted(topic_findings, key=lambda f: f.credibility_score, reverse=True)
        
        # Take top 3 findings per topic
        topic_findings_limited = sorted_findings[:3]
        
        if total_included + len(topic_findings_limited) > max_findings:
            break
        
        topic_part = f"\n**Topic: {topic}** ({len(topic_findings)} total findings)\n"
        
        for i, finding in enumerate(topic_findings_limited, 1):
            # Truncate claim to 200 chars
            claim_preview = finding.claim[:200] + "..." if len(finding.claim) > 200 else finding.claim
            source_preview = finding.citation.source[:50] if finding.citation.source else "Unknown"
            
            topic_part += f"  {i}. {claim_preview} (Score: {finding.credibility_score:.2f}, Source: {source_preview})\n"
        
        formatted_parts.append(topic_part)
        total_included += len(topic_findings_limited)
    
    header = f"\n=== FINDINGS SUMMARY ({len(findings)} total, showing {total_included}) ===\n"
    return header + "".join(formatted_parts)


def supervisor_node(state: ResearchState) -> Dict[str, Any]:
    """Supervisor node for gap analysis and task generation.
    
    Responsibilities:
    1. Analyze current findings vs research brief (LLM-based)
    2. Generate new tasks for identified gaps
    3. Check completion criteria and budget limits
    4. Update iteration counter
    
    Args:
        state: Current research state with brief, findings, tasks, budget
        
    Returns:
        State update with new tasks or completion signal
    """
    logger.info("Supervisor: Starting gap analysis")
    
    # Extract state fields
    brief = state["research_brief"]
    findings = state.get("findings", [])
    completed_tasks = state.get("completed_tasks", [])
    failed_tasks = state.get("failed_tasks", [])
    budget = state["budget"]
    
    # Check if already complete (e.g. mock mode or previous iteration)
    if state.get("is_complete"):
        logger.info("Supervisor: Research already marked complete")
        return {"is_complete": True}
    
    # Increment iteration counter
    current_iteration = budget["iterations"] + 1
    
    # Check hard budget limits first
    budget_exhausted = (
        current_iteration >= budget["max_iterations"] or
        len(findings) >= budget["max_sub_agents"]
    )
    
    if budget_exhausted:
        logger.info(f"Supervisor: Budget exhausted (iterations={current_iteration}, findings={len(findings)})")
        return {
            "budget": {**budget, "iterations": current_iteration},
            "is_complete": True
        }
    
    # Calculate current research statistics
    findings_by_topic = defaultdict(list)
    for finding in findings:
        findings_by_topic[finding.topic].append(finding)
    
    topics_covered_str = ", ".join(sorted(findings_by_topic.keys())) if findings_by_topic else "None"
    
    avg_credibility = (
        sum(f.credibility_score for f in findings) / len(findings)
        if findings else 0.0
    )
    
    # Format findings context for supervisor (Phase 9.5)
    findings_context = _format_findings_for_supervisor(findings)
    
    # Prepare prompt inputs
    prompt_inputs = {
        "scope": brief.scope,
        "sub_topics": ", ".join(brief.sub_topics),
        "constraints": str(brief.constraints),
        "findings_count": len(findings),
        "topics_covered": topics_covered_str,
        "avg_credibility": avg_credibility,
        "iterations": current_iteration,
        "max_iterations": budget["max_iterations"],
        "total_sub_agents": len(findings),
        "max_sub_agents": budget["max_sub_agents"],
        "total_searches": budget.get("total_searches", 0),
        "completed_count": len(completed_tasks),
        "failed_tasks": ", ".join(failed_tasks) if failed_tasks else "None",
        "findings_context": findings_context
    }
    
    # Get LLM with JSON mode (deepseek-reasoner doesn't support tool_choice)
    llm = get_deepseek_reasoner(temperature=0.5)
    
    # Add JSON schema instructions to prompt
    json_schema_instructions = """
You must respond with valid JSON matching this schema:
{
  "has_gaps": boolean,
  "is_complete": boolean,
  "gaps_identified": [string],
  "new_tasks": [
    {
      "task_id": string,
      "topic": string,
      "query": string,
      "priority": number,
      "requested_by": "supervisor"
    }
  ],
  "reasoning": string
}
"""
    prompt_inputs["json_schema"] = json_schema_instructions
    
    # Create chain and invoke with JSON mode
    chain = SUPERVISOR_GAP_ANALYSIS_TEMPLATE | llm
    
    try:
        import json
        import re
        response = chain.invoke(prompt_inputs)
        
        # Extract JSON from response (handle DeepSeek Reasoner format)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Log raw response for debugging
        logger.debug(f"Raw LLM response: {response_text[:500]}")
        
        # Remove thinking/reasoning blocks if present
        response_text = re.sub(r'<thinking>.*?</thinking>', '', response_text, flags=re.DOTALL)
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        
        # Try to find JSON object in the text
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        
        # Clean up the response
        response_text = response_text.strip()
        
        if not response_text:
            raise ValueError("Empty response from LLM")
        
        result_dict = json.loads(response_text)
        
        # Validate and convert to Pydantic model
        result = GapAnalysisOutput(**result_dict)
        logger.info(f"Supervisor: Analysis complete - has_gaps={result.has_gaps}, is_complete={result.is_complete}")
        
        # Update state based on result
        state_update: Dict[str, Any] = {
            "budget": {**budget, "iterations": current_iteration},
            "gaps": {
                "has_gaps": result.has_gaps,
                "gaps_identified": result.gaps_identified,
                "reasoning": result.reasoning
            }
        }
        
        # Add new tasks if gaps exist and not complete
        if result.new_tasks and not result.is_complete:
            state_update["task_history"] = result.new_tasks
            logger.info(f"Supervisor: Generated {len(result.new_tasks)} new tasks")
        
        # Set completion flag
        state_update["is_complete"] = result.is_complete
        
        # If complete, aggregate findings automatically (Phase 9.5)
        if result.is_complete:
            logger.info("Supervisor: Research complete, aggregating findings")
            try:
                aggregated_findings = aggregate_findings(state)
                state_update["findings"] = aggregated_findings
                logger.info(f"Supervisor: Aggregated {len(aggregated_findings)} findings for report")
            except Exception as agg_error:
                logger.error(f"Supervisor: Findings aggregation failed - {agg_error}")
                # Don't fail the entire process, just log the error
        
        return state_update
        
    except json.JSONDecodeError as e:
        logger.error(f"Supervisor: Failed to parse JSON response - {e}")
        return {
            "budget": {**budget, "iterations": current_iteration},
            "is_complete": True,
            "error": [f"Supervisor JSON parsing failed: {str(e)}"]
        }
    except Exception as e:
        logger.error(f"Supervisor gap analysis failed: {e}")
        # On error, mark as complete to avoid infinite loops
        return {
            "budget": {**budget, "iterations": current_iteration},
            "is_complete": True,
            "error": [f"Supervisor analysis failed: {str(e)}"]  # List, not string
        }


def aggregate_findings(state: ResearchState) -> List[Finding]:
    """Aggregate and filter findings for report generation (Phase 8.5).
    
    Applies LLM-based filtering and programmatic deduplication.
    
    Args:
        state: Research state with findings and brief
        
    Returns:
        Filtered and ranked list of findings
    """
    findings = state.get("findings", [])
    brief = state["research_brief"]
    
    if not findings:
        return []
    
    # Step 1: Programmatic filtering (credibility threshold)
    filtered = [f for f in findings if f.credibility_score >= 0.5]
    logger.info(f"Filtered {len(findings)} -> {len(filtered)} findings (credibility >= 0.5)")
    
    # Step 2: Programmatic deduplication
    seen_dois = {}
    seen_urls = {}
    seen_titles = {}
    deduplicated = []
    
    for finding in filtered:
        citation = finding.citation
        
        # Check if duplicate based on any criteria
        is_duplicate = False
        existing_match = None
        
        # 1. Check DOI
        if citation.doi:
            if citation.doi in seen_dois:
                is_duplicate = True
                existing_match = seen_dois[citation.doi]
            else:
                seen_dois[citation.doi] = finding
        
        # 2. Check URL (if not already found as duplicate)
        if not is_duplicate and citation.url:
            if citation.url in seen_urls:
                is_duplicate = True
                existing_match = seen_urls[citation.url]
            else:
                seen_urls[citation.url] = finding
                
        # 3. Check Title + Author (if not already found as duplicate)
        if not is_duplicate and citation.title and citation.authors:
            key = (citation.title, citation.authors[0] if citation.authors else "")
            if key in seen_titles:
                is_duplicate = True
                existing_match = seen_titles[key]
            else:
                seen_titles[key] = finding
                
        # Handle duplicate or new entry
        if is_duplicate and existing_match:
            # If new finding has higher credibility, replace the existing one
            if finding.credibility_score > existing_match.credibility_score:
                if existing_match in deduplicated:
                    deduplicated.remove(existing_match)
                deduplicated.append(finding)
                
                # Update indices to point to new finding
                if citation.doi: seen_dois[citation.doi] = finding
                if citation.url: seen_urls[citation.url] = finding
                if citation.title and citation.authors:
                    key = (citation.title, citation.authors[0] if citation.authors else "")
                    seen_titles[key] = finding
        else:
            deduplicated.append(finding)
    
    logger.info(f"Deduplicated {len(filtered)} -> {len(deduplicated)} findings")
    
    # Step 3: Sort by credibility (highest first)
    sorted_findings = sorted(
        deduplicated,
        key=lambda f: f.credibility_score,
        reverse=True
    )
    
    return sorted_findings
