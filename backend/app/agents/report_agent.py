"""
Report Agent - Generates the final research report.

This agent takes the research brief and aggregated findings to produce a
comprehensive markdown report. It uses a one-shot generation approach with
the DeepSeek Reasoner model.
"""

from typing import List, Any, Dict
import logging
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate

from app.models.schemas import ResearchBrief, Finding, ReportFormat
from app.graphs.state import ResearchState
from app.config import get_deepseek_reasoner
from app.prompts.report_prompts import (
    get_report_generation_prompt,
    format_findings_for_prompt,
    get_summary_format_instructions,
    get_comparison_format_instructions,
    get_literature_review_instructions,
    get_gap_analysis_instructions,
    get_fact_validation_instructions,
    get_ranking_format_instructions,
)

logger = logging.getLogger(__name__)


def _build_report_generation_chain() -> Runnable:
    """
    Build the LangChain runnable for report generation.
    
    Returns:
        Runnable: Chain of prompt | LLM.
    """
    prompt = get_report_generation_prompt()
    llm = get_deepseek_reasoner(temperature=0.5)
    
    chain = prompt | llm
    return chain


async def generate_report(
    brief: ResearchBrief,
    findings: List[Finding],
    reviewer_feedback: str = None
) -> str:
    """
    Generate a markdown report from the research brief and findings.
    
    Args:
        brief: The research brief defining scope and format.
        findings: List of findings with citations.
        reviewer_feedback: Optional feedback for refinement.
        
    Returns:
        str: The generated markdown report.
        
    Raises:
        ValueError: If inputs are invalid.
        Exception: If generation fails.
    """
    if not brief:
        raise ValueError("Research brief cannot be None")
    if not brief.scope:
        raise ValueError("Research brief must have a scope")
    if findings is None:
        raise ValueError("Findings list cannot be None (use empty list if no findings)")
    
    if not findings:
        return _generate_no_findings_report(brief)
    
    brief_subtopics = "\n".join([f"- {topic}" for topic in brief.sub_topics])
    brief_constraints = "\n".join(
        [f"- {key}: {value}" for key, value in brief.constraints.items()]
    ) if brief.constraints else "No specific constraints"
    
    format_type = brief.format or ReportFormat.OTHER
    format_instructions = _get_format_instructions(format_type)
    
    findings_context = format_findings_for_prompt(findings)
    
    try:
        chain = _build_report_generation_chain()
    except ValueError as e:
        raise Exception(f"Failed to initialize LLM: {e}")
    
    try:
        response = await chain.ainvoke({
            "brief_scope": brief.scope,
            "brief_subtopics": brief_subtopics,
            "brief_constraints": brief_constraints,
            "brief_format": format_type.value,
            "findings_context": findings_context,
            "format_instructions": format_instructions,
            "reviewer_feedback": reviewer_feedback or "None",
        })
        
        if hasattr(response, 'content'):
            return response.content
        return str(response)
            
    except Exception as e:
        raise Exception(f"Failed to generate report: {str(e)}")


def _get_format_instructions(format_type: ReportFormat) -> str:
    """
    Get specific instructions based on the requested report format.
    
    Args:
        format_type: The desired format (e.g., SUMMARY, COMPARISON).
        
    Returns:
        str: Format instructions for the prompt.
    """
    format_map = {
        ReportFormat.SUMMARY: get_summary_format_instructions,
        ReportFormat.COMPARISON: get_comparison_format_instructions,
        ReportFormat.RANKING: get_ranking_format_instructions,
        ReportFormat.FACT_VALIDATION: get_fact_validation_instructions,
        ReportFormat.LITERATURE_REVIEW: get_literature_review_instructions,
        ReportFormat.GAP_ANALYSIS: get_gap_analysis_instructions,
        ReportFormat.OTHER: lambda: "Use a general report structure with Introduction, Findings, Conclusion, and References.",
    }
    
    instruction_func = format_map.get(format_type, format_map[ReportFormat.OTHER])
    return instruction_func()


def _generate_no_findings_report(brief: ResearchBrief) -> str:
    """
    Generate a fallback report when no findings are available.
    
    Args:
        brief: The research brief.
        
    Returns:
        str: A minimal report stating no findings were found.
    """
    return f"""# Research Report: {brief.scope}

## Overview

This report was generated based on the following research brief:

**Scope**: {brief.scope}

**Sub-topics**:
{chr(10).join([f'- {topic}' for topic in brief.sub_topics])}

## Findings

No research findings were available to generate this report.

## Conclusion

Unable to complete research due to lack of findings. Please expand the research scope or try different search queries.
"""


async def report_agent_node(state: ResearchState) -> Dict[str, Any]:
    """
    LangGraph node for the Report Agent.
    
    Wraps report generation logic, handling state extraction and updates.
    Incorporates reviewer feedback if present.
    
    Args:
        state: Current research state.
        
    Returns:
        Dict[str, Any]: State update containing the generated report.
    """
    brief = state.get("research_brief")
    findings = state.get("findings", [])
    reviewer_feedback = state.get("reviewer_feedback")
    
    if not brief:
        return {
            "report_content": "Error: No research brief available for report generation.",
            "error": "Missing research brief"
        }
    
    try:
        report_content = await generate_report(brief, findings, reviewer_feedback)
        
        return {
            "report_content": report_content,
            "reviewer_feedback": None
        }
    
    except Exception as e:
        logger.error(f"Report Agent: Error generating report - {e}")
        return {
            "report_content": f"Error generating report: {str(e)}",
            "error": str(e)
        }
