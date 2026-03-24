"""
Academic paper search and retrieval tools.

Provides dedicated LangChain-based tools for querying academic databases:
- ArXiv: Cutting-edge preprints in CS, Math, Physics
- Semantic Scholar: Broad academic coverage with citation counts
- Scopus: Peer-reviewed journal publications (Elsevier)
- Citation Graph: Snowballing via Semantic Scholar

Each tool is a standalone @tool so the LLM can dynamically select the
most appropriate database for each research query.
"""

import logging
from typing import Literal, List, Tuple, Optional, Any

from langchain_core.tools import tool, BaseTool
from langsmith import traceable

from app.tools.academic.arxiv import search_arxiv, fetch_arxiv_content
from app.tools.academic.scopus import search_scopus, fetch_scopus_content
from app.tools.academic.semantic_scholar import (
    search_semantic_scholar,
    get_citation_graph,
    fetch_semantic_scholar_content,
)

logger = logging.getLogger(__name__)


@tool(response_format="content_and_artifact")
def fetch_paper_content(
    source: Literal["arxiv", "semantic_scholar", "scopus"],
    paper_id: str
) -> Tuple[str, Optional[Any]]:
    """Fetch and extract key sections from an academic paper.
    
    Use this tool ONLY for papers selected during Triage (Phase 2).
    
    **RELIABILITY NOTE**: 
    - ArXiv: RELIABLE - always provides full PDF text
    - Semantic Scholar: PARTIAL - only works if paper is open access (~30%)
    - Scopus: PARTIAL - depends on institutional access and open access status
    
    **RECOMMENDATION**: Prefer ArXiv papers for deep reading when possible.
    For non-ArXiv papers, you may only get the abstract.
    
    Args:
        source: The database the paper is from ("arxiv", "semantic_scholar", "scopus")
        paper_id: The paper identifier:
            - ArXiv ID (e.g., "2401.12345" or "1706.03762")
            - Semantic Scholar ID (e.g., "649def34f8be52c8b66281af98ae884c09aef38b")
            - Scopus EID or DOI (e.g., "2-s2.0-85012345678")
        
    Returns:
        Extracted paper sections (Abstract, Intro, Conclusion) for analysis.
    """
    try:
        logger.info(f"Fetching paper content: {source}/{paper_id}")
        
        if source == "arxiv":
            return fetch_arxiv_content(paper_id)
        elif source == "semantic_scholar":
            return fetch_semantic_scholar_content(paper_id)
        elif source == "scopus":
            return fetch_scopus_content(paper_id)
        else:
            return (f"Unknown source: {source}. Use 'arxiv', 'semantic_scholar', or 'scopus'.", None)
        
    except Exception as e:
        logger.error(f"Fetch failed for {source}/{paper_id}: {e}")
        return (f"Failed to fetch paper {paper_id}: {str(e)}", None)


def get_academic_tools() -> List[BaseTool]:
    """
    Get all academic research tools.
    
    Returns:
        List[BaseTool]: List of configured academic tools:
            - search_arxiv: Search ArXiv preprints
            - search_semantic_scholar: Search Semantic Scholar
            - search_scopus: Search Elsevier Scopus
            - get_citation_graph: Snowball citations via S2
            - fetch_paper_content: Download and extract paper sections
    """
    return [
        search_arxiv,
        search_semantic_scholar,
        search_scopus,
        get_citation_graph,
        fetch_paper_content,
    ]
