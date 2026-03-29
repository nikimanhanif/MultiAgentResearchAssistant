"""
ArXiv search and content retrieval tool.

Use this tool to find cutting-edge preprints in CS, Math, and Physics.
High likelihood of full-text PDF availability.
"""

import logging
from typing import List, Tuple, Optional, Any, Dict
from langsmith import traceable

from langchain_core.tools import tool
from langchain_community.document_loaders import ArxivLoader

from app.tools.academic.utils import (
    retry_on_rate_limit,
    _is_rate_limit_error,
    extract_paper_sections,
    format_search_results,
)

logger = logging.getLogger(__name__)


@traceable(run_type="tool", name="Search ArXiv", metadata={"tool": "arxiv_search"})
@retry_on_rate_limit()
def _search_arxiv_internal(query: str, count: int) -> List[Dict[str, Any]]:
    """Search ArXiv and return structured paper list."""
    try:
        import arxiv
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=count,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in client.results(search):
            arxiv_id = result.entry_id.split("/")[-1]
            papers.append({
                "source": "arxiv",
                "paper_id": arxiv_id,
                "title": result.title,
                "authors": ", ".join([a.name for a in result.authors[:3]]),
                "abstract": result.summary[:500] + "..." if len(result.summary) > 500 else result.summary,
                "year": result.published.year if result.published else None,
                "pdf_url": result.pdf_url,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "citation_count": None,  # ArXiv doesn't provide citations
            })
        return papers
        
    except Exception as e:
        if _is_rate_limit_error(e):
            raise
        logger.error(f"ArXiv search failed: {e}")
        return []


@tool(response_format="content_and_artifact")
def search_arxiv(
    query: str,
    count: int = 5,
) -> Tuple[str, Optional[Any]]:
    """Search ArXiv for cutting-edge preprints in CS, Math, and Physics.
    
    High likelihood of full-text PDF availability. Best for recent
    pre-prints and working papers before formal peer review.
    
    Args:
        query: Search query terms (e.g. "transformer attention mechanism")
        count: Maximum number of papers to return (default 5)
        
    Returns:
        Formatted string with paper metadata for agent processing.
    """
    try:
        logger.info(f"Searching ArXiv for: {query} (count={count})")
        papers = _search_arxiv_internal(query, count)
        result = format_search_results(papers, "arxiv", query)
        return (result, None)
    except Exception as e:
        logger.error(f"ArXiv search failed: {e}")
        return (f"ArXiv search error: {str(e)}. Try rephrasing your query.", None)


def fetch_arxiv_content(paper_id: str) -> Tuple[str, Optional[Any]]:
    """
    Fetch and extract key sections from an ArXiv paper.
    
    Args:
        paper_id: ArXiv ID (e.g., "2401.12345" or "1706.03762")
        
    Returns:
        Tuple of (extracted content string, None)
    """
    try:
        loader = ArxivLoader(
            query=paper_id,
            load_max_docs=1,
            doc_content_chars_max=None,
        )
        
        docs = loader.load()
        
        if not docs:
            return (f"Could not load ArXiv paper {paper_id}. Verify the ID is correct.", None)
        
        doc = docs[0]
        full_text = doc.page_content
        metadata = doc.metadata
        
        title = metadata.get('Title', 'Unknown Title')
        authors = metadata.get('Authors', 'Unknown Authors')
        published = metadata.get('Published', 'Unknown Date')
        
        metadata_prefix = f"## {title}\n\n"
        metadata_prefix += f"**Authors**: {authors}\n"
        metadata_prefix += f"**Published**: {published}\n"
        metadata_prefix += f"**ArXiv ID**: {paper_id}\n\n"
        
        if full_text and len(full_text) > 10000:
            extracted = extract_paper_sections(full_text, metadata_prefix)
        else:
            extracted = metadata_prefix + (full_text or "No content available")
        
        return (extracted, None)
        
    except Exception as e:
        logger.error(f"ArXiv fetch failed for {paper_id}: {e}")
        return (f"Failed to fetch ArXiv paper {paper_id}: {str(e)}", None)
