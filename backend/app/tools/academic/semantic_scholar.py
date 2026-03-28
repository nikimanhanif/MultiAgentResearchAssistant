"""
Semantic Scholar search and citation graph tools.

Use search_semantic_scholar for broad academic coverage with citation counts.
Use get_citation_graph to perform 'snowballing' on foundational papers.
"""

import logging
import asyncio
import concurrent.futures
from typing import List, Tuple, Optional, Any, Dict
from langsmith import traceable

from langchain_core.tools import tool

from app.tools.academic.utils import (
    retry_on_rate_limit,
    _is_rate_limit_error,
    download_and_parse_pdf,
    extract_paper_sections,
    format_search_results,
)
from app.config import settings

logger = logging.getLogger(__name__)


# --- Internal sync helpers (run in thread pool to avoid uvloop conflicts) ---

@retry_on_rate_limit()
def _search_semantic_scholar_sync(query: str, count: int) -> List[Dict[str, Any]]:
    """
    Synchronous Semantic Scholar search (runs in thread pool).
    
    The semanticscholar library calls nest_asyncio.apply() in __init__, which
    fails with uvloop. We monkey-patch it to be a no-op since we don't need
    nested event loops in our sync thread context.
    """
    import nest_asyncio
    
    original_apply = nest_asyncio.apply
    nest_asyncio.apply = lambda: None  # Make it a no-op
    
    # Create a new event loop for this thread (standard asyncio, not uvloop)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        from semanticscholar import SemanticScholar
        
        sch = SemanticScholar(timeout=120)
        if settings.SEMANTIC_SCHOLAR_API_KEY:
            sch = SemanticScholar(timeout=120, api_key=settings.SEMANTIC_SCHOLAR_API_KEY)
        
        results = sch.search_paper(query, limit=count, fields=[
            "paperId", "title", "abstract", "authors", "year", 
            "citationCount", "openAccessPdf", "externalIds", "url"
        ])
        
        papers = []
        for paper in results.items:
            pdf_url = None
            if paper.openAccessPdf:
                pdf_url = paper.openAccessPdf.get("url") if isinstance(paper.openAccessPdf, dict) else getattr(paper.openAccessPdf, "url", None)
            
            # Handle Author objects - they have .name attribute, not dict
            authors_list = paper.authors or []
            author_names = []
            for a in authors_list[:3]:
                if hasattr(a, 'name'):
                    author_names.append(a.name)
                elif isinstance(a, dict):
                    author_names.append(a.get("name", ""))
            authors = ", ".join(author_names)
            
            abstract = paper.abstract or ""
            
            # Handle externalIds
            arxiv_id = None
            if paper.externalIds:
                if isinstance(paper.externalIds, dict):
                    arxiv_id = paper.externalIds.get("ArXiv")
                else:
                    arxiv_id = getattr(paper.externalIds, "ArXiv", None)
            
            papers.append({
                "source": "semantic_scholar",
                "paper_id": paper.paperId,
                "title": paper.title,
                "authors": authors,
                "abstract": abstract[:500] + "..." if len(abstract) > 500 else abstract,
                "year": paper.year,
                "pdf_url": pdf_url,
                "url": pdf_url or getattr(paper, 'url', None) or f"https://www.semanticscholar.org/paper/{paper.paperId}",
                "citation_count": paper.citationCount,
                "arxiv_id": arxiv_id,
            })
        return papers
    except Exception as e:
        if _is_rate_limit_error(e):
            raise
        logger.error(f"Semantic Scholar sync search failed: {e}")
        return []
    finally:
        try:
            loop.close()
        except Exception as e:
            logger.debug(f"Failed to close event loop: {e}")
        nest_asyncio.apply = original_apply


def _get_paper_sync(paper_id: str, fields: List[str]) -> Optional[Any]:
    """
    Synchronous Semantic Scholar paper fetch (runs in thread pool).
    """
    import nest_asyncio
    
    original_apply = nest_asyncio.apply
    nest_asyncio.apply = lambda: None
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        from semanticscholar import SemanticScholar
        
        sch = SemanticScholar(timeout=120)
        if settings.SEMANTIC_SCHOLAR_API_KEY:
            sch = SemanticScholar(timeout=120, api_key=settings.SEMANTIC_SCHOLAR_API_KEY)
        
        return sch.get_paper(paper_id, fields=fields)
    finally:
        loop.close()
        nest_asyncio.apply = original_apply


@retry_on_rate_limit()
def _get_citation_graph_sync(paper_id: str, max_results: int = 10) -> Dict[str, Any]:
    """
    Synchronous citation graph fetch (runs in thread pool).
    
    Fetches both citations (papers that cite this paper) and
    references (papers this paper cites), sorted by influence.
    """
    import nest_asyncio
    
    original_apply = nest_asyncio.apply
    nest_asyncio.apply = lambda: None
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        from semanticscholar import SemanticScholar
        
        sch = SemanticScholar(timeout=120)
        if settings.SEMANTIC_SCHOLAR_API_KEY:
            sch = SemanticScholar(timeout=120, api_key=settings.SEMANTIC_SCHOLAR_API_KEY)
        
        # Fetch the paper with citations and references
        paper = sch.get_paper(paper_id, fields=[
            "title", "year", "citationCount", "influentialCitationCount",
            "citations", "citations.title", "citations.year", 
            "citations.citationCount", "citations.influentialCitationCount",
            "citations.authors", "citations.paperId",
            "references", "references.title", "references.year",
            "references.citationCount", "references.influentialCitationCount",
            "references.authors", "references.paperId",
        ])
        
        if not paper:
            return {"error": f"Paper {paper_id} not found"}
        
        def _extract_papers(items: list, max_n: int) -> List[Dict]:
            """Extract and sort papers by influential citation count."""
            extracted = []
            for p in (items or []):
                if not p or not getattr(p, 'title', None):
                    continue
                
                author_names = []
                for a in (getattr(p, 'authors', None) or [])[:3]:
                    if hasattr(a, 'name'):
                        author_names.append(a.name)
                    elif isinstance(a, dict):
                        author_names.append(a.get("name", ""))
                
                extracted.append({
                    "paper_id": getattr(p, 'paperId', None),
                    "title": p.title,
                    "year": getattr(p, 'year', None),
                    "citation_count": getattr(p, 'citationCount', None),
                    "influential_citation_count": getattr(p, 'influentialCitationCount', None),
                    "authors": ", ".join(author_names),
                })
            
            # Sort by influential citation count (desc), then by citation count
            extracted.sort(
                key=lambda x: (x.get("influential_citation_count") or 0, x.get("citation_count") or 0),
                reverse=True
            )
            return extracted[:max_n]
        
        return {
            "seed_paper": {
                "paper_id": paper_id,
                "title": paper.title,
                "year": paper.year,
                "citation_count": paper.citationCount,
                "influential_citation_count": paper.influentialCitationCount,
            },
            "top_citations": _extract_papers(paper.citations, max_results),
            "top_references": _extract_papers(paper.references, max_results),
        }
        
    except Exception as e:
        if _is_rate_limit_error(e):
            raise
        logger.error(f"Citation graph fetch failed: {e}")
        return {"error": str(e)}
    finally:
        try:
            loop.close()
        except Exception as e:
            logger.debug(f"Failed to close event loop: {e}")
        nest_asyncio.apply = original_apply


# --- Thread pool wrappers ---

def _run_in_thread(fn, *args, timeout: int = 120):
    """Run a sync function in a thread pool to avoid uvloop conflicts."""
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn, *args)
            return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        logger.error(f"{fn.__name__} timed out")
        return None
    except Exception as e:
        logger.error(f"{fn.__name__} failed: {e}")
        return None


# --- Public LangChain tools ---

@tool(response_format="content_and_artifact")
def search_semantic_scholar(
    query: str,
    count: int = 5,
) -> Tuple[str, Optional[Any]]:
    """Search Semantic Scholar for broad academic coverage with citation counts.
    
    Good for overarching academic queries that require citation data to
    identify influential and foundational papers across all disciplines.
    
    Args:
        query: Search query terms (e.g. "large language models alignment")
        count: Maximum number of papers to return (default 5)
        
    Returns:
        Formatted string with paper metadata including citation counts.
    """
    try:
        logger.info(f"Searching Semantic Scholar for: {query} (count={count})")
        papers = _run_in_thread(_search_semantic_scholar_sync, query, count)
        if papers is None:
            return ("Semantic Scholar search timed out. Try again or use a different source.", None)
        result = format_search_results(papers, "semantic_scholar", query)
        return (result, None)
    except Exception as e:
        logger.error(f"Semantic Scholar search failed: {e}")
        return (f"Semantic Scholar search error: {str(e)}. Try rephrasing your query.", None)


@tool(response_format="content_and_artifact")
def get_citation_graph(
    paper_id: str,
    max_results: int = 10,
) -> Tuple[str, Optional[Any]]:
    """Perform 'snowballing' on a foundational paper using its Semantic Scholar ID.
    
    Traces how a research topic has developed by examining the paper's
    citations (who cited it) and references (what it cited), sorted by
    influence. Use this after finding a key paper to discover the research
    landscape around it.
    
    Args:
        paper_id: Semantic Scholar paper ID
                  (e.g. "649def34f8be52c8b66281af98ae884c09aef38b")
        max_results: Maximum number of citing/referenced papers to return (default 10)
        
    Returns:
        Formatted citation graph showing most influential citing and referenced papers.
    """
    try:
        logger.info(f"Building citation graph for: {paper_id}")
        graph_data = _run_in_thread(_get_citation_graph_sync, paper_id, max_results)
        
        if graph_data is None:
            return ("Citation graph request timed out. Try again later.", None)
        
        if "error" in graph_data:
            return (f"Citation graph error: {graph_data['error']}", None)
        
        # Format output
        seed = graph_data["seed_paper"]
        output = f"# Citation Graph for: {seed['title']}\n\n"
        output += f"**Year**: {seed.get('year', 'N/A')} | "
        output += f"**Total Citations**: {seed.get('citation_count', 'N/A')} | "
        output += f"**Influential Citations**: {seed.get('influential_citation_count', 'N/A')}\n\n"
        
        # Top citations (papers that cite this paper)
        output += "## 📊 Most Influential Citing Papers\n\n"
        citations = graph_data.get("top_citations", [])
        if citations:
            for i, p in enumerate(citations, 1):
                output += f"### {i}. {p['title']}\n"
                output += f"- **Paper ID**: `{p['paper_id']}`\n"
                if p.get('authors'):
                    output += f"- **Authors**: {p['authors']}\n"
                if p.get('year'):
                    output += f"- **Year**: {p['year']}\n"
                if p.get('citation_count') is not None:
                    output += f"- **Citations**: {p['citation_count']}\n"
                if p.get('influential_citation_count') is not None:
                    output += f"- **Influential Citations**: {p['influential_citation_count']}\n"
                output += "\n"
        else:
            output += "*No citing papers found.*\n\n"
        
        # Top references (papers this paper cites)
        output += "## 📚 Most Influential Referenced Papers\n\n"
        references = graph_data.get("top_references", [])
        if references:
            for i, p in enumerate(references, 1):
                output += f"### {i}. {p['title']}\n"
                output += f"- **Paper ID**: `{p['paper_id']}`\n"
                if p.get('authors'):
                    output += f"- **Authors**: {p['authors']}\n"
                if p.get('year'):
                    output += f"- **Year**: {p['year']}\n"
                if p.get('citation_count') is not None:
                    output += f"- **Citations**: {p['citation_count']}\n"
                if p.get('influential_citation_count') is not None:
                    output += f"- **Influential Citations**: {p['influential_citation_count']}\n"
                output += "\n"
        else:
            output += "*No referenced papers found.*\n\n"
        
        output += "---\n**Use `fetch_paper_content(source='semantic_scholar', paper_id='...')` to read any of these papers.**"
        
        return (output, None)
        
    except Exception as e:
        logger.error(f"Citation graph failed: {e}")
        return (f"Citation graph error: {str(e)}", None)


def fetch_semantic_scholar_content(paper_id: str) -> Tuple[str, Optional[Any]]:
    """
    Fetch and extract content from a Semantic Scholar paper.
    
    Args:
        paper_id: Semantic Scholar paper ID
        
    Returns:
        Tuple of (extracted content string, None)
    """
    try:
        paper = _run_in_thread(
            _get_paper_sync, paper_id,
            ["title", "authors", "year", "abstract", "openAccessPdf"]
        )
        
        if not paper:
            return (f"Could not find Semantic Scholar paper {paper_id}.", None)
        
        title = paper.title or "Unknown Title"
        
        # Handle Author objects
        authors_list = paper.authors or []
        author_names = []
        for a in authors_list[:5]:
            if hasattr(a, 'name'):
                author_names.append(a.name)
            elif isinstance(a, dict):
                author_names.append(a.get("name", ""))
        authors = ", ".join(author_names) or "Unknown Authors"
        
        year = paper.year or "Unknown"
        
        metadata_prefix = f"## {title}\n\n"
        metadata_prefix += f"**Authors**: {authors}\n"
        metadata_prefix += f"**Year**: {year}\n"
        metadata_prefix += f"**S2 ID**: {paper_id}\n\n"
        
        # Try to get full text from PDF
        pdf_url = None
        if paper.openAccessPdf:
            if isinstance(paper.openAccessPdf, dict):
                pdf_url = paper.openAccessPdf.get("url")
            else:
                pdf_url = getattr(paper.openAccessPdf, "url", None)
        
        full_text = None
        if pdf_url:
            full_text = download_and_parse_pdf(pdf_url)
        
        if full_text and len(full_text) > 10000:
            return (extract_paper_sections(full_text, metadata_prefix), None)
        elif full_text:
            return (metadata_prefix + full_text, None)
        
        # Fallback to abstract only
        if paper.abstract:
            return (
                metadata_prefix + f"### Abstract\n{paper.abstract}\n\n"
                "*Note: Full PDF not available. Only abstract provided.*",
                None
            )
        
        return (f"No full-text content available for {paper_id}. Paper may not be open access.", None)
        
    except Exception as e:
        logger.error(f"Semantic Scholar fetch failed for {paper_id}: {e}")
        return (f"Failed to fetch paper {paper_id}: {str(e)}", None)
