"""
Academic paper search and retrieval tools.

Provides dedicated LangChain-based tools for querying ArXiv, Semantic Scholar, 
and PubMed, with unified PDF parsing across all sources.
"""

import logging
import re
import json
import tempfile
import httpx
import concurrent.futures
from typing import Literal, List, Optional, Tuple, Any, Dict
from pathlib import Path

from langchain_core.tools import tool, BaseTool
from langchain_community.document_loaders import ArxivLoader, PyMuPDFLoader
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.utilities.pubmed import PubMedAPIWrapper

logger = logging.getLogger(__name__)

# Initialize API wrappers lazily to avoid uvloop conflicts
_arxiv_api = ArxivAPIWrapper(
    top_k_results=5,
    load_max_docs=5,
    load_all_available_meta=True,
)

# PubMedAPIWrapper is lazily initialized to avoid event loop issues at import time
_pubmed_api = None

def _get_pubmed_api():
    """Lazily initialize PubMed API wrapper."""
    global _pubmed_api
    if _pubmed_api is None:
        _pubmed_api = PubMedAPIWrapper()
    return _pubmed_api


def _extract_paper_sections(full_text: str, metadata_prefix: str = "") -> str:
    """
    Extract key sections from academic paper text.
    
    Strategy:
    - Abstract + Introduction: Find "Abstract" and take 5000 chars
    - Conclusion: 3000 chars before "References" section
    
    Args:
        full_text: Full paper text from PDF.
        metadata_prefix: Optional metadata header to prepend.
        
    Returns:
        str: Extracted sections (~6-8K chars instead of 10-50K).
    """
    paper_text = full_text
    
    # Handle JSON-wrapped content
    if full_text.strip().startswith('{'):
        try:
            data = json.loads(full_text)
            if isinstance(data, dict) and 'text' in data:
                paper_text = data.get('text', '')
        except json.JSONDecodeError:
            pass
    
    # ABSTRACT + INTRODUCTION: Find "Abstract" and take 5000 chars
    abstract_match = re.search(
        r'(?:Abstract|ABSTRACT)\s*[:\n]',
        paper_text, re.IGNORECASE
    )
    if abstract_match:
        start_pos = abstract_match.start()
        beginning = paper_text[start_pos:start_pos + 5000].strip()
    else:
        beginning = paper_text[:5000].strip()
    
    # CONCLUSION: Find "References" section and take 3000 chars before it
    ref_match = re.search(
        r'\n\s*(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*\n',
        paper_text
    )
    
    if ref_match:
        ref_start = ref_match.start()
        conclusion_start = max(5000, ref_start - 3000)
        conclusion = paper_text[conclusion_start:ref_start].strip()
    else:
        ref_match = re.search(r'\n\[\d+\]\s+[A-Z]', paper_text[int(len(paper_text) * 0.7):])
        if ref_match:
            ref_start = int(len(paper_text) * 0.7) + ref_match.start()
            conclusion_start = max(5000, ref_start - 3000)
            conclusion = paper_text[conclusion_start:ref_start].strip()
        else:
            conclusion = paper_text[-3000:].strip()
    
    result = metadata_prefix + f"## ABSTRACT & INTRODUCTION\n{beginning}"
    
    if conclusion and len(conclusion) > 200:
        result += f"\n\n## CONCLUSION\n{conclusion}"
    
    return result


def _download_and_parse_pdf(pdf_url: str) -> Optional[str]:
    """
    Download PDF from URL and parse using PyMuPDF.
    
    Args:
        pdf_url: URL to the PDF file.
        
    Returns:
        Extracted text from PDF, or None if failed.
    """
    # Browser-like headers to avoid bot detection by academic publishers
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            with httpx.Client(timeout=30.0, follow_redirects=True, headers=headers) as client:
                response = client.get(pdf_url)
                response.raise_for_status()
                tmp_file.write(response.content)
                tmp_path = tmp_file.name
        
        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()
        
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)
        
        if docs:
            return "\n\n".join([doc.page_content for doc in docs])
        return None
        
    except Exception as e:
        logger.error(f"PDF download/parse failed: {e}")
        return None


def _search_arxiv(query: str, count: int) -> List[Dict[str, Any]]:
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
            papers.append({
                "source": "arxiv",
                "paper_id": result.entry_id.split("/")[-1],
                "title": result.title,
                "authors": ", ".join([a.name for a in result.authors[:3]]),
                "abstract": result.summary[:500] + "..." if len(result.summary) > 500 else result.summary,
                "year": result.published.year if result.published else None,
                "pdf_url": result.pdf_url,
                "citation_count": None,  # ArXiv doesn't provide citations
            })
        return papers
        
    except Exception as e:
        logger.error(f"ArXiv search failed: {e}")
        return []


def _search_semantic_scholar_sync(query: str, count: int) -> List[Dict[str, Any]]:
    """
    Synchronous Semantic Scholar search (runs in thread pool).
    
    The semanticscholar library calls nest_asyncio.apply() in __init__, which
    fails with uvloop. We monkey-patch it to be a no-op since we don't need
    nested event loops in our sync thread context.
    
    The library also requires an event loop for its async internals, so we
    create a new standard asyncio loop in the thread (not uvloop).
    """
    import asyncio
    import nest_asyncio
    
    original_apply = nest_asyncio.apply
    nest_asyncio.apply = lambda: None  # Make it a no-op
    
    # Create a new event loop for this thread (standard asyncio, not uvloop)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        from semanticscholar import SemanticScholar
        
        # Set a generous timeout for the API client (120s)
        sch = SemanticScholar(timeout=120)
        results = sch.search_paper(query, limit=count, fields=[
            "paperId", "title", "abstract", "authors", "year", 
            "citationCount", "openAccessPdf", "externalIds"
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
                "citation_count": paper.citationCount,
                "arxiv_id": arxiv_id,
            })
        return papers
    finally:
        # Cleanup
        loop.close()
        nest_asyncio.apply = original_apply


def _search_semantic_scholar(query: str, count: int) -> List[Dict[str, Any]]:
    """
    Search Semantic Scholar and return structured paper list with citations.
    
    Runs in a thread pool to avoid uvloop/nest_asyncio conflict.
    """
    try:
        # Run in thread pool to avoid nest_asyncio + uvloop conflict
        # Timeout of 120s to handle slow Semantic Scholar API responses
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_search_semantic_scholar_sync, query, count)
            return future.result(timeout=120)
        
    except concurrent.futures.TimeoutError:
        logger.error("Semantic Scholar search timed out")
        return []
    except Exception as e:
        logger.error(f"Semantic Scholar search failed: {e}")
        return []


def _get_semantic_scholar_paper_sync(paper_id: str, fields: List[str]) -> Optional[Any]:
    """
    Synchronous Semantic Scholar paper fetch (runs in thread pool).
    
    The semanticscholar library calls nest_asyncio.apply() in __init__, which
    fails with uvloop. We monkey-patch it to be a no-op since we don't need
    nested event loops in our sync thread context.
    
    The library also requires an event loop for its async internals, so we
    create a new standard asyncio loop in the thread (not uvloop).
    """
    import asyncio
    import nest_asyncio
    
    original_apply = nest_asyncio.apply
    nest_asyncio.apply = lambda: None  # Make it a no-op
    
    # Create a new event loop for this thread (standard asyncio, not uvloop)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        from semanticscholar import SemanticScholar
        
        # Set a generous timeout for the API client (120s)
        sch = SemanticScholar(timeout=120)
        return sch.get_paper(paper_id, fields=fields)
    finally:
        # Cleanup
        loop.close()
        nest_asyncio.apply = original_apply


def _get_semantic_scholar_paper(paper_id: str, fields: List[str]) -> Optional[Any]:
    """
    Get paper details from Semantic Scholar.
    
    Runs in a thread pool to avoid uvloop/nest_asyncio conflict.
    """
    try:
        # Timeout of 120s to handle slow Semantic Scholar API
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_get_semantic_scholar_paper_sync, paper_id, fields)
            return future.result(timeout=120)
    except concurrent.futures.TimeoutError:
        logger.error(f"Semantic Scholar paper fetch timed out for {paper_id}")
        return None
    except Exception as e:
        logger.error(f"Semantic Scholar paper fetch failed for {paper_id}: {e}")
        return None


def _search_pubmed(query: str, count: int) -> List[Dict[str, Any]]:
    """Search PubMed and return structured paper list."""
    try:
        # PubMedAPIWrapper returns docs with limited metadata
        raw_results = _get_pubmed_api().run(query)
        
        if not raw_results or raw_results == "No good PubMed Result was found":
            return []
        
        # Parse the text output into structured data
        papers = []
        # PubMed results come as text, we'll parse them
        entries = raw_results.split("\n\n")
        
        for entry in entries[:count]:
            title_match = re.search(r"Title:\s*(.+?)(?:\n|$)", entry)
            abstract_match = re.search(r"(?:Abstract|Summary):\s*(.+?)(?:\n\n|$)", entry, re.DOTALL)
            published_match = re.search(r"Published:\s*(\d{4})", entry)
            pmid_match = re.search(r"PMID:\s*(\d+)", entry)
            
            if title_match:
                abstract = abstract_match.group(1).strip() if abstract_match else ""
                papers.append({
                    "source": "pubmed",
                    "paper_id": pmid_match.group(1) if pmid_match else None,
                    "title": title_match.group(1).strip(),
                    "authors": "",  # PubMed wrapper doesn't easily expose authors
                    "abstract": abstract[:500] + "..." if len(abstract) > 500 else abstract,
                    "year": int(published_match.group(1)) if published_match else None,
                    "pdf_url": None,  # PubMed Central requires separate lookup
                    "citation_count": None,
                })
        return papers
        
    except Exception as e:
        logger.error(f"PubMed search failed: {e}")
        return []


def _format_search_results(papers: List[Dict[str, Any]], source: str, query: str) -> str:
    """Format paper list into readable string for agent."""
    if not papers:
        return f"No papers found from {source} for query: {query}"
    
    output = f"## {source.replace('_', ' ').title()} Results for: {query}\n\n"
    
    for i, paper in enumerate(papers, 1):
        output += f"### {i}. {paper['title']}\n"
        output += f"- **Source**: {paper['source']}\n"
        output += f"- **Paper ID**: {paper['paper_id']}\n"
        if paper.get('authors'):
            output += f"- **Authors**: {paper['authors']}\n"
        if paper.get('year'):
            output += f"- **Year**: {paper['year']}\n"
        if paper.get('citation_count') is not None:
            output += f"- **Citations**: {paper['citation_count']}\n"
        if paper.get('pdf_url'):
            output += f"- **PDF Available**: Yes\n"
        if paper.get('abstract'):
            output += f"- **Abstract**: {paper['abstract']}\n"
        output += "\n"
    
    output += "\n**Use `fetch_paper_content(source, paper_id)` to get full text.**"
    return output


@tool(response_format="content_and_artifact")
def search_papers(
    query: str,
    source: Literal["arxiv", "semantic_scholar", "pubmed", "all"] = "all",
    count: int = 5,
    sortBy: Literal["relevance", "citations"] = "relevance"
) -> Tuple[str, Optional[Any]]:
    """Search academic databases for papers matching query.
    
    Use this tool in Phase 1 (Discovery) to find candidate papers.
    Returns paper metadata including titles, authors, abstracts, and citation counts
    for ranking in Phase 2 (Triage).
    
    Args:
        query: Search query terms (e.g. "transformer attention mechanism")
        source: Database to search:
            - "arxiv" for CS/ML/Physics papers
            - "semantic_scholar" for broad academic coverage with citation counts
            - "pubmed" for biomedical/life sciences papers
            - "all" to search all sources (default)
        count: Maximum number of papers per source (default 5)
        sortBy: Sort by "relevance" (default) or "citations" for foundational papers
        
    Returns:
        Formatted string with paper metadata for agent processing.
    """
    try:
        logger.info(f"Searching {source} for: {query} (count={count})")
        
        all_papers = []
        
        if source in ["arxiv", "all"]:
            arxiv_papers = _search_arxiv(query, count)
            all_papers.extend(arxiv_papers)
        
        # TEMPORARILY DISABLED: Semantic Scholar frequently times out
        # if source in ["semantic_scholar", "all"]:
        #     ss_papers = _search_semantic_scholar(query, count)
        #     all_papers.extend(ss_papers)
        if source == "semantic_scholar":
            logger.warning("Semantic Scholar is temporarily disabled due to timeout issues")
            return ("Semantic Scholar is temporarily disabled. Please use 'arxiv' or 'pubmed'.", None)
            
        if source in ["pubmed", "all"]:
            pubmed_papers = _search_pubmed(query, count)
            all_papers.extend(pubmed_papers)
        
        if not all_papers:
            return (f"No papers found for query: {query}. Try different keywords or a specific source.", None)
        
        # Sort by citations if requested
        if sortBy == "citations":
            all_papers.sort(key=lambda p: p.get("citation_count") or 0, reverse=True)
        
        # Group by source for output
        output = f"# Search Results for: {query}\n\n"
        output += f"**Total papers found**: {len(all_papers)}\n\n"
        
        for i, paper in enumerate(all_papers, 1):
            output += f"### {i}. {paper['title']}\n"
            output += f"- **Source**: {paper['source']}\n"
            output += f"- **Paper ID**: `{paper['paper_id']}`\n"
            if paper.get('authors'):
                output += f"- **Authors**: {paper['authors']}\n"
            if paper.get('year'):
                output += f"- **Year**: {paper['year']}\n"
            if paper.get('citation_count') is not None:
                output += f"- **Citations**: {paper['citation_count']} {'📈' if paper['citation_count'] > 100 else ''}\n"
            if paper.get('pdf_url'):
                output += f"- **PDF**: Available\n"
            if paper.get('abstract'):
                output += f"- **Abstract**: {paper['abstract']}\n"
            output += "\n"
        
        output += "\n---\n**Use `fetch_paper_content(source, paper_id)` to get full text for deep reading.**"
        
        return (output, None)
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return (f"Search error: {str(e)}. Try rephrasing your query.", None)


@tool(response_format="content_and_artifact")
def fetch_paper_content(
    source: Literal["arxiv", "semantic_scholar", "pubmed"],
    paper_id: str
) -> Tuple[str, Optional[Any]]:
    """Fetch and extract key sections from an academic paper.
    
    Use this tool in Phase 3 (Deep Reading) ONLY for papers selected during Triage.
    
    **RELIABILITY NOTE**: 
    - ArXiv: RELIABLE - always provides full PDF text
    - Semantic Scholar: PARTIAL - only works if paper is open access (~30%)
    - PubMed: UNRELIABLE - rate-limited, only works if paper is in PMC
    
    **RECOMMENDATION**: Prefer ArXiv papers for deep reading when possible.
    For non-ArXiv papers, you may only get the abstract.
    
    Args:
        source: The database the paper is from ("arxiv", "semantic_scholar", "pubmed")
        paper_id: The paper identifier:
            - ArXiv ID (e.g., "2401.12345" or "1706.03762")
            - Semantic Scholar ID (e.g., "649def34f8be52c8b66281af98ae884c09aef38b")
            - PubMed PMID (e.g., "12345678")
        
    Returns:
        Extracted paper sections (Abstract, Intro, Conclusion) for analysis.
    """
    try:
        logger.info(f"Fetching paper content: {source}/{paper_id}")
        
        # TEMPORARILY DISABLED: Semantic Scholar frequently times out
        if source == "semantic_scholar":
            return ("Semantic Scholar is temporarily disabled due to timeout issues. Please use ArXiv papers.", None)
        
        full_text = None
        metadata_prefix = ""
        
        if source == "arxiv":
            # Use ArxivLoader for reliable PDF parsing
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
            
        elif source == "semantic_scholar":
            # Get paper details and PDF URL from Semantic Scholar
            # Uses thread pool to avoid uvloop/nest_asyncio conflict
            paper = _get_semantic_scholar_paper(paper_id, [
                "title", "authors", "year", "abstract", "openAccessPdf"
            ])
            
            if not paper:
                return (f"Could not find Semantic Scholar paper {paper_id}.", None)
            
            title = paper.title or "Unknown Title"
            
            # Handle Author objects - they have .name attribute, not dict
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
            
            if pdf_url:
                full_text = _download_and_parse_pdf(pdf_url)
            
            if not full_text and paper.abstract:
                # Fallback to abstract only
                return (
                    metadata_prefix + f"### Abstract\n{paper.abstract}\n\n"
                    "*Note: Full PDF not available. Only abstract provided.*",
                    None
                )
            
            if not full_text:
                return (f"No full-text content available for {paper_id}. Paper may not be open access.", None)
                
        elif source == "pubmed":
            # PubMed - try to get from PMC if available
            # First, get the paper metadata
            result = _get_pubmed_api().run(f"{paper_id}[PMID]")
            
            if not result or "No good PubMed Result was found" in result:
                return (f"Could not find PubMed paper with PMID {paper_id}.", None)
            
            # Try PMC Central for full text PDF
            pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/pmid/{paper_id}/pdf"
            full_text = _download_and_parse_pdf(pmc_url)
            
            if not full_text:
                # Return abstract from PubMed results
                return (
                    f"## PubMed Paper (PMID: {paper_id})\n\n{result}\n\n"
                    "*Note: Full PDF not in PMC. Only abstract provided.*",
                    None
                )
            
            metadata_prefix = f"## PubMed Paper (PMID: {paper_id})\n\n"
            
        else:
            return (f"Unknown source: {source}. Use 'arxiv', 'semantic_scholar', or 'pubmed'.", None)
        
        # Extract key sections from full text
        if full_text and len(full_text) > 10000:
            extracted = _extract_paper_sections(full_text, metadata_prefix)
        else:
            extracted = metadata_prefix + (full_text or "No content available")
        
        return (extracted, None)
        
    except Exception as e:
        logger.error(f"Fetch failed for {source}/{paper_id}: {e}")
        return (f"Failed to fetch paper {paper_id}: {str(e)}", None)


def get_academic_tools() -> List[BaseTool]:
    """
    Get all academic research tools.
    
    Returns:
        List[BaseTool]: List of configured academic tools
            - search_papers: Search ArXiv, Semantic Scholar, PubMed
            - fetch_paper_content: Download and extract key sections from papers
    """
    return [search_papers, fetch_paper_content]
