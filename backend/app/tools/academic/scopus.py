"""
Elsevier Scopus search tool.

Use this tool to find highly credible, peer-reviewed academic journal
publications. Use when verification and credibility scoring are paramount.
"""

import logging
import requests
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
from app.tools.academic.semantic_scholar import _run_in_thread, _get_paper_sync
from app.config import settings

logger = logging.getLogger(__name__)

# Elsevier API Base URLs
ELSEVIER_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"
ELSEVIER_ABSTRACT_URL = "https://api.elsevier.com/content/abstract/scopus_id/"

def _get_elsevier_headers() -> Dict[str, str]:
    """Return standard headers for Elsevier API authentication."""
    headers = {
        "Accept": "application/json",
    }
    if settings.SCOPUS_API_KEY:
        headers["X-ELS-APIKey"] = settings.SCOPUS_API_KEY
    if settings.SCOPUS_INST_TOKEN:
        headers["X-ELS-Insttoken"] = settings.SCOPUS_INST_TOKEN
    return headers


@retry_on_rate_limit()
def _search_scopus_sync(query: str, count: int) -> List[Dict[str, Any]]:
    """
    Synchronous Scopus search (runs in thread pool).
    
    Uses direct REST API calls to Elsevier to bypass exhaustive pybliometrics pagination.
    """
    headers = _get_elsevier_headers()
    params = {
        "query": query,
        "count": count,
        "view": "STANDARD" 
    }
    
    try:
        response = requests.get(
            ELSEVIER_SEARCH_URL,
            headers=headers,
            params=params,
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        
        results = data.get("search-results", {}).get("entry", [])
        if not results:
            return []
            
        papers = []
        for result in results:
            # Depending on view, authors might be formatted differently or missing.
            # In STANDARD view they are often missing, we degrade gracefully
            authors = result.get("dc:creator", "Unknown Author")
            
            # Scopus ID is prefixed, clean it up
            eid = result.get("eid", "")
            scopus_id = eid.replace("2-s2.0-", "") if eid else ""
            doi = result.get("prism:doi", "")
            
            pdf_url = None
            if doi:
                pdf_url = f"https://doi.org/{doi}"
                
            cover_date = result.get("prism:coverDate", "")
            year = int(cover_date[:4]) if cover_date else None
            
            papers.append({
                "source": "scopus",
                "paper_id": scopus_id or eid or doi,
                "title": result.get("dc:title", "Unknown Title"),
                "authors": authors,
                "abstract": result.get("dc:description", ""),
                "year": year,
                "pdf_url": pdf_url,
                "url": f"https://doi.org/{doi}" if doi else None,
                "citation_count": int(result.get("citedby-count", 0)),
                "doi": doi,
            })
            
        return papers
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            raise Exception("Rate limit exceeded")
        logger.error(f"Scopus REST search failed ({e.response.status_code}): {e.response.text}")
        return []
    except Exception as e:
        logger.error(f"Scopus REST search failed: {e}")
        return []


@traceable(run_type="tool", name="Search Scopus", metadata={"tool": "scopus_search"})
def _search_scopus(query: str, count: int) -> List[Dict[str, Any]]:
    """
    Search Scopus and return structured paper list.
    
    Runs in a thread pool to avoid potential event loop conflicts.
    """
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_search_scopus_sync, query, count)
            return future.result(timeout=120)
        
    except concurrent.futures.TimeoutError:
        logger.error("Scopus search timed out")
        return []
    except Exception as e:
        logger.error(f"Scopus search failed: {e}")
        return []


@tool(response_format="content_and_artifact")
def search_scopus(
    query: str,
    count: int = 5,
) -> Tuple[str, Optional[Any]]:
    """Search Elsevier Scopus for highly credible, peer-reviewed journal publications.
    
    Use when verification and credibility scoring are paramount.
    Scopus indexes the largest database of peer-reviewed literature including
    scientific journals, books, and conference proceedings.
    
    Args:
        query: Scopus search query (supports Scopus advanced search syntax,
               e.g. "TITLE-ABS-KEY(transformer attention)" or plain text)
        count: Maximum number of papers to return (default 5)
        
    Returns:
        Formatted string with paper metadata for agent processing.
    """
    if not settings.SCOPUS_API_KEY:
        return ("Scopus API key not configured. Please set SCOPUS_API_KEY in environment.", None)
    
    try:
        logger.info(f"Searching Scopus for: {query} (count={count})")
        papers = _search_scopus(query, count)
        result = format_search_results(papers, "scopus", query)
        return (result, None)
    except Exception as e:
        logger.error(f"Scopus search failed: {e}")
        return (f"Scopus search error: {str(e)}. Try rephrasing your query.", None)


def _get_oa_pdf_url_for_doi(doi: str) -> Optional[str]:
    """Look up an open-access PDF URL for a DOI via Semantic Scholar."""
    try:
        paper = _run_in_thread(_get_paper_sync, f"DOI:{doi}", ["openAccessPdf"])
        if not paper or not paper.openAccessPdf:
            return None
        if isinstance(paper.openAccessPdf, dict):
            return paper.openAccessPdf.get("url")
        return getattr(paper.openAccessPdf, "url", None)
    except Exception:
        return None


def fetch_scopus_content(paper_id: str) -> Tuple[str, Optional[Any]]:
    """
    Fetch content for a Scopus paper.
    
    Attempts DOI-based PDF download. Falls back to metadata-only via Abstract API if PDF unavailable.
    
    Args:
        paper_id: Scopus ID (without the 2-s2.0 prefix)
        
    Returns:
        Tuple of (extracted content string, None)
    """
    try:
        # Use Abstract API for full metadata
        headers = _get_elsevier_headers()
        url = f"{ELSEVIER_ABSTRACT_URL}{paper_id}"
        
        response = requests.get(url, headers=headers, params={"view": "FULL"}, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        ab = data.get("abstracts-retrieval-response", {})
        coredata = ab.get("coredata", {})
        
        title = coredata.get("dc:title", "Unknown Title")
        
        # Parse authors (structure can vary)
        authors_list = []
        authors_data = ab.get("authors", {}).get("author", [])
        if isinstance(authors_data, dict):
            authors_data = [authors_data]
            
        for a in authors_data:
            given = a.get("ce:given-name", "")
            sur = a.get("ce:surname", "")
            if given or sur:
                authors_list.append(f"{given} {sur}".strip())
                
        authors = "; ".join(authors_list[:5]) if authors_list else "Unknown"
        
        cover_date = coredata.get("prism:coverDate", "")
        year = cover_date[:4] if cover_date else "Unknown"
        doi = coredata.get("prism:doi", "")
        
        metadata_prefix = f"## {title}\n\n"
        metadata_prefix += f"**Authors**: {authors}\n"
        metadata_prefix += f"**Year**: {year}\n"
        metadata_prefix += f"**Scopus ID**: {paper_id}\n"
        if doi:
            metadata_prefix += f"**DOI**: {doi}\n"
        metadata_prefix += "\n"
        
        # Try PDF: prefer open-access version from Semantic Scholar, fall back to DOI URL
        full_text = None
        if doi:
            oa_url = _get_oa_pdf_url_for_doi(doi)
            if oa_url:
                full_text = download_and_parse_pdf(oa_url)
            if not full_text:
                full_text = download_and_parse_pdf(f"https://doi.org/{doi}")
        
        if full_text and len(full_text) > 10000:
            return (extract_paper_sections(full_text, metadata_prefix), None)
        elif full_text:
            return (metadata_prefix + full_text, None)
        
        # Fallback to abstract
        abstract = coredata.get("dc:description", "")
        if not abstract and "item" in ab:
            # Complex XML structure fallback
            try:
                abstract = ab["item"]["bibrecord"]["head"]["abstracts"]
            except KeyError:
                pass
                
        if abstract:
            # Ensure it's a string not a complex dict structure from XML
            if isinstance(abstract, dict):
                abstract = "\n".join(str(v) for v in abstract.values())
            
            return (
                metadata_prefix + f"### Abstract\n{abstract}\n\n"
                "*Note: Full PDF not available. Only abstract provided.*",
                None
            )
        
        return (metadata_prefix + "No abstract or content available for this Scopus paper.", None)
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"Scopus fetch REST failed ({e.response.status_code}): {e.response.text}")
        return (f"Failed to fetch Scopus paper {paper_id}: HTTP {e.response.status_code}", None)
    except Exception as e:
        logger.error(f"Scopus fetch failed for {paper_id}: {e}")
        return (f"Failed to fetch Scopus paper {paper_id}: {str(e)}", None)
