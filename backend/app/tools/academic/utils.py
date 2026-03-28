"""
Shared utilities for academic tools.

Provides common functions used across all academic database tools:
- Rate limit detection and retry with exponential backoff
- PDF download and parsing
- Paper section extraction (Abstract, Introduction, Conclusion)
- Search result formatting
"""

import logging
import re
import json
import tempfile
import time
import random
import functools
import httpx
from typing import List, Optional, Any, Dict, Callable, TypeVar
from pathlib import Path
from langsmith import traceable

from langchain_community.document_loaders import PyMuPDFLoader

logger = logging.getLogger(__name__)

T = TypeVar('T')


def _is_rate_limit_error(error: Exception) -> bool:
    """
    Check if an exception is a rate limit error.
    
    Uses a multi-layer detection approach:
    1. Check HTTP status code (429) from known exception types
    2. Fall back to string pattern matching for wrapped/unknown errors
    """
    # Layer 1: Check HTTP status codes from known exception types
    status_code = None
    
    # httpx exceptions (used by our PDF downloader)
    if hasattr(error, 'response') and hasattr(error.response, 'status_code'):
        status_code = error.response.status_code
    
    # urllib.error.HTTPError (used by arxiv library)
    elif hasattr(error, 'code'):
        status_code = error.code
    
    # Check for 429 Too Many Requests
    if status_code == 429:
        return True
    
    # Layer 2: String pattern matching for wrapped errors or custom messages
    error_str = str(error).lower()
    rate_limit_patterns = [
        "rate limit",
        "ratelimit", 
        "429",
        "too many requests",
        "quota exceeded",
        "throttl",
        "retry after",
    ]
    return any(pattern in error_str for pattern in rate_limit_patterns)


def retry_on_rate_limit(
    max_retries: int = 3,
    base_delay: float = 3.0,
    backoff_multiplier: float = 1.5,
    jitter_max: float = 1.5
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to retry a function on rate limit errors with exponential backoff.
    
    Includes random jitter to prevent thundering herd when multiple sub-agents
    hit rate limits simultaneously.
    
    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 3.0, matches ArXiv guideline)
        backoff_multiplier: Multiplier for exponential backoff (default: 1.5)
        jitter_max: Maximum random jitter to add to each delay (default: 1.5)
        
    Returns:
        Decorated function that retries on rate limit errors.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error: Optional[Exception] = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if _is_rate_limit_error(e):
                        last_error = e
                        if attempt < max_retries:
                            delay = base_delay * (backoff_multiplier ** attempt)
                            jitter = random.uniform(0, jitter_max)
                            total_delay = delay + jitter
                            logger.warning(
                                f"Rate limit hit in {func.__name__}, "
                                f"retrying in {total_delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                            )
                            time.sleep(total_delay)
                            continue
                    raise

            if last_error:
                raise last_error
        
        return wrapper
    return decorator


def extract_paper_sections(full_text: str, metadata_prefix: str = "") -> str:
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


@traceable(run_type="retriever", name="Download and Parse PDF", metadata={"tool": "pdf_parser"})
def download_and_parse_pdf(pdf_url: str) -> Optional[str]:
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
    
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            with httpx.Client(timeout=30.0, follow_redirects=True, headers=headers) as client:
                response = client.get(pdf_url)
                response.raise_for_status()
                tmp_file.write(response.content)
        
        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()
        
        if docs:
            return "\n\n".join([doc.page_content for doc in docs])
        return None
        
    except Exception as e:
        logger.error(f"PDF download/parse failed: {e}")
        return None
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def format_search_results(papers: List[Dict[str, Any]], source: str, query: str) -> str:
    """Format paper list into readable string for agent."""
    if not papers:
        return f"No papers found from {source} for query: {query}"
    
    output = f"# {source.replace('_', ' ').title()} Results for: {query}\n\n"
    output += f"**Total papers found**: {len(papers)}\n\n"
    
    for i, paper in enumerate(papers, 1):
        output += f"### {i}. {paper['title']}\n"
        output += f"- **Source**: {paper['source']}\n"
        output += f"- **Paper ID**: `{paper['paper_id']}`\n"
        if paper.get('authors'):
            output += f"- **Authors**: {paper['authors']}\n"
        if paper.get('year'):
            output += f"- **Year**: {paper['year']}\n"
        if paper.get('citation_count') is not None:
            output += f"- **Citations**: {paper['citation_count']} {'📈' if paper['citation_count'] > 100 else ''}\n"
        url = paper.get('url') or paper.get('pdf_url')
        if not url and paper.get('doi'):
            url = f"https://doi.org/{paper['doi']}"
        if not url and paper.get('arxiv_id'):
            url = f"https://arxiv.org/abs/{paper['arxiv_id']}"
        if not url and paper.get('paper_id') and paper.get('source') == 'semantic_scholar':
            url = f"https://www.semanticscholar.org/paper/{paper['paper_id']}"
        if url:
            output += f"- **URL**: {url}\n"
        if paper.get('abstract'):
            output += f"- **Abstract**: {paper['abstract']}\n"
        output += "\n"
    
    output += "\n---\n**Use `fetch_paper_content(source, paper_id)` to get full text for deep reading.**"
    return output
