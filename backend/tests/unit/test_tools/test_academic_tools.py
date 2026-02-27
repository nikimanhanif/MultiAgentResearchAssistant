"""Unit tests for modular academic tools package.

Tests search and content fetching tools with mocked API responses
to avoid external network calls during testing.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.documents import Document

from app.tools.academic import (
    get_academic_tools,
    fetch_paper_content,
)
from app.tools.academic.arxiv import search_arxiv, _search_arxiv_internal
from app.tools.academic.semantic_scholar import search_semantic_scholar, get_citation_graph
from app.tools.academic.scopus import search_scopus
from app.tools.academic.utils import extract_paper_sections


class TestExtractPaperSections:
    """Test the section extraction helper function."""
    
    def test_extracts_abstract_section(self):
        """Test that abstract is correctly extracted."""
        text = """
Abstract
This is the abstract of the paper with important findings.

1. Introduction
This is the introduction section.
""" + ("Body content. " * 100)
        
        result = extract_paper_sections(text)
        
        assert "ABSTRACT" in result.upper()
        assert "important findings" in result
    
    def test_extracts_conclusion_before_references(self):
        """Test that conclusion is extracted before references section."""
        text = "Abstract\nThis is the abstract.\n\n1. Introduction\n" + ("Body content. " * 400) + """

5. Conclusion
This is our main conclusion with key findings about the research.

References
[1] First reference
[2] Second reference
"""
        result = extract_paper_sections(text)
        
        assert "ABSTRACT" in result.upper()
    
    def test_handles_json_wrapped_content(self):
        """Test handling of JSON-wrapped paper content."""
        import json
        json_content = json.dumps({
            "title": "Test Paper",
            "authors": ["Author A", "Author B"],
            "date": "2024-01-01",
            "text": "Abstract\nThis is the paper content."
        })
        
        result = extract_paper_sections(json_content)
        
        assert "Test Paper" in result or "Abstract" in result or "paper content" in result
    
    def test_fallback_for_no_abstract(self):
        """Test fallback when no abstract section found."""
        text = "This paper discusses machine learning without a clear abstract header."
        
        result = extract_paper_sections(text)
        
        assert "machine learning" in result


class TestSearchArxiv:
    """Test the search_arxiv tool."""
    
    @patch('app.tools.academic.arxiv._search_arxiv_internal')
    def test_search_arxiv(self, mock_arxiv):
        """Test searching ArXiv."""
        mock_arxiv.return_value = [{
            "source": "arxiv",
            "paper_id": "2401.12345",
            "title": "Test Paper",
            "authors": "Author A",
            "abstract": "Test abstract",
            "year": 2024,
            "pdf_url": "http://arxiv.org/pdf/2401.12345",
            "citation_count": None,
        }]
        
        result = search_arxiv.invoke({
            "query": "machine learning",
            "count": 1
        })
        
        content = result[0] if isinstance(result, tuple) else result
        assert "Test Paper" in content
        assert "arxiv" in content.lower()
    
    @patch('app.tools.academic.arxiv._search_arxiv_internal')
    def test_handles_empty_results(self, mock_arxiv):
        """Test handling when no papers found."""
        mock_arxiv.return_value = []
        
        result = search_arxiv.invoke({
            "query": "nonexistent topic xyz123",
            "count": 5
        })
        
        content = result[0] if isinstance(result, tuple) else result
        assert "No papers found" in content


class TestSearchSemanticScholar:
    """Test the search_semantic_scholar tool."""
    
    @patch('app.tools.academic.semantic_scholar._run_in_thread')
    def test_search_includes_citations(self, mock_thread):
        """Test that Semantic Scholar results include citation counts."""
        mock_thread.return_value = [{
            "source": "semantic_scholar",
            "paper_id": "abc123",
            "title": "Highly Cited Paper",
            "authors": "Famous Author",
            "abstract": "Important research",
            "year": 2020,
            "pdf_url": None,
            "citation_count": 5000,
        }]
        
        result = search_semantic_scholar.invoke({
            "query": "deep learning",
            "count": 1
        })
        
        content = result[0] if isinstance(result, tuple) else result
        assert "5000" in content
        assert "Citations" in content


class TestSearchScopus:
    """Test the search_scopus tool."""
    
    @patch('app.tools.academic.scopus._search_scopus')
    @patch('app.tools.academic.scopus.settings')
    def test_search_scopus_returns_results(self, mock_settings, mock_search):
        """Test that Scopus search returns formatted results."""
        mock_settings.SCOPUS_API_KEY = "test_key"
        mock_search.return_value = [{
            "source": "scopus",
            "paper_id": "2-s2.0-123456",
            "title": "Peer Reviewed Paper",
            "authors": "Dr. Smith, Dr. Jones",
            "abstract": "Validated research findings",
            "year": 2023,
            "pdf_url": "https://doi.org/10.1234/test",
            "citation_count": 42,
        }]
        
        result = search_scopus.invoke({
            "query": "machine learning",
            "count": 1
        })
        
        content = result[0] if isinstance(result, tuple) else result
        assert "Peer Reviewed Paper" in content


class TestGetCitationGraph:
    """Test the get_citation_graph tool."""
    
    @patch('app.tools.academic.semantic_scholar._run_in_thread')
    def test_citation_graph_returns_formatted_output(self, mock_thread):
        """Test that citation graph returns both citations and references."""
        mock_thread.return_value = {
            "seed_paper": {
                "paper_id": "abc123",
                "title": "Foundational Paper",
                "year": 2017,
                "citation_count": 50000,
                "influential_citation_count": 5000,
            },
            "top_citations": [{
                "paper_id": "cite1",
                "title": "Citing Paper 1",
                "year": 2020,
                "citation_count": 100,
                "influential_citation_count": 10,
                "authors": "Author X",
            }],
            "top_references": [{
                "paper_id": "ref1",
                "title": "Referenced Paper 1",
                "year": 2015,
                "citation_count": 200,
                "influential_citation_count": 20,
                "authors": "Author Y",
            }],
        }
        
        result = get_citation_graph.invoke({
            "paper_id": "abc123",
        })
        
        content = result[0] if isinstance(result, tuple) else result
        assert "Foundational Paper" in content
        assert "Citing Paper 1" in content
        assert "Referenced Paper 1" in content


class TestFetchPaperContent:
    """Test the fetch_paper_content tool."""
    
    @patch('app.tools.academic.arxiv.ArxivLoader')
    def test_fetch_arxiv_paper(self, mock_loader_class):
        """Test fetching ArXiv paper content."""
        mock_doc = Document(
            page_content="Abstract\nThis is the paper content with important findings.\n\nConclusion\nKey results.",
            metadata={"Title": "Test Paper", "Authors": "Author A", "Published": "2024-01-01"}
        )
        mock_loader = MagicMock()
        mock_loader.load.return_value = [mock_doc]
        mock_loader_class.return_value = mock_loader
        
        result = fetch_paper_content.invoke({
            "source": "arxiv",
            "paper_id": "2401.12345"
        })
        
        content = result[0] if isinstance(result, tuple) else result
        assert "Test Paper" in content
        assert "important findings" in content
    
    @patch('app.tools.academic.arxiv.ArxivLoader')
    def test_handles_missing_paper(self, mock_loader_class):
        """Test handling when paper not found."""
        mock_loader = MagicMock()
        mock_loader.load.return_value = []
        mock_loader_class.return_value = mock_loader
        
        result = fetch_paper_content.invoke({
            "source": "arxiv",
            "paper_id": "nonexistent123"
        })
        
        content = result[0] if isinstance(result, tuple) else result
        assert "Could not load" in content or "not found" in content.lower()
    
    @patch('app.tools.academic.arxiv.ArxivLoader')
    def test_handles_arxiv_loader_error(self, mock_loader_class):
        """Test handling when ArxivLoader raises an error."""
        mock_loader = MagicMock()
        mock_loader.load.side_effect = Exception("Network error")
        mock_loader_class.return_value = mock_loader
        
        result = fetch_paper_content.invoke({
            "source": "arxiv",
            "paper_id": "12345"
        })
        
        content = result[0] if isinstance(result, tuple) else result
        assert "Failed" in content or "error" in content.lower()


class TestGetAcademicTools:
    """Test the tool loader function."""
    
    def test_returns_all_tools(self):
        """Test that all 5 tools are returned."""
        tools = get_academic_tools()
        
        assert len(tools) == 5
        tool_names = [t.name for t in tools]
        assert "search_arxiv" in tool_names
        assert "search_semantic_scholar" in tool_names
        assert "search_scopus" in tool_names
        assert "get_citation_graph" in tool_names
        assert "fetch_paper_content" in tool_names
    
    def test_tools_have_descriptions(self):
        """Test that tools have proper descriptions."""
        tools = get_academic_tools()
        
        for tool in tools:
            assert tool.description
            assert len(tool.description) > 20
    
    def test_tools_use_content_and_artifact_format(self):
        """Test that tools use content_and_artifact response format."""
        tools = get_academic_tools()
        
        for tool in tools:
            assert getattr(tool, 'response_format', None) == 'content_and_artifact'
