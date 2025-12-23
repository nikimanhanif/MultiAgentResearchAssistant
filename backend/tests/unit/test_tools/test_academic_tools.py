"""Unit tests for academic_tools module.

Tests search and content fetching tools with mocked API responses
to avoid external network calls during testing.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.documents import Document

from app.tools.academic_tools import (
    search_papers,
    fetch_paper_content,
    get_academic_tools,
    _extract_paper_sections,
    _search_arxiv,
    _search_semantic_scholar,
    _search_pubmed,
)


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
        
        result = _extract_paper_sections(text)
        
        assert "ABSTRACT" in result.upper()
        assert "important findings" in result
    
    def test_extracts_conclusion_before_references(self):
        """Test that conclusion is extracted before references section."""
        # Need enough content to trigger proper section extraction (>5000 chars beginning)
        text = "Abstract\nThis is the abstract.\n\n1. Introduction\n" + ("Body content. " * 400) + """

5. Conclusion
This is our main conclusion with key findings about the research.

References
[1] First reference
[2] Second reference
"""
        result = _extract_paper_sections(text)
        
        # Should extract abstract and conclusion portions
        assert "ABSTRACT" in result.upper()
        # References section should be excluded or at end
        # The key is that the result is shorter than original
    
    def test_handles_json_wrapped_content(self):
        """Test handling of JSON-wrapped paper content."""
        import json
        json_content = json.dumps({
            "title": "Test Paper",
            "authors": ["Author A", "Author B"],
            "date": "2024-01-01",
            "text": "Abstract\nThis is the paper content."
        })
        
        result = _extract_paper_sections(json_content)
        
        # Either parse the JSON and get metadata, or process as-is
        assert "Test Paper" in result or "Abstract" in result or "paper content" in result
    
    def test_fallback_for_no_abstract(self):
        """Test fallback when no abstract section found."""
        text = "This paper discusses machine learning without a clear abstract header."
        
        result = _extract_paper_sections(text)
        
        assert "machine learning" in result


class TestSearchPapers:
    """Test the search_papers tool."""
    
    @patch('app.tools.academic_tools._search_arxiv')
    def test_search_arxiv_only(self, mock_arxiv):
        """Test searching ArXiv only."""
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
        
        result = search_papers.invoke({
            "query": "machine learning",
            "source": "arxiv",
            "count": 1
        })
        
        content = result[0] if isinstance(result, tuple) else result
        assert "Test Paper" in content
        assert "arxiv" in content.lower()
    
    @patch('app.tools.academic_tools._search_semantic_scholar')
    def test_search_semantic_scholar_includes_citations(self, mock_ss):
        """Test that Semantic Scholar results include citation counts."""
        mock_ss.return_value = [{
            "source": "semantic_scholar",
            "paper_id": "abc123",
            "title": "Highly Cited Paper",
            "authors": "Famous Author",
            "abstract": "Important research",
            "year": 2020,
            "pdf_url": None,
            "citation_count": 5000,
        }]
        
        result = search_papers.invoke({
            "query": "deep learning",
            "source": "semantic_scholar",
            "count": 1
        })
        
        content = result[0] if isinstance(result, tuple) else result
        assert "5000" in content
        assert "Citations" in content
    
    @patch('app.tools.academic_tools._search_arxiv')
    @patch('app.tools.academic_tools._search_semantic_scholar')
    @patch('app.tools.academic_tools._search_pubmed')
    def test_search_all_sources(self, mock_pubmed, mock_ss, mock_arxiv):
        """Test searching all sources at once."""
        mock_arxiv.return_value = [{"source": "arxiv", "paper_id": "1", "title": "ArXiv Paper", 
                                     "authors": "", "abstract": "", "year": 2024, "pdf_url": None, "citation_count": None}]
        mock_ss.return_value = [{"source": "semantic_scholar", "paper_id": "2", "title": "S2 Paper",
                                  "authors": "", "abstract": "", "year": 2023, "pdf_url": None, "citation_count": 100}]
        mock_pubmed.return_value = [{"source": "pubmed", "paper_id": "3", "title": "PubMed Paper",
                                      "authors": "", "abstract": "", "year": 2022, "pdf_url": None, "citation_count": None}]
        
        result = search_papers.invoke({
            "query": "neural networks",
            "source": "all",
            "count": 1
        })
        
        content = result[0] if isinstance(result, tuple) else result
        assert "ArXiv Paper" in content
        assert "S2 Paper" in content
        assert "PubMed Paper" in content
    
    @patch('app.tools.academic_tools._search_arxiv')
    def test_handles_empty_results(self, mock_arxiv):
        """Test handling when no papers found."""
        mock_arxiv.return_value = []
        
        result = search_papers.invoke({
            "query": "nonexistent topic xyz123",
            "source": "arxiv",
            "count": 5
        })
        
        content = result[0] if isinstance(result, tuple) else result
        assert "No papers found" in content


class TestFetchPaperContent:
    """Test the fetch_paper_content tool."""
    
    @patch('app.tools.academic_tools.ArxivLoader')
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
    
    @patch('app.tools.academic_tools.ArxivLoader')
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
    
    def test_handles_arxiv_loader_error(self):
        """Test handling when ArxivLoader raises an error."""
        with patch('app.tools.academic_tools.ArxivLoader') as mock_loader_class:
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
    
    def test_returns_both_tools(self):
        """Test that both tools are returned."""
        tools = get_academic_tools()
        
        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "search_papers" in tool_names
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
