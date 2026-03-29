import sys
import pytest
from unittest.mock import patch, MagicMock

from app.tools.academic.arxiv import (
    _search_arxiv_internal,
    search_arxiv,
    fetch_arxiv_content
)

class TestArxivSearchInternal:
    def test_search_arxiv_internal_success(self):
        mock_arxiv = MagicMock()
        mock_client = MagicMock()
        mock_arxiv.Client.return_value = mock_client
        mock_arxiv.Search.return_value = "search_obj"
        mock_arxiv.SortCriterion.Relevance = "relevance"
        
        # Mock result
        mock_res1 = MagicMock()
        mock_res1.entry_id = "http://arxiv.org/abs/1234.5678v1"
        mock_res1.title = "Test Paper"
        
        mock_author = MagicMock()
        mock_author.name = "John Doe"
        mock_res1.authors = [mock_author]
        
        mock_res1.summary = "A " * 300 # > 500 chars to trigger truncation
        
        mock_published = MagicMock()
        mock_published.year = 2024
        mock_res1.published = mock_published
        
        mock_res1.pdf_url = "http://pdf"
        
        # Test no published date
        mock_res2 = MagicMock()
        mock_res2.entry_id = "http://arxiv.org/abs/2222.3333"
        mock_res2.title = "Second Paper"
        mock_res2.authors = []
        mock_res2.summary = "Short summary"
        mock_res2.published = None
        mock_res2.pdf_url = None
        
        mock_client.results.return_value = [mock_res1, mock_res2]
        
        with patch.dict('sys.modules', {'arxiv': mock_arxiv}):
            papers = _search_arxiv_internal("query", 2)
        
        assert len(papers) == 2
        assert papers[0]["paper_id"] == "1234.5678v1"
        assert papers[0]["title"] == "Test Paper"
        assert papers[0]["authors"] == "John Doe"
        assert papers[0]["year"] == 2024
        assert len(papers[0]["abstract"]) <= 504 # Truncated
        assert papers[0]["abstract"].endswith("...")
        
        assert papers[1]["paper_id"] == "2222.3333"
        assert papers[1]["year"] is None
        assert papers[1]["abstract"] == "Short summary"

    def test_search_arxiv_internal_rate_limit(self):
        mock_arxiv = MagicMock()
        mock_client = MagicMock()
        mock_arxiv.Client.return_value = mock_client
        
        # Mock rate limit (e.g., 429)
        class RateLimitError(Exception): pass
        mock_client.results.side_effect = RateLimitError("429 Too Many Requests")
        
        with patch.dict('sys.modules', {'arxiv': mock_arxiv}):
            with patch("app.tools.academic.utils.time.sleep"):  # avoid actual sleep
                with pytest.raises(Exception):
                    _search_arxiv_internal("query", 2)
                
    def test_search_arxiv_internal_generic_error(self):
        mock_arxiv = MagicMock()
        mock_client = MagicMock()
        mock_arxiv.Client.return_value = mock_client
        mock_client.results.side_effect = Exception("Generic error")
        
        with patch.dict('sys.modules', {'arxiv': mock_arxiv}):
            papers = _search_arxiv_internal("query", 2)
        assert papers == []

class TestSearchArxivTool:
    @patch("app.tools.academic.arxiv._search_arxiv_internal")
    @patch("app.tools.academic.arxiv.format_search_results")
    def test_search_arxiv_success(self, mock_format, mock_internal):
        mock_internal.return_value = [{"paper_id": "1", "title": "A"}]
        mock_format.return_value = "Formatted Result"
        
        res, art = search_arxiv.func(query="q", count=2)
        assert res == "Formatted Result"
        assert art is None
        
        # Test tool wrapping
        mock_internal.assert_called_with("q", 2)
        mock_format.assert_called_with([{"paper_id": "1", "title": "A"}], "arxiv", "q")

    @patch("app.tools.academic.arxiv._search_arxiv_internal")
    def test_search_arxiv_exception(self, mock_internal):
        mock_internal.side_effect = Exception("Network Error")
        
        res, art = search_arxiv.func(query="q", count=2)
        assert "ArXiv search error: Network Error" in res

class TestFetchArxivContent:
    @patch("app.tools.academic.arxiv.ArxivLoader")
    def test_fetch_arxiv_content_success(self, mock_loader_cls):
        mock_loader = MagicMock()
        mock_loader_cls.return_value = mock_loader
        
        mock_doc = MagicMock()
        mock_doc.page_content = "This is the full text."
        mock_doc.metadata = {"Title": "Paper", "Authors": "Author1", "Published": "2024-01-01"}
        mock_loader.load.return_value = [mock_doc]
        
        # Not greater than 10000, no extraction needed
        content, _ = fetch_arxiv_content("1234.5678")
        
        assert "## Paper" in content
        assert "**Authors**: Author1" in content
        assert "This is the full text." in content

    @patch("app.tools.academic.arxiv.ArxivLoader")
    @patch("app.tools.academic.arxiv.extract_paper_sections")
    def test_fetch_arxiv_content_long_text(self, mock_extract, mock_loader_cls):
        mock_loader = MagicMock()
        mock_loader_cls.return_value = mock_loader
        
        mock_doc = MagicMock()
        mock_doc.page_content = "A" * 10001
        mock_doc.metadata = {}
        mock_loader.load.return_value = [mock_doc]
        
        mock_extract.return_value = "Extracted"
        
        content, _ = fetch_arxiv_content("1234.5678")
        
        assert content == "Extracted"
        mock_extract.assert_called_once()
        args = mock_extract.call_args[0]
        assert len(args[0]) > 10000
        assert "Unknown Title" in args[1]

    @patch("app.tools.academic.arxiv.ArxivLoader")
    def test_fetch_arxiv_content_no_docs(self, mock_loader_cls):
        mock_loader = MagicMock()
        mock_loader_cls.return_value = mock_loader
        mock_loader.load.return_value = []
        
        content, _ = fetch_arxiv_content("1234.5678")
        assert "Could not load" in content

    @patch("app.tools.academic.arxiv.ArxivLoader")
    def test_fetch_arxiv_content_exception(self, mock_loader_cls):
        mock_loader = MagicMock()
        mock_loader_cls.return_value = mock_loader
        mock_loader.load.side_effect = Exception("API down")
        
        content, _ = fetch_arxiv_content("1234.5678")
        assert "Failed to fetch ArXiv paper" in content
        assert "API down" in content
