import pytest
from unittest.mock import patch, MagicMock

from app.tools.academic.semantic_scholar import (
    _search_semantic_scholar_sync,
    _get_paper_sync,
    _get_citation_graph_sync,
    search_semantic_scholar,
    get_citation_graph,
    fetch_semantic_scholar_content,
    _run_in_thread
)
from app.config import settings

class DummyAuthor:
    def __init__(self, name):
        self.name = name

class DummyPdf:
    def __init__(self, url):
        self.url = url

class DummyExternalIds:
    def __init__(self, arxiv):
        self.ArXiv = arxiv

class DummyPaper:
    def __init__(self, paperId, title, abstract, authors, year, citationCount, openAccessPdf, externalIds, influentialCitationCount=None, citations=None, references=None):
        self.paperId = paperId
        self.title = title
        self.abstract = abstract
        self.authors = authors
        self.year = year
        self.citationCount = citationCount
        self.openAccessPdf = openAccessPdf
        self.externalIds = externalIds
        self.influentialCitationCount = influentialCitationCount
        self.citations = citations
        self.references = references

class DummySearchResults:
    def __init__(self, items):
        self.items = items

class TestSemanticScholarSync:
    @patch("semanticscholar.SemanticScholar")
    def test_search_sync_success(self, MockSS):
        mock_ss = MockSS.return_value
        
        # Test 1: Full object properties
        p1 = DummyPaper(
            "p1", "Test Title", "Test Abstract"*100, [DummyAuthor("Author 1"), {"name": "Author 2"}], 
            2020, 10, DummyPdf("url1"), DummyExternalIds("arxiv1")
        )
        # Test 2: Dict properties
        p2 = DummyPaper(
            "p2", "Title 2", "", None, 2021, 0, {"url": "url2"}, {"ArXiv": "arxiv2"}
        )
        
        mock_ss.search_paper.return_value = DummySearchResults([p1, p2])
        
        res = _search_semantic_scholar_sync("query", 2)
        assert len(res) == 2
        
        assert res[0]["paper_id"] == "p1"
        assert "Author 1, Author 2" in res[0]["authors"]
        assert len(res[0]["abstract"]) <= 503
        assert res[0]["pdf_url"] == "url1"
        assert res[0]["arxiv_id"] == "arxiv1"
        
        assert res[1]["paper_id"] == "p2"
        assert res[1]["authors"] == ""
        assert res[1]["pdf_url"] == "url2"
        assert res[1]["arxiv_id"] == "arxiv2"

    @patch("app.tools.academic.utils.time.sleep")
    @patch("semanticscholar.SemanticScholar")
    def test_search_sync_rate_limit(self, MockSS, mock_sleep):
        mock_ss = MockSS.return_value
        # Mock rate limit error by injecting string that satisfies _is_rate_limit_error
        class RateLimitError(Exception): pass
        mock_ss.search_paper.side_effect = RateLimitError("429 Too Many Requests")
        
        with pytest.raises(Exception):
            _search_semantic_scholar_sync("q", 2)

    @patch("semanticscholar.SemanticScholar")
    def test_search_sync_other_error(self, MockSS):
        mock_ss = MockSS.return_value
        mock_ss.search_paper.side_effect = Exception("failed")
        assert _search_semantic_scholar_sync("q", 2) == []
        
    @patch("semanticscholar.SemanticScholar")
    def test_get_paper_sync(self, MockSS):
        mock_ss = MockSS.return_value
        mock_ss.get_paper.return_value = "paper_obj"
        assert _get_paper_sync("id", []) == "paper_obj"
        
    @patch("semanticscholar.SemanticScholar")
    def test_get_citation_graph_sync_success(self, MockSS):
        mock_ss = MockSS.return_value
        
        c1 = DummyPaper("c1", "Cite1", None, [DummyAuthor("CAuthor")], 2022, 5, None, None, influentialCitationCount=2)
        r1 = DummyPaper("r1", "Ref1", None, [{"name": "RAuthor"}], 2018, 50, None, None, influentialCitationCount=20)
        # Add a missing title paper to trigger continue
        r2 = DummyPaper("r2", None, None, None, None, None, None, None)
        
        p = DummyPaper("id", "Seed", None, None, 2020, 10, None, None, influentialCitationCount=5, citations=[c1], references=[r1, r2])
        mock_ss.get_paper.return_value = p
        
        res = _get_citation_graph_sync("id", 10)
        assert res["seed_paper"]["title"] == "Seed"
        assert len(res["top_citations"]) == 1
        assert res["top_citations"][0]["paper_id"] == "c1"
        assert len(res["top_references"]) == 1
        
    @patch("semanticscholar.SemanticScholar")
    def test_get_citation_graph_sync_not_found(self, MockSS):
        mock_ss = MockSS.return_value
        mock_ss.get_paper.return_value = None
        res = _get_citation_graph_sync("id", 10)
        assert "not found" in res["error"]
        
    @patch("semanticscholar.SemanticScholar")
    def test_get_citation_graph_sync_error(self, MockSS):
        mock_ss = MockSS.return_value
        mock_ss.get_paper.side_effect = Exception("failed")
        res = _get_citation_graph_sync("id", 10)
        assert "failed" in res["error"]
        
    @patch("app.tools.academic.utils.time.sleep")
    @patch("semanticscholar.SemanticScholar")
    def test_citation_graph_sync_rate_limit(self, MockSS, mock_sleep):
        mock_ss = MockSS.return_value
        class RateLimitError(Exception): pass
        mock_ss.get_paper.side_effect = RateLimitError("429 Too Many Requests")
        with pytest.raises(Exception):
            _get_citation_graph_sync("id", 10)
            
    @patch("asyncio.new_event_loop")
    @patch("semanticscholar.SemanticScholar")
    @patch("asyncio.set_event_loop")
    def test_close_event_loop_error(self, mock_set, MockSS, mock_new_loop):
        class DummyLoop:
            def close(self): raise Exception("loop close failed")
        mock_new_loop.return_value = DummyLoop()
        # Call them to trigger the finally block error handling lines 106-107 and 223-224
        _search_semantic_scholar_sync("q", 1)
        
        mock_new_loop.return_value = DummyLoop()
        _get_citation_graph_sync("id", 1)

class TestSemanticScholarTools:
    @patch("app.tools.academic.semantic_scholar._run_in_thread")
    def test_search_semantic_scholar(self, mock_run):
        mock_run.return_value = [{"title": "t1", "authors": "a1", "paper_id": "p1", "year": 2020, "source": "semantic_scholar", "citation_count": 0}]
        res = search_semantic_scholar.invoke({"query": "q", "count": 2})
        content = res[0] if isinstance(res, tuple) else res
        assert "t1" in content
        
    @patch("app.tools.academic.semantic_scholar._run_in_thread")
    def test_search_semantic_scholar_timeout(self, mock_run):
        mock_run.return_value = None
        res = search_semantic_scholar.invoke({"query": "q", "count": 2})
        content = res[0] if isinstance(res, tuple) else res
        assert "timed out" in content
        
    @patch("app.tools.academic.semantic_scholar._run_in_thread")
    def test_search_semantic_scholar_error(self, mock_run):
        mock_run.side_effect = Exception("failed")
        res = search_semantic_scholar.invoke({"query": "q", "count": 2})
        content = res[0] if isinstance(res, tuple) else res
        assert "failed" in content
        
    @patch("app.tools.academic.semantic_scholar._run_in_thread")
    def test_get_citation_graph(self, mock_run):
        mock_run.return_value = {
            "seed_paper": {"title": "S", "year": 2020, "citation_count": 5, "influential_citation_count": 1},
            "top_citations": [{"title": "C", "paper_id": "c1", "authors": "A", "year": 2021, "citation_count": 1, "influential_citation_count": 0}],
            "top_references": [{"title": "R", "paper_id": "r1", "authors": "A", "year": 2019, "citation_count": 10, "influential_citation_count": 5}]
        }
        res = get_citation_graph.invoke({"paper_id": "id"})
        content = res[0] if isinstance(res, tuple) else res
        assert "S" in content
        assert "C" in content
        assert "R" in content
        
        # Test empty cites/refs
        mock_run.return_value["top_citations"] = []
        mock_run.return_value["top_references"] = []
        res = get_citation_graph.invoke({"paper_id": "id"})
        content = res[0] if isinstance(res, tuple) else res
        assert "No citing papers" in content
        
    @patch("app.tools.academic.semantic_scholar._run_in_thread")
    def test_get_citation_graph_timeout_and_error(self, mock_run):
        mock_run.return_value = None
        res = get_citation_graph.invoke({"paper_id": "id"})
        content = res[0] if isinstance(res, tuple) else res
        assert "timed out" in content
        
        mock_run.return_value = {"error": "not found"}
        res = get_citation_graph.invoke({"paper_id": "id"})
        content = res[0] if isinstance(res, tuple) else res
        assert "not found" in content
        
        mock_run.side_effect = Exception("failed")
        res = get_citation_graph.invoke({"paper_id": "id"})
        content = res[0] if isinstance(res, tuple) else res
        assert "failed" in content

    def test_run_in_thread_timeout(self):
        def slow_func():
            return True
        with patch("concurrent.futures.Future.result", side_effect=__import__('concurrent').futures.TimeoutError):
            assert _run_in_thread(slow_func) is None
            
    def test_run_in_thread_error(self):
        def error_func():
            raise Exception("failed")
        assert _run_in_thread(error_func) is None

class TestFetchSemanticScholarContent:
    @patch("app.tools.academic.semantic_scholar._run_in_thread")
    @patch("app.tools.academic.semantic_scholar.download_and_parse_pdf")
    @patch("app.tools.academic.semantic_scholar.extract_paper_sections")
    def test_fetch_content_success(self, mock_ext, mock_dl, mock_run):
        # Dict access and object access handles
        p = DummyPaper("1", "T", "abs", [{"name": "A"}, DummyAuthor("B")], 2020, 5, {"url": "dl"}, None)
        mock_run.return_value = p
        mock_dl.return_value = "F" * 15000
        mock_ext.return_value = "Extracted"
        
        res, _ = fetch_semantic_scholar_content("1")
        assert "Extracted" in res
        
        # Short text
        mock_dl.return_value = "Short"
        res, _ = fetch_semantic_scholar_content("1")
        assert "Short" in res

    @patch("app.tools.academic.semantic_scholar._run_in_thread")
    @patch("app.tools.academic.semantic_scholar.download_and_parse_pdf")
    def test_fetch_content_abstract_only(self, mock_dl, mock_run):
        p = DummyPaper("1", "T", "abs", None, 2020, 5, DummyPdf("url"), None)
        mock_run.return_value = p
        mock_dl.return_value = None
        
        res, _ = fetch_semantic_scholar_content("1")
        assert "abs" in res
        assert "Full PDF not available" in res
        
    @patch("app.tools.academic.semantic_scholar._run_in_thread")
    def test_fetch_content_no_abstract(self, mock_run):
        p = DummyPaper("1", "T", None, None, 2020, 5, None, None)
        mock_run.return_value = p
        
        res, _ = fetch_semantic_scholar_content("1")
        assert "No full-text content available" in res
        
    @patch("app.tools.academic.semantic_scholar._run_in_thread")
    def test_fetch_content_not_found_and_errors(self, mock_run):
        mock_run.return_value = None
        res, _ = fetch_semantic_scholar_content("1")
        assert "Could not find" in res
        
        mock_run.side_effect = Exception("failed")
        res, _ = fetch_semantic_scholar_content("1")
        assert "failed" in res
