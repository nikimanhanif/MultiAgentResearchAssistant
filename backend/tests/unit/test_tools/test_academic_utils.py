import pytest
import json
from unittest.mock import patch, MagicMock
from app.tools.academic.utils import (
    _is_rate_limit_error,
    retry_on_rate_limit,
    extract_paper_sections,
    download_and_parse_pdf,
    format_search_results
)

class DummyHttpException(Exception):
    def __init__(self, status_code):
        self.response = MagicMock()
        self.response.status_code = status_code
        super().__init__(f"HTTP {status_code}")

class DummyUrlLibException(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(f"Error {code}")

class TestIsRateLimitError:
    def test_status_codes(self):
        assert _is_rate_limit_error(DummyHttpException(429)) is True
        assert _is_rate_limit_error(DummyUrlLibException(429)) is True
        assert _is_rate_limit_error(DummyHttpException(404)) is False

    def test_string_matching(self):
        assert _is_rate_limit_error(Exception("quota exceeded")) is True
        assert _is_rate_limit_error(Exception("Retry after 5s")) is True
        assert _is_rate_limit_error(Exception("Network timeout")) is False

class TestRetryOnRateLimit:
    @patch("app.tools.academic.utils.time.sleep")
    def test_retry_success_after_failure(self, mock_sleep):
        attempt = 0
        @retry_on_rate_limit(max_retries=2, base_delay=0)
        def flaky_func():
            nonlocal attempt
            attempt += 1
            if attempt < 2:
                raise Exception("rate limit exceeded")
            return "success"
            
        assert flaky_func() == "success"
        assert attempt == 2
        mock_sleep.assert_called_once()

    @patch("app.tools.academic.utils.time.sleep")
    def test_retry_max_retries_exceeded(self, mock_sleep):
        @retry_on_rate_limit(max_retries=1, base_delay=0)
        def failing_func():
            raise Exception("rate limit")
            
        with pytest.raises(Exception, match="rate limit"):
            failing_func()
        mock_sleep.assert_called_once()

    def test_no_retry_on_other_errors(self):
        @retry_on_rate_limit(max_retries=1, base_delay=0)
        def fail_func():
            raise ValueError("bad input")
            
        with pytest.raises(ValueError, match="bad input"):
            fail_func()

class TestExtractPaperSections:
    def test_extracts_json_wrapped(self):
        js = json.dumps({"text": "Abstract: Content"})
        res = extract_paper_sections(js)
        assert "Content" in res

    def test_extract_invalid_json(self):
        res = extract_paper_sections("{ invalid_json }  Abstract: A")
        assert "A" in res

    def test_extract_bracket_references(self):
        text = "Abstract: start" + (" a " * 4000) + "\n[1] Reference A"
        res = extract_paper_sections(text)
        assert "CONCLUSION" in res
        
    def test_extract_fallback_conclusion(self):
        text = "Abstract: start" + (" a " * 4000) + "\nNo references listed here."
        res = extract_paper_sections(text)
        assert "CONCLUSION" in res

class TestDownloadAndParsePdf:
    @patch("app.tools.academic.utils.httpx.Client")
    @patch("app.tools.academic.utils.PyMuPDFLoader")
    def test_download_success(self, MockLoader, MockClient):
        mock_client_instance = MockClient.return_value.__enter__.return_value
        mock_resp = MagicMock()
        mock_resp.content = b"pdf_data"
        mock_client_instance.get.return_value = mock_resp
        
        mock_loader = MockLoader.return_value
        mock_doc = MagicMock()
        mock_doc.page_content = "Parsed text"
        mock_loader.load.return_value = [mock_doc]
        
        res = download_and_parse_pdf("url")
        assert res == "Parsed text"

    @patch("app.tools.academic.utils.httpx.Client")
    @patch("app.tools.academic.utils.PyMuPDFLoader")
    def test_download_empty_docs(self, MockLoader, MockClient):
        mock_client_instance = MockClient.return_value.__enter__.return_value
        mock_client_instance.get.return_value = MagicMock()
        
        MockLoader.return_value.load.return_value = []
        assert download_and_parse_pdf("url") is None

    @patch("app.tools.academic.utils.httpx.Client")
    def test_download_http_error(self, MockClient):
        mock_client_instance = MockClient.return_value.__enter__.return_value
        mock_client_instance.get.side_effect = Exception("conn failed")
        assert download_and_parse_pdf("url") is None

class TestFormatSearchResults:
    def test_format_empty(self):
        assert "No papers found" in format_search_results([], "arxiv", "test")

    def test_format_populated(self):
        papers = [{
            "title": "T", "source": "arxiv", "paper_id": "1", 
            "authors": "A", "year": 2021, "citation_count": 200, 
            "pdf_url": "url", "abstract": "abs"
        }]
        res = format_search_results(papers, "arxiv", "test")
        assert "T" in res
        assert "arxiv" in res
        assert "Authors" in res
        assert "A" in res
        assert "2021" in res
        assert "📈" in res
        assert "Available" in res
        assert "abs" in res
