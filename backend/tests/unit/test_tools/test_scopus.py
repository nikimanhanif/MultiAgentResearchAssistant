import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from requests.exceptions import HTTPError, RequestException

from app.tools.academic.scopus import (
    _get_elsevier_headers,
    _search_scopus_sync,
    search_scopus,
    fetch_scopus_content,
    _search_scopus
)
from app.config import settings

class TestScopusSearchSync:
    @patch("app.tools.academic.scopus.requests.get")
    def test_search_scopus_sync_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "search-results": {
                "entry": [
                    {
                        "dc:creator": "Author A",
                        "eid": "2-s2.0-123456",
                        "prism:doi": "10.1234/test",
                        "prism:coverDate": "2023-01-01",
                        "dc:title": "Test Title",
                        "dc:description": "Test abstract",
                        "citedby-count": "10"
                    },
                    {} # Test empty entry gracefully degrades
                ]
            }
        }
        mock_get.return_value = mock_resp
        
        # Test rate limit logic directly since retry_on_rate_limit catches RateLimitError 
        # Actually _search_scopus_sync will throw Exception("Rate limit exceeded") when 429
        # In success, it just proceeds
        results = _search_scopus_sync("query", 2)
        assert len(results) == 2
        assert results[0]["paper_id"] == "123456"
        assert results[0]["year"] == 2023
        assert results[0]["doi"] == "10.1234/test"
        
        # Empty gracefully
        assert results[1]["paper_id"] == ""
        assert results[1]["year"] is None
        
    @patch("app.tools.academic.scopus.requests.get")
    def test_search_scopus_sync_empty(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"search-results": {}}
        mock_get.return_value = mock_resp
        assert _search_scopus_sync("query", 2) == []
        
    @patch("app.tools.academic.utils.time.sleep")
    @patch("app.tools.academic.scopus.requests.get")
    def test_search_scopus_sync_429(self, mock_get, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_err = HTTPError(response=mock_resp)
        mock_get.side_effect = mock_err
        # Uses custom retry wrapper so the error will bubble
        with pytest.raises(Exception, match="Rate limit exceeded"):
            _search_scopus_sync("query", 2)
            
    @patch("app.tools.academic.scopus.requests.get")
    def test_search_scopus_sync_other_http_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Server Error"
        mock_err = HTTPError(response=mock_resp)
        mock_get.side_effect = mock_err
        # Returns [] on HTTP error != 429
        assert _search_scopus_sync("query", 2) == []
        
    @patch("app.tools.academic.scopus.requests.get")
    def test_search_scopus_sync_exception(self, mock_get):
        mock_get.side_effect = RequestException("Network error")
        assert _search_scopus_sync("query", 2) == []

class TestScopusSearchTool:
    @patch("app.tools.academic.scopus.settings")
    def test_search_scopus_no_api_key(self, mock_settings):
        # Tools uses the original pydantic object for settings, patching settings directly
        mock_settings.SCOPUS_API_KEY = None
        res = search_scopus.invoke({"query": "q", "count": 2})
        content = res[0] if isinstance(res, tuple) else res
        assert "not configured" in content
        
    @patch("app.tools.academic.scopus.settings")
    @patch("app.tools.academic.scopus._search_scopus")
    def test_search_scopus_error(self, mock_search, mock_settings):
        mock_settings.SCOPUS_API_KEY = "test"
        mock_search.side_effect = Exception("error")
        res = search_scopus.invoke({"query": "q", "count": 2})
        content = res[0] if isinstance(res, tuple) else res
        assert "error" in content.lower()

    @patch("app.tools.academic.scopus.settings")
    @patch("app.tools.academic.scopus._search_scopus_sync")
    def test_search_scopus_success(self, mock_sync, mock_settings):
        # Hits lines 158-159 because _search_scopus_sync is patched, format_search_results is executed
        mock_settings.SCOPUS_API_KEY = "test"
        mock_sync.return_value = [{"title": "T1", "authors": "A1", "year": 2023, "paper_id": "1", "source": "scopus", "citation_count": 5}]
        res = search_scopus.invoke({"query": "q", "count": 2})
        content = res[0] if isinstance(res, tuple) else res
        assert "T1" in content

    @patch("app.tools.academic.scopus._search_scopus_sync")
    def test_search_scopus_timeout(self, mock_sync):
        with patch("concurrent.futures.Future.result", side_effect=__import__('concurrent').futures.TimeoutError):
            assert _search_scopus("q", 2) == []
            
    @patch("app.tools.academic.scopus._search_scopus_sync")
    def test_search_scopus_exception(self, mock_sync):
        mock_sync.side_effect = Exception("failed")
        assert _search_scopus("q", 2) == []

class TestFetchScopusContent:
    @patch("app.tools.academic.scopus.requests.get")
    @patch("app.tools.academic.scopus.download_and_parse_pdf")
    @patch("app.tools.academic.scopus.extract_paper_sections")
    def test_fetch_scopus_content_success_pdf(self, mock_extract, mock_download, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "abstracts-retrieval-response": {
                "coredata": {
                    "dc:title": "Test Title",
                    "prism:coverDate": "2023-01-01",
                    "prism:doi": "10.1234/test"
                },
                "authors": {
                    "author": [
                        {"ce:given-name": "John", "ce:surname": "Doe"},
                        {"ce:given-name": "Jane", "ce:surname": "Smith"}
                    ]
                }
            }
        }
        mock_get.return_value = mock_resp
        mock_download.return_value = "A" * 15000  # length > 10000 to trigger extract
        mock_extract.return_value = "Extracted sections"
        
        res, _ = fetch_scopus_content("123456")
        assert "Extracted sections" in res
        mock_extract.assert_called_once()
        
    @patch("app.tools.academic.scopus.requests.get")
    @patch("app.tools.academic.scopus.download_and_parse_pdf")
    def test_fetch_scopus_content_success_pdf_short(self, mock_download, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "abstracts-retrieval-response": {
                "coredata": {
                    "dc:title": "Test Title",
                    "prism:doi": "10.1234/test"
                },
                "authors": {
                    "author": {"ce:given-name": "Single", "ce:surname": "Author"} # dict author to hit line 195
                }
            }
        }
        mock_get.return_value = mock_resp
        mock_download.return_value = "Short pdf content"
        
        res, _ = fetch_scopus_content("123456")
        assert "Short pdf content" in res
        assert "Single Author" in res
        
    @patch("app.tools.academic.scopus.requests.get")
    @patch("app.tools.academic.scopus.download_and_parse_pdf")
    def test_fetch_scopus_content_abstract_fallback(self, mock_download, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "abstracts-retrieval-response": {
                "coredata": {
                    "dc:description": "This is an abstract."
                }
            }
        }
        mock_get.return_value = mock_resp
        mock_download.return_value = None  # PDF failed
        
        res, _ = fetch_scopus_content("123456")
        assert "This is an abstract." in res
        assert "Full PDF not available" in res

    @patch("app.tools.academic.scopus.requests.get")
    @patch("app.tools.academic.scopus.download_and_parse_pdf")
    def test_fetch_scopus_content_abstract_xml_fallback(self, mock_download, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "abstracts-retrieval-response": {
                "coredata": {}, # no dc:description
                "item": {
                    "bibrecord": {
                        "head": {
                            "abstracts": "XML abstract fallback"
                        }
                    }
                }
            }
        }
        mock_get.return_value = mock_resp
        mock_download.return_value = None
        
        res, _ = fetch_scopus_content("123456")
        assert "XML abstract fallback" in res

    @patch("app.tools.academic.scopus.requests.get")
    @patch("app.tools.academic.scopus.download_and_parse_pdf")
    def test_fetch_scopus_content_abstract_xml_fallback_keyerror(self, mock_download, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "abstracts-retrieval-response": {
                "coredata": {},
                "item": {
                    "bibrecord": {
                        "head": {} # Missing 'abstracts' key to hit 234-235
                    }
                }
            }
        }
        mock_get.return_value = mock_resp
        mock_download.return_value = None
        
        res, _ = fetch_scopus_content("123456")
        assert "No abstract or content available" in res

    @patch("app.tools.academic.scopus.requests.get")
    @patch("app.tools.academic.scopus.download_and_parse_pdf")
    def test_fetch_scopus_content_abstract_dict_fallback(self, mock_download, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "abstracts-retrieval-response": {
                "coredata": {},
                "item": {
                    "bibrecord": {
                        "head": {
                            "abstracts": {"p": "XML p dict abstract fallback"}
                        }
                    }
                }
            }
        }
        mock_get.return_value = mock_resp
        mock_download.return_value = None
        
        res, _ = fetch_scopus_content("123456")
        assert "XML p dict abstract fallback" in res
        
    @patch("app.tools.academic.scopus.requests.get")
    @patch("app.tools.academic.scopus.download_and_parse_pdf")
    def test_fetch_scopus_content_no_abstract(self, mock_download, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "abstracts-retrieval-response": {
                "coredata": {}
            }
        }
        mock_get.return_value = mock_resp
        mock_download.return_value = None
        
        res, _ = fetch_scopus_content("123456")
        assert "No abstract or content available" in res
        
    @patch("app.tools.academic.scopus.requests.get")
    def test_fetch_scopus_content_http_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Not found"
        mock_err = HTTPError(response=mock_resp)
        mock_get.side_effect = mock_err
        
        res, _ = fetch_scopus_content("123456")
        assert "HTTP 404" in res
        
    @patch("app.tools.academic.scopus.requests.get")
    def test_fetch_scopus_content_exception(self, mock_get):
        mock_get.side_effect = Exception("failed")
        res, _ = fetch_scopus_content("123456")
        assert "Failed to fetch Scopus paper" in res

def test_get_elsevier_headers():
    with patch("app.tools.academic.scopus.settings") as mock_settings:
        mock_settings.SCOPUS_API_KEY = "key"
        mock_settings.SCOPUS_INST_TOKEN = "token"
        headers = _get_elsevier_headers()
        assert headers["X-ELS-APIKey"] == "key"
        assert headers["X-ELS-Insttoken"] == "token"
