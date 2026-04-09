"""
Unit tests for export API endpoints.

Tests the /exports/ router for PDF and BibTeX export functionality,
following the same patterns as test_conversations.py (TestClient + mocking).
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from app.api.exports import router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)
client = TestClient(app)


# ---------------------------------------------------------------------------
# Sample conversation data used across tests
# ---------------------------------------------------------------------------

SAMPLE_CONVERSATION = {
    "conversation_id": "conv-abc-123",
    "user_query": "What is quantum computing?",
    "report_content": "# Quantum Computing\n\nAn overview of quantum computing.",
    "research_brief": {
        "scope": "Quantum computing fundamentals and applications",
        "sub_topics": ["qubits", "quantum gates"],
    },
    "findings": [
        {
            "claim": "Quantum computers use qubits",
            "citation": {
                "source": "Nature",
                "url": "https://nature.com/quantum",
                "title": "Quantum Computing Fundamentals",
                "authors": ["Smith, John"],
                "year": 2023,
                "credibility_score": 0.95,
                "source_type": "peer_reviewed",
                "doi": "10.1038/quantum",
                "venue": "Nature Physics",
                "is_peer_reviewed": True,
            },
            "topic": "qubits",
            "credibility_score": 0.95,
        }
    ],
    "status": "complete",
    "created_at": "2024-01-15T10:00:00",
}


# ===========================================================================
# TestExportPdf
# ===========================================================================


class TestExportPdf:
    """Tests for GET /{user_id}/{conversation_id}/pdf."""

    @patch("app.api.exports.get_conversation", new_callable=AsyncMock)
    @patch("app.api.exports.generate_pdf_from_markdown")
    def test_export_pdf_success_returns_200_with_pdf_content_type(
        self, mock_gen_pdf, mock_get_conv
    ):
        """Successful PDF export returns 200 with application/pdf content type."""
        mock_get_conv.return_value = SAMPLE_CONVERSATION
        mock_gen_pdf.return_value = b"%PDF-1.4 fake pdf content"

        response = client.get("/exports/user1/conv-abc-123/pdf")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.content == b"%PDF-1.4 fake pdf content"

        # Verify generate_pdf_from_markdown was called with correct args
        mock_gen_pdf.assert_called_once_with(
            SAMPLE_CONVERSATION["report_content"],
            "Quantum Computing",
            "Quantum computing fundamentals and applications",
        )

    @patch("app.api.exports.get_conversation", new_callable=AsyncMock)
    def test_export_pdf_conversation_not_found_returns_404(self, mock_get_conv):
        """Non-existent conversation returns 404."""
        mock_get_conv.return_value = None

        response = client.get("/exports/user1/nonexistent/pdf")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @patch("app.api.exports.get_conversation", new_callable=AsyncMock)
    def test_export_pdf_no_report_content_returns_400(self, mock_get_conv):
        """Conversation without report content returns 400."""
        mock_get_conv.return_value = {
            **SAMPLE_CONVERSATION,
            "report_content": "",
        }

        response = client.get("/exports/user1/conv-abc-123/pdf")

        assert response.status_code == 400
        assert "no report content" in response.json()["detail"].lower()

    @patch("app.api.exports.get_conversation", new_callable=AsyncMock)
    def test_export_pdf_whitespace_report_returns_400(self, mock_get_conv):
        """Conversation with whitespace-only report content returns 400."""
        mock_get_conv.return_value = {
            **SAMPLE_CONVERSATION,
            "report_content": "   \n\n  ",
        }

        response = client.get("/exports/user1/conv-abc-123/pdf")

        assert response.status_code == 400

    @patch("app.api.exports.get_conversation", new_callable=AsyncMock)
    @patch("app.api.exports.generate_pdf_from_markdown")
    def test_export_pdf_includes_correct_content_disposition_header(
        self, mock_gen_pdf, mock_get_conv
    ):
        """Response includes Content-Disposition with correct filename."""
        mock_get_conv.return_value = SAMPLE_CONVERSATION
        mock_gen_pdf.return_value = b"%PDF-1.4 content"

        response = client.get("/exports/user1/conv-abc-123/pdf")

        assert "content-disposition" in response.headers
        disposition = response.headers["content-disposition"]
        assert "attachment" in disposition
        assert "research_report_conv-abc" in disposition
        assert ".pdf" in disposition

    @patch("app.api.exports.get_conversation", new_callable=AsyncMock)
    @patch("app.api.exports.generate_pdf_from_markdown")
    def test_export_pdf_no_research_brief_passes_none_scope(
        self, mock_gen_pdf, mock_get_conv
    ):
        """When research_brief is missing, scope is passed as None."""
        mock_get_conv.return_value = {
            **SAMPLE_CONVERSATION,
            "research_brief": None,
        }
        mock_gen_pdf.return_value = b"%PDF-1.4 content"

        response = client.get("/exports/user1/conv-abc-123/pdf")

        assert response.status_code == 200
        mock_gen_pdf.assert_called_once_with(
            SAMPLE_CONVERSATION["report_content"],
            "Quantum Computing",
            None,
        )


# ===========================================================================
# TestExportBibtex
# ===========================================================================


class TestExportBibtex:
    """Tests for GET /{user_id}/{conversation_id}/bibtex."""

    @patch("app.api.exports.get_conversation", new_callable=AsyncMock)
    @patch("app.api.exports.generate_bibtex")
    def test_export_bibtex_success_returns_200_with_bibtex_content_type(
        self, mock_gen_bib, mock_get_conv
    ):
        """Successful BibTeX export returns 200 with correct content type."""
        mock_get_conv.return_value = SAMPLE_CONVERSATION
        mock_gen_bib.return_value = "% Bibliography\n@article{smith2023,\n  title={Test}\n}"

        response = client.get("/exports/user1/conv-abc-123/bibtex")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/x-bibtex"
        assert "@article{" in response.text

    @patch("app.api.exports.get_conversation", new_callable=AsyncMock)
    def test_export_bibtex_conversation_not_found_returns_404(self, mock_get_conv):
        """Non-existent conversation returns 404."""
        mock_get_conv.return_value = None

        response = client.get("/exports/user1/nonexistent/bibtex")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @patch("app.api.exports.get_conversation", new_callable=AsyncMock)
    def test_export_bibtex_no_findings_returns_400(self, mock_get_conv):
        """Conversation without findings returns 400."""
        mock_get_conv.return_value = {
            **SAMPLE_CONVERSATION,
            "findings": [],
        }

        response = client.get("/exports/user1/conv-abc-123/bibtex")

        assert response.status_code == 400
        assert "no findings" in response.json()["detail"].lower()

    @patch("app.api.exports.get_conversation", new_callable=AsyncMock)
    @patch("app.api.exports.generate_bibtex")
    def test_export_bibtex_includes_correct_content_disposition_header(
        self, mock_gen_bib, mock_get_conv
    ):
        """Response includes Content-Disposition with correct filename."""
        mock_get_conv.return_value = SAMPLE_CONVERSATION
        mock_gen_bib.return_value = "% Bibliography\n"

        response = client.get("/exports/user1/conv-abc-123/bibtex")

        assert "content-disposition" in response.headers
        disposition = response.headers["content-disposition"]
        assert "attachment" in disposition
        assert "references_conv-abc" in disposition
        assert ".bib" in disposition

    @patch("app.api.exports.get_conversation", new_callable=AsyncMock)
    @patch("app.api.exports.generate_bibtex")
    def test_export_bibtex_deserializes_findings_correctly(
        self, mock_gen_bib, mock_get_conv
    ):
        """Raw finding dicts are deserialized into Finding objects before passing."""
        mock_get_conv.return_value = SAMPLE_CONVERSATION
        mock_gen_bib.return_value = "% Bibliography\n"

        response = client.get("/exports/user1/conv-abc-123/bibtex")

        assert response.status_code == 200
        # Verify generate_bibtex was called with Finding objects (not raw dicts)
        call_args = mock_gen_bib.call_args[0][0]
        assert len(call_args) == 1
        from app.models.schemas import Finding
        assert isinstance(call_args[0], Finding)
        assert call_args[0].citation.doi == "10.1038/quantum"
