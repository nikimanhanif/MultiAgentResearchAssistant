"""
Unit tests for PDF export utility.

Tests HTML document construction and PDF generation from markdown content.
"""

import pytest
from app.utils.export_pdf import build_html_document, generate_pdf_from_markdown


# ===========================================================================
# TestBuildHtmlDocument
# ===========================================================================


class TestBuildHtmlDocument:
    """Tests for build_html_document()."""

    def test_build_html_document_contains_title(self):
        """Title appears in the HTML output."""
        html = build_html_document(
            "# Report",
            title="Quantum Computing Survey",
            exported_at="2024-01-15 10:00:00",
        )
        assert "Quantum Computing Survey" in html

    def test_build_html_document_contains_scope_when_provided(self):
        """Scope line is included when a scope value is given."""
        html = build_html_document(
            "Content here",
            title="Title",
            scope="Effects of AI on healthcare",
            exported_at="2024-01-15 10:00:00",
        )
        assert "Effects of AI on healthcare" in html
        assert "Scope:" in html

    def test_build_html_document_omits_scope_when_none(self):
        """No scope div is rendered when scope is None."""
        html = build_html_document(
            "Content here",
            title="Title",
            scope=None,
            exported_at="2024-01-15 10:00:00",
        )
        assert "Scope:" not in html

    def test_build_html_document_contains_report_content_as_html(self):
        """Markdown content is converted to HTML tags."""
        html = build_html_document(
            "## Section Title\n\nA paragraph of text.",
            title="Title",
            exported_at="2024-01-15 10:00:00",
        )
        assert "<h2" in html
        assert "Section Title" in html
        assert "<p>" in html
        assert "A paragraph of text." in html

    def test_build_html_document_contains_css_styles(self):
        """The output includes embedded CSS styling."""
        html = build_html_document(
            "Content",
            title="Title",
            exported_at="2024-01-15 10:00:00",
        )
        assert "<style>" in html
        assert "@page" in html
        assert "font-family" in html

    def test_build_html_document_empty_content_shows_placeholder(self):
        """Empty report content shows a placeholder message."""
        html = build_html_document(
            "",
            title="Title",
            exported_at="2024-01-15 10:00:00",
        )
        assert "No report content available" in html

    def test_build_html_document_whitespace_only_content_shows_placeholder(self):
        """Whitespace-only content is treated as empty."""
        html = build_html_document(
            "   \n\n  ",
            title="Title",
            exported_at="2024-01-15 10:00:00",
        )
        assert "No report content available" in html

    def test_build_html_document_markdown_tables_rendered(self):
        """Markdown tables are converted to HTML table elements."""
        md = "| Col A | Col B |\n|-------|-------|\n| 1     | 2     |"
        html = build_html_document(
            md,
            title="Title",
            exported_at="2024-01-15 10:00:00",
        )
        assert "<table>" in html or "<table" in html
        assert "<th>" in html or "<th" in html
        assert "<td>" in html or "<td" in html

    def test_build_html_document_markdown_code_blocks_rendered(self):
        """Fenced code blocks are rendered as <pre><code> elements."""
        md = "```python\nprint('hello')\n```"
        html = build_html_document(
            md,
            title="Title",
            exported_at="2024-01-15 10:00:00",
        )
        assert "<code" in html
        assert "print" in html

    def test_build_html_document_contains_exported_at_timestamp(self):
        """The exported_at timestamp appears in the output."""
        html = build_html_document(
            "Content",
            title="Title",
            exported_at="2024-03-15 14:30:00",
        )
        assert "2024-03-15 14:30:00" in html

    def test_build_html_document_is_valid_html_structure(self):
        """Output is a well-formed HTML document."""
        html = build_html_document(
            "Content",
            title="Title",
            exported_at="2024-01-15 10:00:00",
        )
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "<head>" in html
        assert "<body>" in html
        assert "</html>" in html


# ===========================================================================
# TestGeneratePdfFromMarkdown
# ===========================================================================


class TestGeneratePdfFromMarkdown:
    """Tests for generate_pdf_from_markdown().

    These tests invoke WeasyPrint and produce real PDF output.
    They are slightly slower (~1-2s) but verify the full pipeline.
    """

    def test_generate_pdf_returns_bytes(self):
        """Output is a bytes object."""
        result = generate_pdf_from_markdown("# Hello World")
        assert isinstance(result, bytes)

    def test_generate_pdf_output_starts_with_pdf_magic_bytes(self):
        """PDF output starts with the %PDF- magic header."""
        result = generate_pdf_from_markdown("# Hello World")
        assert result[:5] == b"%PDF-"

    def test_generate_pdf_empty_content_still_produces_valid_pdf(self):
        """Empty content generates a valid PDF (with placeholder text)."""
        result = generate_pdf_from_markdown("")
        assert result[:5] == b"%PDF-"
        assert len(result) > 100  # Not trivially empty

    def test_generate_pdf_with_scope_included(self):
        """Passing a scope value does not break PDF generation."""
        result = generate_pdf_from_markdown(
            "## Analysis\n\nSome findings here.",
            title="AI Research",
            scope="Impact of LLMs on software engineering",
        )
        assert result[:5] == b"%PDF-"
        assert len(result) > 100

    def test_generate_pdf_with_tables(self):
        """Markdown with tables produces a valid PDF."""
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        result = generate_pdf_from_markdown(md, title="Table Test")
        assert result[:5] == b"%PDF-"
