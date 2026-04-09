"""Utility functions for research pipeline.

This module provides utility functions for various research pipeline operations:
- Credibility scoring for sources
- Research gap detection and analysis
- Research strategy selection
- Markdown formatting for reports
- Paper extraction and compression
- BibTeX and PDF export

"""

from app.utils.export_bibtex import (
    escape_bibtex_chars,
    generate_bibtex,
    generate_citation_key,
    source_type_to_bibtex_entry,
    format_bibtex_entry,
)
from app.utils.export_pdf import (
    build_html_document,
    generate_pdf_from_markdown,
)

