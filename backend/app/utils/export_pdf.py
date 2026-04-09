"""
PDF export utility.

Converts markdown report content into a styled PDF document using
the ``markdown`` library (MD → HTML) and ``weasyprint`` (HTML → PDF).
No FastAPI or persistence dependencies — fully testable in isolation.
"""

from datetime import datetime
from typing import Optional

import markdown
import weasyprint


# Embedded CSS for the PDF document
_PDF_CSS = """\
@page {
    size: A4;
    margin: 3cm 2.5cm 3cm 2.5cm;
    @bottom-center {
        content: "Page " counter(page);
        font-size: 9pt;
        color: #888;
        font-family: "Palatino Linotype", "Palatino", "Book Antiqua", Georgia, serif;
    }
}

/* Section counter reset */
body {
    counter-reset: h2counter;
}

body {
    font-family: "Palatino Linotype", "Palatino", "Book Antiqua", Georgia, serif;
    font-size: 10.5pt;
    line-height: 1.5;
    color: #000;
    text-align: justify;
}

/* Title block */
.title-block {
    text-align: center;
    margin-bottom: 2.5em;
    padding-bottom: 1.2em;
    border-bottom: 1px solid #000;
}

.title-block h1 {
    font-size: 16pt;
    font-weight: bold;
    margin-bottom: 0.5em;
    color: #000;
    text-align: center;
}

.title-block .scope-line {
    font-size: 10pt;
    color: #000;
    margin-bottom: 0.25em;
}

.title-block .date-line {
    font-size: 9pt;
    color: #555;
}

/* Headings — no colors, weight/size only */
h1 {
    font-size: 14pt;
    font-weight: bold;
    margin-top: 1.8em;
    margin-bottom: 0.5em;
    color: #000;
}

/* Auto-number h2 as "1.", "2.", etc. */
h2 {
    font-size: 13pt;
    font-weight: bold;
    margin-top: 1.5em;
    margin-bottom: 0.4em;
    color: #000;
    counter-reset: h3counter;
    counter-increment: h2counter;
}

h2::before {
    content: counter(h2counter) ". ";
}

/* Auto-number h3 as "1.1.", "1.2.", etc. */
h3 {
    font-size: 11pt;
    font-weight: bold;
    margin-top: 1.2em;
    margin-bottom: 0.3em;
    color: #000;
    counter-increment: h3counter;
}

h3::before {
    content: counter(h2counter) "." counter(h3counter) " ";
}

h4 {
    font-size: 10.5pt;
    font-weight: normal;
    font-style: italic;
    margin-top: 1em;
    margin-bottom: 0.3em;
    color: #000;
}

/* Tables — booktabs style (horizontal rules only) */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 1.2em 0;
    font-size: 10pt;
}

th, td {
    border: none;
    padding: 0.4em 0.6em;
    text-align: left;
}

/* Top and bottom rules for the whole table */
thead tr:first-child th {
    border-top: 1.5px solid #000;
    border-bottom: 1px solid #000;
    font-weight: bold;
}

tbody tr:last-child td {
    border-bottom: 1.5px solid #000;
}

/* Code blocks — flat, academic */
pre {
    background-color: #f8f8f8;
    border: none;
    border-left: 2px solid #bbb;
    padding: 0.7em 1em;
    font-family: "Courier New", Courier, monospace;
    font-size: 9pt;
    white-space: pre-wrap;
    word-wrap: break-word;
    margin: 1em 0;
}

code {
    font-family: "Courier New", Courier, monospace;
    font-size: 9pt;
    background-color: #f4f4f4;
    padding: 0.1em 0.25em;
}

pre code {
    background-color: transparent;
    padding: 0;
}

/* Blockquotes */
blockquote {
    border-left: 2px solid #555;
    margin: 1em 2em;
    padding: 0.3em 0.8em;
    color: #333;
    font-style: italic;
}

/* Lists */
ul, ol {
    margin: 0.5em 0;
    padding-left: 2em;
}

li {
    margin-bottom: 0.25em;
}

/* References / bibliography — hanging indent, smaller font */
.references ol,
.references ul {
    font-size: 9.5pt;
    padding-left: 1.5em;
    text-indent: -1.5em;
    margin-left: 1.5em;
}

.references li {
    margin-bottom: 0.5em;
    text-align: left;
}

/* Links — black for print */
a {
    color: #000;
    text-decoration: underline;
}

/* Horizontal rules */
hr {
    border: none;
    border-top: 1px solid #aaa;
    margin: 1.5em 0;
}
"""

# Markdown extensions to enable
_MD_EXTENSIONS = ["extra", "toc", "tables", "fenced_code", "smarty"]


def build_html_document(
    report_content: str,
    title: str,
    scope: Optional[str] = None,
    exported_at: Optional[str] = None,
) -> str:
    """
    Convert markdown report content into a full HTML document with embedded styles.

    Args:
        report_content: Raw markdown string of the report.
        title: Document title (typically the user's research query).
        scope: Optional research scope description.
        exported_at: Optional timestamp string for the export date.

    Returns:
        Complete HTML document string ready for PDF rendering.
    """
    if exported_at is None:
        exported_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert markdown to HTML body
    if report_content and report_content.strip():
        body_html = markdown.markdown(
            report_content, extensions=_MD_EXTENSIONS
        )
    else:
        body_html = "<p><em>No report content available.</em></p>"

    # Build scope line
    scope_html = ""
    if scope:
        scope_html = f'<div class="scope-line"><strong>Scope:</strong> {scope}</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>{_PDF_CSS}</style>
</head>
<body>
    <div class="title-block">
        <h1>{title}</h1>
        {scope_html}
        <div class="date-line">Exported: {exported_at}</div>
    </div>
    {body_html}
</body>
</html>"""


def generate_pdf_from_markdown(
    report_content: str,
    title: str = "Research Report",
    scope: Optional[str] = None,
) -> bytes:
    """
    Generate a PDF document from markdown report content.

    Pipeline: Markdown → HTML (with styles) → PDF (via WeasyPrint).

    Args:
        report_content: Raw markdown string of the report.
        title: Document title.
        scope: Optional research scope description.

    Returns:
        PDF file contents as bytes.
    """
    html_str = build_html_document(report_content, title, scope)
    return weasyprint.HTML(string=html_str).write_pdf()
