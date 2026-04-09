"""
Export API endpoints.

Provides endpoints for exporting research conversations as PDF reports
and BibTeX bibliography files.
"""

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.persistence import get_conversation
from app.models.schemas import Finding
from app.utils.export_pdf import generate_pdf_from_markdown
from app.utils.export_bibtex import generate_bibtex

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/exports", tags=["exports"])


def _extract_report_title(report_content: str) -> str:
    """Extract the first markdown heading from the report, or fall back to a default."""
    for line in report_content.splitlines():
        stripped = line.lstrip("#").strip()
        if line.startswith("#") and stripped:
            return stripped
    return "Research Report"


@router.get("/{user_id}/{conversation_id}/pdf")
async def export_pdf(user_id: str, conversation_id: str):
    """
    Export a conversation's research report as a PDF document.

    Args:
        user_id: The ID of the user.
        conversation_id: The ID of the conversation.

    Returns:
        Response: PDF file as an attachment.

    Raises:
        HTTPException: 404 if conversation not found, 400 if no report content.
    """
    conversation = await get_conversation(user_id, conversation_id)

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    report_content = conversation.get("report_content", "")
    if not report_content or not report_content.strip():
        raise HTTPException(status_code=400, detail="No report content available")

    # Extract scope from research brief if available
    scope = None
    research_brief = conversation.get("research_brief")
    if research_brief and isinstance(research_brief, dict):
        scope = research_brief.get("scope")

    title = _extract_report_title(report_content)

    pdf_bytes = generate_pdf_from_markdown(report_content, title, scope)

    short_id = conversation_id[:8]
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="research_report_{short_id}.pdf"'
        },
    )


@router.get("/{user_id}/{conversation_id}/bibtex")
async def export_bibtex(user_id: str, conversation_id: str):
    """
    Export a conversation's citations as a BibTeX bibliography file.

    Args:
        user_id: The ID of the user.
        conversation_id: The ID of the conversation.

    Returns:
        Response: BibTeX file as an attachment.

    Raises:
        HTTPException: 404 if conversation not found, 400 if no findings.
    """
    conversation = await get_conversation(user_id, conversation_id)

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    raw_findings = conversation.get("findings", [])
    if not raw_findings:
        raise HTTPException(status_code=400, detail="No findings available")

    findings = [Finding(**f) for f in raw_findings]
    bibtex_str = generate_bibtex(findings)

    short_id = conversation_id[:8]
    return Response(
        content=bibtex_str,
        media_type="application/x-bibtex",
        headers={
            "Content-Disposition": f'attachment; filename="references_{short_id}.bib"'
        },
    )
