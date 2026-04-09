'use client'

import { useState } from 'react'
import { useChatContext } from '@/context/chat-context'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
} from '@/components/ui/dropdown-menu'
import { Download, FileText, FileDown, BookOpen, Loader2 } from 'lucide-react'
import { downloadBlob } from '@/lib/export-utils'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'

interface ExportMenuProps {
  scope: 'session' | 'report'
}

export function ExportMenu({ scope }: ExportMenuProps) {
  const {
    messages,
    researchBrief,
    reviewRequest,
    activeReportContent,
    researchProgress,
    threadId,
    userId,
  } = useChatContext()

  const [isExportingPdf, setIsExportingPdf] = useState(false)
  const [isExportingBibtex, setIsExportingBibtex] = useState(false)

  const reportContent = scope === 'report' ? activeReportContent : reviewRequest?.report
  const hasContent = messages.length > 0 || researchBrief || reportContent

  // ── Markdown export (client-side) ──

  const handleExportMarkdown = () => {
    const lines: string[] = []
    const timestamp = new Date().toISOString().slice(0, 19).replace('T', ' ')

    if (scope === 'session') {
      // Full session: brief + conversation + report
      lines.push('# Research Assistant Session')
      lines.push(`\n**Exported:** ${timestamp}\n`)
      lines.push('---\n')

      if (researchBrief) {
        lines.push('## Research Brief\n')
        lines.push(`**Scope:** ${researchBrief.scope}\n`)
        if (researchBrief.subTopics.length > 0) {
          lines.push('\n**Sub-topics:**')
          researchBrief.subTopics.forEach(topic => {
            lines.push(`- ${topic}`)
          })
        }
        lines.push('\n---\n')
      }

      if (messages.length > 0) {
        lines.push('## Conversation\n')
        messages.forEach(msg => {
          const roleLabel = msg.role === 'user' ? 'You' : 'Assistant'
          lines.push(`### ${roleLabel}\n`)
          lines.push(msg.content)
          lines.push('\n')
        })
        lines.push('---\n')
      }

      if (reviewRequest?.report) {
        lines.push('## Generated Report\n')
        lines.push(reviewRequest.report)
        lines.push('\n')
      }
    } else {
      // Report only
      lines.push('# Research Report')
      lines.push(`\n**Exported:** ${timestamp}\n`)

      if (researchBrief) {
        lines.push(`**Research Scope:** ${researchBrief.scope}\n`)
      }

      lines.push('---\n')
      if (activeReportContent) {
        lines.push(activeReportContent)
      }
    }

    const content = lines.join('\n')
    const blob = new Blob([content], { type: 'text/markdown' })
    const prefix = scope === 'session' ? 'research_session' : 'research_report'
    downloadBlob(blob, `${prefix}_${Date.now()}.md`)
  }

  // ── PDF export (backend) ──

  const handleExportPdf = async () => {
    if (!threadId || !userId) return
    setIsExportingPdf(true)
    try {
      const response = await fetch(
        `${API_URL}/exports/${userId}/${threadId}/pdf`
      )
      if (!response.ok) {
        console.error('PDF export failed:', response.status)
        return
      }
      const blob = await response.blob()
      downloadBlob(blob, `research_report_${Date.now()}.pdf`)
    } catch (err) {
      console.error('PDF export error:', err)
    } finally {
      setIsExportingPdf(false)
    }
  }

  // ── BibTeX export (backend) ──

  const handleExportBibtex = async () => {
    if (!threadId || !userId) return
    setIsExportingBibtex(true)
    try {
      const response = await fetch(
        `${API_URL}/exports/${userId}/${threadId}/bibtex`
      )
      if (!response.ok) {
        console.error('BibTeX export failed:', response.status)
        return
      }
      const blob = await response.blob()
      downloadBlob(blob, `references_${Date.now()}.bib`)
    } catch (err) {
      console.error('BibTeX export error:', err)
    } finally {
      setIsExportingBibtex(false)
    }
  }

  if (!hasContent) {
    return null
  }

  const canExportPdf = !!threadId && !!userId && (!!reportContent || researchProgress.phase === 'complete')
  const canExportBibtex = !!threadId && !!userId

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          title="Export options"
        >
          <Download className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-48">
        <DropdownMenuItem onClick={handleExportMarkdown}>
          <FileText className="mr-2 h-4 w-4" />
          Export as Markdown
        </DropdownMenuItem>

        <DropdownMenuSeparator />

        <DropdownMenuItem
          onClick={handleExportPdf}
          disabled={!canExportPdf || isExportingPdf}
        >
          {isExportingPdf ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : (
            <FileDown className="mr-2 h-4 w-4" />
          )}
          Export as PDF
        </DropdownMenuItem>

        <DropdownMenuItem
          onClick={handleExportBibtex}
          disabled={!canExportBibtex || isExportingBibtex}
        >
          {isExportingBibtex ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : (
            <BookOpen className="mr-2 h-4 w-4" />
          )}
          Export as BibTeX
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
