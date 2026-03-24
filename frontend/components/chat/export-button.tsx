'use client'

import { useChatContext } from '@/context/chat-context'
import { Button } from '@/components/ui/button'
import { Download } from 'lucide-react'

export function ExportButton() {
  const { messages, researchBrief, reviewRequest } = useChatContext()

  const handleExport = () => {
    const lines: string[] = []
    const timestamp = new Date().toISOString().slice(0, 19).replace('T', ' ')
    
    // Header
    lines.push('# Research Assistant Session')
    lines.push(`\n**Exported:** ${timestamp}\n`)
    lines.push('---\n')
    
    // Research Brief
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
    
    // Conversation
    if (messages.length > 0) {
      lines.push('## Conversation\n')
      messages.forEach(msg => {
        const roleIcon = msg.role === 'user' ? '👤' : '🤖'
        const roleLabel = msg.role === 'user' ? 'You' : 'Assistant'
        lines.push(`### ${roleIcon} ${roleLabel}\n`)
        lines.push(msg.content)
        lines.push('\n')
      })
      lines.push('---\n')
    }
    
    // Report
    if (reviewRequest?.report) {
      lines.push('## Generated Report\n')
      lines.push(reviewRequest.report)
      lines.push('\n')
    }
    
    // Create and download file
    const content = lines.join('\n')
    const blob = new Blob([content], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `research_session_${Date.now()}.md`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const hasContent = messages.length > 0 || researchBrief || reviewRequest?.report

  if (!hasContent) {
    return null
  }

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={handleExport}
      className="h-8 w-8"
      title="Export to Markdown"
    >
      <Download className="h-4 w-4" />
    </Button>
  )
}
