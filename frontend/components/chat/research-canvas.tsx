'use client'

import { useChatContext } from '@/context/chat-context'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Button } from '@/components/ui/button'
import { MarkdownContent } from './markdown-content'
import { Download, X, Maximize2, Minimize2 } from 'lucide-react'
import { motion } from 'framer-motion'

export function ResearchCanvas() {
  const { 
    activeReportContent, 
    focusMode, 
    closeReport, 
    toggleFocusMode,
    researchBrief
  } = useChatContext()

  if (!activeReportContent) return null

  const handleExport = () => {
    const lines: string[] = []
    const timestamp = new Date().toISOString().slice(0, 19).replace('T', ' ')
    
    // Header
    lines.push('# Research Report')
    lines.push(`\n**Exported:** ${timestamp}\n`)
    
    if (researchBrief) {
      lines.push(`**Research Scope:** ${researchBrief.scope}\n`)
    }
    
    lines.push('---\n')
    lines.push(activeReportContent)
    
    // Create and download file
    const content = lines.join('\n')
    const blob = new Blob([content], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `research_report_${Date.now()}.md`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      transition={{ type: 'spring', damping: 25, stiffness: 200 }}
      className="h-full flex flex-col bg-zinc-950/50 backdrop-blur-md border-l border-white/10"
    >
      {/* Sticky Header */}
      <div className="sticky top-0 z-10 flex items-center justify-between px-4 py-3 border-b border-white/10 bg-zinc-950/80 backdrop-blur-sm">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-foreground">Research Report</h2>
        </div>
        
        <div className="flex items-center gap-1">
          {/* Export Button */}
          <Button
            variant="ghost"
            size="icon"
            onClick={handleExport}
            className="h-8 w-8 text-muted-foreground hover:text-foreground"
            title="Export Report"
          >
            <Download className="h-4 w-4" />
          </Button>
          
          {/* Focus Mode Toggle */}
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleFocusMode}
            className="h-8 w-8 text-muted-foreground hover:text-foreground"
            title={focusMode ? 'Exit Focus Mode' : 'Enter Focus Mode'}
          >
            {focusMode ? (
              <Minimize2 className="h-4 w-4" />
            ) : (
              <Maximize2 className="h-4 w-4" />
            )}
          </Button>
          
          {/* Close Button */}
          <Button
            variant="ghost"
            size="icon"
            onClick={closeReport}
            className="h-8 w-8 text-muted-foreground hover:text-foreground"
            title="Close Panel"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>
      
      {/* Scrollable Report Content */}
      <ScrollArea className="flex-1">
        <div className="px-6 py-6 max-w-3xl mx-auto">
          <MarkdownContent content={activeReportContent} />
        </div>
      </ScrollArea>
    </motion.div>
  )
}
