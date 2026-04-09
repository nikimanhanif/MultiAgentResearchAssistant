'use client'

import { useChatContext } from '@/context/chat-context'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Button } from '@/components/ui/button'
import { MarkdownContent } from './markdown-content'
import { ExportMenu } from './export-menu'
import { X, Maximize2, Minimize2 } from 'lucide-react'
import { motion } from 'framer-motion'

export function ResearchCanvas() {
  const {
    activeReportContent,
    focusMode,
    closeReport,
    toggleFocusMode,
  } = useChatContext()

  if (!activeReportContent) return null

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
          {/* Export Menu */}
          <ExportMenu scope="report" />
          
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
