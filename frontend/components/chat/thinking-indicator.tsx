'use client'

import { useChatContext } from '@/context/chat-context'
import { cn } from '@/lib/utils'
import { Brain, ChevronDown, Search, FileText, Sparkles } from 'lucide-react'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import { useState, useEffect } from 'react'

/**
 * ThinkingIndicator - Displays the AI's internal monologue as a collapsible
 * chat message during research. Shows high-level summaries of what the AI
 * is currently doing.
 */
export function ThinkingIndicator() {
  const { thinking, isStreaming, researchProgress } = useChatContext()
  const [isOpen, setIsOpen] = useState(false)
  const [displayedElapsed, setDisplayedElapsed] = useState(0)

  // Update elapsed time display every second while thinking
  useEffect(() => {
    if (!thinking.isThinking) {
      setDisplayedElapsed(0)
      return
    }

    const interval = setInterval(() => {
      if (thinking.startTime) {
        setDisplayedElapsed(Math.floor((Date.now() - thinking.startTime) / 1000))
      }
    }, 1000)

    return () => clearInterval(interval)
  }, [thinking.isThinking, thinking.startTime])

  // Don't show if not streaming or not thinking
  if (!isStreaming || !thinking.isThinking) {
    return null
  }

  const getStepIcon = (step: string) => {
    switch (step) {
      case 'analyzing':
        return <Brain className="h-3.5 w-3.5" />
      case 'researching':
        return <Search className="h-3.5 w-3.5" />
      case 'generating':
        return <FileText className="h-3.5 w-3.5" />
      default:
        return <Sparkles className="h-3.5 w-3.5" />
    }
  }

  const getAgentLabel = (agent: string) => {
    switch (agent) {
      case 'supervisor':
        return 'Analyzing'
      case 'sub_agent':
        return 'Researching'
      case 'report_agent':
        return 'Writing'
      default:
        return 'Thinking'
    }
  }

  return (
    <div className="flex gap-3 py-4">
      {/* Avatar placeholder for consistency with chat messages */}
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
        <Brain className="h-4 w-4 text-primary animate-pulse" />
      </div>

      {/* Collapsible thinking card */}
      <Collapsible
        open={isOpen}
        onOpenChange={setIsOpen}
        className="flex-1 max-w-2xl"
      >
        <div className={cn(
          "rounded-lg border bg-card/50 backdrop-blur-sm transition-all duration-200",
          isOpen && "border-primary/30"
        )}>
          <CollapsibleTrigger className="w-full px-4 py-3 flex items-center gap-3 hover:bg-muted/50 rounded-lg transition-colors">
            {/* Animated icon */}
            <div className="flex items-center justify-center w-6 h-6 rounded-full bg-primary/10">
              <div className="animate-pulse">
                {getStepIcon(thinking.step)}
              </div>
            </div>

            {/* Label and elapsed time */}
            <div className="flex-1 text-left">
              <span className="text-sm font-medium text-foreground">
                {getAgentLabel(thinking.agent)}
              </span>
              <span className="text-xs text-muted-foreground ml-2">
                {displayedElapsed}s
              </span>
            </div>

            {/* Chevron */}
            <ChevronDown className={cn(
              "h-4 w-4 text-muted-foreground transition-transform duration-200",
              isOpen && "rotate-180"
            )} />
          </CollapsibleTrigger>

          <CollapsibleContent>
            <div className="px-4 pb-3 pt-1 border-t border-subtle">
              {/* Current thought */}
              <p className="text-sm text-muted-foreground leading-relaxed">
                {thinking.thought}
              </p>

              {/* Thought history (last few entries) */}
              {thinking.history.length > 1 && (
                <div className="mt-3 space-y-1.5">
                  <span className="text-xs font-medium text-muted-foreground/70">
                    Recent activity
                  </span>
                  <ul className="space-y-1">
                    {thinking.history.slice(-3).map((entry) => (
                      <li 
                        key={entry.id}
                        className="flex items-start gap-2 text-xs text-muted-foreground/80"
                      >
                        <span className="flex-shrink-0 mt-0.5">
                          {getStepIcon(entry.step)}
                        </span>
                        <span className="line-clamp-1">{entry.thought}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Progress stats if available */}
              {(researchProgress.findingsCount > 0 || researchProgress.tasksCount > 0) && (
                <div className="mt-3 flex gap-4 text-xs text-muted-foreground">
                  {researchProgress.tasksCount > 0 && (
                    <span>{researchProgress.tasksCount} tasks</span>
                  )}
                  {researchProgress.findingsCount > 0 && (
                    <span>{researchProgress.findingsCount} findings</span>
                  )}
                </div>
              )}
            </div>
          </CollapsibleContent>
        </div>
      </Collapsible>
    </div>
  )
}
