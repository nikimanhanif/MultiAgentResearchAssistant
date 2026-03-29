'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, ChevronRight, Brain, Search, FileText, Sparkles, CheckCircle } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useChatContext } from '@/context/chat-context'

interface ReasoningBlockProps {
  phase: string
  isActive: boolean
  isComplete?: boolean
  findingsCount: number
  tasksCount: number
  durationMs: number
}

export function ReasoningBlock({
  phase,
  isActive,
  isComplete = false,
  findingsCount,
  tasksCount,
  durationMs
}: ReasoningBlockProps) {
  const { thinking } = useChatContext()
  const [isExpanded, setIsExpanded] = useState(true)
  const [displayDuration, setDisplayDuration] = useState(durationMs)

  // Drive the timer locally so it updates independently of backend thought events.
  useEffect(() => {
    if (!thinking.isThinking || !thinking.startTime) {
      setDisplayDuration(thinking.elapsedMs || durationMs)
      return
    }

    const updateDuration = () => {
      setDisplayDuration(Date.now() - thinking.startTime)
    }

    updateDuration()

    const interval = setInterval(updateDuration, 100)

    return () => clearInterval(interval)
  }, [thinking.isThinking, thinking.startTime, thinking.elapsedMs, durationMs])

  // Auto-collapse when complete (but not immediately)
  useEffect(() => {
    if (isComplete && thinking.phases.length > 0) {
      const timeout = setTimeout(() => {
        setIsExpanded(false)
      }, 2000)
      return () => clearTimeout(timeout)
    }
  }, [isComplete, thinking.phases.length])

  const phaseLabel = {
    scoping: 'Understanding Query',
    researching: 'Analyzing Sources',
    generating_report: 'Generating Report',
    review: 'Preparing Review',
    complete: 'Research Complete'
  }[phase] || phase

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms}ms`
    return `${(ms / 1000).toFixed(1)}s`
  }

  const getStepIcon = (step: string) => {
    switch (step) {
      case 'planning':
        return <Sparkles className="h-3 w-3" />
      case 'analyzing':
        return <Brain className="h-3 w-3" />
      case 'researching':
        return <Search className="h-3 w-3" />
      case 'generating':
        return <FileText className="h-3 w-3" />
      default:
        return <Sparkles className="h-3 w-3" />
    }
  }

  const getPhaseIcon = (phaseName: string) => {
    switch (phaseName) {
      case 'investigating':
        return <Sparkles className="h-3 w-3" />
      case 'findings':
        return <Search className="h-3 w-3" />
      case 'deepening':
        return <Brain className="h-3 w-3" />
      case 'analyzing':
        return <Brain className="h-3 w-3" />
      default:
        return <Sparkles className="h-3 w-3" />
    }
  }

  const getSummary = () => {
    if (isComplete) {
      return `Completed research with ${findingsCount} sources`
    }
    if (phase === 'researching' && findingsCount > 0) {
      return `Analyzed ${findingsCount} sources in ${formatDuration(displayDuration)}`
    }
    if (phase === 'generating_report') {
      return `Synthesizing ${findingsCount} findings`
    }
    if (phase === 'scoping') {
      return `Processed in ${formatDuration(displayDuration)}`
    }
    return null
  }

  const summary = getSummary()
  const hasPhases = thinking.phases.length > 0

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="mb-4"
    >
      <div 
        className={cn(
          'rounded-lg border overflow-hidden transition-all duration-300',
          'bg-zinc-800/40 border-white/10',
          isActive && 'border-blue-500/30 shadow-[0_0_15px_rgba(59,130,246,0.1)]',
          isComplete && 'border-green-500/20'
        )}
      >
        {/* Header */}
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="w-full flex items-center gap-3 p-3 hover:bg-white/5 transition-colors"
        >
          {/* Brain icon with pulse animation */}
          <div className={cn(
            'p-2 rounded-lg transition-all duration-300',
            isActive 
              ? 'bg-blue-500/20 text-blue-400 animate-pulse' 
              : isComplete
                ? 'bg-green-500/20 text-green-400'
                : 'bg-zinc-700/50 text-zinc-400'
          )}>
            {isComplete ? <CheckCircle className="h-4 w-4" /> : <Brain className="h-4 w-4" />}
          </div>

          <div className="flex flex-col items-start min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span className={cn(
                'text-sm font-medium transition-colors',
                isActive ? 'text-white' : isComplete ? 'text-green-400' : 'text-zinc-400'
              )}>
                {phaseLabel}
              </span>
              {isActive && (
                <span className="flex items-center gap-1.5 text-xs text-blue-400">
                  <span className="h-1.5 w-1.5 bg-blue-400 rounded-full animate-pulse" />
                  {formatDuration(displayDuration)}
                </span>
              )}
            </div>
            
            {/* Current thought preview (when collapsed) */}
            {!isExpanded && thinking.isThinking && thinking.thought && (
              <span className="text-xs text-zinc-500 font-mono mt-0.5 truncate max-w-full">
                {thinking.thought}
              </span>
            )}
            
            {/* Summary line (shown when collapsed or complete) */}
            {!isActive && summary && (
              <span className="text-xs text-zinc-500 font-mono mt-0.5">
                {summary}
              </span>
            )}
          </div>

          <div className="ml-auto text-zinc-500">
            {isExpanded ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </div>
        </button>

        {/* Expandable Content */}
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <div className="px-3 pb-3">
                <div className={cn(
                  'p-3 rounded-md bg-black/40 text-sm font-mono leading-relaxed',
                  'text-zinc-400 space-y-3'
                )}>
                  {/* Current Thought - the internal monologue (only when active) */}
                  {thinking.isThinking && thinking.thought && (
                    <div className="flex items-start gap-2">
                      <div className="flex-shrink-0 mt-0.5 text-blue-400">
                        {getStepIcon(thinking.step)}
                      </div>
                      <p className="text-zinc-300 text-xs leading-relaxed">
                        {thinking.thought}
                      </p>
                    </div>
                  )}

                  {/* Phase-grouped thoughts */}
                  {hasPhases && (
                    <div className="space-y-3">
                      {thinking.phases.map((phaseGroup, phaseIdx) => (
                        <div key={phaseGroup.id} className="space-y-1.5">
                          <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-zinc-500">
                            {getPhaseIcon(phaseGroup.phase)}
                            <span>{phaseGroup.label}</span>
                          </div>
                          <ul className="space-y-1 pl-5">
                            {phaseGroup.thoughts.map((entry, idx) => (
                              <li 
                                key={entry.id}
                                className={cn(
                                  "flex items-start gap-2 text-xs",
                                  idx === phaseGroup.thoughts.length - 1 && phaseIdx === thinking.phases.length - 1
                                    ? "text-zinc-300" 
                                    : "text-zinc-500"
                                )}
                              >
                                <span className="flex-shrink-0 mt-0.5">
                                  {getStepIcon(entry.step)}
                                </span>
                                <span>{entry.thought}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Progress stats */}
                  <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs border-t border-white/5 pt-2">
                    {tasksCount > 0 && (
                      <span className="text-zinc-500">
                        Tasks: <span className="text-zinc-300">{tasksCount}</span>
                      </span>
                    )}
                    {findingsCount > 0 && (
                      <span className="text-zinc-500">
                        Sources: <span className="text-zinc-300">{findingsCount}</span>
                      </span>
                    )}
                    <span className="text-zinc-500">
                      Duration: <span className="text-zinc-300">{formatDuration(displayDuration)}</span>
                    </span>
                  </div>
                  
                  {/* Activity indicator */}
                  {isActive && !thinking.thought && (
                    <div className="flex items-center gap-2 text-xs text-blue-400/80 pt-1">
                      <div className="flex gap-1">
                        <span className="w-1 h-1 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                        <span className="w-1 h-1 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                        <span className="w-1 h-1 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                      </div>
                      Processing...
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}
