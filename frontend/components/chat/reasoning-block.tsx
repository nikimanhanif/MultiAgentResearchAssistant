'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, ChevronRight, Brain, Search, FileText, Sparkles } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useChatContext } from '@/context/chat-context'

interface ReasoningBlockProps {
  phase: string
  isActive: boolean
  findingsCount: number
  tasksCount: number
  durationMs: number
}

export function ReasoningBlock({
  phase,
  isActive,
  findingsCount,
  tasksCount,
  durationMs
}: ReasoningBlockProps) {
  const { thinking } = useChatContext()
  const [isExpanded, setIsExpanded] = useState(true)
  const [displayDuration, setDisplayDuration] = useState(durationMs)

  // Update duration in real-time when active
  useEffect(() => {
    if (!isActive) {
      setDisplayDuration(durationMs)
      return
    }
    
    const interval = setInterval(() => {
      setDisplayDuration(prev => prev + 100)
    }, 100)
    
    return () => clearInterval(interval)
  }, [isActive, durationMs])

  // Auto-collapse when phase completes
  useEffect(() => {
    if (!isActive && findingsCount > 0) {
      const timeout = setTimeout(() => {
        setIsExpanded(false)
      }, 1000)
      return () => clearTimeout(timeout)
    }
  }, [isActive, findingsCount])

  const phaseLabel = {
    scoping: 'Understanding Query',
    researching: 'Analyzing Sources',
    generating_report: 'Generating Report',
    review: 'Preparing Review'
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

  const getSummary = () => {
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
          isActive && 'border-blue-500/30 shadow-[0_0_15px_rgba(59,130,246,0.1)]'
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
              : 'bg-zinc-700/50 text-zinc-400'
          )}>
            <Brain className="h-4 w-4" />
          </div>

          <div className="flex flex-col items-start min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span className={cn(
                'text-sm font-medium transition-colors',
                isActive ? 'text-white' : 'text-zinc-400'
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
                {thinking.thought.slice(0, 60)}...
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
                  {/* Current Thought - the internal monologue */}
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

                  {/* Thought History */}
                  {thinking.history.length > 1 && (
                    <div className="space-y-1.5 border-t border-white/5 pt-2">
                      <span className="text-[10px] uppercase tracking-wider text-zinc-600">
                        Recent Activity
                      </span>
                      <ul className="space-y-1">
                        {thinking.history.slice(-3).reverse().map((entry, idx) => (
                          <li 
                            key={entry.id}
                            className={cn(
                              "flex items-start gap-2 text-xs",
                              idx === 0 ? "text-zinc-400" : "text-zinc-500"
                            )}
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

