'use client'

import { useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Message } from './message'
import { ReasoningBlock } from './reasoning-block'
import { useChatContext } from '@/context/chat-context'
import { Search } from 'lucide-react'
import { ScrollArea } from '@/components/ui/scroll-area'

export function ChatMessages() {
  const { 
    messages, 
    isStreaming, 
    currentStreamingContent, 
    researchProgress,
    activeNode,
    thinking
  } = useChatContext()
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, currentStreamingContent])

  const hasMessages = messages.length > 0 || currentStreamingContent
  
  const isResearchingPhase = researchProgress.phase === 'researching'
  const isGeneratingReport = researchProgress.phase === 'generating_report'
  const isCompletePhase = researchProgress.phase === 'complete' || researchProgress.phase === 'review'
  
  // Show reasoning block if we have phases history, or are actively researching
  const hasThinkingHistory = thinking.phases.length > 0
  const showReasoning = hasThinkingHistory || 
                        (isResearchingPhase && researchProgress.tasksCount > 0) || 
                        (isStreaming && researchProgress.phase === 'scoping')
  
  const showStreamingMessage = currentStreamingContent && (isGeneratingReport || !showReasoning)

  return (
    <ScrollArea className="h-full">
      <div className="px-4 py-6">
        {!hasMessages ? (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-col items-center justify-center min-h-[60vh] text-center px-4"
          >
            <div className="mb-6 p-5 rounded-2xl bg-primary/10 border border-primary/20">
              <Search className="h-10 w-10 text-primary" />
            </div>
            <h2 className="text-2xl font-semibold mb-3">Start Your Research</h2>
            <p className="text-muted-foreground max-w-md leading-relaxed">
              Ask any research question and I&apos;ll help you find comprehensive answers 
              with citations from academic sources.
            </p>
          </motion.div>
        ) : (
          <div className="space-y-1 max-w-4xl mx-auto">
            <AnimatePresence mode="popLayout">
              {messages.map((message, index) => {
                // Detect brief message - multiple formats for live vs restored
                const isBriefMessage = message.content.includes('### Research Brief Created') ||
                                       message.content.includes('Research brief created') ||
                                       message.content.toLowerCase().includes('proceeding with research on:')
                
                return (
                  <div key={message.id}>
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      transition={{ 
                        duration: 0.3, 
                        delay: Math.min(index * 0.05, 0.2) 
                      }}
                    >
                      <Message message={message} />
                    </motion.div>
                    
                    {/* Insert ReasoningBlock after Research Brief message */}
                    {isBriefMessage && showReasoning && (
                      <ReasoningBlock 
                        phase={researchProgress.phase}
                        isActive={isStreaming && (isResearchingPhase || researchProgress.phase === 'scoping')}
                        isComplete={isCompletePhase}
                        findingsCount={researchProgress.findingsCount}
                        tasksCount={researchProgress.tasksCount}
                        durationMs={researchProgress.phaseDurationMs}
                      />
                    )}
                  </div>
                )
              })}
            </AnimatePresence>
            
            
            {/* Fallback: Show reasoning block at end if no brief message yet (e.g., during scoping) */}
            {showReasoning && !messages.some(m => 
              m.content.includes('### Research Brief Created') ||
              m.content.includes('Research brief created') ||
              m.content.toLowerCase().includes('proceeding with research on:')
            ) && (
              <ReasoningBlock 
                phase={researchProgress.phase}
                isActive={isStreaming && (isResearchingPhase || researchProgress.phase === 'scoping')}
                isComplete={isCompletePhase}
                findingsCount={researchProgress.findingsCount}
                tasksCount={researchProgress.tasksCount}
                durationMs={researchProgress.phaseDurationMs}
              />
            )}
            
            {/* Streaming message - for report generation and other streaming content */}
            <AnimatePresence>
              {showStreamingMessage && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <Message
                    message={{
                      id: 'streaming',
                      role: 'assistant',
                      content: currentStreamingContent,
                      timestamp: new Date()
                    }}
                    isStreaming={true}
                  />
                </motion.div>
              )}
            </AnimatePresence>
            
            {/* Typing indicator - only when streaming with no content */}
            <AnimatePresence>
              {isStreaming && !currentStreamingContent && !showReasoning && (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="p-4"
                >
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <div className="flex gap-1">
                      <span className="w-2 h-2 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <span className="w-2 h-2 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <span className="w-2 h-2 bg-primary/60 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                    <span className="text-sm font-mono">Thinking...</span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
            
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>
    </ScrollArea>
  )
}

