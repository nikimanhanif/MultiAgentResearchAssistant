'use client'

import { useRef, useEffect } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { ChatMessages } from './chat-messages'
import { FloatingInput } from './floating-input'
import { ExportButton } from './export-button'
import { ResearchCanvas } from './research-canvas'
import { useChatContext } from '@/context/chat-context'
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from '@/components/ui/resizable'
import type { ImperativePanelHandle } from 'react-resizable-panels'
import { cn } from '@/lib/utils'

export function ChatContainer() {
  const { reportPanelOpen, focusMode, closeReport } = useChatContext()
  
  const chatPanelRef = useRef<ImperativePanelHandle>(null)
  const reportPanelRef = useRef<ImperativePanelHandle>(null)

  // Set panel sizes based on state (after initial mount)
  useEffect(() => {
    // Small delay to let the transition class apply first
    const timer = setTimeout(() => {
      if (reportPanelOpen) {
        if (focusMode) {
          chatPanelRef.current?.resize(0)
          reportPanelRef.current?.resize(100)
        } else {
          chatPanelRef.current?.resize(40)
          reportPanelRef.current?.resize(60)
        }
      } else {
        chatPanelRef.current?.resize(100)
        reportPanelRef.current?.resize(0)
      }
    }, 10)
    return () => clearTimeout(timer)
  }, [reportPanelOpen, focusMode])

  // Handle keyboard escape to close report panel
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && reportPanelOpen) {
        closeReport()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [reportPanelOpen, closeReport])

  return (
    <ResizablePanelGroup 
      direction="horizontal" 
      className="h-full transition-all duration-300 ease-out"
    >
      {/* Chat Panel */}
      <ResizablePanel 
        ref={chatPanelRef}
        defaultSize={100}
        minSize={0}
        order={1}
        className="transition-all duration-300 ease-out"
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-end px-4 py-3 border-b border-subtle">
            <ExportButton />
          </div>
          
          {/* Messages */}
          <div className="flex-1 overflow-hidden">
            <ChatMessages />
          </div>
          
          {/* Floating Input */}
          <FloatingInput />
        </div>
      </ResizablePanel>

      {/* Resizable Handle */}
      <ResizableHandle 
        withHandle 
        className={cn(
          "w-px bg-border/50 hover:bg-primary/50 data-[resize-handle-active]:bg-primary",
          "transition-opacity duration-300 ease-out",
          !reportPanelOpen && "opacity-0 pointer-events-none"
        )}
      />

      {/* Report Panel */}
      <ResizablePanel 
        ref={reportPanelRef}
        defaultSize={0}
        minSize={0}
        order={2}
        collapsible
        collapsedSize={0}
        className="transition-all duration-300 ease-out"
        onCollapse={() => {
          if (reportPanelOpen) {
            closeReport()
          }
        }}
      >
        <div 
          className={cn(
            "h-full transition-opacity duration-300 ease-out",
            reportPanelOpen ? "opacity-100" : "opacity-0"
          )}
        >
          {reportPanelOpen && <ResearchCanvas />}
        </div>
      </ResizablePanel>
    </ResizablePanelGroup>
  )
}
