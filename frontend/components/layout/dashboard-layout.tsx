'use client'

import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from '@/components/ui/resizable'
import { ResearchHistory } from '@/components/sidebar/research-history'
import { ActiveAgentsPanel } from '@/components/agents/active-agents-panel'
import { ChatProvider } from '@/context/chat-context'
import { ReviewModal } from '@/components/chat/review-modal'

interface DashboardLayoutProps {
  children: React.ReactNode
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
  return (
    <ChatProvider>
      <div className="h-screen w-screen overflow-hidden bg-background">
        <ResizablePanelGroup direction="horizontal" className="h-full">
          {/* Left Panel - Research History */}
          <ResizablePanel 
            defaultSize={18} 
            minSize={15} 
            maxSize={25}
            className="border-r border-subtle"
          >
            <ResearchHistory />
          </ResizablePanel>

          <ResizableHandle className="w-px bg-border/50 hover:bg-primary/50 transition-colors" />

          {/* Center Panel - Chat Thread */}
          <ResizablePanel defaultSize={62} minSize={40}>
            <div className="h-full flex flex-col">
              {children}
            </div>
          </ResizablePanel>

          <ResizableHandle className="w-px bg-border/50 hover:bg-primary/50 transition-colors" />

          {/* Right Panel - Active Agents */}
          <ResizablePanel 
            defaultSize={20} 
            minSize={15} 
            maxSize={30}
            className="border-l border-subtle"
          >
            <ActiveAgentsPanel />
          </ResizablePanel>
        </ResizablePanelGroup>

        {/* Review Modal (triggered by HITL) */}
        <ReviewModal />
      </div>
    </ChatProvider>
  )
}
