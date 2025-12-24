'use client'

import { ChatMessages } from './chat-messages'
import { FloatingInput } from './floating-input'
import { ExportButton } from './export-button'

export function ChatContainer() {
  return (
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
  )
}
