'use client'

import { useEffect, useRef } from 'react'
import { Message } from './message'
import type { Message as MessageType } from '@/types/chat'
import { Bot } from 'lucide-react'

export function ChatMessages() {
  // TODO: Fetch messages from state/API
  const messages: MessageType[] = []
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="flex-1 overflow-y-auto scroll-smooth">
      {messages.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-full text-center px-4 animate-in fade-in-0 duration-500">
          <div className="mb-4">
            <Bot className="h-12 w-12 text-muted-foreground" />
          </div>
          <h2 className="text-2xl font-semibold mb-2">Start a conversation</h2>
          <p className="text-muted-foreground max-w-md">
            Ask me anything and I'll help you with your research.
          </p>
        </div>
      ) : (
        <div className="divide-y divide-border">
          {messages.map((message, index) => (
            <div
              key={message.id}
              className="animate-in fade-in-0 slide-in-from-bottom-2 duration-300"
              style={{ animationDelay: `${index * 50}ms` }}
            >
              <Message message={message} />
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      )}
    </div>
  )
}
