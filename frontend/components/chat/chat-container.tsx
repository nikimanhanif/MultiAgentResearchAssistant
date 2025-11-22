'use client'

import { Sidebar } from './sidebar'
import { ChatMessages } from './chat-messages'
import { MessageInput } from './message-input'
import { ChatHeader } from './chat-header'

export function ChatContainer() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex flex-col flex-1 min-w-0">
        <ChatHeader />
        <ChatMessages />
        <MessageInput />
      </div>
    </div>
  )
}
