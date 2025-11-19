'use client'

import { useState } from 'react'
import type { Message } from '@/types/chat'

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)

  const sendMessage = async (content: string) => {
    // TODO: Implement message sending logic
  }

  return {
    messages,
    isLoading,
    sendMessage,
  }
}

