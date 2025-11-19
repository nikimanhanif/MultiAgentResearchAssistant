'use client'

import { Bot } from 'lucide-react'
import { Skeleton } from '@/components/ui/skeleton'

export function TypingIndicator() {
  return (
    <div className="flex w-full items-start gap-4 p-4 bg-muted/30">
      <div className="flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-md border bg-primary text-primary-foreground border-primary shadow-sm">
        <Bot className="h-4 w-4" />
      </div>
      <div className="flex-1 space-y-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold">Assistant</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="flex gap-1">
            <div className="h-2 w-2 rounded-full bg-muted-foreground animate-bounce" style={{ animationDelay: '0ms' }} />
            <div className="h-2 w-2 rounded-full bg-muted-foreground animate-bounce" style={{ animationDelay: '150ms' }} />
            <div className="h-2 w-2 rounded-full bg-muted-foreground animate-bounce" style={{ animationDelay: '300ms' }} />
          </div>
        </div>
      </div>
    </div>
  )
}

