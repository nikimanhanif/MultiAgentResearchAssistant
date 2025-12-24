'use client'

import { useState } from 'react'
import type { Message as MessageType } from '@/types/chat'
import { cn } from '@/lib/utils'
import { User, Bot, Copy, Check } from 'lucide-react'
import { MarkdownContent } from './markdown-content'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'

interface MessageProps {
  message: MessageType
  isStreaming?: boolean
}

export function Message({ message, isStreaming }: MessageProps) {
  const isUser = message.role === 'user'
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div
      className={cn(
        'group relative flex w-full items-start gap-4 p-4 rounded-lg transition-colors',
        isUser 
          ? 'bg-transparent' 
          : 'bg-zinc-900/30 border border-subtle'
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          'flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-lg transition-colors',
          isUser
            ? 'bg-muted/50 border border-subtle text-muted-foreground'
            : 'bg-primary/10 border border-primary/20 text-primary'
        )}
      >
        {isUser ? (
          <User className="h-4 w-4" />
        ) : (
          <Bot className="h-4 w-4" />
        )}
      </div>

      {/* Content */}
      <div className="flex-1 space-y-2 overflow-hidden min-w-0">
        <div className="flex items-center justify-between gap-2">
          <span className={cn(
            'text-sm font-semibold',
            isUser ? 'text-muted-foreground' : 'text-primary'
          )}>
            {isUser ? 'You' : 'Research Assistant'}
          </span>
          {!isUser && (
            <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-muted-foreground hover:text-foreground"
                      onClick={handleCopy}
                    >
                      {copied ? (
                        <Check className="h-3.5 w-3.5" />
                      ) : (
                        <Copy className="h-3.5 w-3.5" />
                      )}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>{copied ? 'Copied!' : 'Copy'}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          )}
        </div>
        <div className={cn(
          'text-sm leading-relaxed',
          isStreaming && 'typewriter-cursor'
        )}>
          {isUser ? (
            <p className="whitespace-pre-wrap break-words text-foreground/90">{message.content}</p>
          ) : (
            <MarkdownContent content={message.content} />
          )}
        </div>
      </div>
    </div>
  )
}
