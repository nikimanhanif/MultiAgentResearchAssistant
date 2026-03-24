'use client'

import { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { ArrowUp, Square } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useChatContext } from '@/context/chat-context'

export function FloatingInput() {
  const [input, setInput] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const { sendMessage, isStreaming } = useChatContext()

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`
    }
  }, [input])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isStreaming) return

    const message = input.trim()
    setInput('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
    
    await sendMessage(message)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <div className="p-4 pt-2">
      <form onSubmit={handleSubmit}>
        <div className={cn(
          'relative flex items-end gap-2 rounded-xl border border-subtle bg-card/80 backdrop-blur-xl p-3',
          'transition-all duration-200 floating-input-glow'
        )}>
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isStreaming ? "Researching..." : "Ask a research question..."}
            disabled={isStreaming}
            rows={1}
            className={cn(
              'flex-1 bg-transparent text-sm resize-none',
              'placeholder:text-muted-foreground/60',
              'focus:outline-none',
              'disabled:cursor-not-allowed disabled:opacity-50',
              'min-h-[24px] max-h-[150px]'
            )}
          />
          <Button
            type="submit"
            size="icon"
            disabled={!input.trim() || isStreaming}
            className={cn(
              'shrink-0 h-8 w-8 rounded-full transition-all duration-200',
              'bg-primary hover:bg-primary/90',
              'disabled:opacity-30'
            )}
          >
            {isStreaming ? (
              <Square className="h-3 w-3 fill-current" />
            ) : (
              <ArrowUp className="h-4 w-4" />
            )}
          </Button>
        </div>
      </form>
    </div>
  )
}
