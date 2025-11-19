'use client'

import { useEffect, useState } from 'react'
import { Copy, Check } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

interface CodeBlockProps {
  language: string
  code: string
  className?: string
}

export function CodeBlock({ language, code, className }: CodeBlockProps) {
  const [copied, setCopied] = useState(false)
  const [highlightedCode, setHighlightedCode] = useState<string>(code)

  useEffect(() => {
    // Simple syntax highlighting fallback
    // In production, you'd use shiki or react-syntax-highlighter
    setHighlightedCode(code)
  }, [code])

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="relative group">
      <div className="flex items-center justify-between px-4 py-2 bg-muted/50 border-b border-border rounded-t-lg">
        <span className="text-xs text-muted-foreground font-mono">
          {language}
        </span>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6"
          onClick={handleCopy}
        >
          {copied ? (
            <Check className="h-3 w-3" />
          ) : (
            <Copy className="h-3 w-3" />
          )}
        </Button>
      </div>
      <pre
        className={cn(
          'overflow-x-auto p-4 rounded-b-lg bg-muted text-sm',
          className
        )}
      >
        <code className="font-mono">{highlightedCode}</code>
      </pre>
    </div>
  )
}

