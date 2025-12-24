'use client'

import { useChatContext } from '@/context/chat-context'
import { cn } from '@/lib/utils'
import { Plus, MessageSquare, MoreVertical, Trash2, BookOpen } from 'lucide-react'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'

export function ResearchHistory() {
  const { conversations, threadId, loadConversation, deleteConversation, startNewChat } = useChatContext()

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    const now = new Date()
    const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24))
    
    if (diffDays === 0) return 'Today'
    if (diffDays === 1) return 'Yesterday'
    if (diffDays < 7) return `${diffDays} days ago`
    return date.toLocaleDateString()
  }

  const truncateQuery = (query: string, maxLength: number = 28) => {
    return query.length > maxLength ? query.slice(0, maxLength) + '...' : query
  }

  return (
    <div className="flex flex-col h-full bg-card/30">
      {/* Header */}
      <div className="p-4 border-b border-subtle">
        <div className="flex items-center gap-2 mb-4">
          <BookOpen className="h-5 w-5 text-primary" />
          <span className="font-semibold text-sm">Research Assistant</span>
        </div>
        <Button
          variant="outline"
          className="w-full justify-start gap-2 border-subtle hover:bg-accent/50"
          onClick={startNewChat}
        >
          <Plus className="h-4 w-4" />
          New Research
        </Button>
      </div>

      {/* History List */}
      <ScrollArea className="flex-1">
        <div className="p-2">
          {conversations.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 px-4 text-center">
              <MessageSquare className="h-8 w-8 text-muted-foreground mb-2" />
              <p className="text-sm text-muted-foreground">No research history</p>
            </div>
          ) : (
            <div className="space-y-1">
              {conversations.map((conv) => (
                <div
                  key={conv.conversation_id}
                  className={cn(
                    'group relative flex items-center rounded-md transition-colors',
                    'hover:bg-accent/50',
                    threadId === conv.conversation_id && 'bg-accent/70'
                  )}
                >
                  <button
                    onClick={() => loadConversation(conv.conversation_id)}
                    className={cn(
                      'flex-1 text-left px-3 py-2.5 rounded-md text-sm transition-colors',
                      'focus:outline-none'
                    )}
                  >
                    <div className="flex items-start gap-2">
                      <MessageSquare className="h-4 w-4 shrink-0 text-muted-foreground mt-0.5" />
                      <div className="flex-1 min-w-0">
                        <span className="truncate block text-foreground/90">
                          {truncateQuery(conv.user_query)}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          {formatDate(conv.created_at)}
                        </span>
                      </div>
                    </div>
                  </button>
                  <div className="opacity-0 group-hover:opacity-100 transition-opacity pr-2">
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="icon" className="h-7 w-7">
                          <MoreVertical className="h-3.5 w-3.5" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem 
                          className="text-destructive"
                          onClick={() => deleteConversation(conv.conversation_id)}
                        >
                          <Trash2 className="h-4 w-4 mr-2" />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  )
}
