'use client'

import type { Chat } from '@/types/chat'
import { cn } from '@/lib/utils'
import { MessageSquare, MoreVertical, Trash2, Edit } from 'lucide-react'
import { useState } from 'react'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Button } from '@/components/ui/button'

export function ChatList() {
  // TODO: Fetch chats from state/API
  const chats: Chat[] = []
  const [selectedChatId, setSelectedChatId] = useState<string | null>(null)

  const handleDelete = (chatId: string) => {
    // TODO: Implement delete functionality
  }

  const handleRename = (chatId: string) => {
    // TODO: Implement rename functionality
  }

  return (
    <div className="flex-1 overflow-y-auto p-2">
      {chats.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-8 px-4 text-center">
          <MessageSquare className="h-8 w-8 text-muted-foreground mb-2" />
          <p className="text-sm text-muted-foreground">No chat history</p>
        </div>
      ) : (
        <div className="space-y-1">
          {chats.map((chat) => (
            <div
              key={chat.id}
              className={cn(
                'group relative flex items-center rounded-md transition-colors',
                'hover:bg-accent',
                selectedChatId === chat.id && 'bg-accent'
              )}
            >
              <button
                onClick={() => setSelectedChatId(chat.id)}
                className={cn(
                  'flex-1 text-left px-3 py-2 rounded-md text-sm transition-colors',
                  'focus:outline-none'
                )}
              >
                <div className="flex items-center gap-2">
                  <MessageSquare className="h-4 w-4 shrink-0 text-muted-foreground" />
                  <span className="truncate">{chat.title}</span>
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
                    <DropdownMenuItem onClick={() => handleRename(chat.id)}>
                      <Edit className="h-4 w-4 mr-2" />
                      Rename
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onClick={() => handleDelete(chat.id)}
                      className="text-destructive"
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
  )
}
