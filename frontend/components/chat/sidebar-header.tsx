'use client'

import { Button } from '@/components/ui/button'
import { Plus } from 'lucide-react'

export function SidebarHeader() {
  const handleNewChat = () => {
    // TODO: Implement new chat creation
  }

  return (
    <div className="p-3">
      <Button
        onClick={handleNewChat}
        className="w-full justify-start gap-2"
        variant="outline"
      >
        <Plus className="h-4 w-4" />
        New Chat
      </Button>
    </div>
  )
}

