'use client'

import { useState } from 'react'
import { SidebarHeader } from './sidebar-header'
import { ChatList } from './chat-list'
import { Separator } from '@/components/ui/separator'
import { Button } from '@/components/ui/button'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'

export function Sidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false)

  return (
    <div className="relative hidden md:flex border-r border-border bg-background flex-col transition-all duration-300 ease-in-out">
      <div
        className={cn(
          'flex flex-col overflow-hidden transition-all duration-300 ease-in-out',
          isCollapsed ? 'w-0' : 'w-64'
        )}
      >
        <SidebarHeader />
        <Separator />
        <ChatList />
      </div>
      <Button
        variant="ghost"
        size="icon"
        className="absolute -right-3 top-4 h-6 w-6 rounded-full border border-border bg-background shadow-sm z-10"
        onClick={() => setIsCollapsed(!isCollapsed)}
        aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        {isCollapsed ? (
          <ChevronRight className="h-3 w-3" />
        ) : (
          <ChevronLeft className="h-3 w-3" />
        )}
      </Button>
    </div>
  )
}
