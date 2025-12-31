'use client'

import { Brain } from 'lucide-react'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { useState } from 'react'

export function ChatHeader() {
  const [deepResearch, setDeepResearch] = useState(false)

  return (
    <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-background">
      <div className="flex-1" />
      <div className="flex items-center gap-2">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex items-center gap-2">
                <Brain className="h-4 w-4 text-muted-foreground" />
                <Label htmlFor="deep-research" className="text-sm cursor-pointer">
                  Deep Research
                </Label>
                <Switch
                  id="deep-research"
                  checked={deepResearch}
                  onCheckedChange={setDeepResearch}
                />
              </div>
            </TooltipTrigger>
            <TooltipContent>
              <p>Enable deep research mode for more thorough analysis</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
    </div>
  )
}

