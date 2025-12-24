'use client'

import { useState } from 'react'
import { useChatContext } from '@/context/chat-context'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { cn } from '@/lib/utils'
import { Check, RefreshCw, Search } from 'lucide-react'
import type { ReviewAction } from '@/types/chat'
import { ScrollArea } from '@/components/ui/scroll-area'

export function ReviewModal() {
  const { reviewRequest, resumeReview, isStreaming } = useChatContext()
  const [feedback, setFeedback] = useState('')
  const [selectedAction, setSelectedAction] = useState<ReviewAction | null>(null)

  const isOpen = reviewRequest?.pending ?? false

  const handleAction = async (action: ReviewAction) => {
    if (action === 'approve') {
      await resumeReview(action)
    } else {
      setSelectedAction(action)
    }
  }

  const handleSubmitFeedback = async () => {
    if (selectedAction && feedback.trim()) {
      await resumeReview(selectedAction, feedback.trim())
      setFeedback('')
      setSelectedAction(null)
    }
  }

  const handleClose = () => {
    resumeReview('approve')
  }

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && handleClose()}>
      <DialogContent className="max-w-3xl h-[80vh] flex flex-col bg-card border-subtle">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Check className="h-5 w-5 text-primary" />
            Review Report
          </DialogTitle>
          <DialogDescription>
            Review the generated report and choose an action
          </DialogDescription>
        </DialogHeader>

        {/* Report Content */}
        <ScrollArea className="flex-1 py-4">
          <div className="prose prose-sm dark:prose-invert max-w-none">
            <div 
              className="whitespace-pre-wrap text-sm font-mono leading-relaxed"
              dangerouslySetInnerHTML={{ __html: reviewRequest?.report || '' }}
            />
          </div>
        </ScrollArea>

        {/* Feedback Input */}
        {selectedAction && (
          <div className="py-4 border-t border-subtle animate-in fade-in-0 slide-in-from-bottom-2 duration-200">
            <label className="text-sm font-medium mb-2 block">
              {selectedAction === 'refine' 
                ? 'What would you like to refine?' 
                : 'What additional research is needed?'}
            </label>
            <textarea
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              placeholder="Describe your feedback..."
              className={cn(
                'w-full min-h-[100px] p-3 rounded-lg border border-subtle bg-background',
                'text-sm resize-none focus:outline-none focus:ring-2 focus:ring-ring'
              )}
            />
            <div className="flex gap-2 mt-3">
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setSelectedAction(null)
                  setFeedback('')
                }}
              >
                Cancel
              </Button>
              <Button
                size="sm"
                onClick={handleSubmitFeedback}
                disabled={!feedback.trim() || isStreaming}
              >
                Submit Feedback
              </Button>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        {!selectedAction && (
          <DialogFooter className="border-t border-subtle pt-4">
            <div className="flex gap-3 w-full">
              <Button
                variant="outline"
                className="flex-1"
                onClick={() => handleAction('re_research')}
                disabled={isStreaming}
              >
                <Search className="h-4 w-4 mr-2" />
                Re-research
              </Button>
              <Button
                variant="outline"
                className="flex-1"
                onClick={() => handleAction('refine')}
                disabled={isStreaming}
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Refine
              </Button>
              <Button
                className="flex-1"
                onClick={() => handleAction('approve')}
                disabled={isStreaming}
              >
                <Check className="h-4 w-4 mr-2" />
                Approve
              </Button>
            </div>
          </DialogFooter>
        )}
      </DialogContent>
    </Dialog>
  )
}
