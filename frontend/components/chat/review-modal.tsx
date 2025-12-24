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
import { Check, RefreshCw, Search, Loader2 } from 'lucide-react'
import type { ReviewAction } from '@/types/chat'
import { ScrollArea } from '@/components/ui/scroll-area'
import { MarkdownContent } from './markdown-content'

export function ReviewModal() {
  const { reviewRequest, resumeReview, isStreaming } = useChatContext()
  const [feedback, setFeedback] = useState('')
  const [selectedAction, setSelectedAction] = useState<ReviewAction | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const isOpen = reviewRequest?.pending ?? false

  const handleAction = async (action: ReviewAction) => {
    if (action === 'approve') {
      setIsSubmitting(true)
      try {
        await resumeReview(action)
      } finally {
        setIsSubmitting(false)
      }
    } else {
      setSelectedAction(action)
    }
  }

  const handleSubmitFeedback = async () => {
    if (selectedAction && feedback.trim()) {
      setIsSubmitting(true)
      try {
        await resumeReview(selectedAction, feedback.trim())
        setFeedback('')
        setSelectedAction(null)
      } finally {
        setIsSubmitting(false)
      }
    }
  }

  const handleClose = () => {
    if (!isSubmitting) {
      resumeReview('approve')
    }
  }

  const handleCancel = () => {
    setSelectedAction(null)
    setFeedback('')
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
          <div className="prose prose-sm dark:prose-invert max-w-none px-1">
            <MarkdownContent content={reviewRequest?.report || ''} />
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
              disabled={isSubmitting}
              className={cn(
                'w-full min-h-[100px] p-3 rounded-lg border border-subtle bg-background',
                'text-sm resize-none focus:outline-none focus:ring-2 focus:ring-ring',
                isSubmitting && 'opacity-50 cursor-not-allowed'
              )}
            />
            <div className="flex gap-2 mt-3">
              <Button
                variant="outline"
                size="sm"
                onClick={handleCancel}
                disabled={isSubmitting}
              >
                Cancel
              </Button>
              <Button
                size="sm"
                onClick={handleSubmitFeedback}
                disabled={!feedback.trim() || isSubmitting || isStreaming}
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Submitting...
                  </>
                ) : (
                  'Submit Feedback'
                )}
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
                disabled={isSubmitting || isStreaming}
              >
                <Search className="h-4 w-4 mr-2" />
                Re-research
              </Button>
              <Button
                variant="outline"
                className="flex-1"
                onClick={() => handleAction('refine')}
                disabled={isSubmitting || isStreaming}
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Refine
              </Button>
              <Button
                className="flex-1"
                onClick={() => handleAction('approve')}
                disabled={isSubmitting || isStreaming}
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Approving...
                  </>
                ) : (
                  <>
                    <Check className="h-4 w-4 mr-2" />
                    Approve
                  </>
                )}
              </Button>
            </div>
          </DialogFooter>
        )}
      </DialogContent>
    </Dialog>
  )
}

