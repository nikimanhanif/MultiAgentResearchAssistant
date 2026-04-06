// @vitest-environment jsdom

import React from 'react'
import { render, screen, within, act, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi, describe, it, expect, beforeEach } from 'vitest'
import { ReviewModal } from '@/components/chat/review-modal'
import { useChatContext } from '@/context/chat-context'

vi.mock('@/context/chat-context', () => ({
  useChatContext: vi.fn(),
}))

vi.mock('@/components/chat/markdown-content', () => ({
  MarkdownContent: ({ content }: any) => <div data-testid="report-content">{content}</div>,
}))

const mockResumeReview = vi.fn()

const pendingReview = {
  pending: true,
  report: 'This is the generated report content.',
}

const defaultContext = {
  reviewRequest: pendingReview,
  resumeReview: mockResumeReview,
  isStreaming: false,
}

beforeEach(() => {
  vi.mocked(useChatContext).mockReturnValue(defaultContext as any)
  mockResumeReview.mockReset()
})

describe('ReviewModal', () => {
  describe('visibility', () => {
    it('renders nothing when reviewRequest is null', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...defaultContext,
        reviewRequest: null,
      } as any)
      render(<ReviewModal />)
      expect(screen.queryByRole('dialog')).toBeNull()
    })

    it('renders nothing when reviewRequest.pending is false', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...defaultContext,
        reviewRequest: { pending: false, report: 'some report' },
      } as any)
      render(<ReviewModal />)
      expect(screen.queryByRole('dialog')).toBeNull()
    })

    it('renders the dialog when reviewRequest.pending is true', () => {
      render(<ReviewModal />)
      expect(screen.queryByRole('dialog')).not.toBeNull()
    })

    it('shows "Review Report" as the dialog title', () => {
      render(<ReviewModal />)
      expect(within(document.body).queryByText('Review Report')).not.toBeNull()
    })

    it('renders the report content inside the dialog', () => {
      render(<ReviewModal />)
      expect(within(document.body).queryByTestId('report-content')).not.toBeNull()
      expect(within(document.body).queryByText('This is the generated report content.')).not.toBeNull()
    })
  })

  describe('action buttons — initial state', () => {
    it('renders Re-research, Refine, and Approve buttons', () => {
      render(<ReviewModal />)
      expect(within(document.body).queryByText('Re-research')).not.toBeNull()
      expect(within(document.body).queryByText('Refine')).not.toBeNull()
      expect(within(document.body).queryByText('Approve')).not.toBeNull()
    })

    it('all three buttons are enabled when not streaming and not submitting', () => {
      render(<ReviewModal />)
      const reResearch = within(document.body).getByText('Re-research').closest('button') as HTMLButtonElement
      const refine = within(document.body).getByText('Refine').closest('button') as HTMLButtonElement
      const approve = within(document.body).getByText('Approve').closest('button') as HTMLButtonElement
      expect(reResearch.disabled).toBe(false)
      expect(refine.disabled).toBe(false)
      expect(approve.disabled).toBe(false)
    })

    it('all three buttons are disabled when isStreaming is true', () => {
      vi.mocked(useChatContext).mockReturnValue({ ...defaultContext, isStreaming: true } as any)
      render(<ReviewModal />)
      const reResearch = within(document.body).getByText('Re-research').closest('button') as HTMLButtonElement
      const refine = within(document.body).getByText('Refine').closest('button') as HTMLButtonElement
      const approve = within(document.body).getByText('Approve').closest('button') as HTMLButtonElement
      expect(reResearch.disabled).toBe(true)
      expect(refine.disabled).toBe(true)
      expect(approve.disabled).toBe(true)
    })
  })

  describe('Approve action', () => {
    it('calls resumeReview("approve") when Approve is clicked', async () => {
      mockResumeReview.mockResolvedValue(undefined)
      const user = userEvent.setup()
      render(<ReviewModal />)
      const approveBtn = within(document.body).getByText('Approve').closest('button')!
      await user.click(approveBtn)
      expect(mockResumeReview).toHaveBeenCalledWith('approve')
    })

    it('shows "Approving..." text while submitting', async () => {
      // Never-resolving promise keeps isSubmitting=true
      mockResumeReview.mockReturnValue(new Promise(() => {}))
      render(<ReviewModal />)
      const approveBtn = within(document.body).getByText('Approve').closest('button')!
      await act(async () => {
        fireEvent.click(approveBtn)
        await Promise.resolve()
      })
      expect(within(document.body).queryByText('Approving...')).not.toBeNull()
    })
  })

  describe('Refine action', () => {
    it('shows the feedback textarea after clicking Refine', async () => {
      const user = userEvent.setup()
      render(<ReviewModal />)
      const refineBtn = within(document.body).getByText('Refine').closest('button')!
      await user.click(refineBtn)
      expect(within(document.body).queryByPlaceholderText('Describe your feedback...')).not.toBeNull()
    })

    it('hides the three main action buttons after clicking Refine', async () => {
      const user = userEvent.setup()
      render(<ReviewModal />)
      const refineBtn = within(document.body).getByText('Refine').closest('button')!
      await user.click(refineBtn)
      expect(within(document.body).queryByText('Re-research')).toBeNull()
      expect(within(document.body).queryByText('Approve')).toBeNull()
    })

    it('shows "What would you like to refine?" as the label', async () => {
      const user = userEvent.setup()
      render(<ReviewModal />)
      const refineBtn = within(document.body).getByText('Refine').closest('button')!
      await user.click(refineBtn)
      expect(within(document.body).queryByText('What would you like to refine?')).not.toBeNull()
    })

    it('Submit Feedback button is disabled when textarea is empty', async () => {
      const user = userEvent.setup()
      render(<ReviewModal />)
      const refineBtn = within(document.body).getByText('Refine').closest('button')!
      await user.click(refineBtn)
      const submitBtn = within(document.body).getByText('Submit Feedback').closest('button') as HTMLButtonElement
      expect(submitBtn.disabled).toBe(true)
    })

    it('Submit Feedback button is enabled when textarea has content', async () => {
      const user = userEvent.setup()
      render(<ReviewModal />)
      const refineBtn = within(document.body).getByText('Refine').closest('button')!
      await user.click(refineBtn)
      const textarea = within(document.body).getByPlaceholderText('Describe your feedback...')
      await user.type(textarea, 'Please add more citations')
      const submitBtn = within(document.body).getByText('Submit Feedback').closest('button') as HTMLButtonElement
      expect(submitBtn.disabled).toBe(false)
    })

    it('calls resumeReview("refine", feedback) with trimmed feedback on submit', async () => {
      mockResumeReview.mockResolvedValue(undefined)
      const user = userEvent.setup()
      render(<ReviewModal />)
      const refineBtn = within(document.body).getByText('Refine').closest('button')!
      await user.click(refineBtn)
      const textarea = within(document.body).getByPlaceholderText('Describe your feedback...')
      await user.type(textarea, 'Add more citations')
      const submitBtn = within(document.body).getByText('Submit Feedback').closest('button')!
      await user.click(submitBtn)
      expect(mockResumeReview).toHaveBeenCalledWith('refine', 'Add more citations')
    })

    it('clears feedback and returns to main buttons after successful submit', async () => {
      mockResumeReview.mockResolvedValue(undefined)
      const user = userEvent.setup()
      render(<ReviewModal />)
      const refineBtn = within(document.body).getByText('Refine').closest('button')!
      await user.click(refineBtn)
      const textarea = within(document.body).getByPlaceholderText('Describe your feedback...')
      await user.type(textarea, 'Some feedback')
      const submitBtn = within(document.body).getByText('Submit Feedback').closest('button')!
      await user.click(submitBtn)
      // Main buttons should be visible again
      expect(within(document.body).queryByText('Re-research')).not.toBeNull()
      expect(within(document.body).queryByText('Approve')).not.toBeNull()
    })

    it('Cancel button returns to the main action buttons without calling resumeReview', async () => {
      const user = userEvent.setup()
      render(<ReviewModal />)
      const refineBtn = within(document.body).getByText('Refine').closest('button')!
      await user.click(refineBtn)
      const cancelBtn = within(document.body).getByText('Cancel').closest('button')!
      await user.click(cancelBtn)
      expect(within(document.body).queryByText('Re-research')).not.toBeNull()
      expect(within(document.body).queryByText('Approve')).not.toBeNull()
      expect(mockResumeReview).not.toHaveBeenCalled()
    })
  })

  describe('Re-research action', () => {
    it('shows the feedback textarea after clicking Re-research', async () => {
      const user = userEvent.setup()
      render(<ReviewModal />)
      const reBtn = within(document.body).getByText('Re-research').closest('button')!
      await user.click(reBtn)
      expect(within(document.body).queryByPlaceholderText('Describe your feedback...')).not.toBeNull()
    })

    it('shows "What additional research is needed?" as the label', async () => {
      const user = userEvent.setup()
      render(<ReviewModal />)
      const reBtn = within(document.body).getByText('Re-research').closest('button')!
      await user.click(reBtn)
      expect(within(document.body).queryByText('What additional research is needed?')).not.toBeNull()
    })

    it('calls resumeReview("re_research", feedback) on submit', async () => {
      mockResumeReview.mockResolvedValue(undefined)
      const user = userEvent.setup()
      render(<ReviewModal />)
      const reBtn = within(document.body).getByText('Re-research').closest('button')!
      await user.click(reBtn)
      const textarea = within(document.body).getByPlaceholderText('Describe your feedback...')
      await user.type(textarea, 'Need more papers on topic X')
      const submitBtn = within(document.body).getByText('Submit Feedback').closest('button')!
      await user.click(submitBtn)
      expect(mockResumeReview).toHaveBeenCalledWith('re_research', 'Need more papers on topic X')
    })
  })
})
