// @vitest-environment jsdom

import React from 'react'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi, describe, it, expect, beforeEach } from 'vitest'
import { ResearchCanvas } from '@/components/chat/research-canvas'
import { useChatContext } from '@/context/chat-context'

vi.mock('@/context/chat-context', () => ({
  useChatContext: vi.fn(),
}))

vi.mock('@/components/chat/export-menu', () => ({
  ExportMenu: () => null,
}))

vi.mock('@/components/chat/markdown-content', () => ({
  MarkdownContent: ({ content }: any) => <div data-testid="markdown-content">{content}</div>,
}))

const mockCloseReport = vi.fn()
const mockToggleFocusMode = vi.fn()

const defaultContext = {
  activeReportContent: 'Report content here',
  focusMode: false,
  closeReport: mockCloseReport,
  toggleFocusMode: mockToggleFocusMode,
}

beforeEach(() => {
  vi.mocked(useChatContext).mockReturnValue(defaultContext as any)
  mockCloseReport.mockReset()
  mockToggleFocusMode.mockReset()
})

describe('ResearchCanvas', () => {
  describe('null guard', () => {
    it('renders nothing when activeReportContent is null', () => {
      vi.mocked(useChatContext).mockReturnValue({ ...defaultContext, activeReportContent: null } as any)
      const { container } = render(<ResearchCanvas />)
      expect(container.firstChild).toBeNull()
    })
  })

  describe('rendering with content', () => {
    it('renders "Research Report" heading when activeReportContent is set', () => {
      render(<ResearchCanvas />)
      expect(screen.queryByText('Research Report')).not.toBeNull()
    })

    it('renders the report content via MarkdownContent', () => {
      render(<ResearchCanvas />)
      expect(screen.queryByTestId('markdown-content')).not.toBeNull()
      expect(screen.queryByText('Report content here')).not.toBeNull()
    })

    it('shows Minimize2 icon when focusMode is true', () => {
      vi.mocked(useChatContext).mockReturnValue({ ...defaultContext, focusMode: true } as any)
      const { container } = render(<ResearchCanvas />)
      // lucide-react v0.400: Minimize2 class = lucide-minimize2 (no hyphen before digit)
      expect(container.querySelector('.lucide-minimize2')).not.toBeNull()
      expect(container.querySelector('.lucide-maximize2')).toBeNull()
    })

    it('shows Maximize2 icon when focusMode is false', () => {
      const { container } = render(<ResearchCanvas />)
      expect(container.querySelector('.lucide-maximize2')).not.toBeNull()
      expect(container.querySelector('.lucide-minimize2')).toBeNull()
    })
  })

  describe('controls', () => {
    it('calls closeReport when the close button is clicked', async () => {
      const user = userEvent.setup()
      render(<ResearchCanvas />)
      const closeButton = screen.getByTitle('Close Panel')
      await user.click(closeButton)
      expect(mockCloseReport).toHaveBeenCalledTimes(1)
    })

    it('calls toggleFocusMode when the focus mode button is clicked', async () => {
      const user = userEvent.setup()
      render(<ResearchCanvas />)
      const focusButton = screen.getByTitle('Enter Focus Mode')
      await user.click(focusButton)
      expect(mockToggleFocusMode).toHaveBeenCalledTimes(1)
    })
  })
})
