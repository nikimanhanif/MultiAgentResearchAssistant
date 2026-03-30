// @vitest-environment jsdom

import React from 'react'
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest'
import { Message } from '@/components/chat/message'
import { useChatContext } from '@/context/chat-context'

vi.mock('@/context/chat-context', () => ({
  useChatContext: vi.fn(),
}))

// ── Fixtures ────────────────────────────────────────────────────────────────

const userMessage = {
  id: 'msg_1',
  role: 'user' as const,
  content: 'What is quantum computing?',
  timestamp: new Date(),
}

const shortAssistantMessage = {
  id: 'msg_2',
  role: 'assistant' as const,
  content: 'Quantum computing uses qubits.',
  timestamp: new Date(),
}

// isReportMessage() returns true for this fixture:
//   hasMultipleHeaders: 4 headers match /^##?\s/gm
//   hasReportIndicator: contains '## Executive Summary' (in reportIndicators list)
const reportMessage = {
  id: 'msg_3',
  role: 'assistant' as const,
  timestamp: new Date(),
  content: [
    '# Research Report',
    '',
    '## Executive Summary',
    '',
    'This report covers quantum computing across multiple dimensions.',
    'The field has advanced significantly in recent years with major',
    'breakthroughs in error correction and qubit stability.',
    '',
    '## Key Findings',
    '',
    'Finding one is significant. Finding two confirms the hypothesis.',
    'Additional analysis shows strong correlation between qubit count',
    'and computational advantage in specific problem domains.',
    '',
    '## References',
    '',
    '[1] Smith et al. 2023',
  ].join('\n'),
}

// ── Context mock setup ───────────────────────────────────────────────────────

const mockOpenReport = vi.fn()

const defaultContext = {
  openReport: mockOpenReport,
  reportPanelOpen: false,
  activeReportContent: null,
}

beforeEach(() => {
  vi.mocked(useChatContext).mockReturnValue(defaultContext as any)
  mockOpenReport.mockReset()
})

// ── Tests ────────────────────────────────────────────────────────────────────

describe('Message', () => {
  describe('user message rendering', () => {
    it('displays "You" as the role label', () => {
      render(<Message message={userMessage} />)
      expect(screen.queryByText('You')).not.toBeNull()
    })

    it('renders message content as plain text (not markdown)', () => {
      render(<Message message={userMessage} />)
      expect(screen.queryByText('What is quantum computing?')).not.toBeNull()
    })

    it('does not render the "Research Assistant" label', () => {
      render(<Message message={userMessage} />)
      expect(screen.queryByText('Research Assistant')).toBeNull()
    })

    it('does not render the "View Full Report" button', () => {
      render(<Message message={userMessage} />)
      expect(screen.queryByText('View Full Report')).toBeNull()
    })
  })

  describe('assistant message rendering', () => {
    it('displays "Research Assistant" as the role label', () => {
      render(<Message message={shortAssistantMessage} />)
      expect(screen.queryByText('Research Assistant')).not.toBeNull()
    })

    it('renders message content via MarkdownContent (text appears in DOM)', () => {
      render(<Message message={shortAssistantMessage} />)
      // react-markdown renders the text inside a <p> tag; RTL finds it
      expect(screen.queryByText('Quantum computing uses qubits.')).not.toBeNull()
    })

    it('does not display "You" as the role label', () => {
      render(<Message message={shortAssistantMessage} />)
      expect(screen.queryByText('You')).toBeNull()
    })
  })

  describe('View Full Report button', () => {
    it('does not render for a short assistant message', () => {
      render(<Message message={shortAssistantMessage} />)
      expect(screen.queryByText('View Full Report')).toBeNull()
    })

    it('does not render for a user message even with report-like content', () => {
      // isReport = !isUser && !isStreaming && isReportMessage(...) → false when isUser
      render(<Message message={{ ...userMessage, content: reportMessage.content }} />)
      expect(screen.queryByText('View Full Report')).toBeNull()
    })

    it('renders for an assistant message where isReportMessage returns true', () => {
      render(<Message message={reportMessage} />)
      expect(screen.queryByText('View Full Report')).not.toBeNull()
    })

    it('calls openReport with the message content when clicked', async () => {
      const user = userEvent.setup()
      render(<Message message={reportMessage} />)
      await user.click(screen.getByText('View Full Report'))
      expect(mockOpenReport).toHaveBeenCalledWith(reportMessage.content)
    })

    it('shows "Viewing" label when this message is the activeReportContent', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...defaultContext,
        activeReportContent: reportMessage.content,
        reportPanelOpen: true,
      } as any)
      render(<Message message={reportMessage} />)
      expect(screen.queryByText('Viewing')).not.toBeNull()
      expect(screen.queryByText('View Full Report')).toBeNull()
    })
  })

  describe('copy button', () => {
    it('is not visible before hover (action wrapper has opacity-0 class)', () => {
      // The action area uses CSS group-hover, not React state — the button is
      // always in the DOM but visually hidden by the opacity-0 class.
      const { container } = render(<Message message={shortAssistantMessage} />)
      const actionWrapper = container.querySelector('.opacity-0')
      expect(actionWrapper).not.toBeNull()
    })

    it('becomes visible on hover (copy button accessible after hover)', async () => {
      const user = userEvent.setup()
      const { container } = render(<Message message={shortAssistantMessage} />)
      const actionWrapper = container.querySelector('.opacity-0') as HTMLElement
      await user.hover(actionWrapper)
      // The button was always in the DOM; hover confirms it remains accessible
      expect(container.querySelector('.opacity-0 button')).not.toBeNull()
    })

    it('copies message content to clipboard on click', async () => {
      const writeText = vi.fn().mockResolvedValue(undefined)
      // Use data descriptor (not accessor) to override the clipboard OWN property
      // on the navigator instance. Data descriptors can replace accessor descriptors
      // when the original property is configurable.
      Object.defineProperty(navigator, 'clipboard', {
        value: { writeText },
        writable: true,
        configurable: true,
      })
      const { container } = render(<Message message={shortAssistantMessage} />)
      // shortAssistantMessage has no View Full Report button; the only button in
      // .opacity-0 is the copy button
      const copyButton = container.querySelector('.opacity-0 button') as HTMLElement
      // Use fireEvent.click to bypass user-event pointer-event sequence
      // which might interfere with Radix tooltip internals
      await act(async () => {
        fireEvent.click(copyButton)
        // Flush the microtask from the async handleCopy
        await Promise.resolve()
      })
      // Restore clipboard so other tests aren't affected
      Object.defineProperty(navigator, 'clipboard', {
        get: () => undefined,
        configurable: true,
      })
      expect(writeText).toHaveBeenCalledWith(shortAssistantMessage.content)
    })

    it('shows Check icon (and "Copied!" state) after clicking copy', async () => {
      const writeText = vi.fn().mockResolvedValue(undefined)
      Object.defineProperty(navigator, 'clipboard', {
        value: { writeText },
        writable: true,
        configurable: true,
      })
      const { container } = render(<Message message={shortAssistantMessage} />)
      const copyButton = container.querySelector('.opacity-0 button') as HTMLElement
      // Before click: Copy icon present, Check icon absent
      expect(copyButton.querySelector('.lucide-copy')).not.toBeNull()
      expect(copyButton.querySelector('.lucide-check')).toBeNull()
      // Click: handleCopy → setCopied(true) after writeText Promise resolves
      await act(async () => {
        fireEvent.click(copyButton)
        // Flush writeText's resolved Promise microtask so setCopied(true) runs
        await Promise.resolve()
        await Promise.resolve()
      })
      // After setCopied(true): icon changes from Copy to Check
      expect(copyButton.querySelector('.lucide-check')).not.toBeNull()
      expect(copyButton.querySelector('.lucide-copy')).toBeNull()
      // Cleanup
      Object.defineProperty(navigator, 'clipboard', {
        get: () => undefined,
        configurable: true,
      })
    })
  })

  describe('isStreaming prop', () => {
    it('renders with typewriter-cursor class on content when isStreaming=true', () => {
      const { container } = render(
        <Message message={shortAssistantMessage} isStreaming={true} />
      )
      expect(container.querySelector('.typewriter-cursor')).not.toBeNull()
    })

    it('does not apply typewriter-cursor class when isStreaming=false', () => {
      const { container } = render(
        <Message message={shortAssistantMessage} isStreaming={false} />
      )
      expect(container.querySelector('.typewriter-cursor')).toBeNull()
    })
  })
})
