// @vitest-environment jsdom

import React from 'react'
import { render, screen, within, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi, describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest'
import { ResearchHistory } from '@/components/sidebar/research-history'
import { useChatContext } from '@/context/chat-context'

vi.mock('@/context/chat-context', () => ({
  useChatContext: vi.fn(),
}))

// @radix-ui/react-scroll-area uses ResizeObserver which is not in jsdom.
// Must be a real class so `new ResizeObserver(cb)` works.
globalThis.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

// ── Deterministic date anchor (matches format-date.test.ts) ─────────────────
// now = 2026-03-30T12:00:00Z
// conv_1 created_at 06:00Z same day → diffDays=0 → "Today"
// conv_2 created_at 29T06:00Z     → diffDays=1 → "Yesterday"
// conv_3 created_at 27T06:00Z     → diffDays=3 → "3 days ago"

const FIXED_NOW = new Date('2026-03-30T12:00:00.000Z')

beforeAll(() => {
  vi.useFakeTimers()
  vi.setSystemTime(FIXED_NOW)
})

afterAll(() => {
  vi.useRealTimers()
})

// ── Mock conversations ───────────────────────────────────────────────────────

const mockConversations = [
  {
    conversation_id: 'conv_1',
    user_query: 'What is quantum computing and how does it work in practice?',
    created_at: '2026-03-30T06:00:00Z',
    status: 'complete' as const,
  },
  {
    conversation_id: 'conv_2',
    user_query: 'Short query',
    created_at: '2026-03-29T06:00:00Z',
    status: 'in_progress' as const,
    phase: 'researching',
  },
  {
    conversation_id: 'conv_3',
    user_query: 'Another query',
    created_at: '2026-03-27T06:00:00Z',
    status: 'waiting_review' as const,
  },
]

// ── Context mock setup ───────────────────────────────────────────────────────

const mockLoadConversation = vi.fn()
const mockDeleteConversation = vi.fn()
const mockStartNewChat = vi.fn()

const baseContext = {
  conversations: mockConversations,
  threadId: null,
  loadConversation: mockLoadConversation,
  deleteConversation: mockDeleteConversation,
  startNewChat: mockStartNewChat,
}

beforeEach(() => {
  vi.mocked(useChatContext).mockReturnValue(baseContext as any)
  mockLoadConversation.mockReset()
  mockDeleteConversation.mockReset()
  mockStartNewChat.mockReset()
})

// ── Helper: find the outer .group container for a conversation ───────────────
function getConvContainer(queryText: string): HTMLElement {
  const textEl = screen.getByText(queryText)
  return textEl.closest('[class*="group"]') as HTMLElement
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe('ResearchHistory', () => {
  describe('empty state', () => {
    beforeEach(() => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        conversations: [],
      } as any)
    })

    it('renders "No research history" when conversations is empty', () => {
      render(<ResearchHistory />)
      expect(screen.queryByText('No research history')).not.toBeNull()
    })

    it('renders the "New Research" button regardless of conversation count', () => {
      render(<ResearchHistory />)
      expect(screen.queryByText('New Research')).not.toBeNull()
    })

    it('does not render any conversation list items when conversations is empty', () => {
      render(<ResearchHistory />)
      expect(screen.queryByText('What is quantum computing an...')).toBeNull()
      expect(screen.queryByText('Short query')).toBeNull()
      expect(screen.queryByText('Another query')).toBeNull()
    })
  })

  describe('conversation list rendering', () => {
    it('renders one list item per conversation', () => {
      render(<ResearchHistory />)
      // Each conversation shows its (possibly truncated) query text
      expect(screen.queryByText('What is quantum computing an...')).not.toBeNull()
      expect(screen.queryByText('Short query')).not.toBeNull()
      expect(screen.queryByText('Another query')).not.toBeNull()
    })

    it('truncates long query text to 28 characters with ellipsis', () => {
      render(<ResearchHistory />)
      // 'What is quantum computing and how does it work in practice?' slice(0,28) = 'What is quantum computing an'
      expect(screen.queryByText('What is quantum computing an...')).not.toBeNull()
      // Full text should NOT appear
      expect(
        screen.queryByText('What is quantum computing and how does it work in practice?')
      ).toBeNull()
    })

    it('does not truncate short query text', () => {
      render(<ResearchHistory />)
      expect(screen.queryByText('Short query')).not.toBeNull()
    })

    it('renders "Today" for a conversation created today', () => {
      render(<ResearchHistory />)
      expect(screen.queryByText('Today')).not.toBeNull()
    })

    it('renders "Yesterday" for a conversation created yesterday', () => {
      render(<ResearchHistory />)
      expect(screen.queryByText('Yesterday')).not.toBeNull()
    })

    it('renders "3 days ago" for a conversation created 3 days ago', () => {
      render(<ResearchHistory />)
      expect(screen.queryByText('3 days ago')).not.toBeNull()
    })
  })

  describe('status badges', () => {
    it('does not render a status badge for complete conversations', () => {
      render(<ResearchHistory />)
      const conv1Container = getConvContainer('What is quantum computing an...')
      // No badge text for complete status
      expect(within(conv1Container).queryByText('researching')).toBeNull()
      expect(within(conv1Container).queryByText('Needs Review')).toBeNull()
      expect(within(conv1Container).queryByText('In Progress')).toBeNull()
    })

    it('renders a badge with phase text for in_progress conversations', () => {
      render(<ResearchHistory />)
      // conv_2 has phase: 'researching'
      expect(screen.queryByText('researching')).not.toBeNull()
    })

    it('renders a "Needs Review" badge for waiting_review conversations', () => {
      render(<ResearchHistory />)
      expect(screen.queryByText('Needs Review')).not.toBeNull()
    })

    it('renders a left border highlight for waiting_review conversations', () => {
      render(<ResearchHistory />)
      const conv3Container = getConvContainer('Another query')
      expect(conv3Container.className).toContain('border-orange-500')
    })
  })

  describe('active conversation highlighting', () => {
    it('applies active styling to the conversation matching threadId', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        threadId: 'conv_1',
      } as any)
      render(<ResearchHistory />)
      const conv1Container = getConvContainer('What is quantum computing an...')
      expect(conv1Container.className).toContain('bg-accent')
    })

    it('does not apply active styling to non-matching conversations', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        threadId: 'conv_1',
      } as any)
      render(<ResearchHistory />)
      const conv2Container = getConvContainer('Short query')
      // conv_2 container should not have the active bg-accent/70 class
      expect(conv2Container.className).not.toContain('bg-accent/70')
    })
  })

  describe('callbacks', () => {
    // Fake timers (from outer beforeAll) cause userEvent to hang because
    // user-event's internal setTimeout delays never advance automatically
    // when advanceTimers is not set up. These tests don't assert on date
    // text so real timers are safe to use here.
    beforeEach(() => vi.useRealTimers())
    afterEach(() => {
      vi.useFakeTimers()
      vi.setSystemTime(FIXED_NOW)
    })

    it('calls loadConversation with the conversation_id when clicked', async () => {
      const user = userEvent.setup()
      render(<ResearchHistory />)
      // The inner <button> triggers loadConversation
      const conv1Button = screen.getByText('What is quantum computing an...').closest('button')!
      await user.click(conv1Button)
      expect(mockLoadConversation).toHaveBeenCalledWith('conv_1')
    })

    it('calls startNewChat when "New Research" button is clicked', async () => {
      const user = userEvent.setup()
      render(<ResearchHistory />)
      const newResearchBtn = screen.getByText('New Research').closest('button')!
      await user.click(newResearchBtn)
      expect(mockStartNewChat).toHaveBeenCalled()
    })

    it('calls deleteConversation with the conversation_id when delete is clicked', async () => {
      const user = userEvent.setup()
      render(<ResearchHistory />)
      // The DropdownMenuTrigger is the second button within the conv_1 container
      const conv1Container = getConvContainer('What is quantum computing an...')
      const buttons = within(conv1Container).getAllByRole('button')
      // buttons[0] = loadConversation button; buttons[last] = DropdownMenuTrigger
      const dropdownTrigger = buttons[buttons.length - 1]
      await user.click(dropdownTrigger)
      // DropdownMenuContent renders in a portal under document.body
      await waitFor(() =>
        expect(within(document.body).queryByText('Delete')).not.toBeNull()
      )
      const deleteItem = within(document.body).getByText('Delete')
      await user.click(deleteItem)
      expect(mockDeleteConversation).toHaveBeenCalledWith('conv_1')
    })
  })
})
