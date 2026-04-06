// @vitest-environment jsdom

import React from 'react'
import { render, screen, act } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest'
import { ReasoningBlock } from '@/components/chat/reasoning-block'
import { useChatContext } from '@/context/chat-context'

vi.mock('@/context/chat-context', () => ({
  useChatContext: vi.fn(),
}))

const idleThinking = {
  isThinking: false,
  agent: '',
  thought: '',
  step: '',
  startTime: 0,
  elapsedMs: 1500,
  history: [],
  phases: [],
  currentPhase: '',
}

const defaultProps = {
  phase: 'researching',
  isActive: false,
  isComplete: false,
  findingsCount: 5,
  tasksCount: 3,
  durationMs: 1500,
}

beforeEach(() => {
  vi.mocked(useChatContext).mockReturnValue({ thinking: idleThinking } as any)
})

afterEach(() => {
  vi.useRealTimers()
})

describe('ReasoningBlock', () => {
  describe('phase label rendering', () => {
    it('shows "Analyzing Sources" for phase="researching"', () => {
      render(<ReasoningBlock {...defaultProps} phase="researching" />)
      expect(screen.queryByText('Analyzing Sources')).not.toBeNull()
    })

    it('shows "Generating Report" for phase="generating_report"', () => {
      render(<ReasoningBlock {...defaultProps} phase="generating_report" />)
      expect(screen.queryByText('Generating Report')).not.toBeNull()
    })

    it('shows "Understanding Query" for phase="scoping"', () => {
      render(<ReasoningBlock {...defaultProps} phase="scoping" />)
      expect(screen.queryByText('Understanding Query')).not.toBeNull()
    })

    it('shows "Research Complete" for phase="complete"', () => {
      render(<ReasoningBlock {...defaultProps} phase="complete" />)
      expect(screen.queryByText('Research Complete')).not.toBeNull()
    })

    it('shows "Preparing Review" for phase="review"', () => {
      render(<ReasoningBlock {...defaultProps} phase="review" />)
      expect(screen.queryByText('Preparing Review')).not.toBeNull()
    })
  })

  describe('expand/collapse', () => {
    it('is expanded by default', () => {
      render(<ReasoningBlock {...defaultProps} />)
      expect(screen.queryByText(/Duration:/)).not.toBeNull()
    })

    it('collapses when the header button is clicked', async () => {
      const user = userEvent.setup()
      render(<ReasoningBlock {...defaultProps} />)
      expect(screen.queryByText(/Duration:/)).not.toBeNull()
      const headerBtn = screen.getByRole('button')
      await user.click(headerBtn)
      expect(screen.queryByText(/Duration:/)).toBeNull()
    })

    it('expands again when clicked while collapsed', async () => {
      const user = userEvent.setup()
      render(<ReasoningBlock {...defaultProps} />)
      const headerBtn = screen.getByRole('button')
      await user.click(headerBtn)
      expect(screen.queryByText(/Duration:/)).toBeNull()
      await user.click(headerBtn)
      expect(screen.queryByText(/Duration:/)).not.toBeNull()
    })

    it('hides phase content when collapsed', async () => {
      const user = userEvent.setup()
      render(<ReasoningBlock {...defaultProps} />)
      const headerBtn = screen.getByRole('button')
      await user.click(headerBtn)
      expect(screen.queryByText(/Tasks:/)).toBeNull()
      expect(screen.queryByText(/Sources:/)).toBeNull()
    })
  })

  describe('active state', () => {
    it('shows the elapsed time display when isActive is true', () => {
      const { container } = render(<ReasoningBlock {...defaultProps} isActive={true} />)
      expect(container.querySelector('.text-blue-400')).not.toBeNull()
    })

    it('does not show elapsed time when isActive is false', () => {
      const { container } = render(<ReasoningBlock {...defaultProps} isActive={false} />)
      expect(container.querySelector('.text-blue-400')).toBeNull()
    })

    it('shows current thought text when thinking.isThinking is true and thinking.thought is non-empty', () => {
      vi.mocked(useChatContext).mockReturnValue({
        thinking: { ...idleThinking, isThinking: true, thought: 'Analyzing paper X', startTime: Date.now() },
      } as any)
      render(<ReasoningBlock {...defaultProps} isActive={true} />)
      expect(screen.queryByText('Analyzing paper X')).not.toBeNull()
    })

    it('does not show thought text when thinking.isThinking is false', () => {
      render(<ReasoningBlock {...defaultProps} />)
      expect(screen.queryByText('Analyzing paper X')).toBeNull()
    })
  })

  describe('complete state', () => {
    it('shows CheckCircle icon when isComplete is true', () => {
      const { container } = render(<ReasoningBlock {...defaultProps} isComplete={true} />)
      // CheckCircle in lucide-react v0.400 has displayName CircleCheckBig → class lucide-circle-check-big
      expect(container.querySelector('.lucide-circle-check-big')).not.toBeNull()
      expect(container.querySelector('.lucide-brain')).toBeNull()
    })

    it('shows Brain icon when isComplete is false', () => {
      const { container } = render(<ReasoningBlock {...defaultProps} isComplete={false} />)
      expect(container.querySelector('.lucide-brain')).not.toBeNull()
      expect(container.querySelector('.lucide-circle-check-big')).toBeNull()
    })

    it('shows "Completed research with N sources" summary when isComplete', () => {
      render(<ReasoningBlock {...defaultProps} isComplete={true} isActive={false} />)
      expect(screen.queryByText(/Completed research with 5 sources/)).not.toBeNull()
    })
  })

  describe('progress stats', () => {
    it('shows "Tasks: 3" when tasksCount=3', () => {
      render(<ReasoningBlock {...defaultProps} />)
      expect(screen.queryByText(/Tasks:/)).not.toBeNull()
      expect(screen.queryByText('3')).not.toBeNull()
    })

    it('shows "Sources: 5" when findingsCount=5', () => {
      render(<ReasoningBlock {...defaultProps} />)
      expect(screen.queryByText(/Sources:/)).not.toBeNull()
      expect(screen.queryByText('5')).not.toBeNull()
    })

    it('does not show Tasks stat when tasksCount=0', () => {
      render(<ReasoningBlock {...defaultProps} tasksCount={0} />)
      expect(screen.queryByText(/Tasks:/)).toBeNull()
    })

    it('does not show Sources stat when findingsCount=0', () => {
      render(<ReasoningBlock {...defaultProps} findingsCount={0} />)
      expect(screen.queryByText(/Sources:/)).toBeNull()
    })

    it('always shows Duration stat', () => {
      render(<ReasoningBlock {...defaultProps} tasksCount={0} findingsCount={0} />)
      expect(screen.queryByText(/Duration:/)).not.toBeNull()
    })
  })

  describe('phase groups', () => {
    const thinkingWithPhases = {
      ...idleThinking,
      phases: [
        {
          id: 'phase-1',
          phase: 'investigating',
          label: 'Investigating',
          thoughts: [
            { id: 't1', agent: 'supervisor', thought: 'First thought', step: 'analyzing', phase: 'investigating', timestamp: new Date() },
            { id: 't2', agent: 'supervisor', thought: 'Second thought', step: 'planning', phase: 'investigating', timestamp: new Date() },
          ],
          startTime: Date.now(),
        },
      ],
    }

    it('renders phase group labels when thinking.phases is non-empty', () => {
      vi.mocked(useChatContext).mockReturnValue({ thinking: thinkingWithPhases } as any)
      render(<ReasoningBlock {...defaultProps} />)
      expect(screen.queryByText('Investigating')).not.toBeNull()
    })

    it('renders thought entries within each phase group', () => {
      vi.mocked(useChatContext).mockReturnValue({ thinking: thinkingWithPhases } as any)
      render(<ReasoningBlock {...defaultProps} />)
      expect(screen.queryByText('First thought')).not.toBeNull()
      expect(screen.queryByText('Second thought')).not.toBeNull()
    })

    it('highlights the last thought in the last phase group', () => {
      vi.mocked(useChatContext).mockReturnValue({ thinking: thinkingWithPhases } as any)
      render(<ReasoningBlock {...defaultProps} />)
      const lastThought = screen.getByText('Second thought')
      // Last thought in last phase should have text-zinc-300 class (highlighted)
      expect(lastThought.closest('li')?.className).toContain('text-zinc-300')
    })
  })

  describe('duration formatting', () => {
    // elapsedMs must be 0 so displayDuration falls back to durationMs prop
    const thinkingNoElapsed = { ...idleThinking, elapsedMs: 0 }

    it('shows duration in ms format when durationMs < 1000', () => {
      vi.mocked(useChatContext).mockReturnValue({ thinking: thinkingNoElapsed } as any)
      render(<ReasoningBlock {...defaultProps} durationMs={500} />)
      expect(screen.queryByText('500ms')).not.toBeNull()
    })

    it('shows duration in seconds format when durationMs >= 1000', () => {
      vi.mocked(useChatContext).mockReturnValue({ thinking: thinkingNoElapsed } as any)
      render(<ReasoningBlock {...defaultProps} durationMs={2500} />)
      expect(screen.queryByText('2.5s')).not.toBeNull()
    })
  })

  describe('auto-collapse on complete', () => {
    it('auto-collapses after 2000ms when isComplete becomes true and phases exist', () => {
      vi.useFakeTimers()
      const phases = [
        {
          id: 'p1',
          phase: 'researching',
          label: 'Researching',
          thoughts: [
            { id: 't1', agent: 'a', thought: 'thinking...', step: 'analyzing', phase: 'researching', timestamp: new Date() },
          ],
          startTime: Date.now(),
        },
      ]
      vi.mocked(useChatContext).mockReturnValue({ thinking: { ...idleThinking, phases } } as any)
      render(<ReasoningBlock {...defaultProps} isComplete={true} />)

      // Initially expanded
      expect(screen.queryByText(/Duration:/)).not.toBeNull()

      // Advance timer by 2000ms
      act(() => { vi.advanceTimersByTime(2000) })

      // Now collapsed
      expect(screen.queryByText(/Duration:/)).toBeNull()
    })
  })
})
