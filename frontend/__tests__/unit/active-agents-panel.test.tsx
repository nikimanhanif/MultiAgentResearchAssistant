// @vitest-environment jsdom

import React from 'react'
import { render, screen } from '@testing-library/react'
import { vi, describe, it, expect, beforeEach } from 'vitest'
import { ActiveAgentsPanel } from '@/components/agents/active-agents-panel'
import { useChatContext } from '@/context/chat-context'

vi.mock('@/context/chat-context', () => ({
  useChatContext: vi.fn(),
}))

const baseContext = {
  researchProgress: {
    phase: 'idle',
    tasksCount: 0,
    findingsCount: 0,
    iterations: 0,
    phaseDurationMs: 0,
  },
  isStreaming: false,
  activeNode: null,
  isReportStreaming: false,
}

beforeEach(() => {
  vi.mocked(useChatContext).mockReturnValue(baseContext as any)
})

describe('ActiveAgentsPanel', () => {
  describe('rendering', () => {
    it('renders all four agent cards: Scope Agent, Research, Report Agent, Reviewer', () => {
      render(<ActiveAgentsPanel />)
      expect(screen.queryByText('Scope Agent')).not.toBeNull()
      expect(screen.queryByText('Research')).not.toBeNull()
      expect(screen.queryByText('Report Agent')).not.toBeNull()
      expect(screen.queryByText('Reviewer')).not.toBeNull()
    })

    it('renders "Active Agents" heading', () => {
      render(<ActiveAgentsPanel />)
      expect(screen.queryByText('Active Agents')).not.toBeNull()
    })

    it('shows "Live" indicator when isStreaming is true', () => {
      vi.mocked(useChatContext).mockReturnValue({ ...baseContext, isStreaming: true } as any)
      render(<ActiveAgentsPanel />)
      expect(screen.queryByText('Live')).not.toBeNull()
    })

    it('does not show "Live" indicator when isStreaming is false', () => {
      render(<ActiveAgentsPanel />)
      expect(screen.queryByText('Live')).toBeNull()
    })

    it('shows progress stats section when tasksCount > 0', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        researchProgress: { ...baseContext.researchProgress, tasksCount: 5 },
      } as any)
      render(<ActiveAgentsPanel />)
      expect(screen.queryByText('Tasks')).not.toBeNull()
    })

    it('shows progress stats section when findingsCount > 0', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        researchProgress: { ...baseContext.researchProgress, findingsCount: 3 },
      } as any)
      render(<ActiveAgentsPanel />)
      expect(screen.queryByText('Findings')).not.toBeNull()
    })

    it('does not show progress stats when both counts are 0', () => {
      render(<ActiveAgentsPanel />)
      expect(screen.queryByText('Tasks')).toBeNull()
      expect(screen.queryByText('Findings')).toBeNull()
    })
  })

  describe('agent status — isAgentActive logic', () => {
    it('marks Report Agent as active when isReportStreaming is true', () => {
      vi.mocked(useChatContext).mockReturnValue({ ...baseContext, isReportStreaming: true } as any)
      render(<ActiveAgentsPanel />)
      // Find the Report Agent card's status text
      const reportAgentName = screen.getByText('Report Agent')
      const card = reportAgentName.closest('[class*="rounded-lg"]')!
      expect(card.textContent).toContain('In Progress')
    })

    it('marks Research agent as active when phase is "researching"', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        researchProgress: { ...baseContext.researchProgress, phase: 'researching' },
      } as any)
      render(<ActiveAgentsPanel />)
      const researchName = screen.getByText('Research')
      const card = researchName.closest('[class*="rounded-lg"]')!
      expect(card.textContent).toContain('In Progress')
    })

    it('marks Scope Agent as active when phase is "scoping" and no activeNode', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        researchProgress: { ...baseContext.researchProgress, phase: 'scoping' },
        activeNode: null,
      } as any)
      render(<ActiveAgentsPanel />)
      const scopeName = screen.getByText('Scope Agent')
      const card = scopeName.closest('[class*="rounded-lg"]')!
      expect(card.textContent).toContain('In Progress')
    })

    it('marks agent matching activeNode as active', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        activeNode: 'scope',
      } as any)
      render(<ActiveAgentsPanel />)
      const scopeName = screen.getByText('Scope Agent')
      const card = scopeName.closest('[class*="rounded-lg"]')!
      expect(card.textContent).toContain('In Progress')
    })

    it('marks supervisor as active when activeNode is "sub_agent"', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        activeNode: 'sub_agent',
      } as any)
      render(<ActiveAgentsPanel />)
      const researchName = screen.getByText('Research')
      const card = researchName.closest('[class*="rounded-lg"]')!
      expect(card.textContent).toContain('In Progress')
    })

    it('no agent is active when phase is "idle" and no activeNode', () => {
      render(<ActiveAgentsPanel />)
      const allCards = screen.getAllByText('Pending')
      // All 4 agents should be Pending when idle
      expect(allCards.length).toBe(4)
    })
  })

  describe('agent status — isAgentComplete logic', () => {
    it('no agent is complete when phase is "idle"', () => {
      render(<ActiveAgentsPanel />)
      expect(screen.queryByText('Completed')).toBeNull()
    })

    it('marks Scope Agent as complete when phase is "researching"', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        researchProgress: { ...baseContext.researchProgress, phase: 'researching' },
      } as any)
      render(<ActiveAgentsPanel />)
      const scopeName = screen.getByText('Scope Agent')
      const card = scopeName.closest('[class*="rounded-lg"]')!
      expect(card.textContent).toContain('Completed')
    })

    it('marks Scope Agent and Research as complete when phase is "generating_report"', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        researchProgress: { ...baseContext.researchProgress, phase: 'generating_report' },
      } as any)
      render(<ActiveAgentsPanel />)
      const completedAgents = screen.getAllByText('Completed')
      expect(completedAgents.length).toBe(2)
    })

    it('marks Scope Agent and Research as complete when isReportStreaming is true', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        isReportStreaming: true,
        // phase must be non-idle so isAgentComplete's idle guard doesn't short-circuit
        researchProgress: { ...baseContext.researchProgress, phase: 'generating_report' },
      } as any)
      render(<ActiveAgentsPanel />)
      const scopeName = screen.getByText('Scope Agent')
      const scopeCard = scopeName.closest('[class*="rounded-lg"]')!
      expect(scopeCard.textContent).toContain('Completed')
      const researchName = screen.getByText('Research')
      const researchCard = researchName.closest('[class*="rounded-lg"]')!
      expect(researchCard.textContent).toContain('Completed')
    })

    it('all agents are complete when phase is "complete"', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        researchProgress: { ...baseContext.researchProgress, phase: 'complete' },
      } as any)
      render(<ActiveAgentsPanel />)
      const completedAgents = screen.getAllByText('Completed')
      expect(completedAgents.length).toBe(4)
    })
  })

  describe('progress stats display', () => {
    it('displays the correct tasksCount number', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        researchProgress: { ...baseContext.researchProgress, tasksCount: 7 },
      } as any)
      render(<ActiveAgentsPanel />)
      expect(screen.queryByText('7')).not.toBeNull()
    })

    it('displays the correct findingsCount number', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        researchProgress: { ...baseContext.researchProgress, findingsCount: 12 },
      } as any)
      render(<ActiveAgentsPanel />)
      expect(screen.queryByText('12')).not.toBeNull()
    })
  })
})
