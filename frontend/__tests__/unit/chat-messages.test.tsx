// @vitest-environment jsdom

import React from 'react'
import { render, screen } from '@testing-library/react'
import { vi, describe, it, expect, beforeEach } from 'vitest'
import { ChatMessages } from '@/components/chat/chat-messages'
import { useChatContext } from '@/context/chat-context'

vi.mock('@/context/chat-context', () => ({
  useChatContext: vi.fn(),
}))

vi.mock('@/components/chat/message', () => ({
  Message: ({ message }: any) => <div data-testid="message">{message.content}</div>,
}))

vi.mock('@/components/chat/reasoning-block', () => ({
  ReasoningBlock: () => <div data-testid="reasoning-block" />,
}))

const baseContext = {
  messages: [],
  isStreaming: false,
  currentStreamingContent: '',
  researchProgress: {
    phase: 'idle',
    tasksCount: 0,
    findingsCount: 0,
    iterations: 0,
    phaseDurationMs: 0,
  },
  activeNode: null,
  thinking: {
    isThinking: false,
    phases: [],
    thought: '',
    step: '',
    agent: '',
    startTime: 0,
    elapsedMs: 0,
    history: [],
    currentPhase: '',
  },
}

const makeMessage = (id: string, content: string, role: 'user' | 'assistant' = 'user') => ({
  id,
  role,
  content,
  timestamp: new Date(),
})

beforeEach(() => {
  vi.mocked(useChatContext).mockReturnValue(baseContext as any)
})

describe('ChatMessages', () => {
  describe('empty state', () => {
    it('shows "Start Your Research" heading when there are no messages', () => {
      render(<ChatMessages />)
      expect(screen.queryByText('Start Your Research')).not.toBeNull()
    })

    it('shows the descriptive subtitle text in empty state', () => {
      render(<ChatMessages />)
      expect(screen.queryByText(/Ask any research question/)).not.toBeNull()
    })

    it('does not show "Start Your Research" when messages exist', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        messages: [makeMessage('1', 'What is AI?')],
      } as any)
      render(<ChatMessages />)
      expect(screen.queryByText('Start Your Research')).toBeNull()
    })
  })

  describe('message rendering', () => {
    it('renders one Message component per message', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        messages: [
          makeMessage('1', 'First question'),
          makeMessage('2', 'Second question'),
        ],
      } as any)
      render(<ChatMessages />)
      expect(screen.getAllByTestId('message').length).toBe(2)
    })

    it('renders messages in order', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        messages: [
          makeMessage('1', 'First'),
          makeMessage('2', 'Second'),
        ],
      } as any)
      render(<ChatMessages />)
      const messages = screen.getAllByTestId('message')
      expect(messages[0].textContent).toBe('First')
      expect(messages[1].textContent).toBe('Second')
    })

    it('renders a streaming message when currentStreamingContent is non-empty and phase is generating_report', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        messages: [makeMessage('1', 'question')],
        currentStreamingContent: 'Streaming report...',
        researchProgress: { ...baseContext.researchProgress, phase: 'generating_report' },
      } as any)
      render(<ChatMessages />)
      expect(screen.queryByText('Streaming report...')).not.toBeNull()
    })

    it('does not render a streaming message when currentStreamingContent is empty', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        messages: [makeMessage('1', 'question')],
        currentStreamingContent: '',
      } as any)
      render(<ChatMessages />)
      // Only the one regular message is rendered
      expect(screen.getAllByTestId('message').length).toBe(1)
    })
  })

  describe('typing indicator', () => {
    it('shows "Thinking..." when isStreaming is true and no content and no reasoning', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        messages: [makeMessage('1', 'question')],
        isStreaming: true,
        currentStreamingContent: '',
        researchProgress: { ...baseContext.researchProgress, phase: 'idle' },
        thinking: { ...baseContext.thinking, phases: [] },
      } as any)
      render(<ChatMessages />)
      expect(screen.queryByText('Thinking...')).not.toBeNull()
    })

    it('does not show "Thinking..." when isStreaming is false', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        messages: [makeMessage('1', 'question')],
        isStreaming: false,
      } as any)
      render(<ChatMessages />)
      expect(screen.queryByText('Thinking...')).toBeNull()
    })

    it('does not show "Thinking..." when currentStreamingContent is non-empty', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        messages: [makeMessage('1', 'question')],
        isStreaming: true,
        currentStreamingContent: 'Some content',
      } as any)
      render(<ChatMessages />)
      expect(screen.queryByText('Thinking...')).toBeNull()
    })
  })

  describe('ReasoningBlock insertion', () => {
    it('inserts ReasoningBlock after a brief message when showReasoning is true', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        messages: [makeMessage('1', '### Research Brief Created\nSome brief content', 'assistant')],
        researchProgress: { ...baseContext.researchProgress, phase: 'researching', tasksCount: 3 },
      } as any)
      render(<ChatMessages />)
      expect(screen.queryByTestId('reasoning-block')).not.toBeNull()
    })

    it('does not insert ReasoningBlock after a non-brief message', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        messages: [makeMessage('1', 'Regular assistant response', 'assistant')],
        researchProgress: { ...baseContext.researchProgress, phase: 'idle', tasksCount: 0 },
      } as any)
      render(<ChatMessages />)
      expect(screen.queryByTestId('reasoning-block')).toBeNull()
    })

    it('shows fallback ReasoningBlock at end when showReasoning is true but no brief message exists', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        messages: [makeMessage('1', 'A question', 'user')],
        researchProgress: { ...baseContext.researchProgress, phase: 'researching', tasksCount: 2 },
      } as any)
      render(<ChatMessages />)
      expect(screen.queryByTestId('reasoning-block')).not.toBeNull()
    })

    it('does not show fallback ReasoningBlock when phases empty, not researching, not streaming-scoping', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        messages: [makeMessage('1', 'question')],
        isStreaming: false,
        researchProgress: { ...baseContext.researchProgress, phase: 'idle', tasksCount: 0 },
        thinking: { ...baseContext.thinking, phases: [] },
      } as any)
      render(<ChatMessages />)
      expect(screen.queryByTestId('reasoning-block')).toBeNull()
    })
  })

  describe('showReasoning conditions', () => {
    it('showReasoning is true when thinking.phases is non-empty', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        messages: [makeMessage('1', 'question')],
        thinking: {
          ...baseContext.thinking,
          phases: [{ id: 'p1', phase: 'researching', label: 'R', thoughts: [], startTime: 0 }],
        },
      } as any)
      render(<ChatMessages />)
      expect(screen.queryByTestId('reasoning-block')).not.toBeNull()
    })

    it('showReasoning is true when phase=researching and tasksCount>0', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        messages: [makeMessage('1', 'question')],
        researchProgress: { ...baseContext.researchProgress, phase: 'researching', tasksCount: 1 },
      } as any)
      render(<ChatMessages />)
      expect(screen.queryByTestId('reasoning-block')).not.toBeNull()
    })

    it('showReasoning is true when isStreaming and phase=scoping', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        messages: [makeMessage('1', 'question')],
        isStreaming: true,
        researchProgress: { ...baseContext.researchProgress, phase: 'scoping' },
      } as any)
      render(<ChatMessages />)
      expect(screen.queryByTestId('reasoning-block')).not.toBeNull()
    })

    it('showReasoning is false when phases empty, not researching, not streaming-scoping', () => {
      vi.mocked(useChatContext).mockReturnValue({
        ...baseContext,
        messages: [makeMessage('1', 'question')],
        isStreaming: false,
        researchProgress: { ...baseContext.researchProgress, phase: 'complete', tasksCount: 0 },
        thinking: { ...baseContext.thinking, phases: [] },
      } as any)
      render(<ChatMessages />)
      expect(screen.queryByTestId('reasoning-block')).toBeNull()
    })
  })
})
