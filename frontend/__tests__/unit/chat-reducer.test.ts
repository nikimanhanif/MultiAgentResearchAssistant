import { describe, it, expect, beforeAll, afterAll, vi } from 'vitest'
import { chatReducer, initialState } from '@/context/chat-context'
import type { ChatAction } from '@/context/chat-context'
import type { ChatState, Message, ThinkingState } from '@/types/chat'

// Freeze time for deterministic IDs and timing
const FIXED_NOW = new Date('2026-01-01T12:00:00.000Z')
const FIXED_MS = FIXED_NOW.getTime()

beforeAll(() => {
  vi.useFakeTimers()
  vi.setSystemTime(FIXED_NOW)
})

afterAll(() => {
  vi.useRealTimers()
})

// Helper: produce a state with optional overrides
function makeState(overrides: Partial<ChatState> = {}): ChatState {
  return { ...initialState, ...overrides }
}

// Helper: minimal valid message
function makeMessage(overrides: Partial<Message> = {}): Message {
  return {
    id: 'msg_test',
    role: 'assistant',
    content: 'hello',
    timestamp: new Date(),
    ...overrides
  }
}

describe('chatReducer', () => {
  describe('ADD_MESSAGE', () => {
    it('appends message to empty messages array', () => {
      const msg = makeMessage()
      const next = chatReducer(initialState, { type: 'ADD_MESSAGE', payload: msg })
      expect(next.messages).toHaveLength(1)
      expect(next.messages[0]).toBe(msg)
    })

    it('appends message to non-empty messages array preserving existing items', () => {
      const existing = makeMessage({ id: 'existing' })
      const state = makeState({ messages: [existing] })
      const newMsg = makeMessage({ id: 'new' })
      const next = chatReducer(state, { type: 'ADD_MESSAGE', payload: newMsg })
      expect(next.messages).toHaveLength(2)
      expect(next.messages[0]).toBe(existing)
      expect(next.messages[1]).toBe(newMsg)
    })

    it('returns a new array reference (immutability)', () => {
      const state = makeState({ messages: [makeMessage()] })
      const next = chatReducer(state, { type: 'ADD_MESSAGE', payload: makeMessage({ id: 'new' }) })
      expect(next.messages).not.toBe(state.messages)
    })

    it('does not mutate other state keys', () => {
      const state = makeState({ isStreaming: true, error: 'oops' })
      const next = chatReducer(state, { type: 'ADD_MESSAGE', payload: makeMessage() })
      expect(next.isStreaming).toBe(true)
      expect(next.error).toBe('oops')
    })
  })

  describe('UPDATE_LAST_MESSAGE', () => {
    it('replaces content of the last message', () => {
      const state = makeState({ messages: [makeMessage({ content: 'old' })] })
      const next = chatReducer(state, { type: 'UPDATE_LAST_MESSAGE', payload: 'new' })
      expect(next.messages[0].content).toBe('new')
    })

    it('preserves all other fields on the last message (id, role, timestamp)', () => {
      const msg = makeMessage({ id: 'keep-me', role: 'user', content: 'old' })
      const state = makeState({ messages: [msg] })
      const next = chatReducer(state, { type: 'UPDATE_LAST_MESSAGE', payload: 'new' })
      expect(next.messages[0].id).toBe('keep-me')
      expect(next.messages[0].role).toBe('user')
    })

    it('does not affect messages other than the last', () => {
      const first = makeMessage({ id: 'first', content: 'untouched' })
      const last  = makeMessage({ id: 'last',  content: 'old' })
      const state = makeState({ messages: [first, last] })
      const next = chatReducer(state, { type: 'UPDATE_LAST_MESSAGE', payload: 'new' })
      expect(next.messages[0].content).toBe('untouched')
      expect(next.messages[1].content).toBe('new')
    })

    it('returns state unchanged when messages is empty (no-op)', () => {
      const next = chatReducer(initialState, { type: 'UPDATE_LAST_MESSAGE', payload: 'new' })
      expect(next.messages).toHaveLength(0)
    })
  })

  describe('SET_STREAMING', () => {
    it('sets isStreaming to true', () => {
      const next = chatReducer(initialState, { type: 'SET_STREAMING', payload: true })
      expect(next.isStreaming).toBe(true)
    })

    it('sets isStreaming to false', () => {
      const state = makeState({ isStreaming: true })
      const next = chatReducer(state, { type: 'SET_STREAMING', payload: false })
      expect(next.isStreaming).toBe(false)
    })
  })

  describe('SET_STREAMING_CONTENT', () => {
    it('sets currentStreamingContent to the given string', () => {
      const next = chatReducer(initialState, { type: 'SET_STREAMING_CONTENT', payload: 'hello' })
      expect(next.currentStreamingContent).toBe('hello')
    })

    it('sets currentStreamingContent to empty string', () => {
      const state = makeState({ currentStreamingContent: 'stuff' })
      const next = chatReducer(state, { type: 'SET_STREAMING_CONTENT', payload: '' })
      expect(next.currentStreamingContent).toBe('')
    })
  })

  describe('APPEND_STREAMING_CONTENT', () => {
    it('concatenates payload onto existing currentStreamingContent', () => {
      const state = makeState({ currentStreamingContent: 'hello ' })
      const next = chatReducer(state, { type: 'APPEND_STREAMING_CONTENT', payload: 'world' })
      expect(next.currentStreamingContent).toBe('hello world')
    })

    it('appends to empty string (initial accumulation)', () => {
      const next = chatReducer(initialState, { type: 'APPEND_STREAMING_CONTENT', payload: 'first' })
      expect(next.currentStreamingContent).toBe('first')
    })

    it('sequential appends produce correct final string', () => {
      let state = initialState
      for (const chunk of ['a', 'b', 'c']) {
        state = chatReducer(state, { type: 'APPEND_STREAMING_CONTENT', payload: chunk })
      }
      expect(state.currentStreamingContent).toBe('abc')
    })

    it('does not mutate other state keys', () => {
      const state = makeState({ isStreaming: true })
      const next = chatReducer(state, { type: 'APPEND_STREAMING_CONTENT', payload: 'x' })
      expect(next.isStreaming).toBe(true)
    })
  })

  describe('FINALIZE_STREAMING_MESSAGE', () => {
    it('creates a new assistant message with content from currentStreamingContent', () => {
      const state = makeState({ currentStreamingContent: 'final text', isStreaming: true })
      const next = chatReducer(state, { type: 'FINALIZE_STREAMING_MESSAGE' })
      expect(next.messages).toHaveLength(1)
      expect(next.messages[0].content).toBe('final text')
      expect(next.messages[0].role).toBe('assistant')
    })

    it('appends the new message to existing messages', () => {
      const existing = makeMessage({ id: 'existing' })
      const state = makeState({ messages: [existing], currentStreamingContent: 'new' })
      const next = chatReducer(state, { type: 'FINALIZE_STREAMING_MESSAGE' })
      expect(next.messages).toHaveLength(2)
      expect(next.messages[0]).toBe(existing)
    })

    it('clears currentStreamingContent to empty string', () => {
      const state = makeState({ currentStreamingContent: 'text' })
      const next = chatReducer(state, { type: 'FINALIZE_STREAMING_MESSAGE' })
      expect(next.currentStreamingContent).toBe('')
    })

    it('sets isStreaming to false', () => {
      const state = makeState({ currentStreamingContent: 'text', isStreaming: true })
      const next = chatReducer(state, { type: 'FINALIZE_STREAMING_MESSAGE' })
      expect(next.isStreaming).toBe(false)
    })

    it('generated message id starts with "msg_"', () => {
      const state = makeState({ currentStreamingContent: 'text' })
      const next = chatReducer(state, { type: 'FINALIZE_STREAMING_MESSAGE' })
      expect(next.messages[0].id).toMatch(/^msg_/)
    })

    it('does not add a message when currentStreamingContent is empty string', () => {
      const next = chatReducer(initialState, { type: 'FINALIZE_STREAMING_MESSAGE' })
      expect(next.messages).toHaveLength(0)
    })

    it('returns a new messages array reference', () => {
      const state = makeState({ currentStreamingContent: 'text' })
      const next = chatReducer(state, { type: 'FINALIZE_STREAMING_MESSAGE' })
      expect(next.messages).not.toBe(state.messages)
    })
  })

  describe('SET_PROGRESS', () => {
    it('merges partial payload into existing researchProgress', () => {
      const state = makeState({ researchProgress: { ...initialState.researchProgress, tasksCount: 5 } })
      const next = chatReducer(state, { type: 'SET_PROGRESS', payload: { findingsCount: 3 } })
      expect(next.researchProgress.tasksCount).toBe(5)
      expect(next.researchProgress.findingsCount).toBe(3)
    })

    it('preserves fields not in the payload', () => {
      const state = makeState({ researchProgress: { ...initialState.researchProgress, iterations: 2 } })
      const next = chatReducer(state, { type: 'SET_PROGRESS', payload: { phase: 'researching' } })
      expect(next.researchProgress.iterations).toBe(2)
    })

    it('overwrites all fields when full payload is provided', () => {
      const full = { phase: 'complete' as const, tasksCount: 10, findingsCount: 20, iterations: 3, phaseDurationMs: 5000 }
      const next = chatReducer(initialState, { type: 'SET_PROGRESS', payload: full })
      expect(next.researchProgress).toEqual(full)
    })

    it('does not mutate the researchProgress object reference', () => {
      const state = makeState()
      const next = chatReducer(state, { type: 'SET_PROGRESS', payload: { tasksCount: 1 } })
      expect(next.researchProgress).not.toBe(state.researchProgress)
    })
  })

  describe('SET_BRIEF', () => {
    it('sets researchBrief to the payload value', () => {
      const brief = { scope: 'AI safety', subTopics: ['alignment', 'robustness'] }
      const next = chatReducer(initialState, { type: 'SET_BRIEF', payload: brief })
      expect(next.researchBrief).toEqual(brief)
    })

    it('does not mutate other state keys', () => {
      const state = makeState({ isStreaming: true })
      const next = chatReducer(state, { type: 'SET_BRIEF', payload: { scope: 'x', subTopics: [] } })
      expect(next.isStreaming).toBe(true)
    })
  })

  describe('SET_REVIEW_REQUEST', () => {
    it('sets reviewRequest to a ReviewRequest object', () => {
      const req = { report: 'Report text here', pending: true }
      const next = chatReducer(initialState, { type: 'SET_REVIEW_REQUEST', payload: req })
      expect(next.reviewRequest).toEqual(req)
    })

    it('sets reviewRequest to null (clearing it)', () => {
      const state = makeState({ reviewRequest: { report: 'r', pending: true } })
      const next = chatReducer(state, { type: 'SET_REVIEW_REQUEST', payload: null })
      expect(next.reviewRequest).toBeNull()
    })
  })

  describe('SET_CONVERSATIONS', () => {
    it('sets conversations array', () => {
      const convs = [{ conversation_id: 'c1', user_query: 'q', created_at: '2026-01-01', status: 'complete' as const }]
      const next = chatReducer(initialState, { type: 'SET_CONVERSATIONS', payload: convs })
      expect(next.conversations).toEqual(convs)
    })

    it('replaces an existing conversations array', () => {
      const old = [{ conversation_id: 'old', user_query: 'q', created_at: '2026-01-01', status: 'complete' as const }]
      const state = makeState({ conversations: old })
      const next = chatReducer(state, { type: 'SET_CONVERSATIONS', payload: [] })
      expect(next.conversations).toHaveLength(0)
    })
  })

  describe('SET_ERROR', () => {
    it('sets the error message', () => {
      const next = chatReducer(initialState, { type: 'SET_ERROR', payload: 'Something went wrong' })
      expect(next.error).toBe('Something went wrong')
    })

    it('forces isStreaming to false regardless of prior value', () => {
      const state = makeState({ isStreaming: true })
      const next = chatReducer(state, { type: 'SET_ERROR', payload: 'err' })
      expect(next.isStreaming).toBe(false)
    })

    it('forces activeNode to null regardless of prior value', () => {
      const state = makeState({ activeNode: 'scope' })
      const next = chatReducer(state, { type: 'SET_ERROR', payload: 'err' })
      expect(next.activeNode).toBeNull()
    })

    it('sets error to null when payload is null', () => {
      const state = makeState({ error: 'old error' })
      const next = chatReducer(state, { type: 'SET_ERROR', payload: null })
      expect(next.error).toBeNull()
    })
  })

  describe('RESET_CHAT', () => {
    const populatedState = makeState({
      messages: [makeMessage()],
      threadId: 'thread_123',
      userId: 'user_abc',
      isStreaming: true,
      currentStreamingContent: 'partial',
      conversations: [{ conversation_id: 'c1', user_query: 'q', created_at: '2026-01-01', status: 'complete' as const }],
      error: 'oops',
      activeNode: 'scope',
    })

    it('resets messages to empty array', () => {
      localStorage.setItem('active_thread_id', 'thread_123')
      const next = chatReducer(populatedState, { type: 'RESET_CHAT' })
      expect(next.messages).toHaveLength(0)
    })

    it('resets threadId to null', () => {
      const next = chatReducer(populatedState, { type: 'RESET_CHAT' })
      expect(next.threadId).toBeNull()
    })

    it('resets isStreaming to false', () => {
      const next = chatReducer(populatedState, { type: 'RESET_CHAT' })
      expect(next.isStreaming).toBe(false)
    })

    it('resets currentStreamingContent to empty string', () => {
      const next = chatReducer(populatedState, { type: 'RESET_CHAT' })
      expect(next.currentStreamingContent).toBe('')
    })

    it('removes "active_thread_id" key from localStorage', () => {
      localStorage.setItem('active_thread_id', 'thread_123')
      chatReducer(populatedState, { type: 'RESET_CHAT' })
      expect(localStorage.getItem('active_thread_id')).toBeNull()
    })

    it('PRESERVES userId from prior state', () => {
      const next = chatReducer(populatedState, { type: 'RESET_CHAT' })
      expect(next.userId).toBe('user_abc')
    })

    it('PRESERVES conversations array from prior state', () => {
      const next = chatReducer(populatedState, { type: 'RESET_CHAT' })
      expect(next.conversations).toHaveLength(1)
    })

    it('resets researchProgress to initial values', () => {
      const next = chatReducer(populatedState, { type: 'RESET_CHAT' })
      expect(next.researchProgress).toEqual(initialState.researchProgress)
    })

    it('resets thinking to initial values', () => {
      const state = makeState({ thinking: { ...initialState.thinking, isThinking: true } })
      const next = chatReducer(state, { type: 'RESET_CHAT' })
      expect(next.thinking).toEqual(initialState.thinking)
    })
  })

  describe('SET_ACTIVE_NODE', () => {
    it('sets activeNode to a string value', () => {
      const next = chatReducer(initialState, { type: 'SET_ACTIVE_NODE', payload: 'scope' })
      expect(next.activeNode).toBe('scope')
    })

    it('sets activeNode to null', () => {
      const state = makeState({ activeNode: 'scope' })
      const next = chatReducer(state, { type: 'SET_ACTIVE_NODE', payload: null })
      expect(next.activeNode).toBeNull()
    })
  })

  describe('SET_REPORT_STREAMING', () => {
    it('sets isReportStreaming to true', () => {
      const next = chatReducer(initialState, { type: 'SET_REPORT_STREAMING', payload: true })
      expect(next.isReportStreaming).toBe(true)
    })

    it('sets isReportStreaming to false', () => {
      const state = makeState({ isReportStreaming: true })
      const next = chatReducer(state, { type: 'SET_REPORT_STREAMING', payload: false })
      expect(next.isReportStreaming).toBe(false)
    })
  })

  describe('FINALIZE_AND_SWITCH_TO_REPORT', () => {
    it('when currentStreamingContent is non-empty: creates an assistant message from it', () => {
      const state = makeState({ currentStreamingContent: 'scope summary' })
      const next = chatReducer(state, { type: 'FINALIZE_AND_SWITCH_TO_REPORT' })
      expect(next.messages).toHaveLength(1)
      expect(next.messages[0].content).toBe('scope summary')
      expect(next.messages[0].role).toBe('assistant')
    })

    it('when currentStreamingContent is non-empty: clears currentStreamingContent', () => {
      const state = makeState({ currentStreamingContent: 'scope summary' })
      const next = chatReducer(state, { type: 'FINALIZE_AND_SWITCH_TO_REPORT' })
      expect(next.currentStreamingContent).toBe('')
    })

    it('when currentStreamingContent is non-empty: sets isReportStreaming to true', () => {
      const state = makeState({ currentStreamingContent: 'scope summary' })
      const next = chatReducer(state, { type: 'FINALIZE_AND_SWITCH_TO_REPORT' })
      expect(next.isReportStreaming).toBe(true)
    })

    it('when currentStreamingContent is empty: sets isReportStreaming to true', () => {
      const next = chatReducer(initialState, { type: 'FINALIZE_AND_SWITCH_TO_REPORT' })
      expect(next.isReportStreaming).toBe(true)
    })

    it('when currentStreamingContent is empty: messages array is unchanged', () => {
      const next = chatReducer(initialState, { type: 'FINALIZE_AND_SWITCH_TO_REPORT' })
      expect(next.messages).toHaveLength(0)
    })
  })

  describe('OPEN_REPORT', () => {
    it('sets reportPanelOpen to true', () => {
      const next = chatReducer(initialState, { type: 'OPEN_REPORT', payload: 'report text' })
      expect(next.reportPanelOpen).toBe(true)
    })

    it('sets focusMode to true', () => {
      const next = chatReducer(initialState, { type: 'OPEN_REPORT', payload: 'report text' })
      expect(next.focusMode).toBe(true)
    })

    it('sets activeReportContent to the payload string', () => {
      const next = chatReducer(initialState, { type: 'OPEN_REPORT', payload: 'report text' })
      expect(next.activeReportContent).toBe('report text')
    })

    it('does not mutate other state keys', () => {
      const state = makeState({ isStreaming: true })
      const next = chatReducer(state, { type: 'OPEN_REPORT', payload: 'r' })
      expect(next.isStreaming).toBe(true)
    })
  })

  describe('CLOSE_REPORT', () => {
    it('sets reportPanelOpen to false', () => {
      const state = makeState({ reportPanelOpen: true })
      const next = chatReducer(state, { type: 'CLOSE_REPORT' })
      expect(next.reportPanelOpen).toBe(false)
    })

    it('sets focusMode to false', () => {
      const state = makeState({ focusMode: true })
      const next = chatReducer(state, { type: 'CLOSE_REPORT' })
      expect(next.focusMode).toBe(false)
    })

    it('sets activeReportContent to null', () => {
      const state = makeState({ activeReportContent: 'some report' })
      const next = chatReducer(state, { type: 'CLOSE_REPORT' })
      expect(next.activeReportContent).toBeNull()
    })
  })

  describe('TOGGLE_FOCUS_MODE', () => {
    it('toggles focusMode from false to true', () => {
      const next = chatReducer(initialState, { type: 'TOGGLE_FOCUS_MODE' })
      expect(next.focusMode).toBe(true)
    })

    it('toggles focusMode from true to false', () => {
      const state = makeState({ focusMode: true })
      const next = chatReducer(state, { type: 'TOGGLE_FOCUS_MODE' })
      expect(next.focusMode).toBe(false)
    })

    it('does not mutate other state keys', () => {
      const state = makeState({ isStreaming: true })
      const next = chatReducer(state, { type: 'TOGGLE_FOCUS_MODE' })
      expect(next.isStreaming).toBe(true)
    })
  })

  describe('SET_THINKING', () => {
    const thinkingPayload = { agent: 'scope', thought: 'Thinking...', step: 'step1', phase: 'investigating' }

    it('sets isThinking to true', () => {
      const next = chatReducer(initialState, { type: 'SET_THINKING', payload: thinkingPayload })
      expect(next.thinking.isThinking).toBe(true)
    })

    it('sets agent, thought, step, currentPhase from payload', () => {
      const next = chatReducer(initialState, { type: 'SET_THINKING', payload: thinkingPayload })
      expect(next.thinking.agent).toBe('scope')
      expect(next.thinking.thought).toBe('Thinking...')
      expect(next.thinking.step).toBe('step1')
      expect(next.thinking.currentPhase).toBe('investigating')
    })

    it('creates a ThoughtEntry with a "thought_" prefixed id', () => {
      const next = chatReducer(initialState, { type: 'SET_THINKING', payload: thinkingPayload })
      expect(next.thinking.history[0].id).toMatch(/^thought_/)
    })

    it('appends the ThoughtEntry to thinking.history', () => {
      const next = chatReducer(initialState, { type: 'SET_THINKING', payload: thinkingPayload })
      expect(next.thinking.history).toHaveLength(1)
      expect(next.thinking.history[0].agent).toBe('scope')
      expect(next.thinking.history[0].thought).toBe('Thinking...')
    })

    it('creates a new ThinkingPhase when phase is new', () => {
      const next = chatReducer(initialState, { type: 'SET_THINKING', payload: thinkingPayload })
      expect(next.thinking.phases).toHaveLength(1)
      expect(next.thinking.phases[0].phase).toBe('investigating')
    })

    it('new ThinkingPhase has label from getPhaseLabel(phase)', () => {
      const next = chatReducer(initialState, { type: 'SET_THINKING', payload: thinkingPayload })
      expect(next.thinking.phases[0].label).toBe('Initial Investigation')
    })

    it('new ThinkingPhase has the new ThoughtEntry in its thoughts array', () => {
      const next = chatReducer(initialState, { type: 'SET_THINKING', payload: thinkingPayload })
      expect(next.thinking.phases[0].thoughts).toHaveLength(1)
    })

    it('appends ThoughtEntry to an existing phase when phase already exists', () => {
      const s1 = chatReducer(initialState, { type: 'SET_THINKING', payload: thinkingPayload })
      const s2 = chatReducer(s1, { type: 'SET_THINKING', payload: { ...thinkingPayload, thought: 'Second thought' } })
      expect(s2.thinking.phases).toHaveLength(1)
      expect(s2.thinking.phases[0].thoughts).toHaveLength(2)
    })

    it('does NOT create a duplicate phase when same phase repeats', () => {
      const s1 = chatReducer(initialState, { type: 'SET_THINKING', payload: thinkingPayload })
      const s2 = chatReducer(s1, { type: 'SET_THINKING', payload: thinkingPayload })
      expect(s2.thinking.phases).toHaveLength(1)
    })

    it('creates a second phase when a new phase value is dispatched', () => {
      const s1 = chatReducer(initialState, { type: 'SET_THINKING', payload: thinkingPayload })
      const s2 = chatReducer(s1, { type: 'SET_THINKING', payload: { ...thinkingPayload, phase: 'findings' } })
      expect(s2.thinking.phases).toHaveLength(2)
    })

    it('preserves startTime when thinking.startTime was already set (non-zero)', () => {
      const state = makeState({ thinking: { ...initialState.thinking, startTime: 1000 } })
      const next = chatReducer(state, { type: 'SET_THINKING', payload: thinkingPayload })
      expect(next.thinking.startTime).toBe(1000)
    })

    it('initialises startTime via Date.now() when thinking.startTime was 0', () => {
      const next = chatReducer(initialState, { type: 'SET_THINKING', payload: thinkingPayload })
      expect(next.thinking.startTime).toBe(FIXED_MS)
    })

    it('does not create a phase entry when phase is empty string', () => {
      const next = chatReducer(initialState, { type: 'SET_THINKING', payload: { ...thinkingPayload, phase: '' } })
      expect(next.thinking.phases).toHaveLength(0)
    })

    it('still appends to history even when phase is empty string', () => {
      const next = chatReducer(initialState, { type: 'SET_THINKING', payload: { ...thinkingPayload, phase: '' } })
      expect(next.thinking.history).toHaveLength(1)
    })
  })

  describe('STOP_THINKING', () => {
    it('sets isThinking to false', () => {
      const state = makeState({ thinking: { ...initialState.thinking, isThinking: true } })
      const next = chatReducer(state, { type: 'STOP_THINKING' })
      expect(next.thinking.isThinking).toBe(false)
    })

    it('calculates elapsedMs = Date.now() - startTime when startTime is non-zero', () => {
      const startTime = FIXED_MS - 2000
      const state = makeState({ thinking: { ...initialState.thinking, isThinking: true, startTime } })
      const next = chatReducer(state, { type: 'STOP_THINKING' })
      expect(next.thinking.elapsedMs).toBe(2000)
    })

    it('keeps prior elapsedMs unchanged when startTime is 0', () => {
      const state = makeState({ thinking: { ...initialState.thinking, startTime: 0, elapsedMs: 999 } })
      const next = chatReducer(state, { type: 'STOP_THINKING' })
      expect(next.thinking.elapsedMs).toBe(999)
    })

    it('uses Math.max so elapsedMs never regresses on a second STOP_THINKING call', () => {
      // First stop: sets elapsedMs = 2000
      const startTime = FIXED_MS - 2000
      const s1 = chatReducer(makeState({ thinking: { ...initialState.thinking, startTime } }), { type: 'STOP_THINKING' })
      // Second stop on already-stopped state (startTime still set, elapsedMs=2000)
      // Date.now() - startTime is still 2000, Math.max(2000, 2000) = 2000
      const s2 = chatReducer(s1, { type: 'STOP_THINKING' })
      expect(s2.thinking.elapsedMs).toBeGreaterThanOrEqual(s1.thinking.elapsedMs)
    })
  })

  describe('RESTORE_THINKING', () => {
    it('replaces entire thinking state with the payload', () => {
      const newThinking: ThinkingState = {
        isThinking: false,
        agent: 'restored_agent',
        thought: 'restored thought',
        step: 'restored step',
        startTime: 12345,
        elapsedMs: 5000,
        history: [],
        phases: [{ id: 'phase_1', phase: 'findings', label: 'Research Findings', thoughts: [], startTime: 12345 }],
        currentPhase: 'findings'
      }
      const next = chatReducer(initialState, { type: 'RESTORE_THINKING', payload: newThinking })
      expect(next.thinking).toEqual(newThinking)
    })

    it('full replacement — not a merge with existing thinking state', () => {
      const state = makeState({ thinking: { ...initialState.thinking, isThinking: true, agent: 'old' } })
      const replacement: ThinkingState = { ...initialState.thinking, agent: 'new' }
      const next = chatReducer(state, { type: 'RESTORE_THINKING', payload: replacement })
      expect(next.thinking.isThinking).toBe(false)
      expect(next.thinking.agent).toBe('new')
    })
  })

  describe('state isolation', () => {
    it('SET_USER_ID does not change messages, threadId, isStreaming, thinking', () => {
      const state = makeState({ messages: [makeMessage()], threadId: 't', isStreaming: true })
      const next = chatReducer(state, { type: 'SET_USER_ID', payload: 'new_user' })
      expect(next.messages).toBe(state.messages)
      expect(next.threadId).toBe('t')
      expect(next.isStreaming).toBe(true)
    })

    it('TOGGLE_FOCUS_MODE does not change messages, thinking, isStreaming, error', () => {
      const state = makeState({ messages: [makeMessage()], isStreaming: true, error: 'e' })
      const next = chatReducer(state, { type: 'TOGGLE_FOCUS_MODE' })
      expect(next.messages).toBe(state.messages)
      expect(next.isStreaming).toBe(true)
      expect(next.error).toBe('e')
    })

    it('unknown action type returns state unchanged', () => {
      const next = chatReducer(initialState, { type: 'NONEXISTENT' } as unknown as ChatAction)
      expect(next).toBe(initialState)
    })
  })
})
