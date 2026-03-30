import { describe, it, expect, beforeAll, afterAll, vi } from 'vitest'
import { getActionsForEvent } from '@/context/chat-context'
import type { StreamEvent } from '@/types/chat'

// Freeze time for deterministic IDs in brief_created / clarification_request
const FIXED_NOW = new Date('2026-01-01T12:00:00.000Z')
const FIXED_MS = FIXED_NOW.getTime()

beforeAll(() => {
  vi.useFakeTimers()
  vi.setSystemTime(FIXED_NOW)
})

afterAll(() => {
  vi.useRealTimers()
})

describe('getActionsForEvent', () => {
  describe('token event', () => {
    const event: StreamEvent = { type: 'token', content: 'hello', node: 'scope' }

    it('returns exactly 1 action: APPEND_STREAMING_CONTENT with event.content', () => {
      const { actions } = getActionsForEvent(event, false)
      expect(actions).toHaveLength(1)
      expect(actions[0]).toEqual({ type: 'APPEND_STREAMING_CONTENT', payload: 'hello' })
    })

    it('nextIsReportStreaming stays false when passed false', () => {
      const { nextIsReportStreaming } = getActionsForEvent(event, false)
      expect(nextIsReportStreaming).toBe(false)
    })

    it('nextIsReportStreaming stays true when passed true', () => {
      const { nextIsReportStreaming } = getActionsForEvent(event, true)
      expect(nextIsReportStreaming).toBe(true)
    })
  })

  describe('report_token event — first token (isReportStreaming = false)', () => {
    const event: StreamEvent = { type: 'report_token', content: 'Report start' }

    it('returns STOP_THINKING as action[0]', () => {
      const { actions } = getActionsForEvent(event, false)
      expect(actions[0]).toEqual({ type: 'STOP_THINKING' })
    })

    it('returns FINALIZE_AND_SWITCH_TO_REPORT as action[1]', () => {
      const { actions } = getActionsForEvent(event, false)
      expect(actions[1]).toEqual({ type: 'FINALIZE_AND_SWITCH_TO_REPORT' })
    })

    it('returns SET_PROGRESS { phase: "generating_report" } as action[2]', () => {
      const { actions } = getActionsForEvent(event, false)
      expect(actions[2]).toEqual({ type: 'SET_PROGRESS', payload: { phase: 'generating_report' } })
    })

    it('returns APPEND_STREAMING_CONTENT with event.content as action[3]', () => {
      const { actions } = getActionsForEvent(event, false)
      expect(actions[3]).toEqual({ type: 'APPEND_STREAMING_CONTENT', payload: 'Report start' })
    })

    it('returns exactly 4 actions', () => {
      const { actions } = getActionsForEvent(event, false)
      expect(actions).toHaveLength(4)
    })

    it('sets nextIsReportStreaming to true', () => {
      const { nextIsReportStreaming } = getActionsForEvent(event, false)
      expect(nextIsReportStreaming).toBe(true)
    })
  })

  describe('report_token event — subsequent tokens (isReportStreaming = true)', () => {
    const event: StreamEvent = { type: 'report_token', content: 'More report content' }

    it('returns exactly 1 action: APPEND_STREAMING_CONTENT with event.content', () => {
      const { actions } = getActionsForEvent(event, true)
      expect(actions).toHaveLength(1)
      expect(actions[0]).toEqual({ type: 'APPEND_STREAMING_CONTENT', payload: 'More report content' })
    })

    it('nextIsReportStreaming stays true', () => {
      const { nextIsReportStreaming } = getActionsForEvent(event, true)
      expect(nextIsReportStreaming).toBe(true)
    })
  })

  describe('progress event', () => {
    const event: StreamEvent = {
      type: 'progress',
      phase: 'researching',
      tasks_count: 5,
      findings_count: 3,
      iterations: 2,
      phase_duration_ms: 1500
    }

    it('returns exactly 1 action: SET_PROGRESS', () => {
      const { actions } = getActionsForEvent(event, false)
      expect(actions).toHaveLength(1)
      expect(actions[0].type).toBe('SET_PROGRESS')
    })

    it('maps tasks_count → tasksCount', () => {
      const { actions } = getActionsForEvent(event, false)
      expect((actions[0] as any).payload.tasksCount).toBe(5)
    })

    it('maps findings_count → findingsCount', () => {
      const { actions } = getActionsForEvent(event, false)
      expect((actions[0] as any).payload.findingsCount).toBe(3)
    })

    it('maps phase_duration_ms → phaseDurationMs', () => {
      const { actions } = getActionsForEvent(event, false)
      expect((actions[0] as any).payload.phaseDurationMs).toBe(1500)
    })

    it('defaults phaseDurationMs to 0 when phase_duration_ms is 0', () => {
      const e: StreamEvent = { ...event, phase_duration_ms: 0 }
      const { actions } = getActionsForEvent(e, false)
      expect((actions[0] as any).payload.phaseDurationMs).toBe(0)
    })
  })

  describe('brief_created event', () => {
    const event: StreamEvent = {
      type: 'brief_created',
      scope: 'AI safety',
      sub_topics: ['alignment', 'robustness']
    }

    it('returns FINALIZE_STREAMING_MESSAGE as action[0]', () => {
      const { actions } = getActionsForEvent(event, false)
      expect(actions[0]).toEqual({ type: 'FINALIZE_STREAMING_MESSAGE' })
    })

    it('returns SET_BRIEF with scope and subTopics as action[1]', () => {
      const { actions } = getActionsForEvent(event, false)
      expect(actions[1]).toEqual({ type: 'SET_BRIEF', payload: { scope: 'AI safety', subTopics: ['alignment', 'robustness'] } })
    })

    it('returns ADD_MESSAGE as action[2]', () => {
      const { actions } = getActionsForEvent(event, false)
      expect(actions[2].type).toBe('ADD_MESSAGE')
    })

    it('ADD_MESSAGE content contains scope value from event', () => {
      const { actions } = getActionsForEvent(event, false)
      const msg = (actions[2] as any).payload
      expect(msg.content).toContain('AI safety')
    })

    it('ADD_MESSAGE content contains each sub_topic as a bullet point', () => {
      const { actions } = getActionsForEvent(event, false)
      const msg = (actions[2] as any).payload
      expect(msg.content).toContain('- alignment')
      expect(msg.content).toContain('- robustness')
    })

    it('ADD_MESSAGE id starts with "msg_brief_"', () => {
      const { actions } = getActionsForEvent(event, false)
      const msg = (actions[2] as any).payload
      expect(msg.id).toMatch(/^msg_brief_/)
    })

    it('returns exactly 3 actions', () => {
      const { actions } = getActionsForEvent(event, false)
      expect(actions).toHaveLength(3)
    })
  })

  describe('clarification_request event', () => {
    const event: StreamEvent = { type: 'clarification_request', questions: 'What is your focus area?' }

    it('returns ADD_MESSAGE as action[0] with event.questions as content', () => {
      const { actions } = getActionsForEvent(event, false)
      expect(actions[0].type).toBe('ADD_MESSAGE')
      expect((actions[0] as any).payload.content).toBe('What is your focus area?')
    })

    it('added message has role: "assistant" and node: "scope"', () => {
      const { actions } = getActionsForEvent(event, false)
      const msg = (actions[0] as any).payload
      expect(msg.role).toBe('assistant')
      expect(msg.node).toBe('scope')
    })

    it('returns SET_STREAMING false as action[1]', () => {
      const { actions } = getActionsForEvent(event, false)
      expect(actions[1]).toEqual({ type: 'SET_STREAMING', payload: false })
    })

    it('returns SET_STREAMING_CONTENT "" as action[2]', () => {
      const { actions } = getActionsForEvent(event, false)
      expect(actions[2]).toEqual({ type: 'SET_STREAMING_CONTENT', payload: '' })
    })

    it('returns exactly 3 actions', () => {
      const { actions } = getActionsForEvent(event, false)
      expect(actions).toHaveLength(3)
    })
  })

  describe('review_request event', () => {
    const event: StreamEvent = { type: 'review_request', report: 'Final report text here.' }

    it('returns FINALIZE_STREAMING_MESSAGE as action[0]', () => {
      const { actions } = getActionsForEvent(event, true)
      expect(actions[0]).toEqual({ type: 'FINALIZE_STREAMING_MESSAGE' })
    })

    it('returns SET_STREAMING false as action[1]', () => {
      const { actions } = getActionsForEvent(event, true)
      expect(actions[1]).toEqual({ type: 'SET_STREAMING', payload: false })
    })

    it('returns SET_REPORT_STREAMING false as action[2]', () => {
      const { actions } = getActionsForEvent(event, true)
      expect(actions[2]).toEqual({ type: 'SET_REPORT_STREAMING', payload: false })
    })

    it('returns SET_REVIEW_REQUEST { report, pending: true } as action[3]', () => {
      const { actions } = getActionsForEvent(event, true)
      expect(actions[3]).toEqual({
        type: 'SET_REVIEW_REQUEST',
        payload: { report: 'Final report text here.', pending: true }
      })
    })

    it('returns exactly 4 actions', () => {
      const { actions } = getActionsForEvent(event, true)
      expect(actions).toHaveLength(4)
    })

    it('sets nextIsReportStreaming to false', () => {
      const { nextIsReportStreaming } = getActionsForEvent(event, true)
      expect(nextIsReportStreaming).toBe(false)
    })
  })

  describe('state_update event', () => {
    const event: StreamEvent = { type: 'state_update', node: 'supervisor', is_complete: false }

    it('returns exactly 1 action: SET_ACTIVE_NODE with event.node', () => {
      const { actions } = getActionsForEvent(event, false)
      expect(actions).toHaveLength(1)
      expect(actions[0]).toEqual({ type: 'SET_ACTIVE_NODE', payload: 'supervisor' })
    })
  })

  describe('thought event', () => {
    const event: StreamEvent = {
      type: 'thought',
      agent: 'scope',
      thought: 'I should search for papers on...',
      step: 'planning',
      elapsed_ms: 200,
      phase: 'investigating'
    }

    it('returns exactly 1 action: SET_THINKING', () => {
      const { actions } = getActionsForEvent(event, false)
      expect(actions).toHaveLength(1)
      expect(actions[0].type).toBe('SET_THINKING')
    })

    it('SET_THINKING payload has agent, thought, step, phase from event', () => {
      const { actions } = getActionsForEvent(event, false)
      const payload = (actions[0] as any).payload
      expect(payload.agent).toBe('scope')
      expect(payload.thought).toBe('I should search for papers on...')
      expect(payload.step).toBe('planning')
      expect(payload.phase).toBe('investigating')
    })

    it('uses empty string for phase when event.phase is empty', () => {
      const e: StreamEvent = { ...event, phase: '' }
      const { actions } = getActionsForEvent(e, false)
      expect((actions[0] as any).payload.phase).toBe('')
    })
  })

  describe('complete event', () => {
    const event: StreamEvent = { type: 'complete', message: 'Research complete' }

    it('returns FINALIZE_STREAMING_MESSAGE as action[0]', () => {
      const { actions } = getActionsForEvent(event, true)
      expect(actions[0]).toEqual({ type: 'FINALIZE_STREAMING_MESSAGE' })
    })

    it('returns SET_REPORT_STREAMING false as action[1]', () => {
      const { actions } = getActionsForEvent(event, true)
      expect(actions[1]).toEqual({ type: 'SET_REPORT_STREAMING', payload: false })
    })

    it('returns SET_PROGRESS { phase: "complete" } as action[2]', () => {
      const { actions } = getActionsForEvent(event, true)
      expect(actions[2]).toEqual({ type: 'SET_PROGRESS', payload: { phase: 'complete' } })
    })

    it('returns STOP_THINKING as action[3]', () => {
      const { actions } = getActionsForEvent(event, true)
      expect(actions[3]).toEqual({ type: 'STOP_THINKING' })
    })

    it('returns exactly 4 actions', () => {
      const { actions } = getActionsForEvent(event, true)
      expect(actions).toHaveLength(4)
    })

    it('sets nextIsReportStreaming to false', () => {
      const { nextIsReportStreaming } = getActionsForEvent(event, true)
      expect(nextIsReportStreaming).toBe(false)
    })
  })

  describe('error event', () => {
    const event: StreamEvent = { type: 'error', error: 'Backend connection failed' }

    it('returns SET_ERROR with event.error as action[0]', () => {
      const { actions } = getActionsForEvent(event, true)
      expect(actions[0]).toEqual({ type: 'SET_ERROR', payload: 'Backend connection failed' })
    })

    it('returns SET_REPORT_STREAMING false as action[1]', () => {
      const { actions } = getActionsForEvent(event, true)
      expect(actions[1]).toEqual({ type: 'SET_REPORT_STREAMING', payload: false })
    })

    it('returns STOP_THINKING as action[2]', () => {
      const { actions } = getActionsForEvent(event, true)
      expect(actions[2]).toEqual({ type: 'STOP_THINKING' })
    })

    it('returns exactly 3 actions', () => {
      const { actions } = getActionsForEvent(event, true)
      expect(actions).toHaveLength(3)
    })

    it('sets nextIsReportStreaming to false', () => {
      const { nextIsReportStreaming } = getActionsForEvent(event, true)
      expect(nextIsReportStreaming).toBe(false)
    })
  })
})
