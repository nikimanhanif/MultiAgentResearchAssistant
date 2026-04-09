import { describe, it, expect } from 'vitest'
import { getPhaseLabel } from '@/context/chat-context'

describe('getPhaseLabel', () => {
  describe('known phase strings', () => {
    it('returns "Initial Investigation" for "investigating"', () => {
      expect(getPhaseLabel('investigating')).toBe('Initial Investigation')
    })

    it('returns "Research Findings" for "findings"', () => {
      expect(getPhaseLabel('findings')).toBe('Research Findings')
    })

    it('returns "Deepening Analysis" for "deepening"', () => {
      expect(getPhaseLabel('deepening')).toBe('Deepening Analysis')
    })

    it('returns "Gap Analysis" for "analyzing"', () => {
      expect(getPhaseLabel('analyzing')).toBe('Gap Analysis')
    })
  })

  describe('default branch (unknown strings)', () => {
    it('capitalises first letter: "scoping" → "Scoping"', () => {
      expect(getPhaseLabel('scoping')).toBe('Scoping')
    })

    it('leaves subsequent chars unchanged: "myPhase" → "MyPhase"', () => {
      expect(getPhaseLabel('myPhase')).toBe('MyPhase')
    })

    it('returns empty string for empty string input (no throw)', () => {
      expect(getPhaseLabel('')).toBe('')
    })

    it('handles already-capitalised input: "Reviewing" → "Reviewing"', () => {
      expect(getPhaseLabel('Reviewing')).toBe('Reviewing')
    })
  })
})
