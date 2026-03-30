import { describe, it, expect } from 'vitest'
import { isReportMessage } from '@/lib/report-utils'

// Logic: returns true iff hasMultipleHeaders (≥2 matches of /^##?\s/gm)
//        AND (hasReportIndicator OR content.length > 500)

const twoHeaders = '## Header One\n## Header Two\n'      // 29 chars, 2 valid H2 headers
const oneHeader  = '## Only Header\n'                    // 15 chars, 1 header
const longBody   = 'x'.repeat(502)                       // 502 chars, no headers, no indicator

describe('isReportMessage', () => {
  describe('returns false', () => {
    it('returns false for empty string', () => {
      expect(isReportMessage('')).toBe(false)
    })

    it('returns false for short content with no headers (<500 chars, no indicator)', () => {
      expect(isReportMessage('Just a short answer.')).toBe(false)
    })

    it('returns false for content with exactly 1 header, no indicator, length ≤500', () => {
      expect(isReportMessage(oneHeader + 'Some text.')).toBe(false)
    })

    it('returns true for 2+ H2 headers even at short length (## always contains "# " indicator at index 1)', () => {
      // This documents a subtle behaviour: "## Foo" includes "# " at position 1,
      // so hasReportIndicator is always true when H2 headers exist.
      // The condition reduces to: hasMultipleHeaders && true = hasMultipleHeaders.
      expect(isReportMessage(twoHeaders)).toBe(true)
    })

    it('returns false for content with an indicator keyword mid-line but only 1 real header', () => {
      // oneHeader has 1 line-start header; the indicator "## Executive Summary"
      // appears mid-sentence (not at line-start), so header count stays at 1.
      const content = oneHeader + 'See ## Executive Summary for full details.'
      expect(isReportMessage(content)).toBe(false)
    })

    it('returns false for content where "##" is mid-line (not at line start)', () => {
      const midLine = 'text ## not a real header\nmore text ## another fake\n'
      expect(isReportMessage(midLine)).toBe(false)
    })
  })

  describe('returns true', () => {
    it('returns true for 2+ headers + report indicator (any length)', () => {
      const content = twoHeaders + '## Executive Summary\nShort but has indicator.'
      expect(isReportMessage(content)).toBe(true)
    })

    it('returns true for 2+ headers + length >500 (no indicator keyword needed)', () => {
      const content = twoHeaders + longBody   // 29 + 502 = 531 chars, 2 headers
      expect(isReportMessage(content)).toBe(true)
    })

    it('returns true when indicator is "## Executive Summary"', () => {
      const content = twoHeaders + '## Executive Summary\nDetails here.'
      expect(isReportMessage(content)).toBe(true)
    })

    it('returns true when indicator is "## Key Findings"', () => {
      const content = twoHeaders + '## Key Findings\nDetails here.'
      expect(isReportMessage(content)).toBe(true)
    })

    it('returns true when indicator is "# " prefix on a line (H1)', () => {
      // Two headers: one H1, one H2
      const content = '# Main Title\n## Section Two\n## Key Findings\nSome text.'
      expect(isReportMessage(content)).toBe(true)
    })

    it('returns true for a realistic multi-section report string', () => {
      const report = [
        '# Research Report: Topic',
        '',
        '## Executive Summary',
        'This report examines...',
        '',
        '## Key Findings',
        '1. Finding one',
        '2. Finding two',
        '',
        '## Conclusion',
        'In conclusion...',
      ].join('\n')
      expect(isReportMessage(report)).toBe(true)
    })
  })

  describe('header regex edge cases', () => {
    it('counts "## Header" as header (H2 with space)', () => {
      const content = '## First\n## Second\n## Key Findings\n'
      expect(isReportMessage(content)).toBe(true)
    })

    it('counts "# Header" as header (H1 with space)', () => {
      const content = '# Title\n## Section\n## Key Findings\n'
      expect(isReportMessage(content)).toBe(true)
    })

    it('does NOT count "##Header" (no space after hashes) as a header', () => {
      // Two "##Header" patterns — neither matches /^##?\s/gm because no space
      const content = '##NoSpace\n##AlsoNoSpace\n## Executive Summary\n'
      // Only "## Executive Summary" matches → count = 1 → false
      expect(isReportMessage(content)).toBe(false)
    })

    it('requires header to be at start of line — "text ## header" is not counted', () => {
      const content = 'text ## fake\nmore text ## also fake\n## Executive Summary\n'
      // Only the last one starts at line start → count = 1 → false
      expect(isReportMessage(content)).toBe(false)
    })
  })
})
