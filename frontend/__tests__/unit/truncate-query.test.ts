import { describe, it, expect } from 'vitest'
import { truncateQuery } from '@/components/sidebar/research-history'

describe('truncateQuery', () => {
  describe('no truncation (default maxLength=28)', () => {
    it('returns string unchanged when length < 28', () => {
      expect(truncateQuery('short query')).toBe('short query')
    })

    it('returns string unchanged when length === 28 (boundary — strict > not >=)', () => {
      const exactly28 = 'a'.repeat(28)
      expect(truncateQuery(exactly28)).toBe(exactly28)
    })

    it('returns empty string unchanged', () => {
      expect(truncateQuery('')).toBe('')
    })
  })

  describe('truncation occurs (default maxLength=28)', () => {
    it('truncates and appends "..." when length is 29 (one over limit)', () => {
      const input = 'a'.repeat(29)
      expect(truncateQuery(input)).toBe('a'.repeat(28) + '...')
    })

    it('truncated portion is exactly 28 characters long', () => {
      const result = truncateQuery('a'.repeat(50))
      expect(result.replace('...', '')).toHaveLength(28)
    })

    it('result length equals maxLength + 3 ("..." length)', () => {
      expect(truncateQuery('a'.repeat(50))).toHaveLength(31)
    })

    it('truncates a realistic query string correctly', () => {
      const query = 'What is the economic impact of climate change on coastal cities?'
      expect(truncateQuery(query)).toBe(query.slice(0, 28) + '...')
    })
  })

  describe('custom maxLength', () => {
    it('respects custom maxLength of 10', () => {
      expect(truncateQuery('Hello World!', 10)).toBe('Hello Worl...')
    })

    it('does not truncate when string length equals custom maxLength exactly', () => {
      expect(truncateQuery('Hello', 5)).toBe('Hello')
    })

    it('truncates when string is one character over custom maxLength', () => {
      expect(truncateQuery('Hello!', 5)).toBe('Hello...')
    })
  })
})
