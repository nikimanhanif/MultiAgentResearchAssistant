import { describe, it, expect, beforeAll, afterAll, vi } from 'vitest'
import { formatDate } from '@/components/sidebar/research-history'

// Anchor: now = 2026-03-30T12:00:00Z
const FIXED_NOW = new Date('2026-03-30T12:00:00.000Z')

beforeAll(() => {
  vi.useFakeTimers()
  vi.setSystemTime(FIXED_NOW)
})

afterAll(() => {
  vi.useRealTimers()
})

describe('formatDate', () => {
  it('returns "Today" for a date earlier the same day', () => {
    expect(formatDate('2026-03-30T06:00:00Z')).toBe('Today')
  })

  it('returns "Today" for a date at the very start of the same day', () => {
    expect(formatDate('2026-03-30T00:01:00Z')).toBe('Today')
  })

  it('returns "Yesterday" for a date 1 day ago', () => {
    expect(formatDate('2026-03-29T12:00:00Z')).toBe('Yesterday')
  })

  it('returns "3 days ago" for a date 3 days ago', () => {
    expect(formatDate('2026-03-27T12:00:00Z')).toBe('3 days ago')
  })

  it('returns "6 days ago" for a date 6 days ago (boundary: last relative label)', () => {
    expect(formatDate('2026-03-24T12:00:00Z')).toBe('6 days ago')
  })

  it('returns toLocaleDateString() for 7 days ago (boundary: first locale fallback)', () => {
    const input = '2026-03-23T12:00:00Z'
    expect(formatDate(input)).toBe(new Date(input).toLocaleDateString())
  })

  it('returns toLocaleDateString() for a date far in the past', () => {
    const input = '2025-01-01T00:00:00Z'
    expect(formatDate(input)).toBe(new Date(input).toLocaleDateString())
  })
})
