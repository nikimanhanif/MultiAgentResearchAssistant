import React from 'react'

// Make `window` available so reducer guards like `typeof window !== 'undefined'` work
if (typeof window === 'undefined') {
  ;(globalThis as any).window = globalThis
}

// Mock localStorage — used by SET_THREAD_ID and RESET_CHAT reducer cases
const store: Record<string, string> = {}
const localStorageMock = {
  getItem: (key: string) => store[key] ?? null,
  setItem: (key: string, value: string) => { store[key] = value },
  removeItem: (key: string) => { delete store[key] },
  clear: () => { Object.keys(store).forEach(k => delete store[k]) },
}
Object.defineProperty(globalThis, 'localStorage', { value: localStorageMock, writable: true })

beforeEach(() => localStorageMock.clear())

// ResizeObserver polyfill — needed by Radix UI ScrollArea and similar components
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

// scrollIntoView polyfill — jsdom does not implement scroll APIs
// Guard needed because non-jsdom test files run in Node where Element is undefined
if (typeof Element !== 'undefined') {
  Element.prototype.scrollIntoView = vi.fn()
}

// framer-motion mock — avoids browser animation API errors in jsdom
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => React.createElement('div', {}, children),
  },
  AnimatePresence: ({ children }: any) => children ?? null,
}))
