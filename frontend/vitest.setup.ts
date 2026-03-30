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
