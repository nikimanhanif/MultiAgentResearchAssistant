// @vitest-environment jsdom

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { downloadBlob } from '@/lib/export-utils'

describe('downloadBlob', () => {
  let createObjectURLSpy: ReturnType<typeof vi.fn>
  let revokeObjectURLSpy: ReturnType<typeof vi.fn>
  let clickSpy: ReturnType<typeof vi.fn>
  let appendChildSpy: ReturnType<typeof vi.spyOn>
  let removeChildSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    createObjectURLSpy = vi.fn(() => 'blob:mock-url')
    revokeObjectURLSpy = vi.fn()
    clickSpy = vi.fn()

    globalThis.URL.createObjectURL = createObjectURLSpy
    globalThis.URL.revokeObjectURL = revokeObjectURLSpy

    appendChildSpy = vi.spyOn(document.body, 'appendChild').mockImplementation((node) => node)
    removeChildSpy = vi.spyOn(document.body, 'removeChild').mockImplementation((node) => node)

    vi.spyOn(document, 'createElement').mockImplementation((tag: string) => {
      const el = {
        href: '',
        download: '',
        click: clickSpy,
        tagName: tag.toUpperCase()
      } as unknown as HTMLElement
      return el
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('creates an object URL from the blob', () => {
    const blob = new Blob(['test'], { type: 'text/plain' })
    downloadBlob(blob, 'file.txt')

    expect(createObjectURLSpy).toHaveBeenCalledWith(blob)
  })

  it('creates an anchor element with correct download attribute', () => {
    const blob = new Blob(['test'], { type: 'text/plain' })
    downloadBlob(blob, 'my-file.md')

    expect(document.createElement).toHaveBeenCalledWith('a')
    expect(clickSpy).toHaveBeenCalled()
  })

  it('revokes the object URL after click', () => {
    const blob = new Blob(['test'], { type: 'text/plain' })
    downloadBlob(blob, 'file.txt')

    expect(revokeObjectURLSpy).toHaveBeenCalledWith('blob:mock-url')
  })

  it('appends and removes the anchor from document body', () => {
    const blob = new Blob(['test'], { type: 'text/plain' })
    downloadBlob(blob, 'file.txt')

    expect(appendChildSpy).toHaveBeenCalled()
    expect(removeChildSpy).toHaveBeenCalled()
  })
})
