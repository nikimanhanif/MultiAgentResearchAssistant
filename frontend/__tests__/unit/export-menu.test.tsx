// @vitest-environment jsdom

import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi, describe, it, expect, beforeEach } from 'vitest'
import { ExportMenu } from '@/components/chat/export-menu'
import { useChatContext } from '@/context/chat-context'

vi.mock('@/context/chat-context', () => ({
  useChatContext: vi.fn(),
}))

// Mock the export-utils downloadBlob
vi.mock('@/lib/export-utils', () => ({
  downloadBlob: vi.fn(),
}))

import { downloadBlob } from '@/lib/export-utils'

// ── Fixtures ────────────────────────────────────────────────────────────────

const mockMessages = [
  {
    id: 'msg_1',
    role: 'user' as const,
    content: 'What is AI?',
    timestamp: new Date(),
  },
  {
    id: 'msg_2',
    role: 'assistant' as const,
    content: 'AI is artificial intelligence.',
    timestamp: new Date(),
  },
]

const mockBrief = {
  scope: 'Artificial Intelligence overview',
  subTopics: ['Machine Learning', 'Deep Learning'],
}

const mockReviewRequest = {
  report: '# AI Report\n\n## Summary\n\nAI is transforming the world.',
  pending: false,
}

const fullContext = {
  messages: mockMessages,
  researchBrief: mockBrief,
  reviewRequest: mockReviewRequest,
  activeReportContent: '# Active Report\n\nReport body here.',
  threadId: 'thread-123',
  userId: 'user-456',
  researchProgress: { phase: 'idle', tasksCount: 0, findingsCount: 0, iterations: 0, phaseDurationMs: 0 },
}

const emptyContext = {
  messages: [],
  researchBrief: null,
  reviewRequest: null,
  activeReportContent: null,
  threadId: null,
  userId: '',
  researchProgress: { phase: 'idle', tasksCount: 0, findingsCount: 0, iterations: 0, phaseDurationMs: 0 },
}

// ── Setup ───────────────────────────────────────────────────────────────────

beforeEach(() => {
  vi.mocked(useChatContext).mockReturnValue(fullContext as any)
  vi.mocked(downloadBlob).mockClear()
  vi.spyOn(globalThis, 'fetch').mockResolvedValue({
    ok: true,
    blob: () => Promise.resolve(new Blob(['mock'])),
  } as Response)
})

// ── Tests ───────────────────────────────────────────────────────────────────

describe('ExportMenu', () => {
  it('renders nothing when there is no content', () => {
    vi.mocked(useChatContext).mockReturnValue(emptyContext as any)

    const { container } = render(<ExportMenu scope="session" />)
    expect(container.innerHTML).toBe('')
  })

  it('renders the download button when content exists', () => {
    render(<ExportMenu scope="session" />)
    expect(screen.getByTitle('Export options')).toBeTruthy()
  })

  it('session scope: markdown export downloads with session filename', async () => {
    const user = userEvent.setup()
    render(<ExportMenu scope="session" />)

    await user.click(screen.getByTitle('Export options'))
    const mdOption = await screen.findByText('Export as Markdown')
    await user.click(mdOption)

    expect(downloadBlob).toHaveBeenCalledTimes(1)
    const [blob, filename] = vi.mocked(downloadBlob).mock.calls[0]
    expect(filename).toMatch(/^research_session_\d+\.md$/)
    expect(blob).toBeInstanceOf(Blob)
    expect((blob as Blob).type).toBe('text/markdown')
  })

  it('report scope: markdown export downloads with report filename', async () => {
    const user = userEvent.setup()
    render(<ExportMenu scope="report" />)

    await user.click(screen.getByTitle('Export options'))
    const mdOption = await screen.findByText('Export as Markdown')
    await user.click(mdOption)

    expect(downloadBlob).toHaveBeenCalledTimes(1)
    const [blob, filename] = vi.mocked(downloadBlob).mock.calls[0]
    expect(filename).toMatch(/^research_report_\d+\.md$/)
    expect(blob).toBeInstanceOf(Blob)
    expect((blob as Blob).type).toBe('text/markdown')
  })

  it('PDF option is disabled when there is no threadId', async () => {
    vi.mocked(useChatContext).mockReturnValue({
      ...fullContext,
      threadId: null,
    } as any)

    const user = userEvent.setup()
    render(<ExportMenu scope="session" />)

    await user.click(screen.getByTitle('Export options'))
    const pdfOption = await screen.findByText('Export as PDF')

    // The parent menu item should have data-disabled attribute
    expect(pdfOption.closest('[data-disabled]')).toBeTruthy()
  })

  it('PDF option is disabled when there is no report', async () => {
    vi.mocked(useChatContext).mockReturnValue({
      ...fullContext,
      reviewRequest: null,
    } as any)

    const user = userEvent.setup()
    render(<ExportMenu scope="session" />)

    await user.click(screen.getByTitle('Export options'))
    const pdfOption = await screen.findByText('Export as PDF')
    expect(pdfOption.closest('[data-disabled]')).toBeTruthy()
  })

  it('BibTeX option is disabled when there is no threadId', async () => {
    vi.mocked(useChatContext).mockReturnValue({
      ...fullContext,
      threadId: null,
    } as any)

    const user = userEvent.setup()
    render(<ExportMenu scope="session" />)

    await user.click(screen.getByTitle('Export options'))
    const bibOption = await screen.findByText('Export as BibTeX')
    expect(bibOption.closest('[data-disabled]')).toBeTruthy()
  })

  it('PDF export calls backend with correct URL', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      blob: () => Promise.resolve(new Blob(['%PDF-1.4'])),
    } as Response)

    const user = userEvent.setup()
    render(<ExportMenu scope="session" />)

    await user.click(screen.getByTitle('Export options'))
    const pdfOption = await screen.findByText('Export as PDF')
    await user.click(pdfOption)

    await waitFor(() => {
      expect(fetchSpy).toHaveBeenCalledWith(
        expect.stringContaining('/exports/user-456/thread-123/pdf')
      )
    })
  })

  it('BibTeX export calls backend with correct URL', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      blob: () => Promise.resolve(new Blob(['@article{test}'])),
    } as Response)

    const user = userEvent.setup()
    render(<ExportMenu scope="session" />)

    await user.click(screen.getByTitle('Export options'))
    const bibOption = await screen.findByText('Export as BibTeX')
    await user.click(bibOption)

    await waitFor(() => {
      expect(fetchSpy).toHaveBeenCalledWith(
        expect.stringContaining('/exports/user-456/thread-123/bibtex')
      )
    })
  })

  it('handles PDF fetch error gracefully', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: false,
      status: 500,
    } as Response)

    const user = userEvent.setup()
    render(<ExportMenu scope="session" />)

    await user.click(screen.getByTitle('Export options'))
    const pdfOption = await screen.findByText('Export as PDF')
    await user.click(pdfOption)

    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith('PDF export failed:', 500)
    })

    consoleSpy.mockRestore()
  })

  it('handles BibTeX fetch error gracefully', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: false,
      status: 400,
    } as Response)

    const user = userEvent.setup()
    render(<ExportMenu scope="session" />)

    await user.click(screen.getByTitle('Export options'))
    const bibOption = await screen.findByText('Export as BibTeX')
    await user.click(bibOption)

    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith('BibTeX export failed:', 400)
    })

    consoleSpy.mockRestore()
  })
})
