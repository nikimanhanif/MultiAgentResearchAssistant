// @vitest-environment jsdom

import React from 'react'
import { render, screen, act, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest'
import { CodeBlock } from '@/components/chat/code-block'

beforeEach(() => {
  Object.defineProperty(navigator, 'clipboard', {
    value: { writeText: vi.fn().mockResolvedValue(undefined) },
    configurable: true,
    writable: true,
  })
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('CodeBlock', () => {
  describe('rendering', () => {
    it('displays the language label', () => {
      render(<CodeBlock language="typescript" code="const x = 1" />)
      expect(screen.queryByText('typescript')).not.toBeNull()
    })

    it('renders the code content inside a code element', () => {
      const { container } = render(<CodeBlock language="python" code="print('hello')" />)
      const codeEl = container.querySelector('code')
      expect(codeEl).not.toBeNull()
      expect(codeEl!.textContent).toBe("print('hello')")
    })

    it('renders a copy button', () => {
      render(<CodeBlock language="js" code="let x = 1" />)
      expect(screen.queryByRole('button')).not.toBeNull()
    })
  })

  describe('copy behaviour', () => {
    it('copies code to clipboard when copy button is clicked', async () => {
      const code = 'const answer = 42'
      render(<CodeBlock language="js" code={code} />)
      await act(async () => {
        fireEvent.click(screen.getByRole('button'))
        await Promise.resolve()
      })
      expect(navigator.clipboard.writeText).toHaveBeenCalledWith(code)
    })

    it('shows Check icon after copying', async () => {
      const { container } = render(<CodeBlock language="js" code="x = 1" />)
      expect(container.querySelector('.lucide-copy')).not.toBeNull()
      expect(container.querySelector('.lucide-check')).toBeNull()
      await act(async () => {
        fireEvent.click(screen.getByRole('button'))
        await Promise.resolve()
        await Promise.resolve()
      })
      expect(container.querySelector('.lucide-check')).not.toBeNull()
      expect(container.querySelector('.lucide-copy')).toBeNull()
    })

    it('reverts to Copy icon after 2000ms', async () => {
      vi.useFakeTimers()
      const { container } = render(<CodeBlock language="js" code="x = 1" />)
      await act(async () => {
        fireEvent.click(screen.getByRole('button'))
        await Promise.resolve()
        await Promise.resolve()
      })
      expect(container.querySelector('.lucide-check')).not.toBeNull()
      act(() => { vi.advanceTimersByTime(2000) })
      expect(container.querySelector('.lucide-copy')).not.toBeNull()
      expect(container.querySelector('.lucide-check')).toBeNull()
      vi.useRealTimers()
    })
  })
})
