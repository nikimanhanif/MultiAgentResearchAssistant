// @vitest-environment jsdom

import React from 'react'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi, describe, it, expect, beforeEach } from 'vitest'
import { FloatingInput } from '@/components/chat/floating-input'
import { useChatContext } from '@/context/chat-context'

vi.mock('@/context/chat-context', () => ({
  useChatContext: vi.fn(),
}))

const mockSendMessage = vi.fn()

const defaultContext = { sendMessage: mockSendMessage, isStreaming: false }

beforeEach(() => {
  vi.mocked(useChatContext).mockReturnValue(defaultContext as any)
  mockSendMessage.mockReset()
})

describe('FloatingInput', () => {
  describe('rendering', () => {
    it('renders a textarea with placeholder "Ask a research question..."', () => {
      render(<FloatingInput />)
      expect(screen.queryByPlaceholderText('Ask a research question...')).not.toBeNull()
    })

    it('renders a submit button', () => {
      render(<FloatingInput />)
      expect(screen.queryByRole('button')).not.toBeNull()
    })

    it('shows "Researching..." placeholder when isStreaming is true', () => {
      vi.mocked(useChatContext).mockReturnValue({ ...defaultContext, isStreaming: true } as any)
      render(<FloatingInput />)
      expect(screen.queryByPlaceholderText('Researching...')).not.toBeNull()
    })

    it('disables the textarea when isStreaming is true', () => {
      vi.mocked(useChatContext).mockReturnValue({ ...defaultContext, isStreaming: true } as any)
      render(<FloatingInput />)
      const textarea = screen.getByPlaceholderText('Researching...') as HTMLTextAreaElement
      expect(textarea.disabled).toBe(true)
    })

    it('disables the submit button when input is empty', () => {
      render(<FloatingInput />)
      const button = screen.getByRole('button') as HTMLButtonElement
      expect(button.disabled).toBe(true)
    })

    it('disables the submit button when isStreaming is true', () => {
      vi.mocked(useChatContext).mockReturnValue({ ...defaultContext, isStreaming: true } as any)
      render(<FloatingInput />)
      const button = screen.getByRole('button') as HTMLButtonElement
      expect(button.disabled).toBe(true)
    })

    it('enables the submit button when input has non-whitespace content', async () => {
      const user = userEvent.setup()
      render(<FloatingInput />)
      const textarea = screen.getByPlaceholderText('Ask a research question...')
      await user.type(textarea, 'hello')
      const button = screen.getByRole('button') as HTMLButtonElement
      expect(button.disabled).toBe(false)
    })
  })

  describe('input behaviour', () => {
    it('updates textarea value as user types', async () => {
      const user = userEvent.setup()
      render(<FloatingInput />)
      const textarea = screen.getByPlaceholderText('Ask a research question...') as HTMLTextAreaElement
      await user.type(textarea, 'test query')
      expect(textarea.value).toBe('test query')
    })

    it('trims whitespace — input of only spaces does not enable the button', async () => {
      const user = userEvent.setup()
      render(<FloatingInput />)
      const textarea = screen.getByPlaceholderText('Ask a research question...')
      await user.type(textarea, '   ')
      const button = screen.getByRole('button') as HTMLButtonElement
      expect(button.disabled).toBe(true)
    })
  })

  describe('submission', () => {
    it('calls sendMessage with the trimmed input on form submit', async () => {
      const user = userEvent.setup()
      render(<FloatingInput />)
      const textarea = screen.getByPlaceholderText('Ask a research question...')
      await user.type(textarea, 'hello world')
      await user.click(screen.getByRole('button'))
      expect(mockSendMessage).toHaveBeenCalledWith('hello world')
    })

    it('clears the input after submission', async () => {
      const user = userEvent.setup()
      render(<FloatingInput />)
      const textarea = screen.getByPlaceholderText('Ask a research question...') as HTMLTextAreaElement
      await user.type(textarea, 'my question')
      await user.click(screen.getByRole('button'))
      expect(textarea.value).toBe('')
    })

    it('calls sendMessage when Enter is pressed without Shift', async () => {
      const user = userEvent.setup()
      render(<FloatingInput />)
      const textarea = screen.getByPlaceholderText('Ask a research question...')
      await user.type(textarea, 'my question')
      await user.keyboard('{Enter}')
      expect(mockSendMessage).toHaveBeenCalledWith('my question')
    })

    it('does NOT call sendMessage when Shift+Enter is pressed', async () => {
      const user = userEvent.setup()
      render(<FloatingInput />)
      const textarea = screen.getByPlaceholderText('Ask a research question...')
      await user.type(textarea, 'my question')
      await user.keyboard('{Shift>}{Enter}{/Shift}')
      expect(mockSendMessage).not.toHaveBeenCalled()
    })

    it('does NOT call sendMessage when input is empty and Enter is pressed', async () => {
      const user = userEvent.setup()
      render(<FloatingInput />)
      const textarea = screen.getByPlaceholderText('Ask a research question...')
      await user.click(textarea)
      await user.keyboard('{Enter}')
      expect(mockSendMessage).not.toHaveBeenCalled()
    })

    it('does NOT call sendMessage when isStreaming is true', () => {
      vi.mocked(useChatContext).mockReturnValue({ ...defaultContext, isStreaming: true } as any)
      render(<FloatingInput />)
      const button = screen.getByRole('button') as HTMLButtonElement
      expect(button.disabled).toBe(true)
      expect(mockSendMessage).not.toHaveBeenCalled()
    })
  })

  describe('streaming state icon', () => {
    it('shows Square icon (stop indicator) when isStreaming is true', () => {
      vi.mocked(useChatContext).mockReturnValue({ ...defaultContext, isStreaming: true } as any)
      const { container } = render(<FloatingInput />)
      expect(container.querySelector('.lucide-square')).not.toBeNull()
      expect(container.querySelector('.lucide-arrow-up')).toBeNull()
    })

    it('shows ArrowUp icon (send indicator) when isStreaming is false', () => {
      const { container } = render(<FloatingInput />)
      expect(container.querySelector('.lucide-arrow-up')).not.toBeNull()
      expect(container.querySelector('.lucide-square')).toBeNull()
    })
  })
})
