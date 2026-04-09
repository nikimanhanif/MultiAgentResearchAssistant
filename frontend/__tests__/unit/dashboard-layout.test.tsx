// @vitest-environment jsdom

import React from 'react'
import { render, screen } from '@testing-library/react'
import { vi, describe, it, expect } from 'vitest'
import { DashboardLayout } from '@/components/layout/dashboard-layout'

// Mock child panels — tested in isolation elsewhere
vi.mock('@/components/sidebar/research-history', () => ({
  ResearchHistory: () => <div data-testid="research-history" />,
}))

vi.mock('@/components/agents/active-agents-panel', () => ({
  ActiveAgentsPanel: () => <div data-testid="active-agents-panel" />,
}))

vi.mock('@/components/chat/review-modal', () => ({
  ReviewModal: () => <div data-testid="review-modal" />,
}))

// ChatProvider requires useChatContext consumers — mock it as a passthrough
vi.mock('@/context/chat-context', () => ({
  ChatProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  useChatContext: vi.fn(),
}))

// react-resizable-panels uses DOM measurement APIs not present in jsdom
vi.mock('@/components/ui/resizable', () => ({
  ResizablePanelGroup: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="panel-group">{children}</div>
  ),
  ResizablePanel: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="panel">{children}</div>
  ),
  ResizableHandle: () => <div data-testid="resize-handle" />,
}))

describe('DashboardLayout', () => {
  it('renders ResearchHistory panel', () => {
    render(<DashboardLayout>content</DashboardLayout>)
    expect(screen.queryByTestId('research-history')).not.toBeNull()
  })

  it('renders ActiveAgentsPanel', () => {
    render(<DashboardLayout>content</DashboardLayout>)
    expect(screen.queryByTestId('active-agents-panel')).not.toBeNull()
  })

  it('renders ReviewModal', () => {
    render(<DashboardLayout>content</DashboardLayout>)
    expect(screen.queryByTestId('review-modal')).not.toBeNull()
  })

  it('renders children in the center panel', () => {
    render(<DashboardLayout><span data-testid="child-content">hello</span></DashboardLayout>)
    expect(screen.queryByTestId('child-content')).not.toBeNull()
  })

  it('renders two resize handles', () => {
    render(<DashboardLayout>content</DashboardLayout>)
    expect(screen.getAllByTestId('resize-handle')).toHaveLength(2)
  })
})
