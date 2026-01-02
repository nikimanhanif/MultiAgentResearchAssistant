'use client'

import React, { createContext, useContext, useReducer, useCallback, useEffect, useRef, ReactNode } from 'react'
import type { 
  Message, 
  ChatState, 
  StreamEvent, 
  ResearchProgress,
  ResearchBrief,
  ReviewRequest,
  ConversationSummary,
  ReviewAction,
  ThinkingState,
  ThoughtEntry,
  ThinkingPhase
} from '@/types/chat'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'

// Initial state
const initialProgress: ResearchProgress = {
  phase: 'idle',
  tasksCount: 0,
  findingsCount: 0,
  iterations: 0,
  phaseDurationMs: 0
}

const initialThinking: ThinkingState = {
  isThinking: false,
  agent: '',
  thought: '',
  step: '',
  startTime: 0,
  elapsedMs: 0,
  history: [],
  phases: [],
  currentPhase: ''
}

// Helper to get human-readable phase labels
function getPhaseLabel(phase: string): string {
  switch (phase) {
    case 'investigating': return 'Initial Investigation'
    case 'findings': return 'Research Findings'
    case 'deepening': return 'Deepening Analysis'
    case 'analyzing': return 'Gap Analysis'
    default: return phase.charAt(0).toUpperCase() + phase.slice(1)
  }
}

const initialState: ChatState = {
  messages: [],
  threadId: null,
  userId: '',
  isStreaming: false,
  currentStreamingContent: '',
  researchProgress: initialProgress,
  researchBrief: null,
  reviewRequest: null,
  conversations: [],
  error: null,
  activeNode: null,
  isReportStreaming: false,
  reportPanelOpen: false,
  focusMode: false,
  activeReportContent: null,
  thinking: initialThinking
}

// Action types
type ChatAction =
  | { type: 'SET_USER_ID'; payload: string }
  | { type: 'SET_THREAD_ID'; payload: string }
  | { type: 'ADD_MESSAGE'; payload: Message }
  | { type: 'UPDATE_LAST_MESSAGE'; payload: string }
  | { type: 'SET_STREAMING'; payload: boolean }
  | { type: 'SET_STREAMING_CONTENT'; payload: string }
  | { type: 'APPEND_STREAMING_CONTENT'; payload: string }
  | { type: 'SET_PROGRESS'; payload: Partial<ResearchProgress> }
  | { type: 'SET_BRIEF'; payload: ResearchBrief }
  | { type: 'SET_REVIEW_REQUEST'; payload: ReviewRequest | null }
  | { type: 'SET_CONVERSATIONS'; payload: ConversationSummary[] }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'RESET_CHAT' }
  | { type: 'FINALIZE_STREAMING_MESSAGE' }
  | { type: 'SET_ACTIVE_NODE'; payload: string | null }
  | { type: 'SET_REPORT_STREAMING'; payload: boolean }
  | { type: 'FINALIZE_AND_SWITCH_TO_REPORT' }
  | { type: 'OPEN_REPORT'; payload: string }
  | { type: 'CLOSE_REPORT' }
  | { type: 'TOGGLE_FOCUS_MODE' }
  | { type: 'SET_THINKING'; payload: { agent: string; thought: string; step: string; elapsedMs: number; phase: string } }
  | { type: 'STOP_THINKING' }

// Reducer
function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case 'SET_USER_ID':
      return { ...state, userId: action.payload }
    
    case 'SET_THREAD_ID':
      if (typeof window !== 'undefined') {
        localStorage.setItem('active_thread_id', action.payload)
      }
      return { ...state, threadId: action.payload }
    
    case 'ADD_MESSAGE':
      return { ...state, messages: [...state.messages, action.payload] }
    
    case 'UPDATE_LAST_MESSAGE': {
      const messages = [...state.messages]
      if (messages.length > 0) {
        messages[messages.length - 1] = {
          ...messages[messages.length - 1],
          content: action.payload
        }
      }
      return { ...state, messages }
    }
    
    case 'SET_STREAMING':
      return { ...state, isStreaming: action.payload }
    
    case 'SET_STREAMING_CONTENT':
      return { ...state, currentStreamingContent: action.payload }
    
    case 'APPEND_STREAMING_CONTENT':
      return { 
        ...state, 
        currentStreamingContent: state.currentStreamingContent + action.payload 
      }
    
    case 'FINALIZE_STREAMING_MESSAGE': {
      if (!state.currentStreamingContent) return state
      const newMessage: Message = {
        id: `msg_${Date.now()}`,
        role: 'assistant',
        content: state.currentStreamingContent,
        timestamp: new Date()
      }
      return {
        ...state,
        messages: [...state.messages, newMessage],
        currentStreamingContent: '',
        isStreaming: false
      }
    }
    
    case 'SET_PROGRESS':
      return { 
        ...state, 
        researchProgress: { ...state.researchProgress, ...action.payload } 
      }
    
    case 'SET_BRIEF':
      return { ...state, researchBrief: action.payload }
    
    case 'SET_REVIEW_REQUEST':
      return { ...state, reviewRequest: action.payload }
    
    case 'SET_CONVERSATIONS':
      return { ...state, conversations: action.payload }
    
    case 'SET_ERROR':
      return { ...state, error: action.payload, isStreaming: false, activeNode: null }
    
    case 'RESET_CHAT':
      if (typeof window !== 'undefined') {
        localStorage.removeItem('active_thread_id')
      }
      return {
        ...initialState,
        userId: state.userId,
        conversations: state.conversations
      }
    
    case 'SET_ACTIVE_NODE':
      return { ...state, activeNode: action.payload }
    
    case 'SET_REPORT_STREAMING':
      return { ...state, isReportStreaming: action.payload }
    
    case 'FINALIZE_AND_SWITCH_TO_REPORT': {
      if (state.currentStreamingContent) {
        const scopeMessage: Message = {
          id: `msg_${Date.now()}`,
          role: 'assistant',
          content: state.currentStreamingContent,
          timestamp: new Date()
        }
        return {
          ...state,
          messages: [...state.messages, scopeMessage],
          currentStreamingContent: '',
          isReportStreaming: true
        }
      }
      return { ...state, isReportStreaming: true }
    }
    
    case 'OPEN_REPORT':
      return { 
        ...state, 
        reportPanelOpen: true, 
        focusMode: true,  // Enable focus mode by default when opening
        activeReportContent: action.payload 
      }
    
    case 'CLOSE_REPORT':
      return { 
        ...state, 
        reportPanelOpen: false, 
        focusMode: false,
        activeReportContent: null 
      }
    
    case 'TOGGLE_FOCUS_MODE':
      return { ...state, focusMode: !state.focusMode }
    
    case 'SET_THINKING': {
      const { agent, thought, step, elapsedMs, phase } = action.payload
      const newEntry: ThoughtEntry = {
        id: `thought_${Date.now()}`,
        agent,
        thought,
        step,
        phase,
        timestamp: new Date()
      }
      
      // Group thoughts by phase
      let phases = [...state.thinking.phases]
      const phaseLabel = getPhaseLabel(phase)
      
      const phaseIdx = phases.findIndex(p => p.phase === phase)
      if (phaseIdx === -1 && phase) {
        // New phase - add it
        phases.push({
          id: `phase_${Date.now()}`,
          phase,
          label: phaseLabel,
          thoughts: [newEntry],
          startTime: Date.now()
        })
      } else if (phaseIdx !== -1) {
        // Existing phase - append thought
        phases[phaseIdx] = {
          ...phases[phaseIdx],
          thoughts: [...phases[phaseIdx].thoughts, newEntry]
        }
      }
      
      return {
        ...state,
        thinking: {
          isThinking: true,
          agent,
          thought,
          step,
          startTime: state.thinking.startTime || Date.now(),
          elapsedMs,
          history: [...state.thinking.history, newEntry],
          phases,
          currentPhase: phase
        }
      }
    }
    
    case 'STOP_THINKING':
      return {
        ...state,
        thinking: {
          ...state.thinking,
          isThinking: false
        }
      }
    
    default:
      return state
  }
}

// Context type
interface ChatContextType extends ChatState {
  sendMessage: (content: string) => Promise<void>
  resumeReview: (action: ReviewAction, feedback?: string) => Promise<void>
  loadConversations: () => Promise<void>
  loadConversation: (conversationId: string) => Promise<void>
  deleteConversation: (conversationId: string) => Promise<void>
  startNewChat: () => void
  openReport: (content: string) => void
  closeReport: () => void
  toggleFocusMode: () => void
}

const ChatContext = createContext<ChatContextType | undefined>(undefined)

// Provider component
export function ChatProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(chatReducer, initialState)
  
  // Ref to track report streaming state (avoids stale closure issues in handleStreamEvent)
  const isReportStreamingRef = useRef(false)

  useEffect(() => {
    const userId = 'test_user'
    dispatch({ type: 'SET_USER_ID', payload: userId })
  }, [])



  // Parse SSE event data
  const parseSSEEvent = useCallback((data: string): StreamEvent | null => {
    try {
      return JSON.parse(data) as StreamEvent
    } catch {
      console.error('Failed to parse SSE event:', data)
      return null
    }
  }, [])

  // Handle stream events
  const handleStreamEvent = useCallback((event: StreamEvent) => {
    switch (event.type) {
      case 'token':
        dispatch({ type: 'APPEND_STREAMING_CONTENT', payload: event.content })
        break
      
      case 'report_token':
        if (!isReportStreamingRef.current) {
          isReportStreamingRef.current = true
          dispatch({ type: 'FINALIZE_AND_SWITCH_TO_REPORT' })
          dispatch({ type: 'SET_PROGRESS', payload: { phase: 'generating_report' } })
        }
        dispatch({ type: 'APPEND_STREAMING_CONTENT', payload: event.content })
        break
      
      case 'progress':
        dispatch({ 
          type: 'SET_PROGRESS', 
          payload: {
            phase: event.phase,
            tasksCount: event.tasks_count,
            findingsCount: event.findings_count,
            iterations: event.iterations,
            phaseDurationMs: event.phase_duration_ms || 0
          }
        })
        break
      
      case 'brief_created': {
        dispatch({ type: 'FINALIZE_STREAMING_MESSAGE' })
        dispatch({ 
          type: 'SET_BRIEF', 
          payload: { scope: event.scope, subTopics: event.sub_topics }
        })
        
        const briefMessage: Message = {
          id: `msg_brief_${Date.now()}`,
          role: 'assistant',
          content: `### Research Brief Created\n\n**Scope:** ${event.scope}\n\n**Sub-topics:**\n${event.sub_topics ? event.sub_topics.map((t: string) => `- ${t}`).join('\n') : ''}`,
          timestamp: new Date()
        }
        dispatch({ type: 'ADD_MESSAGE', payload: briefMessage })
        break
      }
      
      case 'clarification_request': {
        const clarificationMessage: Message = {
          id: `msg_${Date.now()}`,
          role: 'assistant',
          content: event.questions,
          timestamp: new Date(),
          node: 'scope'
        }
        dispatch({ type: 'ADD_MESSAGE', payload: clarificationMessage })
        dispatch({ type: 'SET_STREAMING', payload: false })
        dispatch({ type: 'SET_STREAMING_CONTENT', payload: '' })
        break
      }
      
      case 'review_request':
        isReportStreamingRef.current = false
        dispatch({ type: 'FINALIZE_STREAMING_MESSAGE' })
        dispatch({ type: 'SET_STREAMING', payload: false })
        dispatch({ type: 'SET_REPORT_STREAMING', payload: false })
        dispatch({ 
          type: 'SET_REVIEW_REQUEST', 
          payload: { report: event.report, pending: true }
        })
        break
      
      case 'state_update':
        dispatch({ type: 'SET_ACTIVE_NODE', payload: event.node })
        // Don't stop thinking here - let 'complete' event handle final state reset
        break
      
      case 'thought':
        dispatch({ 
          type: 'SET_THINKING', 
          payload: {
            agent: event.agent,
            thought: event.thought,
            step: event.step,
            elapsedMs: event.elapsed_ms,
            phase: event.phase || ''
          }
        })
        break
      
      case 'complete':
        isReportStreamingRef.current = false  // Reset ref
        dispatch({ type: 'FINALIZE_STREAMING_MESSAGE' })
        dispatch({ type: 'SET_REPORT_STREAMING', payload: false }) // Reset for next conversation
        dispatch({ type: 'SET_PROGRESS', payload: { phase: 'complete' } }) // Signal research is complete
        dispatch({ type: 'STOP_THINKING' })
        break
      
      case 'error':
        isReportStreamingRef.current = false  // Reset ref
        dispatch({ type: 'SET_ERROR', payload: event.error })
        dispatch({ type: 'SET_REPORT_STREAMING', payload: false }) // Reset on error
        dispatch({ type: 'STOP_THINKING' })
        break
    }
  }, [])  // No dependency on state.isReportStreaming - using ref instead

  // Send a message
  const sendMessage = useCallback(async (content: string) => {
    // Add user message immediately
    const userMessage: Message = {
      id: `msg_${Date.now()}`,
      role: 'user',
      content,
      timestamp: new Date()
    }
    dispatch({ type: 'ADD_MESSAGE', payload: userMessage })
    dispatch({ type: 'SET_STREAMING', payload: true })
    dispatch({ type: 'SET_STREAMING_CONTENT', payload: '' })
    dispatch({ type: 'SET_REPORT_STREAMING', payload: false })
    dispatch({ type: 'SET_ERROR', payload: null })

    try {
      const messagesHistory = state.messages.map(msg => ({
        role: msg.role,
        content: msg.content
      }))

      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: content,
          messages: messagesHistory,  // Full conversation history
          thread_id: state.threadId,
          user_id: state.userId
        })
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`)
      }

      // Get thread ID from header
      const threadId = response.headers.get('X-Thread-ID')
      if (threadId && threadId !== state.threadId) {
        dispatch({ type: 'SET_THREAD_ID', payload: threadId })
      }

      // Read SSE stream
      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (!reader) {
        throw new Error('No response body')
      }

      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        
        // Parse SSE events from buffer
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''  // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            const event = parseSSEEvent(data)
            if (event) handleStreamEvent(event)
          }
        }
      }

      // Handle any remaining data
      if (buffer.startsWith('data: ')) {
        const data = buffer.slice(6)
        const event = parseSSEEvent(data)
        if (event) {
          handleStreamEvent(event)
        }
      }

    } catch (error) {
      console.error('Error sending message:', error)
      dispatch({ 
        type: 'SET_ERROR', 
        payload: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }, [state.messages, state.threadId, state.userId, parseSSEEvent, handleStreamEvent])

  // Resume review with action
  const resumeReview = useCallback(async (action: ReviewAction, feedback?: string) => {
    if (!state.threadId) {
      dispatch({ type: 'SET_ERROR', payload: 'No active thread' })
      return
    }

    dispatch({ type: 'SET_STREAMING', payload: true })
    dispatch({ type: 'FINALIZE_STREAMING_MESSAGE' })
    dispatch({ type: 'SET_STREAMING_CONTENT', payload: '' })
    dispatch({ type: 'SET_REVIEW_REQUEST', payload: null })

    try {
      const response = await fetch(`${API_URL}/chat/${state.threadId}/resume`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, feedback })
      })

      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`)
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (!reader) {
        throw new Error('No response body')
      }

      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const event = parseSSEEvent(line.slice(6))
            if (event) handleStreamEvent(event)
          }
        }
      }
    } catch (error) {
      console.error('Error resuming review:', error)
      dispatch({ 
        type: 'SET_ERROR', 
        payload: error instanceof Error ? error.message : 'Unknown error'
      })
    } finally {
      // Ensure streaming state is reset even if no complete event was received
      dispatch({ type: 'SET_STREAMING', payload: false })
      // Refresh conversation list to show updated status
      if (state.userId) {
        try {
          const refreshResponse = await fetch(`${API_URL}/conversations/${state.userId}`)
          if (refreshResponse.ok) {
            const conversations = await refreshResponse.json()
            dispatch({ type: 'SET_CONVERSATIONS', payload: conversations })
          }
        } catch (refreshError) {
          console.error('Error refreshing conversations:', refreshError)
        }
      }
    }
  }, [state.threadId, state.userId, parseSSEEvent, handleStreamEvent])

  // Load conversation list
  const loadConversations = useCallback(async () => {
    if (!state.userId) return

    try {
      const response = await fetch(`${API_URL}/conversations/${state.userId}`)
      if (response.ok) {
        const data = await response.json()
        dispatch({ type: 'SET_CONVERSATIONS', payload: data })
      }
    } catch (error) {
      console.error('Error loading conversations:', error)
    }
  }, [state.userId])

  // Load specific conversation
  const loadConversation = useCallback(async (conversationId: string) => {
    if (!state.userId) return

    try {
      const response = await fetch(
        `${API_URL}/conversations/${state.userId}/${conversationId}`
      )
      if (response.ok) {
        const data = await response.json()
        dispatch({ type: 'RESET_CHAT' })
        dispatch({ type: 'SET_THREAD_ID', payload: conversationId })
        
        // Reconstruct messages from saved data
        // Priority: Use full message history if available
        if (data.messages && data.messages.length > 0) {
          data.messages.forEach((msg: any) => {
             const restoredMessage: Message = {
              id: `msg_restored_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
              role: msg.role as 'user' | 'assistant',
              content: msg.content,
              timestamp: new Date(data.created_at) // Approximate timestamp
            }
            dispatch({ type: 'ADD_MESSAGE', payload: restoredMessage })
          })
        } else {
          // Fallback: Add original user query if no history
          if (data.user_query) {
            const userMessage: Message = {
              id: `msg_restored_user_${Date.now()}`,
              role: 'user',
              content: data.user_query,
              timestamp: new Date(data.created_at)
            }
            dispatch({ type: 'ADD_MESSAGE', payload: userMessage })
          }
        }
        
        const status = data.status || 'complete'
        
        if (status === 'waiting_review') {
          // Fetch the conversation state to check for pending interrupt
          try {
            const stateResponse = await fetch(
              `${API_URL}/conversations/${state.userId}/${conversationId}/state`
            )
            if (stateResponse.ok) {
              const stateData = await stateResponse.json()
              
              if (stateData.report_content) {
                dispatch({ 
                  type: 'SET_REVIEW_REQUEST', 
                  payload: { report: stateData.report_content, pending: true }
                })
                dispatch({ type: 'SET_PROGRESS', payload: { phase: 'review' } })
                
                const hasReportMessage = data.messages?.some((m: any) => m.content === stateData.report_content);
                if (!hasReportMessage) {
                  const reportMessage: Message = {
                    id: `msg_restored_report_${Date.now()}`,
                    role: 'assistant',
                    content: stateData.report_content,
                    timestamp: new Date(data.created_at)
                  }
                  dispatch({ type: 'ADD_MESSAGE', payload: reportMessage })
                }
              }
            }
          } catch (stateError) {
            console.error('Error fetching conversation state:', stateError)
          }
        } else if (status === 'in_progress') {
          dispatch({ type: 'SET_PROGRESS', payload: { phase: data.phase || 'scoping' } })
          
          const messages = data.messages || []
          const lastMessage = messages.length > 0 ? messages[messages.length - 1] : null
          const shouldContinue = lastMessage && lastMessage.role === 'user'

          if (!shouldContinue) {
             console.log('Restored conversation waiting for user input. Not triggering continue.')
             return 
          }

          dispatch({ type: 'SET_STREAMING', payload: true })
          
          // Call the continue endpoint to resume graph execution
          try {
            const continueResponse = await fetch(
              `${API_URL}/chat/${conversationId}/continue`,
              { method: 'POST' }
            )
            
            if (continueResponse.ok && continueResponse.body) {
              const reader = continueResponse.body.getReader()
              const decoder = new TextDecoder()
              let buffer = ''
              
              while (true) {
                const { done, value } = await reader.read()
                if (done) break
                
                buffer += decoder.decode(value, { stream: true })
                const lines = buffer.split('\n')
                buffer = lines.pop() || ''
                
                for (const line of lines) {
                  if (line.startsWith('data: ')) {
                    const eventData = parseSSEEvent(line.slice(6))
                    if (eventData) {
                      handleStreamEvent(eventData)
                    }
                  }
                }
              }
            }
          } catch (continueError) {
            console.error('Error continuing conversation:', continueError)
            // Fallback to showing a note if continue fails
            const noteMessage: Message = {
              id: `msg_continue_error_${Date.now()}`,
              role: 'assistant',
              content: `*Could not resume research. Please try starting a new conversation.*`,
              timestamp: new Date()
            }
            dispatch({ type: 'ADD_MESSAGE', payload: noteMessage })
          } finally {
            dispatch({ type: 'SET_STREAMING', payload: false })
          }
        } else {
          // Complete conversation - show the report
          if (data.report_content) {
            // Check if report is already in messages
             const hasReportMessage = data.messages?.some((m: any) => m.content === data.report_content);
             if (!hasReportMessage) {
                const reportMessage: Message = {
                  id: `msg_restored_report_${Date.now()}`,
                  role: 'assistant',
                  content: data.report_content,
                  timestamp: new Date(data.created_at)
                }
                dispatch({ type: 'ADD_MESSAGE', payload: reportMessage })
             }
          }
          dispatch({ type: 'SET_PROGRESS', payload: { phase: 'complete' } })
        }
      }
    } catch (error) {
      console.error('Error loading conversation:', error)
      // If we failed to load the saved thread, clear it to prevent bad state
      if (typeof window !== 'undefined' && localStorage.getItem('active_thread_id') === conversationId) {
        localStorage.removeItem('active_thread_id')
        dispatch({ type: 'SET_ERROR', payload: 'Could not restore previous session' })
      }
    }
  }, [state.userId, parseSSEEvent, handleStreamEvent])

  // Start new chat
  const startNewChat = useCallback(() => {
    dispatch({ type: 'RESET_CHAT' })
  }, [])

  // Research workspace panel controls
  const openReport = useCallback((content: string) => {
    dispatch({ type: 'OPEN_REPORT', payload: content })
  }, [])

  const closeReport = useCallback(() => {
    dispatch({ type: 'CLOSE_REPORT' })
  }, [])

  const toggleFocusMode = useCallback(() => {
    dispatch({ type: 'TOGGLE_FOCUS_MODE' })
  }, [])

  // Delete a conversation
  const deleteConversation = useCallback(async (conversationId: string) => {
    if (!state.userId) return

    try {
      const response = await fetch(
        `${API_URL}/conversations/${state.userId}/${conversationId}`,
        { method: 'DELETE' }
      )
      if (response.ok) {
        // Refresh the conversation list
        await loadConversations()
        // If we deleted the currently active conversation, reset chat
        if (state.threadId === conversationId) {
          dispatch({ type: 'RESET_CHAT' })
        }
      }
    } catch (error) {
      console.error('Error deleting conversation:', error)
    }
  }, [state.userId, state.threadId, loadConversations])

  // Load conversations on mount
  useEffect(() => {
    if (state.userId) {
      loadConversations()
    }
  }, [state.userId, loadConversations])

  // Restore conversation from localStorage
  useEffect(() => {
    if (!state.userId) return

    if (typeof window !== 'undefined') {
      const savedThreadId = localStorage.getItem('active_thread_id')
      if (savedThreadId && !state.threadId) {
        loadConversation(savedThreadId)
      }
    }
  }, [state.userId, loadConversation, state.threadId])

  const value: ChatContextType = {
    ...state,
    sendMessage,
    resumeReview,
    loadConversations,
    loadConversation,
    deleteConversation,
    startNewChat,
    openReport,
    closeReport,
    toggleFocusMode
  }

  return (
    <ChatContext.Provider value={value}>
      {children}
    </ChatContext.Provider>
  )
}

// Hook to use chat context
export function useChatContext() {
  const context = useContext(ChatContext)
  if (context === undefined) {
    throw new Error('useChatContext must be used within a ChatProvider')
  }
  return context
}
