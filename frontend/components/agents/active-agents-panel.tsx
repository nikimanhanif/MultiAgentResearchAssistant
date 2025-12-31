'use client'

import { useChatContext } from '@/context/chat-context'
import { cn } from '@/lib/utils'
import { 
  Search, 
  FileText, 
  Brain, 
  CheckCircle2, 
  Circle,
  Activity,
  TrendingUp
} from 'lucide-react'
import type { ResearchPhase } from '@/types/chat'

interface AgentConfig {
  id: string
  name: string
  icon: React.ElementType
  description: string
  phases: ResearchPhase[]
}

const agents: AgentConfig[] = [
  {
    id: 'scope',
    name: 'Scope Agent',
    icon: Search,
    description: 'Understanding your research query',
    phases: ['scoping']
  },
  {
    id: 'supervisor',
    name: 'Research',
    icon: Brain,
    description: 'Analyzing and researching topics',
    phases: ['researching']
  },
  {
    id: 'report_agent',
    name: 'Report Agent',
    icon: FileText,
    description: 'Generating final report',
    phases: ['generating_report']
  },
  {
    id: 'reviewer',
    name: 'Reviewer',
    icon: CheckCircle2,
    description: 'Awaiting your review',
    phases: ['review']
  }
]

export function ActiveAgentsPanel() {
  const { researchProgress, isStreaming, activeNode } = useChatContext()
  const { phase, tasksCount, findingsCount, iterations } = researchProgress

  // Use phase-based detection for research phase (supervisor loop transitions rapidly)
  const isAgentActive = (agent: AgentConfig) => {
    if (!isStreaming) return false
    // Research phase uses phase-based detection - supervisor loop has rapid transitions
    if (phase === 'researching' && agent.id === 'supervisor') {
      return true
    }
    // For other phases, use activeNode or phase-based inference
    if (activeNode) {
      return agent.id === activeNode || 
             (activeNode === 'sub_agent' && agent.id === 'supervisor')
    }
    return agent.phases.includes(phase)
  }

  const isAgentComplete = (agent: AgentConfig) => {
    if (phase === 'idle') return false
    const agentPhaseIndex = agents.findIndex(a => a.id === agent.id)
    const currentPhaseIndex = agents.findIndex(a => a.phases.includes(phase))
    return agentPhaseIndex < currentPhaseIndex || phase === 'complete'
  }

  return (
    <div className="flex flex-col h-full bg-card/30">
      {/* Header */}
      <div className="p-4 border-b border-subtle">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-primary" />
          <span className="font-semibold text-sm">Active Agents</span>
          {isStreaming && (
            <span className="ml-auto flex items-center gap-1.5 text-xs text-muted-foreground">
              <span className="h-2 w-2 bg-green-500 rounded-full animate-pulse" />
              Live
            </span>
          )}
        </div>
      </div>

      {/* Agent List */}
      <div className="flex-1 overflow-auto p-4">
        <div className="space-y-2">
          {agents.map((agent) => {
            const isActive = isAgentActive(agent)
            const isComplete = isAgentComplete(agent)
            const Icon = agent.icon

            return (
              <div
                key={agent.id}
                className={cn(
                  'p-3 rounded-lg border transition-all duration-300',
                  isActive && 'border-primary/50 bg-primary/5 agent-active',
                  isComplete && 'border-green-500/30 bg-green-500/5',
                  !isActive && !isComplete && 'border-subtle bg-transparent opacity-50'
                )}
              >
                <div className="flex items-start gap-3">
                  <div className={cn(
                    'p-2 rounded-md transition-colors',
                    isActive && 'bg-primary/20 text-primary',
                    isComplete && 'bg-green-500/20 text-green-500',
                    !isActive && !isComplete && 'bg-muted text-muted-foreground'
                  )}>
                    <Icon className="h-4 w-4" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className={cn(
                        'text-sm font-medium',
                        isActive && 'text-primary',
                        isComplete && 'text-green-500'
                      )}>
                        {agent.name}
                      </span>
                      {isComplete && (
                        <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground mt-0.5 font-mono">
                      {isActive ? 'In Progress' : isComplete ? 'Completed' : 'Pending'}
                    </p>
                  </div>
                </div>
              </div>
            )
          })}
        </div>

        {/* Progress Stats */}
        {(tasksCount > 0 || findingsCount > 0) && (
          <div className="mt-6 pt-4 border-t border-subtle">
            <div className="flex items-center gap-2 mb-3">
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium text-muted-foreground">Progress</span>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-lg bg-muted/30 border border-subtle">
                <div className="text-2xl font-bold font-mono text-foreground">
                  {tasksCount}
                </div>
                <div className="text-xs text-muted-foreground">Tasks</div>
              </div>
              <div className="p-3 rounded-lg bg-muted/30 border border-subtle">
                <div className="text-2xl font-bold font-mono text-foreground">
                  {findingsCount}
                </div>
                <div className="text-xs text-muted-foreground">Findings</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
