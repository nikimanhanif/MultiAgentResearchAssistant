// Chat and streaming types for the research assistant

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  node?: string; // Which node generated this message (scope, supervisor, report_agent, etc.)
}

export interface Chat {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

// SSE Stream Event Types
export type StreamEventType =
  | "token"
  | "progress"
  | "state_update"
  | "brief_created"
  | "report_token"
  | "clarification_request"
  | "review_request"
  | "complete"
  | "error";

export interface BaseStreamEvent {
  type: StreamEventType;
}

export interface TokenEvent extends BaseStreamEvent {
  type: "token";
  content: string;
  node: string;
}

export interface ProgressEvent extends BaseStreamEvent {
  type: "progress";
  phase: ResearchPhase;
  tasks_count: number;
  findings_count: number;
  iterations: number;
  phase_duration_ms: number;
}

export interface StateUpdateEvent extends BaseStreamEvent {
  type: "state_update";
  node: string;
  is_complete: boolean;
}

export interface BriefCreatedEvent extends BaseStreamEvent {
  type: "brief_created";
  scope: string;
  sub_topics: string[];
}

export interface ReportTokenEvent extends BaseStreamEvent {
  type: "report_token";
  content: string;
}

export interface ClarificationRequestEvent extends BaseStreamEvent {
  type: "clarification_request";
  questions: string;
}

export interface ReviewRequestEvent extends BaseStreamEvent {
  type: "review_request";
  report: string;
}

export interface CompleteEvent extends BaseStreamEvent {
  type: "complete";
  message: string;
}

export interface ErrorEvent extends BaseStreamEvent {
  type: "error";
  error: string;
}

export type StreamEvent =
  | TokenEvent
  | ProgressEvent
  | StateUpdateEvent
  | BriefCreatedEvent
  | ReportTokenEvent
  | ClarificationRequestEvent
  | ReviewRequestEvent
  | CompleteEvent
  | ErrorEvent;

// Research State Types
export type ResearchPhase =
  | "idle"
  | "scoping"
  | "researching"
  | "generating_report"
  | "review"
  | "complete";

export interface ResearchProgress {
  phase: ResearchPhase;
  tasksCount: number;
  findingsCount: number;
  iterations: number;
  phaseDurationMs: number;
}

export interface ResearchBrief {
  scope: string;
  subTopics: string[];
}

// Review Types
export type ReviewAction = "approve" | "refine" | "re_research";

export interface ReviewRequest {
  report: string;
  pending: boolean;
}

// Conversation History Types
export type ConversationStatus = "in_progress" | "waiting_review" | "complete";

export interface ConversationSummary {
  conversation_id: string;
  user_query: string;
  created_at: string;
  status: ConversationStatus;
  phase?: string;
}

export interface ConversationDetail {
  conversation_id: string;
  user_query: string;
  report_content: string;
  findings_count: number;
  created_at: string;
  status: ConversationStatus;
  phase?: string;
}

// Chat Context State
export interface ChatState {
  messages: Message[];
  threadId: string | null;
  userId: string;
  isStreaming: boolean;
  currentStreamingContent: string;
  researchProgress: ResearchProgress;
  researchBrief: ResearchBrief | null;
  reviewRequest: ReviewRequest | null;
  conversations: ConversationSummary[];
  error: string | null;
  activeNode: string | null;
  isReportStreaming: boolean;
  reportPanelOpen: boolean;
  focusMode: boolean;
  activeReportContent: string | null;
}
