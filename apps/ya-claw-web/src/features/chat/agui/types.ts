import type { AguiEvent, InputPart } from '../../../types'

export type TimelineBlock =
  | UserInputBlock
  | AssistantMessageBlock
  | ReasoningBlock
  | ToolCallBlock
  | RuntimeEventBlock
  | TaskBoardBlock
  | ContextMeterBlock
  | UsageBlock
  | SubagentBlock
  | FileChangeBlock
  | NoteSnapshotBlock
  | SteeringBlock
  | RawCustomBlock

export type UserInputBlock = {
  kind: 'user_input'
  id: string
  runId: string
  parts: InputPart[]
}

export type AssistantMessageBlock = {
  kind: 'assistant_message'
  id: string
  messageId: string
  role: string
  name?: string
  content: string
}

export type ReasoningBlock = {
  kind: 'reasoning'
  id: string
  messageId: string
  content: string
}

export type ToolCallBlock = {
  kind: 'tool_call'
  id: string
  toolCallId: string
  name?: string
  args: string
  result?: string
  status: 'calling' | 'completed' | 'failed'
}

export type RuntimeEventBlock = {
  kind: 'runtime_event'
  id: string
  name: string
  title: string
  status: 'info' | 'running' | 'success' | 'warning' | 'error'
  payload: unknown
}

export type TaskInfo = {
  id: string
  subject: string
  description?: string
  status: 'pending' | 'in_progress' | 'completed'
  active_form?: string | null
  owner?: string | null
  blocked_by?: string[]
  blocks?: string[]
}

export type TaskBoardBlock = {
  kind: 'task_board'
  id: string
  tasks: TaskInfo[]
}

export type ContextMeterBlock = {
  kind: 'context_meter'
  id: string
  totalTokens: number
  contextWindowSize: number
}

export type UsageBlock = {
  kind: 'usage'
  id: string
  payload: unknown
}

export type SubagentBlock = {
  kind: 'subagent'
  id: string
  agentId: string
  agentName: string
  status: 'running' | 'completed' | 'failed'
  promptPreview?: string
  resultPreview?: string
  error?: string
  durationSeconds?: number
  requestCount?: number
}

export type FileChangeBlock = {
  kind: 'file_change'
  id: string
  changes: unknown[]
  toolName?: string
}

export type NoteSnapshotBlock = {
  kind: 'note_snapshot'
  id: string
  entries: Record<string, string>
}

export type SteeringBlock = {
  kind: 'steering'
  id: string
  title: string
  status: 'delivered' | 'injected'
  inputParts: InputPart[]
  prompt?: unknown
  delivery?: string
  payload: unknown
}

export type RawCustomBlock = {
  kind: 'raw_custom'
  id: string
  name: string
  payload: unknown
}

export type AguiTimelineState = {
  blocks: TimelineBlock[]
  rawEvents: AguiEvent[]
}
