import type { AguiEvent, InputPart } from '../../../types'
import { safeJsonStringify } from '../../../lib/utils'
import type {
  AguiTimelineState,
  ContextMeterBlock,
  FileChangeBlock,
  NoteSnapshotBlock,
  RawCustomBlock,
  RuntimeEventBlock,
  SteeringBlock,
  SubagentBlock,
  TaskBoardBlock,
  TaskInfo,
  TimelineBlock,
  ToolCallBlock,
  UsageBlock,
} from './types'

export function createInitialTimelineState(): AguiTimelineState {
  return { blocks: [], rawEvents: [] }
}

export type TimelineReduceOptions = {
  includeRuntimeEvents?: boolean
}

export function buildTimeline(
  events: AguiEvent[],
  inputParts: InputPart[] = [],
  runId = 'run',
  options: TimelineReduceOptions = {},
): AguiTimelineState {
  let state = createInitialTimelineState()
  if (inputParts.length > 0) {
    state = appendBlock(state, {
      kind: 'user_input',
      id: `${runId}:input`,
      runId,
      parts: inputParts,
    })
  }
  for (const event of events) {
    state = reduceAguiEvent(state, event, options)
  }
  return state
}

export function buildTimelineFromRuns(
  runs: Array<{
    id: string
    input_parts?: InputPart[] | null
    message?: AguiEvent[] | null
  }>,
  options: TimelineReduceOptions = {},
): AguiTimelineState {
  return runs.reduce((state, run) => {
    let nextState = state
    if (run.input_parts?.length) {
      nextState = appendBlock(nextState, {
        kind: 'user_input',
        id: `${run.id}:input`,
        runId: run.id,
        parts: run.input_parts,
      })
    }
    return (run.message ?? []).reduce(
      (runState, event) => reduceAguiEvent(runState, event, options),
      nextState,
    )
  }, createInitialTimelineState())
}

export function reduceAguiEvent(
  state: AguiTimelineState,
  event: AguiEvent,
  options: TimelineReduceOptions = {},
): AguiTimelineState {
  const nextState: AguiTimelineState = {
    blocks: [...state.blocks],
    rawEvents: [...state.rawEvents, event],
  }

  const eventType = stringField(event, 'type')
  if (eventType === 'TEXT_MESSAGE_CHUNK') {
    return mergeAssistantMessage(nextState, event)
  }
  if (eventType === 'REASONING_MESSAGE_CHUNK') {
    return mergeReasoning(nextState, event)
  }
  if (eventType === 'TOOL_CALL_CHUNK') {
    return mergeToolCall(nextState, event)
  }
  if (eventType === 'TOOL_CALL_RESULT') {
    return mergeToolResult(nextState, event)
  }
  if (
    eventType === 'RUN_STARTED' ||
    eventType === 'RUN_FINISHED' ||
    eventType === 'RUN_ERROR'
  ) {
    return options.includeRuntimeEvents === false
      ? nextState
      : appendBlock(nextState, runtimeEventFromAgui(eventType, event))
  }
  if (eventType === 'CUSTOM') {
    const block = blockFromCustomEvent(event)
    return options.includeRuntimeEvents === false &&
      block.kind === 'runtime_event'
      ? nextState
      : appendBlock(nextState, block)
  }
  return nextState
}

function mergeAssistantMessage(
  state: AguiTimelineState,
  event: AguiEvent,
): AguiTimelineState {
  const messageId =
    identifierField(event, 'messageId', 'message_id') ??
    `message:${state.blocks.length}`
  const delta = stringField(event, 'delta') ?? ''
  const existingIndex = state.blocks.findIndex(
    (block) =>
      block.kind === 'assistant_message' && block.messageId === messageId,
  )
  if (existingIndex >= 0) {
    const existing = state.blocks[existingIndex]
    if (existing.kind === 'assistant_message') {
      state.blocks[existingIndex] = {
        ...existing,
        content: `${existing.content}${delta}`,
      }
    }
    return state
  }
  return appendBlock(state, {
    kind: 'assistant_message',
    id: `assistant:${messageId}`,
    messageId,
    role: stringField(event, 'role') ?? 'assistant',
    name: stringField(event, 'name') ?? undefined,
    content: delta,
  })
}

function mergeReasoning(
  state: AguiTimelineState,
  event: AguiEvent,
): AguiTimelineState {
  const messageId =
    identifierField(event, 'messageId', 'message_id') ??
    `reasoning:${state.blocks.length}`
  const delta = stringField(event, 'delta') ?? ''
  const existingIndex = state.blocks.findIndex(
    (block) => block.kind === 'reasoning' && block.messageId === messageId,
  )
  if (existingIndex >= 0) {
    const existing = state.blocks[existingIndex]
    if (existing.kind === 'reasoning') {
      state.blocks[existingIndex] = {
        ...existing,
        content: `${existing.content}${delta}`,
      }
    }
    return state
  }
  return appendBlock(state, {
    kind: 'reasoning',
    id: `reasoning:${messageId}`,
    messageId,
    content: delta,
  })
}

function mergeToolCall(
  state: AguiTimelineState,
  event: AguiEvent,
): AguiTimelineState {
  const toolCallId =
    identifierField(event, 'toolCallId', 'tool_call_id') ??
    `tool:${state.blocks.length}`
  const delta = stringField(event, 'delta') ?? ''
  const existingIndex = state.blocks.findIndex(
    (block) => block.kind === 'tool_call' && block.toolCallId === toolCallId,
  )
  if (existingIndex >= 0) {
    const existing = state.blocks[existingIndex]
    if (existing.kind === 'tool_call') {
      state.blocks[existingIndex] = {
        ...existing,
        name:
          existing.name ??
          stringField(event, 'toolCallName', 'tool_call_name') ??
          undefined,
        args: `${existing.args}${delta}`,
      }
    }
    return state
  }
  return appendBlock(state, {
    kind: 'tool_call',
    id: `tool:${toolCallId}`,
    toolCallId,
    name: stringField(event, 'toolCallName', 'tool_call_name') ?? undefined,
    args: delta,
    status: 'calling',
  })
}

function mergeToolResult(
  state: AguiTimelineState,
  event: AguiEvent,
): AguiTimelineState {
  const toolCallId =
    identifierField(event, 'toolCallId', 'tool_call_id') ??
    `tool:${state.blocks.length}`
  const content =
    event.content === undefined
      ? safeJsonStringify(event)
      : stringifyValue(event.content)
  const existingIndex = state.blocks.findIndex(
    (block) => block.kind === 'tool_call' && block.toolCallId === toolCallId,
  )
  if (existingIndex >= 0) {
    const existing = state.blocks[existingIndex]
    if (existing.kind === 'tool_call') {
      state.blocks[existingIndex] = {
        ...existing,
        result: content,
        status: event.error ? 'failed' : 'completed',
      } satisfies ToolCallBlock
    }
    return state
  }
  return appendBlock(state, {
    kind: 'tool_call',
    id: `tool:${toolCallId}`,
    toolCallId,
    args: '',
    result: content,
    status: event.error ? 'failed' : 'completed',
  })
}

function runtimeEventFromAgui(
  eventType: string,
  event: AguiEvent,
): RuntimeEventBlock {
  const status =
    eventType === 'RUN_ERROR'
      ? 'error'
      : eventType === 'RUN_FINISHED'
        ? 'success'
        : 'running'
  return {
    kind: 'runtime_event',
    id: `${eventType}:${event.timestamp ?? Date.now()}:${Math.random()}`,
    name: eventType,
    title: titleFromName(eventType),
    status,
    payload: event,
  }
}

function blockFromCustomEvent(event: AguiEvent): TimelineBlock {
  const name = stringField(event, 'name') ?? 'custom'
  const value = event.value
  const payload = extractCustomPayload(value)
  const id = `${name}:${event.timestamp ?? Date.now()}:${Math.random()}`

  if (name === 'ya_agent.task_event') {
    return {
      kind: 'task_board',
      id,
      tasks: Array.isArray(payload.tasks) ? (payload.tasks as TaskInfo[]) : [],
    } satisfies TaskBoardBlock
  }
  if (
    name === 'yaacli.context_update_event' ||
    name === 'ya_agent.context_update_event'
  ) {
    return {
      kind: 'context_meter',
      id,
      totalTokens: numberField(payload, 'total_tokens'),
      contextWindowSize: numberField(payload, 'context_window_size'),
    } satisfies ContextMeterBlock
  }
  if (name.includes('usage')) {
    return { kind: 'usage', id, payload } satisfies UsageBlock
  }
  if (name === 'ya_agent.subagent_start_event') {
    return {
      kind: 'subagent',
      id,
      agentId: stringField(payload, 'agent_id') ?? 'subagent',
      agentName: stringField(payload, 'agent_name') ?? 'subagent',
      status: 'running',
      promptPreview: stringField(payload, 'prompt_preview') ?? undefined,
    } satisfies SubagentBlock
  }
  if (name === 'ya_agent.subagent_complete_event') {
    return {
      kind: 'subagent',
      id,
      agentId: stringField(payload, 'agent_id') ?? 'subagent',
      agentName: stringField(payload, 'agent_name') ?? 'subagent',
      status: payload.success === false ? 'failed' : 'completed',
      requestCount: numberField(payload, 'request_count'),
      resultPreview: stringField(payload, 'result_preview') ?? undefined,
      error: stringField(payload, 'error') ?? undefined,
      durationSeconds: numberField(payload, 'duration_seconds'),
    } satisfies SubagentBlock
  }
  if (name === 'ya_agent.file_change_event') {
    return {
      kind: 'file_change',
      id,
      changes: Array.isArray(payload.changes) ? payload.changes : [],
      toolName: stringField(payload, 'tool_name') ?? undefined,
    } satisfies FileChangeBlock
  }
  if (name === 'ya_agent.note_event') {
    const entries =
      payload.entries && typeof payload.entries === 'object'
        ? payload.entries
        : {}
    return {
      kind: 'note_snapshot',
      id,
      entries: entries as Record<string, string>,
    } satisfies NoteSnapshotBlock
  }
  if (name === 'ya_claw.run_steered') {
    return steeringBlockFromPayload({
      id,
      title: 'Steer delivered',
      status: 'delivered',
      payload: { delivery: 'runtime_state', ...payload },
    })
  }
  if (name === 'ya_agent.message_received') {
    return steeringBlockFromMessageReceivedPayload({
      id,
      payload,
    })
  }
  if (name.startsWith('ya_agent.') || name.startsWith('ya_claw.')) {
    return {
      kind: 'runtime_event',
      id,
      name,
      title: titleFromName(name),
      status: statusFromCustomName(name),
      payload,
    } satisfies RuntimeEventBlock
  }
  return { kind: 'raw_custom', id, name, payload } satisfies RawCustomBlock
}

function steeringBlockFromPayload({
  id,
  title,
  status,
  payload,
}: {
  id: string
  title: string
  status: SteeringBlock['status']
  payload: Record<string, unknown>
}): SteeringBlock {
  return {
    kind: 'steering',
    id,
    title,
    status,
    inputParts: Array.isArray(payload.input_parts)
      ? (payload.input_parts as SteeringBlock['inputParts'])
      : [],
    prompt: payload.prompt,
    delivery: stringField(payload, 'delivery') ?? undefined,
    payload,
  }
}

function steeringBlockFromMessageReceivedPayload({
  id,
  payload,
}: {
  id: string
  payload: Record<string, unknown>
}): SteeringBlock {
  const messages = Array.isArray(payload.messages)
    ? (payload.messages as Array<Record<string, unknown>>)
    : []
  const renderedMessages = messages
    .map((message) => message.rendered_content ?? message.content_text)
    .filter((message) => message !== undefined)
  return {
    kind: 'steering',
    id,
    title: 'Steer injected',
    status: 'injected',
    inputParts: [],
    prompt:
      renderedMessages.length === 1
        ? renderedMessages[0]
        : renderedMessages.length > 1
          ? renderedMessages
          : undefined,
    delivery: 'message_bus',
    payload,
  }
}

function extractCustomPayload(value: unknown): Record<string, unknown> {
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>
    const payload = record.payload
    if (payload && typeof payload === 'object')
      return payload as Record<string, unknown>
    return record
  }
  return { value }
}

function appendBlock<T extends TimelineBlock>(
  state: AguiTimelineState,
  block: T,
): AguiTimelineState {
  return { ...state, blocks: [...state.blocks, block] }
}

function identifierField(event: Record<string, unknown>, ...names: string[]) {
  for (const name of names) {
    const value = event[name]
    if (typeof value === 'string' && value.trim()) return value
  }
  return null
}

function stringField(event: Record<string, unknown>, ...names: string[]) {
  return identifierField(event, ...names)
}

function numberField(event: Record<string, unknown>, name: string) {
  const value = event[name]
  return typeof value === 'number' ? value : 0
}

function stringifyValue(value: unknown) {
  return typeof value === 'string' ? value : safeJsonStringify(value)
}

function titleFromName(name: string) {
  return name
    .replace(/^ya_agent\./, '')
    .replace(/^ya_claw\./, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

function statusFromCustomName(name: string): RuntimeEventBlock['status'] {
  if (name.includes('failed') || name.includes('error')) return 'error'
  if (name.includes('complete') || name.includes('finished')) return 'success'
  if (name.includes('start') || name.includes('running')) return 'running'
  if (name.includes('interrupt') || name.includes('cancel')) return 'warning'
  return 'info'
}
