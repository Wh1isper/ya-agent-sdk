import { describe, expect, it } from 'vitest'

import { buildTimeline } from './eventReducer'

const custom = (name: string, payload: Record<string, unknown>) => ({
  type: 'CUSTOM',
  name,
  value: {
    run_id: 'run-a',
    session_id: 'session-a',
    agent_id: 'main',
    agent_name: 'main',
    payload,
  },
})

describe('AGUI event reducer', () => {
  it('merges text chunks into assistant messages', () => {
    const timeline = buildTimeline([
      { type: 'TEXT_MESSAGE_CHUNK', messageId: 'm1', delta: 'Hello' },
      { type: 'TEXT_MESSAGE_CHUNK', messageId: 'm1', delta: ' world' },
    ])

    expect(timeline.blocks).toHaveLength(1)
    expect(timeline.blocks[0]).toMatchObject({
      kind: 'assistant_message',
      content: 'Hello world',
    })
  })

  it('renders task snapshots from custom events', () => {
    const timeline = buildTimeline([
      custom('ya_agent.task_event', {
        tasks: [
          { id: '1', subject: 'Design', status: 'completed' },
          {
            id: '2',
            subject: 'Build',
            status: 'in_progress',
            active_form: 'Building',
          },
        ],
      }),
    ])

    expect(timeline.blocks[0]).toMatchObject({
      kind: 'task_board',
      tasks: [{ id: '1' }, { id: '2' }],
    })
  })

  it('renders context usage custom events', () => {
    const timeline = buildTimeline([
      custom('ya_agent.context_update_event', {
        total_tokens: 180000,
        context_window_size: 270000,
      }),
    ])

    expect(timeline.blocks[0]).toMatchObject({
      kind: 'context_meter',
      totalTokens: 180000,
      contextWindowSize: 270000,
    })
  })

  it('keeps runtime custom events as visible runtime cards by default', () => {
    const timeline = buildTimeline([
      custom('ya_claw.run_queued', { run_id: 'run-a', status: 'queued' }),
    ])

    expect(timeline.blocks[0]).toMatchObject({
      kind: 'runtime_event',
      name: 'ya_claw.run_queued',
    })
  })

  it('can hide runtime events for chat rendering', () => {
    const timeline = buildTimeline(
      [
        custom('ya_claw.run_queued', { run_id: 'run-a', status: 'queued' }),
        { type: 'TEXT_MESSAGE_CHUNK', messageId: 'm1', delta: 'Done' },
        { type: 'RUN_FINISHED', result: 'Done' },
      ],
      [],
      'run-a',
      { includeRuntimeEvents: false },
    )

    expect(timeline.blocks).toHaveLength(1)
    expect(timeline.blocks[0]).toMatchObject({
      kind: 'assistant_message',
      content: 'Done',
    })
  })

  it('renders steering delivery and SDK message injection events', () => {
    const timeline = buildTimeline(
      [
        custom('ya_claw.run_steered', {
          input_parts: [{ type: 'text', text: 'please continue' }],
        }),
        custom('ya_agent.message_received', {
          messages: [
            {
              source: 'user',
              target: 'main',
              content_text: 'please continue',
              rendered_content: 'please continue',
            },
          ],
        }),
      ],
      [],
      'run-a',
      { includeRuntimeEvents: false },
    )

    expect(timeline.blocks).toHaveLength(2)
    expect(timeline.blocks[0]).toMatchObject({
      kind: 'steering',
      title: 'Steer delivered',
      status: 'delivered',
      delivery: 'runtime_state',
    })
    expect(timeline.blocks[1]).toMatchObject({
      kind: 'steering',
      title: 'Steer injected',
      status: 'injected',
      delivery: 'message_bus',
      prompt: 'please continue',
    })
  })
})
