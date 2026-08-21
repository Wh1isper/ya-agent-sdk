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
      custom('ya_agent.task', {
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
      custom('ya_agent.model_request_complete', {
        context_tokens: 180000,
        context_window_size: 270000,
      }),
    ])

    expect(timeline.blocks[0]).toMatchObject({
      kind: 'context_meter',
      totalTokens: 180000,
      contextWindowSize: 270000,
    })
  })

  it('renders canonical subagent lifecycle events', () => {
    const timeline = buildTimeline([
      custom('ya_agent.subagent_start', {
        agent_id: 'worker-bg-a7b9',
        agent_name: 'worker',
        prompt_preview: 'inspect code',
      }),
      custom('ya_agent.subagent_complete', {
        agent_id: 'worker-bg-a7b9',
        agent_name: 'worker',
        success: true,
        result_preview: 'done',
      }),
    ])

    expect(timeline.blocks).toMatchObject([
      { kind: 'subagent', status: 'running', agentId: 'worker-bg-a7b9' },
      { kind: 'subagent', status: 'completed', agentId: 'worker-bg-a7b9' },
    ])
  })

  it('replaces prior cost snapshots by transport run instead of summing them', () => {
    const snapshot = (sdkRunId: string, totalAmount: string) =>
      custom('ya_agent.usage_snapshot', {
        run_id: sdkRunId,
        total_usage: {},
        total_cost_estimate: {
          currency: 'USD',
          input_amount: totalAmount,
          output_amount: '0',
          total_amount: totalAmount,
          priced_requests: 1,
          unpriced_requests: 0,
          basis: 'api_list_price',
          source: 'genai_prices',
        },
        entries: [],
        agent_usages: {},
        model_usages: {},
        model_cost_estimates: {},
      })
    const timeline = buildTimeline([
      snapshot('sdk-segment-1', '0.001'),
      snapshot('sdk-segment-2', '0.003'),
    ])

    expect(timeline.blocks).toHaveLength(1)
    expect(timeline.blocks[0]).toMatchObject({
      kind: 'usage',
      id: 'usage:run-a',
      snapshot: {
        run_id: 'sdk-segment-2',
        total_cost_estimate: { total_amount: '0.003' },
      },
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

  it('renders durable steering acceptance and application events', () => {
    const timeline = buildTimeline(
      [
        custom('ya_claw.input_accepted', { disposition: 'accepted' }),
        custom('ya_claw.input_enqueued', { disposition: 'enqueued' }),
        custom('ya_claw.input_applied', { disposition: 'applied' }),
      ],
      [],
      'run-a',
      { includeRuntimeEvents: false },
    )

    expect(timeline.blocks).toHaveLength(3)
    expect(timeline.blocks[0]).toMatchObject({
      kind: 'steering',
      title: 'Steer accepted',
      status: 'delivered',
    })
    expect(timeline.blocks[1]).toMatchObject({
      kind: 'steering',
      title: 'Steer enqueued',
      status: 'delivered',
    })
    expect(timeline.blocks[2]).toMatchObject({
      kind: 'steering',
      title: 'Steer injected',
      status: 'injected',
    })
  })
})
