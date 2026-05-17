import { describe, expect, it } from 'vitest'

import {
  collectTextFromReplay,
  isRunErrorEvent,
  isRunFinishedEvent,
  parseStreamMessage,
  streamErrorMessage,
  streamRunId,
  streamSessionId,
  streamTextDelta,
} from './streamEvents'
import type { ClawStreamEvent } from './types'

describe('stream event utilities', () => {
  it('parses SSE messages into stream events', () => {
    const event = parseStreamMessage({
      id: '7',
      event: 'TEXT_MESSAGE_CHUNK',
      data: JSON.stringify({ type: 'TEXT_MESSAGE_CHUNK', delta: 'hello' }),
    })

    expect(event.id).toBe('7')
    expect(event.event).toBe('TEXT_MESSAGE_CHUNK')
    expect(event.payload).toMatchObject({
      type: 'TEXT_MESSAGE_CHUNK',
      delta: 'hello',
    })
  })

  it('extracts text chunks from delta and content payloads', () => {
    expect(
      streamTextDelta(
        streamEvent('TEXT_MESSAGE_CHUNK', {
          type: 'TEXT_MESSAGE_CHUNK',
          delta: 'hello',
        }),
      ),
    ).toBe('hello')

    expect(
      streamTextDelta(
        streamEvent('TEXT_MESSAGE_CHUNK', {
          type: 'TEXT_MESSAGE_CHUNK',
          content: 'world',
        }),
      ),
    ).toBe('world')
  })

  it('extracts final result output text from custom events', () => {
    expect(
      streamTextDelta(
        streamEvent('CUSTOM', {
          type: 'CUSTOM',
          value: {
            name: 'ya_agent.final_result',
            payload: { output_text: 'done' },
          },
        }),
      ),
    ).toBe('done')
  })

  it('detects terminal run events', () => {
    expect(isRunFinishedEvent(streamEvent('RUN_FINISHED', {}))).toBe(true)
    expect(isRunErrorEvent(streamEvent('RUN_ERROR', { message: 'failed' }))).toBe(
      true,
    )
  })

  it('extracts nested error messages', () => {
    expect(
      streamErrorMessage(
        streamEvent('CUSTOM', {
          type: 'RUN_ERROR',
          value: { payload: { message: 'nested failure' } },
        }),
      ),
    ).toBe('nested failure')
  })

  it('extracts run and session ids from top-level and nested payloads', () => {
    expect(streamRunId(streamEvent('RUN_STARTED', { runId: 'run_1' }))).toBe(
      'run_1',
    )
    expect(
      streamRunId(streamEvent('CUSTOM', { value: { run_id: 'run_2' } })),
    ).toBe('run_2')
    expect(
      streamSessionId(streamEvent('RUN_STARTED', { threadId: 'session_1' })),
    ).toBe('session_1')
    expect(
      streamSessionId(
        streamEvent('CUSTOM', { value: { session_id: 'session_2' } }),
      ),
    ).toBe('session_2')
  })

  it('collects assistant text from replay events', () => {
    expect(
      collectTextFromReplay([
        { type: 'TEXT_MESSAGE_CHUNK', delta: 'hello ' },
        { type: 'TEXT_MESSAGE_CHUNK', delta: 'world' },
      ]),
    ).toBe('hello world')
  })
})

function streamEvent(event: string, payload: Record<string, unknown>): ClawStreamEvent {
  return {
    id: '1',
    event,
    data: JSON.stringify(payload),
    payload,
  }
}
