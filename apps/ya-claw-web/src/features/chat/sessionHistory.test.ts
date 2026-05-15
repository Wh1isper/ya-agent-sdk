import { describe, expect, it } from 'vitest'

import type { SessionGetResponse } from '../../types'
import { mergeSessionHistoryPages } from './sessionHistory'

function page(
  runs: Array<{ id: string; sequence_no: number; content: string }>,
  options: { hasMore?: boolean; total?: number } = {},
): SessionGetResponse {
  return {
    session: {
      id: 'session-a',
      session_type: 'conversation',
      metadata: {},
      created_at: '2026-01-01T00:00:00Z',
      updated_at: '2026-01-01T00:00:00Z',
      status: 'completed',
      run_count: options.total ?? runs.length,
      runs_limit: 3,
      runs_has_more: options.hasMore ?? false,
      runs_next_before_sequence_no: options.hasMore
        ? runs[runs.length - 1]?.sequence_no
        : null,
      runs: runs.map((run) => ({
        id: run.id,
        session_id: 'session-a',
        sequence_no: run.sequence_no,
        status: 'completed',
        trigger_type: 'manual',
        input_parts: [{ type: 'text', text: `Prompt ${run.sequence_no}` }],
        message: [
          {
            type: 'TEXT_MESSAGE_CHUNK',
            messageId: `message-${run.sequence_no}`,
            delta: run.content,
          },
        ],
        created_at: '2026-01-01T00:00:00Z',
      })),
    },
  }
}

describe('session history merging', () => {
  it('orders paged latest-first runs into readable oldest-to-newest timeline', () => {
    const history = mergeSessionHistoryPages([
      page(
        [
          { id: 'run-5', sequence_no: 5, content: 'five' },
          { id: 'run-4', sequence_no: 4, content: 'four' },
          { id: 'run-3', sequence_no: 3, content: 'three' },
        ],
        { hasMore: true, total: 5 },
      ),
      page([
        { id: 'run-2', sequence_no: 2, content: 'two' },
        { id: 'run-1', sequence_no: 1, content: 'one' },
      ]),
    ])

    expect(history.runs.map((run) => run.sequence_no)).toEqual([1, 2, 3, 4, 5])
    expect(history.loadedRunCount).toBe(5)
    expect(history.totalRunCount).toBe(5)
    expect(history.hasMore).toBe(false)
    expect(history.timeline.blocks.map((block) => block.id)).toEqual([
      'run-1:input',
      'assistant:message-1',
      'run-2:input',
      'assistant:message-2',
      'run-3:input',
      'assistant:message-3',
      'run-4:input',
      'assistant:message-4',
      'run-5:input',
      'assistant:message-5',
    ])
  })

  it('keeps hasMore from the last fetched page and deduplicates overlap', () => {
    const history = mergeSessionHistoryPages([
      page(
        [
          { id: 'run-3', sequence_no: 3, content: 'three' },
          { id: 'run-2', sequence_no: 2, content: 'two' },
        ],
        { hasMore: true, total: 4 },
      ),
      page(
        [
          { id: 'run-2', sequence_no: 2, content: 'two' },
          { id: 'run-1', sequence_no: 1, content: 'one' },
        ],
        { hasMore: true, total: 4 },
      ),
    ])

    expect(history.runs.map((run) => run.id)).toEqual([
      'run-1',
      'run-2',
      'run-3',
    ])
    expect(history.hasMore).toBe(true)
    expect(history.loadedRunCount).toBe(3)
    expect(history.oldestSequenceNo).toBe(1)
  })
})
