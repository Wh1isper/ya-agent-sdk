import { describe, expect, it } from 'vitest'

import { validateActivityRunSelection } from './activityRunSelection'

describe('Activity run ownership validation', () => {
  it.each([
    {
      name: 'accepts a run owned by the selected session',
      requestedRunId: 'run-1',
      selectedSessionId: 'session-1',
      run: { id: 'run-1', session_id: 'session-1' },
      expected: { validatedRunId: 'run-1', ownershipMismatch: false },
    },
    {
      name: 'rejects a run owned by another session',
      requestedRunId: 'run-1',
      selectedSessionId: 'session-1',
      run: { id: 'run-1', session_id: 'session-2' },
      expected: { validatedRunId: null, ownershipMismatch: true },
    },
    {
      name: 'waits for the requested run instead of trusting stale data',
      requestedRunId: 'run-1',
      selectedSessionId: 'session-1',
      run: { id: 'run-previous', session_id: 'session-1' },
      expected: { validatedRunId: null, ownershipMismatch: false },
    },
    {
      name: 'does not validate a run without a selected session',
      requestedRunId: 'run-1',
      selectedSessionId: null,
      run: { id: 'run-1', session_id: 'session-1' },
      expected: { validatedRunId: null, ownershipMismatch: true },
    },
  ])('$name', ({ requestedRunId, selectedSessionId, run, expected }) => {
    expect(
      validateActivityRunSelection(requestedRunId, selectedSessionId, run),
    ).toEqual(expected)
  })
})
