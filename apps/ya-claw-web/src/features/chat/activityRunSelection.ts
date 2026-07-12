import type { RunSummary } from '../../types'

type RunIdentity = Pick<RunSummary, 'id' | 'session_id'>

export type ActivityRunSelection = {
  validatedRunId: string | null
  ownershipMismatch: boolean
}

export function validateActivityRunSelection(
  requestedRunId: string | null,
  selectedSessionId: string | null,
  run: RunIdentity | null | undefined,
): ActivityRunSelection {
  if (!requestedRunId || run?.id !== requestedRunId) {
    return { validatedRunId: null, ownershipMismatch: false }
  }
  if (!selectedSessionId || run.session_id !== selectedSessionId) {
    return { validatedRunId: null, ownershipMismatch: true }
  }
  return { validatedRunId: requestedRunId, ownershipMismatch: false }
}
