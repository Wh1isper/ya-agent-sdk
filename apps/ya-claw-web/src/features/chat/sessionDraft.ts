import { useCallback, useSyncExternalStore } from 'react'

const NEW_SESSION_DRAFT_KEY = '__new_session__'
const EMPTY_DRAFT: DraftValue = { text: '', revision: 0 }

type DraftValue = {
  text: string
  revision: number
}

type ComposerRoute = 'chat' | 'debug'

export type DraftSnapshot = DraftValue

let drafts: Record<string, DraftValue> = {}
const listeners = new Set<() => void>()

function draftKey(sessionId: string | null, composerRoute: ComposerRoute) {
  return `${composerRoute}:${sessionId ?? NEW_SESSION_DRAFT_KEY}`
}

function subscribe(listener: () => void) {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

function updateDraft(
  key: string,
  update: (current: DraftValue | undefined) => DraftValue | undefined,
) {
  const nextDraft = update(drafts[key])
  if (!nextDraft) return
  drafts = { ...drafts, [key]: nextDraft }
  listeners.forEach((listener) => listener())
}

export function resetSessionDrafts() {
  if (Object.keys(drafts).length === 0) return
  drafts = {}
  listeners.forEach((listener) => listener())
}

export function isSubmissionTargetActive(
  currentRoute: string,
  currentSessionId: string | null,
  targetSessionId: string | null,
  composerRoute: ComposerRoute,
) {
  return currentRoute === composerRoute && currentSessionId === targetSessionId
}

export function useSessionDraft(
  sessionId: string | null,
  composerRoute: ComposerRoute = 'chat',
) {
  const key = draftKey(sessionId, composerRoute)
  const draft = useSyncExternalStore(
    subscribe,
    () => drafts[key] ?? EMPTY_DRAFT,
    () => EMPTY_DRAFT,
  )

  const setText = useCallback(
    (value: string) => {
      updateDraft(key, (current) => ({
        text: value,
        revision: (current?.revision ?? 0) + 1,
      }))
    },
    [key],
  )

  const clear = useCallback(() => {
    updateDraft(key, (current) =>
      current
        ? {
            text: '',
            revision: current.revision + 1,
          }
        : undefined,
    )
  }, [key])

  const clearIfUnchanged = useCallback(
    (submittedDraft: DraftSnapshot) => {
      updateDraft(key, (current) => {
        if (
          current?.text !== submittedDraft.text ||
          current.revision !== submittedDraft.revision
        ) {
          return undefined
        }
        return {
          text: '',
          revision: current.revision + 1,
        }
      })
    },
    [key],
  )

  return {
    text: draft.text,
    revision: draft.revision,
    setText,
    clear,
    clearIfUnchanged,
  }
}
