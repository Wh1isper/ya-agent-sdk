import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import {
  isSubmissionTargetActive,
  resetSessionDrafts,
  useSessionDraft,
} from './sessionDraft'

afterEach(() => act(() => resetSessionDrafts()))

describe('useSessionDraft', () => {
  it('isolates drafts by session and restores each draft when switching back', () => {
    const { result, rerender } = renderHook(
      ({ sessionId }: { sessionId: string | null }) =>
        useSessionDraft(sessionId),
      { initialProps: { sessionId: 'session-a' as string | null } },
    )

    act(() => result.current.setText('message for session A'))
    expect(result.current.text).toBe('message for session A')

    rerender({ sessionId: 'session-b' })
    expect(result.current.text).toBe('')
    act(() => result.current.setText('message for session B'))

    rerender({ sessionId: null })
    expect(result.current.text).toBe('')
    act(() => result.current.setText('new conversation draft'))

    rerender({ sessionId: 'session-a' })
    expect(result.current.text).toBe('message for session A')
    rerender({ sessionId: 'session-b' })
    expect(result.current.text).toBe('message for session B')
    rerender({ sessionId: null })
    expect(result.current.text).toBe('new conversation draft')
  })

  it('restores a draft after the composer unmounts and remounts', () => {
    const first = renderHook(() => useSessionDraft('session-a', 'chat'))
    act(() => first.result.current.setText('survives route navigation'))
    const revision = first.result.current.revision
    first.unmount()

    const second = renderHook(() => useSessionDraft('session-a', 'chat'))

    expect(second.result.current.text).toBe('survives route navigation')
    expect(second.result.current.revision).toBe(revision)
  })

  it('keeps chat and debug drafts isolated for the same session', () => {
    const chat = renderHook(() => useSessionDraft('session-a', 'chat'))
    const debug = renderHook(() => useSessionDraft('session-a', 'debug'))

    act(() => chat.result.current.setText('chat draft'))
    act(() => debug.result.current.setText('debug draft'))

    expect(chat.result.current.text).toBe('chat draft')
    expect(debug.result.current.text).toBe('debug draft')
  })

  it('keeps an edited draft even when its text returns to the submitted value', () => {
    const { result, rerender } = renderHook(
      ({ sessionId }: { sessionId: string | null }) =>
        useSessionDraft(sessionId),
      { initialProps: { sessionId: 'session-a' as string | null } },
    )

    act(() => result.current.setText('submitted text'))
    const submittedDraft = {
      text: result.current.text,
      revision: result.current.revision,
    }
    const finishSubmission = result.current.clearIfUnchanged

    rerender({ sessionId: 'session-b' })
    act(() => result.current.setText('session B draft'))
    rerender({ sessionId: 'session-a' })
    act(() => result.current.setText('newer session A draft'))
    act(() => result.current.setText('submitted text'))
    act(() => finishSubmission(submittedDraft))

    expect(result.current.text).toBe('submitted text')
    expect(result.current.revision).toBeGreaterThan(submittedDraft.revision)
    rerender({ sessionId: 'session-b' })
    expect(result.current.text).toBe('session B draft')
  })

  it('clears an unchanged submitted draft and advances its revision', () => {
    const { result } = renderHook(() => useSessionDraft('session-a'))
    act(() => result.current.setText('submitted text'))
    const submittedDraft = {
      text: result.current.text,
      revision: result.current.revision,
    }

    act(() => result.current.clearIfUnchanged(submittedDraft))

    expect(result.current.text).toBe('')
    expect(result.current.revision).toBeGreaterThan(submittedDraft.revision)
  })

  it('clears only the active session draft after a successful send', () => {
    const { result, rerender } = renderHook(
      ({ sessionId }: { sessionId: string | null }) =>
        useSessionDraft(sessionId),
      { initialProps: { sessionId: 'session-a' as string | null } },
    )

    act(() => result.current.setText('send this'))
    rerender({ sessionId: 'session-b' })
    act(() => result.current.setText('keep this'))
    rerender({ sessionId: 'session-a' })
    act(() => result.current.clear())

    expect(result.current.text).toBe('')
    rerender({ sessionId: 'session-b' })
    expect(result.current.text).toBe('keep this')
  })
})

describe('isSubmissionTargetActive', () => {
  it.each([
    ['chat', 'session-a', 'session-a', 'chat', true],
    ['chat', 'session-b', 'session-a', 'chat', false],
    ['debug', 'session-a', 'session-a', 'chat', false],
    ['debug', 'session-a', 'session-a', 'debug', true],
    ['settings', 'session-a', 'session-a', 'debug', false],
    ['chat', null, null, 'chat', true],
    ['chat', 'session-a', null, 'chat', false],
  ] as const)(
    'checks route %s and current session %s against target %s for %s',
    (
      currentRoute,
      currentSessionId,
      targetSessionId,
      composerRoute,
      expected,
    ) => {
      expect(
        isSubmissionTargetActive(
          currentRoute,
          currentSessionId,
          targetSessionId,
          composerRoute,
        ),
      ).toBe(expected)
    },
  )
})
