import { Link } from '@tanstack/react-router'
import { Plus, Send, Wrench } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'

import {
  useCreateSessionMutation,
  useProfilesQuery,
  useSubmitSessionInputMutation,
} from '../../../api/hooks'
import { QueryError } from '../../../components/ui'
import { cn } from '../../../lib/utils'
import { useLayoutStore } from '../../../stores/layoutStore'
import type { InputPart, RunSummary } from '../../../types'
import { isSubmissionTargetActive, useSessionDraft } from '../sessionDraft'
import { DEBUG_METADATA } from './constants'

export function Composer({
  selectedSessionId,
  selectedProfile,
  activeRun,
}: {
  selectedSessionId: string | null
  selectedProfile: string | null
  activeRun: RunSummary | null
}) {
  const {
    text,
    revision,
    setText,
    clearIfUnchanged: clearDraftIfUnchanged,
  } = useSessionDraft(selectedSessionId, 'debug')
  const sendingRef = useRef(false)
  const createSession = useCreateSessionMutation()
  const submitInput = useSubmitSessionInputMutation(selectedSessionId)
  const profiles = useProfilesQuery()
  const profileOptions = profiles.data ?? []
  const defaultProfileName = profileOptions[0]?.name ?? ''
  const [profileName, setProfileName] = useState(
    selectedProfile ?? defaultProfileName,
  )
  const selectSession = useLayoutStore((store) => store.selectSession)
  const selectRun = useLayoutStore((store) => store.selectRun)
  const canAppend = activeRun?.status === 'queued'
  const canSteer = activeRun?.status === 'running'

  useEffect(() => {
    setProfileName(selectedProfile ?? defaultProfileName)
  }, [defaultProfileName, selectedProfile])

  const isPending = createSession.isPending || submitInput.isPending
  const profilesReady = Boolean(selectedSessionId) || !profiles.isLoading
  const canSend =
    text.trim().length > 0 && !isPending && profilesReady && !profiles.isError

  async function send() {
    if (!canSend || sendingRef.current) return
    sendingRef.current = true
    const submittedDraft = { text, revision }
    const normalizedText = submittedDraft.text.trim()
    const inputParts: InputPart[] = [{ type: 'text', text: normalizedText }]
    const targetSessionId = selectedSessionId
    try {
      if (targetSessionId) {
        const response = await submitInput.mutateAsync({
          input_parts: inputParts,
          metadata: DEBUG_METADATA,
        })
        const current = useLayoutStore.getState()
        if (
          isSubmissionTargetActive(
            current.route,
            current.selectedSessionId,
            targetSessionId,
            'debug',
          )
        ) {
          selectRun(response.run_id)
        }
      } else {
        const response = await createSession.mutateAsync({
          profile_name: profileName.trim() || null,
          input_parts: inputParts,
          metadata: DEBUG_METADATA,
        })
        const current = useLayoutStore.getState()
        if (
          isSubmissionTargetActive(
            current.route,
            current.selectedSessionId,
            targetSessionId,
            'debug',
          )
        ) {
          selectSession(response.session.id)
          selectRun(
            response.run?.id ??
              response.session.active_run_id ??
              response.session.head_run_id ??
              null,
          )
        }
      }
      clearDraftIfUnchanged(submittedDraft)
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : 'Failed to send message',
      )
    } finally {
      sendingRef.current = false
    }
  }

  return (
    <div className="border-t border-slate-200 bg-white p-3 sm:p-4">
      <div className="mx-auto max-w-4xl">
        {profiles.isError ? (
          <div className="mb-3">
            <QueryError
              compact
              title="Could not load agent profiles"
              error={profiles.error}
              onRetry={() => void profiles.refetch()}
            />
          </div>
        ) : null}
        {activeRun ? (
          <div
            className={cn(
              'mb-3 rounded-2xl border px-4 py-3 text-sm',
              canSteer
                ? 'border-blue-200 bg-blue-50 text-blue-800'
                : 'border-amber-200 bg-amber-50 text-amber-800',
            )}
          >
            {canSteer
              ? 'Active run is streaming. New input will steer the current run.'
              : 'This session is queued. New input will be appended to the queued run.'}
          </div>
        ) : null}
        <div className="rounded-3xl border border-slate-200 bg-white p-3 shadow-sm ring-1 ring-slate-100 transition focus-within:border-blue-200 focus-within:ring-blue-100">
          <textarea
            aria-label="Debug prompt"
            className="max-h-48 min-h-24 w-full resize-none rounded-2xl border-0 p-2 text-sm leading-6 text-slate-900 outline-none placeholder:text-slate-400 disabled:bg-white disabled:text-slate-400"
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder={
              canSteer
                ? 'Steer the active run...'
                : canAppend
                  ? 'Append to the queued run...'
                  : 'Send a debug prompt to YA Claw...'
            }
            onKeyDown={(event) => {
              if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
                event.preventDefault()
                void send()
              }
            }}
          />
          <div className="flex flex-col gap-3 border-t border-slate-100 pt-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex min-w-0 flex-1 items-center gap-2">
              {profiles.isLoading && profiles.data === undefined ? (
                <span
                  className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-500"
                  role="status"
                >
                  Loading profiles…
                </span>
              ) : profileOptions.length > 0 ? (
                <select
                  aria-label="Agent profile"
                  className="max-w-52 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-700 outline-none ring-blue-600 focus:ring-2 disabled:text-slate-400"
                  value={profileName}
                  onChange={(event) => setProfileName(event.target.value)}
                  disabled={Boolean(selectedSessionId) || Boolean(activeRun)}
                >
                  {profileOptions.map((profile) => (
                    <option key={profile.name} value={profile.name}>
                      {profile.name}
                    </option>
                  ))}
                </select>
              ) : !profiles.isError ? (
                <span className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-500">
                  No profiles
                  <Link className="font-semibold text-blue-700" to="/agents">
                    Create one
                  </Link>
                </span>
              ) : null}
              <span className="hidden text-xs text-slate-400 lg:inline">
                Cmd/Ctrl + Enter to send
              </span>
            </div>
            <button
              type="button"
              className={cn(
                'inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold text-white shadow-sm transition disabled:bg-slate-300',
                canSteer
                  ? 'bg-amber-600 hover:bg-amber-700'
                  : 'bg-blue-600 hover:bg-blue-700',
              )}
              disabled={!canSend}
              onClick={() => void send()}
            >
              {canSteer ? (
                <Wrench className="h-4 w-4" />
              ) : selectedSessionId ? (
                <Send className="h-4 w-4" />
              ) : (
                <Plus className="h-4 w-4" />
              )}
              {isPending
                ? canSteer
                  ? 'Steering'
                  : 'Sending'
                : canSteer
                  ? 'Steer run'
                  : selectedSessionId
                    ? canAppend
                      ? 'Append'
                      : 'Send'
                    : 'New debug run'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
