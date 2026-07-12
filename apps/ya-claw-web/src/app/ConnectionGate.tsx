import {
  AlertCircle,
  CheckCircle2,
  LoaderCircle,
  ShieldCheck,
} from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'

import {
  getConnectionErrorMessage,
  validateConnection,
} from '../api/connection'
import { cn } from '../lib/utils'
import { useConnectionStore } from '../stores/connectionStore'

export function ConnectionGate({ children }: { children: ReactNode }) {
  const baseUrl = useConnectionStore((state) => state.baseUrl)
  const apiToken = useConnectionStore((state) => state.apiToken)
  const connectionIssue = useConnectionStore((state) => state.connectionIssue)
  const setConnection = useConnectionStore((state) => state.setConnection)
  const [draftBaseUrl, setDraftBaseUrl] = useState(baseUrl)
  const [draftToken, setDraftToken] = useState('')
  const [showToken, setShowToken] = useState(false)
  const [error, setError] = useState<string | null>(connectionIssue)
  const [isValidating, setIsValidating] = useState(false)

  useEffect(() => {
    if (!apiToken) {
      setDraftToken('')
      setShowToken(false)
      setError(connectionIssue)
    }
  }, [apiToken, connectionIssue])

  if (apiToken.trim()) return <>{children}</>

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (isValidating) return
    setError(null)
    setIsValidating(true)
    try {
      const normalizedBaseUrl = draftBaseUrl.trim()
      const normalizedToken = draftToken.trim()
      await validateConnection({
        baseUrl: normalizedBaseUrl,
        apiToken: normalizedToken,
      })
      setConnection({
        baseUrl: normalizedBaseUrl,
        apiToken: normalizedToken,
      })
    } catch (validationError) {
      setError(getConnectionErrorMessage(validationError))
    } finally {
      setIsValidating(false)
    }
  }

  return (
    <main className="flex h-dvh min-h-0 items-start justify-center overflow-y-auto bg-[var(--canvas)] px-4 py-6 text-[var(--foreground)] sm:px-6 sm:py-10">
      <div className="my-auto w-full max-w-lg rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-6 shadow-[var(--shadow-lg)] sm:p-8">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[var(--primary)] text-sm font-semibold text-white shadow-sm">
            YA
          </div>
          <div>
            <p className="text-sm font-medium text-[var(--primary)]">YA Claw</p>
            <h1 className="text-2xl font-semibold tracking-tight">
              Connect to your runtime
            </h1>
          </div>
        </div>

        <p className="mt-5 text-sm leading-6 text-[var(--muted-foreground)]">
          Verify the runtime and authenticate before opening the workspace. Your
          token stays in memory and is cleared when you disconnect.
        </p>

        {error ? (
          <div
            className="mt-5 flex gap-3 rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800"
            role="alert"
          >
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            <div>
              <p className="font-semibold">Connection failed</p>
              <p className="mt-1 leading-5">{error}</p>
            </div>
          </div>
        ) : null}

        <form className="mt-6 space-y-5" onSubmit={submit}>
          <label className="block">
            <span className="text-sm font-medium">Runtime URL</span>
            <input
              className="mt-2 h-11 w-full rounded-lg border border-[var(--border)] bg-[var(--subtle)] px-3 text-sm outline-none transition focus:border-[var(--primary)] focus:bg-white focus:ring-2 focus:ring-[var(--focus)]"
              value={draftBaseUrl}
              onChange={(event) => setDraftBaseUrl(event.target.value)}
              placeholder="http://127.0.0.1:9042"
              type="url"
              required
              autoComplete="url"
              disabled={isValidating}
            />
            <span className="mt-1.5 block text-xs text-[var(--subtle-foreground)]">
              The URL is remembered. Credentials are not persisted.
            </span>
          </label>

          <label className="block">
            <span className="text-sm font-medium">API token</span>
            <div className="mt-2 flex h-11 rounded-lg border border-[var(--border)] bg-[var(--subtle)] transition focus-within:border-[var(--primary)] focus-within:bg-white focus-within:ring-2 focus-within:ring-[var(--focus)]">
              <input
                className="min-w-0 flex-1 rounded-l-lg bg-transparent px-3 text-sm outline-none"
                value={draftToken}
                onChange={(event) => setDraftToken(event.target.value)}
                type={showToken ? 'text' : 'password'}
                placeholder="YA_CLAW_API_TOKEN"
                required
                autoComplete="off"
                disabled={isValidating}
              />
              <button
                type="button"
                className="rounded-r-lg border-l border-[var(--border)] px-3 text-xs font-medium text-[var(--muted-foreground)] transition hover:bg-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus)]"
                onClick={() => setShowToken((current) => !current)}
                disabled={isValidating}
              >
                {showToken ? 'Hide' : 'Show'}
              </button>
            </div>
          </label>

          <button
            type="submit"
            className={cn(
              'inline-flex h-11 w-full items-center justify-center gap-2 rounded-lg bg-[var(--primary)] px-4 text-sm font-semibold text-white shadow-sm transition hover:bg-[var(--primary-hover)]',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus)] focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60',
            )}
            disabled={isValidating}
          >
            {isValidating ? (
              <LoaderCircle className="h-4 w-4 animate-spin" aria-hidden />
            ) : (
              <ShieldCheck className="h-4 w-4" aria-hidden />
            )}
            {isValidating ? 'Testing connection…' : 'Test and connect'}
          </button>
        </form>

        <div className="mt-6 grid gap-2 text-xs text-[var(--muted-foreground)] sm:grid-cols-3">
          {[
            'Runtime reachable',
            'Authentication accepted',
            'Workspace ready',
          ].map((label) => (
            <span key={label} className="inline-flex items-center gap-1.5">
              {isValidating ? (
                <LoaderCircle className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <CheckCircle2 className="h-3.5 w-3.5 text-slate-300" />
              )}
              {label}
            </span>
          ))}
        </div>
      </div>
    </main>
  )
}
