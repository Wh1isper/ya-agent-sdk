import { AlertTriangle, RefreshCcw } from 'lucide-react'
import type { ReactNode } from 'react'

import { cn } from '../../lib/utils'
import { Button } from './Button'

export type QueryStateLike<T> = {
  data: T | undefined
  isLoading: boolean
  isError: boolean
  isFetching?: boolean
  error: unknown
  refetch: () => Promise<unknown>
}

export function QueryState<T>({
  query,
  children,
  loading,
  empty,
  isEmpty,
  errorTitle = 'Could not load this content',
}: {
  query: QueryStateLike<T>
  children: (data: T) => ReactNode
  loading?: ReactNode
  empty?: ReactNode
  isEmpty?: (data: T) => boolean
  errorTitle?: string
}) {
  if (query.isLoading && query.data === undefined) {
    return loading ?? <QuerySkeleton />
  }

  if (query.isError) {
    return (
      <QueryError
        title={errorTitle}
        error={query.error}
        onRetry={() => void query.refetch()}
      />
    )
  }

  if (query.data === undefined) {
    return loading ?? <QuerySkeleton />
  }

  if (isEmpty?.(query.data)) return empty ?? null
  return <>{children(query.data)}</>
}

export function QueryError({
  title,
  error,
  onRetry,
  compact = false,
}: {
  title: string
  error: unknown
  onRetry?: () => void
  compact?: boolean
}) {
  return (
    <div
      className={cn(
        'flex flex-col rounded-xl border border-rose-200 bg-rose-50/70',
        compact
          ? 'min-h-0 items-stretch p-3 text-left'
          : 'min-h-48 items-center justify-center p-6 text-center',
      )}
      role="alert"
    >
      {!compact ? (
        <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-rose-100 text-rose-700">
          <AlertTriangle className="h-5 w-5" aria-hidden />
        </span>
      ) : null}
      <h2
        className={cn(
          'text-sm font-semibold text-rose-950',
          !compact && 'mt-4',
        )}
      >
        {title}
      </h2>
      <p className="mt-2 max-w-lg text-sm leading-6 text-rose-800">
        {readErrorMessage(error)}
      </p>
      <details className="mt-3 w-full max-w-lg rounded-lg border border-rose-200 bg-white/70 px-3 py-2 text-left">
        <summary className="cursor-pointer text-xs font-semibold text-rose-900">
          Technical details
        </summary>
        <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-words text-xs leading-5 text-rose-800">
          {readTechnicalDetails(error)}
        </pre>
      </details>
      {onRetry ? (
        <Button
          className={compact ? 'mt-3 self-start' : 'mt-4'}
          size="sm"
          variant="secondary"
          leadingIcon={<RefreshCcw className="h-3.5 w-3.5" aria-hidden />}
          onClick={onRetry}
        >
          Try again
        </Button>
      ) : null}
    </div>
  )
}

export function QuerySkeleton({ rows = 3 }: { rows?: number }) {
  return (
    <div className="space-y-3" aria-label="Loading" role="status">
      {Array.from({ length: rows }).map((_, index) => (
        <div
          key={index}
          className="h-20 animate-pulse rounded-xl border border-[var(--border)] bg-[var(--subtle)] motion-reduce:animate-none"
        />
      ))}
      <span className="sr-only">Loading</span>
    </div>
  )
}

function readErrorMessage(error: unknown) {
  if (error instanceof Error) return error.message
  return 'The runtime did not return usable data.'
}

function readTechnicalDetails(error: unknown) {
  if (error instanceof Error) {
    const details: Record<string, unknown> = {
      name: error.name,
      message: error.message,
    }
    for (const key of ['status', 'code', 'requestId', 'detail', 'details']) {
      if (key in error) details[key] = Reflect.get(error, key)
    }
    if (import.meta.env.DEV && error.stack) details.stack = error.stack
    try {
      return JSON.stringify(details, null, 2)
    } catch {
      return `${error.name}: ${error.message}`
    }
  }
  try {
    return JSON.stringify(error, null, 2)
  } catch {
    return String(error)
  }
}
