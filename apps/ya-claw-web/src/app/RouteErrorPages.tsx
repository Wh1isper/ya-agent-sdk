import { Link } from '@tanstack/react-router'

export function NotFoundPage() {
  return (
    <div className="flex h-full items-center justify-center p-6">
      <div className="max-w-md text-center">
        <p className="text-sm font-semibold text-[var(--primary)]">404</p>
        <h1 className="mt-2 text-2xl font-semibold">This page was not found</h1>
        <p className="mt-2 text-sm text-[var(--muted-foreground)]">
          The link may be outdated or the resource may have been removed.
        </p>
        <Link
          to="/"
          className="mt-5 inline-flex rounded-lg bg-[var(--primary)] px-4 py-2 text-sm font-semibold text-white"
        >
          Return home
        </Link>
      </div>
    </div>
  )
}

export function RouteErrorPage({
  error,
  onRetry,
}: {
  error: Error
  onRetry: () => void
}) {
  return (
    <div className="flex h-full items-center justify-center p-6">
      <div className="max-w-lg rounded-xl border border-rose-200 bg-rose-50 p-5">
        <h1 className="text-lg font-semibold text-rose-900">
          This workspace view could not be opened
        </h1>
        <p className="mt-2 text-sm leading-6 text-rose-700">
          Try again. If the problem continues, check the runtime connection and
          recent activity.
        </p>
        <details className="mt-3 rounded-lg border border-rose-200 bg-white/70 px-3 py-2 text-xs text-rose-800">
          <summary className="cursor-pointer font-semibold">
            Technical details
          </summary>
          <p className="mono mt-2 break-words">{error.message}</p>
        </details>
        <button
          type="button"
          className="mt-4 rounded-lg border border-rose-300 bg-white px-3 py-2 text-sm font-semibold text-rose-800"
          onClick={onRetry}
        >
          Try again
        </button>
      </div>
    </div>
  )
}
