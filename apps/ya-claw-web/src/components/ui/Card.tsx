import { forwardRef, type HTMLAttributes, type ReactNode } from 'react'

import { cn } from '../../lib/utils'

export type CardProps = HTMLAttributes<HTMLDivElement> & {
  interactive?: boolean
}

export const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ className, interactive = false, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        'rounded-xl border border-[var(--border)] bg-[var(--surface)]',
        interactive &&
          'transition-colors hover:border-[var(--border-strong)] hover:bg-[var(--subtle)]',
        className,
      )}
      {...props}
    />
  ),
)
Card.displayName = 'Card'

export function CardHeader({
  title,
  description,
  action,
  className,
}: {
  title: ReactNode
  description?: ReactNode
  action?: ReactNode
  className?: string
}) {
  return (
    <div
      className={cn('flex items-start justify-between gap-4 p-5', className)}
    >
      <div className="min-w-0">
        <h2 className="text-base font-semibold tracking-tight text-[var(--foreground)]">
          {title}
        </h2>
        {description ? (
          <p className="mt-1 text-sm leading-5 text-[var(--muted-foreground)]">
            {description}
          </p>
        ) : null}
      </div>
      {action ? <div className="shrink-0">{action}</div> : null}
    </div>
  )
}

export function CardContent({
  className,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('px-5 pb-5', className)} {...props} />
}

export function CardFooter({
  className,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        'flex items-center justify-end gap-2 border-t border-[var(--border)] px-5 py-4',
        className,
      )}
      {...props}
    />
  )
}
