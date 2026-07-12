import {
  forwardRef,
  type HTMLAttributes,
  type InputHTMLAttributes,
  type ReactNode,
  type TextareaHTMLAttributes,
} from 'react'

import { cn } from '../../lib/utils'

export function Field({
  label,
  htmlFor,
  hint,
  error,
  required,
  children,
  className,
}: {
  label: ReactNode
  htmlFor: string
  hint?: ReactNode
  error?: ReactNode
  required?: boolean
  children: ReactNode
  className?: string
}) {
  const descriptionId = hint ? `${htmlFor}-hint` : undefined
  const errorId = error ? `${htmlFor}-error` : undefined
  return (
    <div className={cn('space-y-2', className)}>
      <label htmlFor={htmlFor} className="block text-sm font-medium">
        {label}
        {required ? (
          <span className="ml-1 text-rose-600" aria-hidden>
            *
          </span>
        ) : null}
      </label>
      {children}
      {hint ? (
        <p
          id={descriptionId}
          className="text-xs text-[var(--subtle-foreground)]"
        >
          {hint}
        </p>
      ) : null}
      {error ? (
        <p
          id={errorId}
          className="text-xs font-medium text-rose-700"
          role="alert"
        >
          {error}
        </p>
      ) : null}
    </div>
  )
}

export type InputProps = InputHTMLAttributes<HTMLInputElement> & {
  invalid?: boolean
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, invalid, ...props }, ref) => (
    <input
      ref={ref}
      aria-invalid={invalid || undefined}
      className={cn(
        'h-10 w-full rounded-lg border border-[var(--border)] bg-[var(--subtle)] px-3 text-sm text-[var(--foreground)] outline-none transition placeholder:text-[var(--subtle-foreground)] focus:border-[var(--primary)] focus:bg-[var(--surface)] focus:ring-2 focus:ring-[var(--focus)] disabled:cursor-not-allowed disabled:opacity-60',
        invalid && 'border-rose-300 focus:border-rose-500 focus:ring-rose-100',
        className,
      )}
      {...props}
    />
  ),
)
Input.displayName = 'Input'

export type TextareaProps = TextareaHTMLAttributes<HTMLTextAreaElement> & {
  invalid?: boolean
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, invalid, ...props }, ref) => (
    <textarea
      ref={ref}
      aria-invalid={invalid || undefined}
      className={cn(
        'min-h-28 w-full resize-y rounded-lg border border-[var(--border)] bg-[var(--subtle)] px-3 py-2 text-sm leading-6 text-[var(--foreground)] outline-none transition placeholder:text-[var(--subtle-foreground)] focus:border-[var(--primary)] focus:bg-[var(--surface)] focus:ring-2 focus:ring-[var(--focus)] disabled:cursor-not-allowed disabled:opacity-60',
        invalid && 'border-rose-300 focus:border-rose-500 focus:ring-rose-100',
        className,
      )}
      {...props}
    />
  ),
)
Textarea.displayName = 'Textarea'

export function FieldGroup({
  className,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn('grid grid-cols-1 gap-4 sm:grid-cols-2', className)}
      {...props}
    />
  )
}
