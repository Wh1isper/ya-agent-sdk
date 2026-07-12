import type { VariantProps } from 'class-variance-authority'
import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react'

import { cn } from '../../lib/utils'
import { buttonVariants } from './buttonVariants'

export type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> &
  VariantProps<typeof buttonVariants> & {
    loading?: boolean
    loadingLabel?: string
    leadingIcon?: ReactNode
  }

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant,
      size,
      loading = false,
      loadingLabel,
      leadingIcon,
      disabled,
      children,
      type = 'button',
      ...props
    },
    ref,
  ) => (
    <button
      ref={ref}
      type={type}
      className={cn(buttonVariants({ variant, size }), className)}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      {...props}
    >
      {loading ? (
        <span
          className="h-4 w-4 animate-spin rounded-full border-2 border-current border-r-transparent motion-reduce:animate-none"
          aria-hidden
        />
      ) : (
        leadingIcon
      )}
      {loading && loadingLabel ? loadingLabel : children}
    </button>
  ),
)

Button.displayName = 'Button'
