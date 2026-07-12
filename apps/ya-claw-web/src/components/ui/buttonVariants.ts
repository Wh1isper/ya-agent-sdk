import { cva } from 'class-variance-authority'

export const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus)] focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        primary:
          'bg-[var(--primary)] text-white shadow-sm hover:bg-[var(--primary-hover)]',
        secondary:
          'border border-[var(--border)] bg-[var(--surface)] text-[var(--foreground)] hover:bg-[var(--subtle)]',
        ghost:
          'text-[var(--muted-foreground)] hover:bg-[var(--subtle)] hover:text-[var(--foreground)]',
        danger:
          'bg-[var(--danger)] text-white shadow-sm hover:bg-[var(--danger-hover)]',
        dangerOutline:
          'border border-rose-200 bg-[var(--surface)] text-rose-700 hover:bg-rose-50',
      },
      size: {
        sm: 'h-8 px-3 text-xs',
        md: 'h-10 px-4',
        lg: 'h-11 px-5',
        icon: 'h-10 w-10 p-0',
        iconSm: 'h-8 w-8 p-0',
      },
    },
    defaultVariants: {
      variant: 'primary',
      size: 'md',
    },
  },
)
