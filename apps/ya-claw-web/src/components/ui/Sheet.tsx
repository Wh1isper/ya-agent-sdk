import * as Dialog from '@radix-ui/react-dialog'
import { X } from 'lucide-react'
import {
  forwardRef,
  type ComponentPropsWithoutRef,
  type ComponentRef,
  type ReactNode,
} from 'react'

import { cn } from '../../lib/utils'

export const Sheet = Dialog.Root
export const SheetTrigger = Dialog.Trigger
export const SheetClose = Dialog.Close

export type SheetContentProps = ComponentPropsWithoutRef<
  typeof Dialog.Content
> & {
  side?: 'left' | 'right' | 'bottom'
  hideClose?: boolean
}

export const SheetContent = forwardRef<
  ComponentRef<typeof Dialog.Content>,
  SheetContentProps
>(
  (
    { side = 'right', hideClose = false, className, children, ...props },
    ref,
  ) => (
    <Dialog.Portal>
      <Dialog.Overlay className="fixed inset-0 z-40 bg-slate-950/45 backdrop-blur-[1px] data-[state=closed]:animate-out data-[state=open]:animate-in motion-reduce:animate-none" />
      <Dialog.Content
        ref={ref}
        className={cn(
          'fixed z-50 flex bg-[var(--surface)] shadow-[var(--shadow-lg)] outline-none data-[state=closed]:animate-out data-[state=open]:animate-in motion-reduce:animate-none',
          side === 'left' &&
            'inset-y-0 left-0 w-[min(22rem,92vw)] flex-col border-r border-[var(--border)]',
          side === 'right' &&
            'inset-y-0 right-0 w-[min(28rem,94vw)] flex-col border-l border-[var(--border)]',
          side === 'bottom' &&
            'inset-x-0 bottom-0 max-h-[88dvh] flex-col rounded-t-2xl border-t border-[var(--border)]',
          className,
        )}
        {...props}
      >
        {children}
        {!hideClose ? (
          <Dialog.Close className="absolute right-4 top-4 inline-flex h-9 w-9 items-center justify-center rounded-lg text-[var(--muted-foreground)] transition hover:bg-[var(--subtle)] hover:text-[var(--foreground)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus)]">
            <X className="h-4 w-4" aria-hidden />
            <span className="sr-only">Close</span>
          </Dialog.Close>
        ) : null}
      </Dialog.Content>
    </Dialog.Portal>
  ),
)
SheetContent.displayName = 'SheetContent'

export function SheetHeader({
  title,
  description,
  className,
}: {
  title: ReactNode
  description?: ReactNode
  className?: string
}) {
  return (
    <div
      className={cn(
        'border-b border-[var(--border)] px-5 py-4 pr-14',
        className,
      )}
    >
      <Dialog.Title className="text-base font-semibold tracking-tight">
        {title}
      </Dialog.Title>
      {description ? (
        <Dialog.Description className="mt-1 text-sm leading-5 text-[var(--muted-foreground)]">
          {description}
        </Dialog.Description>
      ) : null}
    </div>
  )
}

export function SheetBody({
  className,
  ...props
}: ComponentPropsWithoutRef<'div'>) {
  return (
    <div
      className={cn(
        'min-h-0 flex-1 overflow-y-auto overscroll-contain p-5',
        className,
      )}
      {...props}
    />
  )
}

export function SheetFooter({
  className,
  ...props
}: ComponentPropsWithoutRef<'div'>) {
  return (
    <div
      className={cn(
        'flex flex-wrap items-center justify-end gap-2 border-t border-[var(--border)] p-4 pb-[max(1rem,env(safe-area-inset-bottom))]',
        className,
      )}
      {...props}
    />
  )
}
