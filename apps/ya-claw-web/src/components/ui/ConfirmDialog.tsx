import * as Dialog from '@radix-ui/react-dialog'
import { AlertTriangle } from 'lucide-react'
import { useState, type ReactNode } from 'react'

import { Button } from './Button'

export function ConfirmDialog({
  open,
  onOpenChange,
  trigger,
  title,
  description,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  danger = false,
  pending = false,
  onConfirm,
}: {
  open?: boolean
  onOpenChange?: (open: boolean) => void
  trigger?: ReactNode
  title: ReactNode
  description?: ReactNode
  confirmLabel?: string
  cancelLabel?: string
  danger?: boolean
  pending?: boolean
  onConfirm: () => void | Promise<void>
}) {
  const [internalOpen, setInternalOpen] = useState(false)
  const [confirmError, setConfirmError] = useState<string | null>(null)
  const resolvedOpen = open ?? internalOpen
  const setOpen = (nextOpen: boolean) => {
    if (!nextOpen) setConfirmError(null)
    if (open === undefined) setInternalOpen(nextOpen)
    onOpenChange?.(nextOpen)
  }
  async function confirm() {
    setConfirmError(null)
    try {
      await onConfirm()
      setOpen(false)
    } catch (error) {
      setConfirmError(
        error instanceof Error ? error.message : 'The operation failed.',
      )
    }
  }

  return (
    <Dialog.Root open={resolvedOpen} onOpenChange={setOpen}>
      {trigger ? <Dialog.Trigger asChild>{trigger}</Dialog.Trigger> : null}
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-slate-950/45 backdrop-blur-[1px] data-[state=closed]:animate-out data-[state=open]:animate-in motion-reduce:animate-none" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-[min(30rem,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-5 shadow-[var(--shadow-lg)] outline-none">
          <div className="flex gap-3">
            <span
              className={
                danger
                  ? 'flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-rose-100 text-rose-700'
                  : 'flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-amber-100 text-amber-700'
              }
            >
              <AlertTriangle className="h-5 w-5" aria-hidden />
            </span>
            <div className="min-w-0">
              <Dialog.Title className="text-base font-semibold tracking-tight">
                {title}
              </Dialog.Title>
              {description ? (
                <Dialog.Description className="mt-2 text-sm leading-6 text-[var(--muted-foreground)]">
                  {description}
                </Dialog.Description>
              ) : null}
            </div>
          </div>
          {confirmError ? (
            <p
              className="mt-4 rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-800"
              role="alert"
            >
              {confirmError}
            </p>
          ) : null}
          <div className="mt-6 flex justify-end gap-2">
            <Dialog.Close asChild>
              <Button variant="secondary" disabled={pending}>
                {cancelLabel}
              </Button>
            </Dialog.Close>
            <Button
              variant={danger ? 'danger' : 'primary'}
              loading={pending}
              loadingLabel={confirmLabel}
              onClick={() => void confirm()}
            >
              {confirmLabel}
            </Button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
