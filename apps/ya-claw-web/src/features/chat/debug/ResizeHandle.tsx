import { Separator } from 'react-resizable-panels'

export function ResizeHandle() {
  return (
    <Separator className="group relative w-1 shrink-0 bg-slate-100 transition hover:bg-blue-100">
      <div className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-slate-200 group-hover:bg-blue-300" />
    </Separator>
  )
}
