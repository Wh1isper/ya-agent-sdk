export function accentFromRuntimeStatus(
  status: 'info' | 'running' | 'success' | 'warning' | 'error',
) {
  if (status === 'running') return 'amber'
  if (status === 'success') return 'emerald'
  if (status === 'warning') return 'amber'
  if (status === 'error') return 'rose'
  return 'slate'
}
