export const AUTOMATION_LIST_LIMIT = 500

export function mayHaveMoreAutomationRows(rowCount: number) {
  return rowCount >= AUTOMATION_LIST_LIMIT
}
