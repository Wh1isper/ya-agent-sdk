const ISO_DATE_TIME_PREFIX = /^\d{4}-\d{2}-\d{2}T/
const EXPLICIT_TIME_ZONE_SUFFIX = /(?:Z|[+-]\d{2}:?\d{2})$/i
const API_TIMESTAMP_PARTS =
  /^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:?\d{2})?$/i

/**
 * Parse timestamps returned by YA Claw.
 *
 * SQLite-backed responses currently serialize UTC datetimes without an
 * explicit offset. Browsers otherwise interpret those values as local time,
 * so add the UTC designator only when an ISO datetime has no timezone.
 */
export function parseApiDate(value: string | number | Date) {
  if (value instanceof Date || typeof value === 'number') return new Date(value)
  const normalized =
    ISO_DATE_TIME_PREFIX.test(value) && !EXPLICIT_TIME_ZONE_SUFFIX.test(value)
      ? `${value}Z`
      : value
  return new Date(normalized)
}

export function apiTimestamp(value: string | number | Date) {
  return parseApiDate(value).getTime()
}

export function isNewerApiTimestamp(
  candidate: string | null,
  baseline: string | null,
) {
  if (!candidate) return false
  if (!baseline) return true
  const candidateOrder = apiTimestampOrder(candidate)
  const baselineOrder = apiTimestampOrder(baseline)
  if (candidateOrder && baselineOrder) {
    if (candidateOrder.epochSecond !== baselineOrder.epochSecond) {
      return candidateOrder.epochSecond > baselineOrder.epochSecond
    }
    return candidateOrder.fraction > baselineOrder.fraction
  }
  return candidate !== baseline
}

function apiTimestampOrder(value: string) {
  const match = API_TIMESTAMP_PARTS.exec(value)
  if (!match) return null
  const [, dateTime, fraction = '', explicitTimeZone] = match
  const epochSecond = Date.parse(`${dateTime}${explicitTimeZone ?? 'Z'}`) / 1000
  if (!Number.isFinite(epochSecond)) return null
  return {
    epochSecond,
    fraction: fraction.padEnd(9, '0').slice(0, 9),
  }
}
