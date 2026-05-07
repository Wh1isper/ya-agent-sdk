const DATETIME_LOCAL_PATTERN =
  /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?$/

type DateTimeParts = {
  year: number
  month: number
  day: number
  hour: number
  minute: number
  second: number
}

export function getBrowserTimeZone() {
  return Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC'
}

export function getSupportedTimeZones() {
  const intlWithSupportedValues = Intl as typeof Intl & {
    supportedValuesOf?: (key: 'timeZone') => string[]
  }
  return intlWithSupportedValues.supportedValuesOf?.('timeZone') ?? []
}

export function formatDateTime(
  value?: string | Date | null,
  options: Intl.DateTimeFormatOptions = {},
) {
  if (!value) return 'not scheduled'
  const date = value instanceof Date ? value : new Date(value)
  if (Number.isNaN(date.getTime())) return 'invalid date'
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
    ...options,
  }).format(date)
}

export function formatDateTimeInTimeZone(
  value?: string | Date | null,
  timeZone = getBrowserTimeZone(),
  options: Intl.DateTimeFormatOptions = {},
) {
  return formatDateTime(value, { timeZone, ...options })
}

export function formatUtcDateTime(value?: string | Date | null) {
  return formatDateTime(value, {
    timeZone: 'UTC',
    timeZoneName: 'short',
  })
}

export function toZonedDatetimeLocalValue(
  value?: string | null,
  timeZone = getBrowserTimeZone(),
) {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  const parts = getDateTimeParts(date, timeZone)
  return `${padYear(parts.year)}-${pad(parts.month)}-${pad(parts.day)}T${pad(
    parts.hour,
  )}:${pad(parts.minute)}`
}

export function zonedDatetimeLocalToIso(value: string, timeZone: string) {
  if (!value) return null
  const parts = parseDatetimeLocal(value)
  const utcTimestamp = zonedDateTimePartsToUtcTimestamp(parts, timeZone)
  return new Date(utcTimestamp).toISOString()
}

export function describeScheduledAndLocalDateTime(
  value?: string | null,
  scheduledTimeZone = getBrowserTimeZone(),
) {
  if (!value) return 'not scheduled'
  const browserTimeZone = getBrowserTimeZone()
  return `Schedule: ${formatDateTimeInTimeZone(
    value,
    scheduledTimeZone,
  )} ${scheduledTimeZone} · Local: ${formatDateTimeInTimeZone(
    value,
    browserTimeZone,
  )} ${browserTimeZone}`
}

export function describeBrowserDateTime(value?: string | null) {
  if (!value) return 'not scheduled'
  const browserTimeZone = getBrowserTimeZone()
  return `${formatDateTimeInTimeZone(value, browserTimeZone)} ${browserTimeZone}`
}

function parseDatetimeLocal(value: string): DateTimeParts {
  const match = DATETIME_LOCAL_PATTERN.exec(value)
  if (!match) {
    throw new Error(`Invalid datetime-local value: ${value}`)
  }
  return {
    year: Number(match[1]),
    month: Number(match[2]),
    day: Number(match[3]),
    hour: Number(match[4]),
    minute: Number(match[5]),
    second: match[6] ? Number(match[6]) : 0,
  }
}

function zonedDateTimePartsToUtcTimestamp(
  parts: DateTimeParts,
  timeZone: string,
) {
  const localTimestamp = Date.UTC(
    parts.year,
    parts.month - 1,
    parts.day,
    parts.hour,
    parts.minute,
    parts.second,
  )
  let utcTimestamp = localTimestamp
  for (let index = 0; index < 4; index += 1) {
    const offset = getTimeZoneOffsetMs(new Date(utcTimestamp), timeZone)
    const nextTimestamp = localTimestamp - offset
    if (Math.abs(nextTimestamp - utcTimestamp) < 1) {
      return nextTimestamp
    }
    utcTimestamp = nextTimestamp
  }
  return utcTimestamp
}

function getTimeZoneOffsetMs(date: Date, timeZone: string) {
  const parts = getDateTimeParts(date, timeZone)
  const zonedTimestamp = Date.UTC(
    parts.year,
    parts.month - 1,
    parts.day,
    parts.hour,
    parts.minute,
    parts.second,
  )
  return zonedTimestamp - date.getTime()
}

function getDateTimeParts(date: Date, timeZone: string): DateTimeParts {
  const formatter = new Intl.DateTimeFormat('en-US', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hourCycle: 'h23',
  })
  const values = Object.fromEntries(
    formatter
      .formatToParts(date)
      .filter((part) => part.type !== 'literal')
      .map((part) => [part.type, part.value]),
  )
  return {
    year: Number(values.year),
    month: Number(values.month),
    day: Number(values.day),
    hour: Number(values.hour),
    minute: Number(values.minute),
    second: Number(values.second),
  }
}

function pad(value: number) {
  return value.toString().padStart(2, '0')
}

function padYear(value: number) {
  return value.toString().padStart(4, '0')
}
