export type SimpleFrequency =
  | 'hourly'
  | 'daily'
  | 'weekdays'
  | 'weekly'
  | 'custom'

const SIMPLE_CRON = {
  hourly: /^0 \* \* \* \*$/,
  daily: /^(\d{1,2}) (\d{1,2}) \* \* \*$/,
  weekdays: /^(\d{1,2}) (\d{1,2}) \* \* 1-5$/,
  weekly: /^(\d{1,2}) (\d{1,2}) \* \* 1$/,
} as const

export function parseSimpleCron(cron: string): {
  frequency: SimpleFrequency
  time: string
} {
  const normalized = cron.trim().replace(/\s+/g, ' ')
  if (SIMPLE_CRON.hourly.test(normalized)) {
    return { frequency: 'hourly', time: '00:00' }
  }
  for (const frequency of ['daily', 'weekdays', 'weekly'] as const) {
    const match = normalized.match(SIMPLE_CRON[frequency])
    if (match) {
      const minute = Number(match[1])
      const hour = Number(match[2])
      if (minute < 60 && hour < 24) {
        return {
          frequency,
          time: `${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}`,
        }
      }
    }
  }
  return { frequency: 'custom', time: '09:00' }
}

export function buildSimpleCron(frequency: SimpleFrequency, time: string) {
  if (frequency === 'custom') return null
  if (frequency === 'hourly') return '0 * * * *'
  const [hour = '9', minute = '0'] = time.split(':')
  const suffix =
    frequency === 'weekdays' ? '1-5' : frequency === 'weekly' ? '1' : '*'
  return `${Number(minute)} ${Number(hour)} * * ${suffix}`
}

export function describeSimpleRecurrence(
  frequency: SimpleFrequency,
  time: string,
  timezone: string,
) {
  const at = time || '09:00'
  const cadence =
    frequency === 'hourly'
      ? 'at the start of every hour'
      : frequency === 'daily'
        ? `every day at ${at}`
        : frequency === 'weekdays'
          ? `every weekday at ${at}`
          : frequency === 'weekly'
            ? `every Monday at ${at}`
            : 'using the advanced cron expression'
  return `Runs ${cadence} in ${timezone || 'your browser timezone'}.`
}

export function nextSimpleOccurrences(
  cron: string,
  timezone: string,
  from = new Date(),
  count = 3,
): Date[] | null {
  const parsed = parseSimpleCron(cron)
  if (parsed.frequency === 'custom') return null

  const formatter = new Intl.DateTimeFormat('en-US', {
    timeZone: timezone,
    weekday: 'short',
    hour: '2-digit',
    minute: '2-digit',
    hourCycle: 'h23',
  })
  const [targetHour, targetMinute] = parsed.time.split(':').map(Number)
  const start = new Date(from)
  start.setUTCSeconds(0, 0)
  start.setUTCMinutes(start.getUTCMinutes() + 1)
  const result: Date[] = []
  // The supported presets fire at least weekly, so eight days is enough for
  // three hourly/daily fires but not three weekly fires. Scan 22 days to cover
  // DST transitions and three Monday occurrences without guessing UTC offsets.
  const maxMinutes = 22 * 24 * 60

  for (let index = 0; index < maxMinutes && result.length < count; index += 1) {
    const candidate = new Date(start.getTime() + index * 60_000)
    const parts = Object.fromEntries(
      formatter
        .formatToParts(candidate)
        .filter((part) => part.type !== 'literal')
        .map((part) => [part.type, part.value]),
    )
    const hour = Number(parts.hour)
    const minute = Number(parts.minute)
    const weekday = parts.weekday
    const matchesTime =
      parsed.frequency === 'hourly'
        ? minute === 0
        : hour === targetHour && minute === targetMinute
    const matchesDay =
      parsed.frequency === 'weekdays'
        ? !['Sat', 'Sun'].includes(weekday)
        : parsed.frequency === 'weekly'
          ? weekday === 'Mon'
          : true
    if (matchesTime && matchesDay) result.push(candidate)
  }

  return result.length === count ? result : null
}
