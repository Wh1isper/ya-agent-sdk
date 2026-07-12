import { afterAll, beforeAll, describe, expect, it } from 'vitest'

import {
  formatDateTimeInTimeZone,
  toZonedDatetimeLocalValue,
  zonedDatetimeLocalToIso,
} from './timezone'

declare const process: { env: Record<string, string | undefined> }

const originalTimeZone = process.env.TZ

beforeAll(() => {
  process.env.TZ = 'America/Los_Angeles'
})

afterAll(() => {
  if (originalTimeZone === undefined) {
    delete process.env.TZ
  } else {
    process.env.TZ = originalTimeZone
  }
})

describe('API datetime timezone handling', () => {
  it('treats offset-less server datetimes as UTC when formatting', () => {
    expect(
      formatDateTimeInTimeZone('2026-01-15T17:30:00', 'America/New_York', {
        dateStyle: undefined,
        timeStyle: undefined,
        hour: '2-digit',
        minute: '2-digit',
        hourCycle: 'h23',
      }),
    ).toBe('12:30')
  })

  it('round-trips a loaded once schedule without changing its instant', () => {
    const serverRunAt = '2026-01-15T17:30:00'
    const scheduleTimeZone = 'America/New_York'

    const formValue = toZonedDatetimeLocalValue(serverRunAt, scheduleTimeZone)

    expect(formValue).toBe('2026-01-15T12:30')
    expect(zonedDatetimeLocalToIso(formValue, scheduleTimeZone)).toBe(
      '2026-01-15T17:30:00.000Z',
    )
  })

  it('rejects a wall-clock time in a daylight-saving gap', () => {
    expect(() =>
      zonedDatetimeLocalToIso('2026-03-08T02:30', 'America/New_York'),
    ).toThrow(
      'The local time 2026-03-08 02:30:00 does not exist in America/New_York because of a timezone transition. Choose another time.',
    )
  })

  it('uses the earlier instant when a fall-back wall-clock time occurs twice', () => {
    // 01:30 occurs first in EDT (UTC-04:00), then in EST (UTC-05:00).
    expect(
      zonedDatetimeLocalToIso('2026-11-01T01:30', 'America/New_York'),
    ).toBe('2026-11-01T05:30:00.000Z')
  })
})
