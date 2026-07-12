import { describe, expect, it } from 'vitest'

import { apiTimestamp, isNewerApiTimestamp, parseApiDate } from './date'

describe('parseApiDate', () => {
  it('treats an offset-less API datetime as UTC', () => {
    expect(parseApiDate('2026-07-11T15:51:07.265887').toISOString()).toBe(
      '2026-07-11T15:51:07.265Z',
    )
  })

  it('preserves explicit UTC and numeric-offset datetimes', () => {
    expect(parseApiDate('2026-07-11T15:51:07Z').toISOString()).toBe(
      '2026-07-11T15:51:07.000Z',
    )
    expect(parseApiDate('2026-07-11T23:51:07+08:00').toISOString()).toBe(
      '2026-07-11T15:51:07.000Z',
    )
  })

  it('returns comparable timestamps and retains invalid-date behavior', () => {
    expect(apiTimestamp('2026-07-11T15:51:07')).toBe(
      Date.parse('2026-07-11T15:51:07Z'),
    )
    expect(Number.isNaN(apiTimestamp('not-a-date'))).toBe(true)
  })

  it('accepts only a genuinely newer server version', () => {
    expect(isNewerApiTimestamp('2026-07-11T15:51:08Z', null)).toBe(true)
    expect(
      isNewerApiTimestamp('2026-07-11T15:51:08Z', '2026-07-11T15:51:07Z'),
    ).toBe(true)
    expect(
      isNewerApiTimestamp('2026-07-11T15:51:07Z', '2026-07-11T15:51:08Z'),
    ).toBe(false)
    expect(
      isNewerApiTimestamp(
        '2026-07-11T15:51:07.265900Z',
        '2026-07-11T15:51:07.265100Z',
      ),
    ).toBe(true)
    expect(isNewerApiTimestamp(null, '2026-07-11T15:51:08Z')).toBe(false)
  })
})
