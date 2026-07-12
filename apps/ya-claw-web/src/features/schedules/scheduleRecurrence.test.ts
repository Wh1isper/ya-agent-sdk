import { describe, expect, it } from 'vitest'

import {
  buildSimpleCron,
  describeSimpleRecurrence,
  nextSimpleOccurrences,
  parseSimpleCron,
} from './scheduleRecurrence'

describe('simple schedule recurrence', () => {
  it('round-trips common frequency and time choices', () => {
    expect(parseSimpleCron('30 14 * * 1-5')).toEqual({
      frequency: 'weekdays',
      time: '14:30',
    })
    expect(buildSimpleCron('weekly', '08:15')).toBe('15 8 * * 1')
    expect(buildSimpleCron('hourly', '12:34')).toBe('0 * * * *')
  })

  it('uses an explicit custom fallback for arbitrary cron', () => {
    expect(parseSimpleCron('*/15 9-17 * * 1-5')).toEqual({
      frequency: 'custom',
      time: '09:00',
    })
    expect(
      nextSimpleOccurrences(
        '*/15 9-17 * * 1-5',
        'UTC',
        new Date('2026-01-01T00:00:00Z'),
      ),
    ).toBeNull()
    expect(describeSimpleRecurrence('custom', '09:00', 'UTC')).toContain(
      'advanced cron expression',
    )
  })

  it('calculates three daily runs in the selected timezone', () => {
    const occurrences = nextSimpleOccurrences(
      '0 9 * * *',
      'America/New_York',
      new Date('2026-03-07T15:00:00Z'),
    )
    expect(occurrences?.map((date) => date.toISOString())).toEqual([
      '2026-03-08T13:00:00.000Z',
      '2026-03-09T13:00:00.000Z',
      '2026-03-10T13:00:00.000Z',
    ])
  })

  it('calculates weekday and weekly presets accurately', () => {
    expect(
      nextSimpleOccurrences(
        '0 9 * * 1-5',
        'UTC',
        new Date('2026-01-02T10:00:00Z'),
      )?.map((date) => date.toISOString()),
    ).toEqual([
      '2026-01-05T09:00:00.000Z',
      '2026-01-06T09:00:00.000Z',
      '2026-01-07T09:00:00.000Z',
    ])
    expect(
      nextSimpleOccurrences(
        '0 9 * * 1',
        'UTC',
        new Date('2026-01-01T00:00:00Z'),
      )?.map((date) => date.toISOString()),
    ).toEqual([
      '2026-01-05T09:00:00.000Z',
      '2026-01-12T09:00:00.000Z',
      '2026-01-19T09:00:00.000Z',
    ])
  })
})
