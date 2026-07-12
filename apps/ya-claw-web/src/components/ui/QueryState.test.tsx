import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { ApiError } from '../../api/client'
import { QueryError } from './QueryState'

describe('QueryError', () => {
  it('shows actionable technical details and retries', async () => {
    const user = userEvent.setup()
    const onRetry = vi.fn()
    const error = new ApiError('Runtime unavailable', 503, {
      request_id: 'request-123',
      reason: 'maintenance',
    })

    render(
      <QueryError
        compact
        title="Could not load data"
        error={error}
        onRetry={onRetry}
      />,
    )

    expect(screen.getByRole('alert')).toHaveTextContent('Runtime unavailable')
    await user.click(screen.getByText('Technical details'))
    expect(screen.getByText(/"status": 503/)).toBeVisible()
    expect(screen.getByText(/"request_id": "request-123"/)).toBeVisible()

    await user.click(screen.getByRole('button', { name: 'Try again' }))
    expect(onRetry).toHaveBeenCalledOnce()
  })
})
