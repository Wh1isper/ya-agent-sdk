import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { PageLoading } from './AppShell'

describe('AppShell page loading fallback', () => {
  it('announces route loading as a polite live status', () => {
    render(<PageLoading />)

    const status = screen.getByRole('status')
    expect(status).toHaveTextContent('Loading workspace…')
    expect(status).toHaveAttribute('aria-live', 'polite')
    expect(status).toHaveAttribute('aria-atomic', 'true')
    expect(status).toHaveAttribute('aria-busy', 'true')
  })
})
