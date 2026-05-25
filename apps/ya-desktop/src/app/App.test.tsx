import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { App } from './App'

describe('App', () => {
  it('renders the native agent workspace', () => {
    render(<App />)

    expect(screen.getByText('YA Desktop')).toBeInTheDocument()
    expect(screen.getByText('Native Agent Workspace')).toBeInTheDocument()
    expect(screen.queryByText('Agency')).not.toBeInTheDocument()
  })
})
