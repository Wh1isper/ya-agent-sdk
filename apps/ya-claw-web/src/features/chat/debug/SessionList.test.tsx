import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { SessionList } from './SessionList'

describe('Activity session source filters', () => {
  it('offers every supported source with human-readable labels', async () => {
    const user = userEvent.setup()
    const onFilterChange = vi.fn()

    render(
      <SessionList
        sessions={[]}
        selectedSessionId={null}
        search=""
        loading={false}
        loadingMore={false}
        hasMore={false}
        error={null}
        filters={{ status: 'all', source: 'all', profile: 'all', time: 'all' }}
        profileOptions={[]}
        onRetry={vi.fn()}
        onLoadMore={vi.fn()}
        onSearchChange={vi.fn()}
        onFilterChange={onFilterChange}
        onClearFilters={vi.fn()}
        onSelect={vi.fn()}
      />,
    )

    const sourceFilter = screen.getByRole('combobox', {
      name: 'Filter activity by source',
    })
    expect(sourceFilter).toHaveTextContent('Web chat')
    expect(sourceFilter).toHaveTextContent('Connected channel')
    expect(sourceFilter).toHaveTextContent('Schedule')
    expect(sourceFilter).toHaveTextContent('Workflow')
    expect(sourceFilter).toHaveTextContent('Heartbeat')
    expect(sourceFilter).toHaveTextContent('Agency / proactive')
    expect(sourceFilter).toHaveTextContent('Memory / system')
    expect(sourceFilter).toHaveTextContent('API')
    expect(screen.queryByText(/New debug run/i)).not.toBeInTheDocument()

    await user.selectOptions(sourceFilter, 'workflow')
    expect(onFilterChange).toHaveBeenCalledWith('source', 'workflow')
  })
})
