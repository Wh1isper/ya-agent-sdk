import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { CostEstimateDisplay } from './CostEstimateDisplay'

const estimate = {
  currency: 'USD' as const,
  input_amount: '0.002',
  output_amount: '0.001',
  total_amount: '0.003',
  priced_requests: 1,
  unpriced_requests: 1,
  basis: 'api_list_price' as const,
  source: 'genai_prices' as const,
}

describe('CostEstimateDisplay', () => {
  it('labels partial list-price estimates and their request coverage', () => {
    render(<CostEstimateDisplay estimate={estimate} />)

    expect(screen.getByText('Estimated API list price')).toBeVisible()
    expect(screen.getByText('~$0.003')).toBeVisible()
    expect(screen.getByText('1/2 requests priced')).toBeVisible()
    expect(screen.getByText('partial')).toBeVisible()
  })

  it('does not present unavailable pricing as zero cost', () => {
    render(<CostEstimateDisplay estimate={null} />)

    expect(screen.getByText('Estimated API list price')).toBeVisible()
    expect(screen.getByText('—')).toBeVisible()
    expect(screen.getByText('Pricing unavailable')).toBeVisible()
  })
})
