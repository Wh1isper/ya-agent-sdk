import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { WorkflowBuilder } from './WorkflowBuilder'

const singleStep = JSON.stringify({
  schema: 'ya-claw.workflow.v1',
  nodes: {
    draft: {
      profile: 'Self',
      mode: 'isolate',
      prompt: 'Draft the answer.',
    },
  },
  result: { from_node: 'draft' },
})

describe('WorkflowBuilder', () => {
  it('shows a readable graph and validation summary for a valid workflow', () => {
    render(<WorkflowBuilder value={singleStep} onChange={vi.fn()} />)

    expect(screen.getByText('1 workflow step ready')).toBeInTheDocument()
    expect(screen.getByText('Graph preview')).toBeInTheDocument()
    expect(screen.getAllByText('draft').length).toBeGreaterThan(0)
  })

  it('applies a multi-step template through the guided builder', () => {
    const onChange = vi.fn()
    render(<WorkflowBuilder value={singleStep} onChange={onChange} />)

    fireEvent.click(screen.getByRole('button', { name: /Research pipeline/i }))

    expect(onChange).toHaveBeenCalledTimes(1)
    const definition = JSON.parse(onChange.mock.calls[0]?.[0] as string) as {
      nodes: Record<string, { needs?: string[] }>
      result: { from_node: string }
    }
    expect(Object.keys(definition.nodes)).toEqual(['research', 'synthesize'])
    expect(definition.nodes.synthesize?.needs).toEqual(['research'])
    expect(definition.result.from_node).toBe('synthesize')
  })

  it('preserves custom definition and node fields when editing and renaming a step', () => {
    const onChange = vi.fn()
    const customized = JSON.stringify({
      schema: 'ya-claw.workflow.v1',
      custom_definition: { owner: 'ops' },
      nodes: {
        draft: {
          profile: 'Self',
          mode: 'isolate',
          prompt: 'Draft the answer.',
          custom_node: { retries: 3 },
        },
      },
      result: { from_node: 'draft', custom_result: true },
    })
    render(<WorkflowBuilder value={customized} onChange={onChange} />)

    fireEvent.change(screen.getByLabelText('Step ID'), {
      target: { value: 'compose' },
    })

    const definition = JSON.parse(onChange.mock.calls[0]?.[0] as string) as {
      custom_definition: { owner: string }
      nodes: Record<string, { custom_node: { retries: number } }>
      result: { from_node: string; custom_result: boolean }
    }
    expect(definition.custom_definition).toEqual({ owner: 'ops' })
    expect(definition.nodes.compose?.custom_node).toEqual({ retries: 3 })
    expect(definition.nodes.draft).toBeUndefined()
    expect(definition.result).toEqual({
      from_node: 'compose',
      custom_result: true,
    })
  })

  it('reports invalid dependencies instead of silently accepting them', () => {
    const invalid = JSON.stringify({
      schema: 'ya-claw.workflow.v1',
      nodes: {
        finish: {
          needs: ['missing'],
          prompt: 'Finish.',
        },
      },
      result: { from_node: 'finish' },
    })
    render(<WorkflowBuilder value={invalid} onChange={vi.fn()} />)

    expect(
      screen.getByText(/depends on unknown step “missing”/i),
    ).toBeInTheDocument()
  })
})
