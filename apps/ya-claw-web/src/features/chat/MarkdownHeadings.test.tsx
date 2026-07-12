import { render, screen } from '@testing-library/react'
import type { ComponentType } from 'react'
import { describe, expect, it } from 'vitest'

import { formatA11yViolations, getA11yViolations } from '../../test/a11y'
import { MarkdownMessage as ChatMarkdownMessage } from './ChatPage'
import { MarkdownMessage as ActivityMarkdownMessage } from './debug/MarkdownMessage'

describe('route message Markdown heading ownership', () => {
  it.each([
    ['Chat', ChatMarkdownMessage],
    ['Activity', ActivityMarkdownMessage],
  ] as Array<[string, ComponentType<{ content: string }>]>)(
    '%s shifts message headings below the route heading',
    (_surface, MarkdownMessage) => {
      const { container } = render(
        <MarkdownMessage
          content={[
            '# Level one',
            '## Level two',
            '### Level three',
            '#### Level four',
            '##### Level five',
            '###### Level six',
          ].join('\n\n')}
        />,
      )

      expect(container.querySelector('h1')).not.toBeInTheDocument()
      expect(
        screen.getByRole('heading', { level: 2, name: 'Level one' }),
      ).toBeInTheDocument()
      expect(
        screen.getByRole('heading', { level: 3, name: 'Level two' }),
      ).toBeInTheDocument()
      expect(
        screen.getByRole('heading', { level: 4, name: 'Level three' }),
      ).toBeInTheDocument()
      expect(
        screen.getByRole('heading', { level: 5, name: 'Level four' }),
      ).toBeInTheDocument()
      expect(
        screen.getByRole('heading', { level: 6, name: 'Level five' }),
      ).toBeInTheDocument()
      expect(
        screen.getByRole('heading', { level: 6, name: 'Level six' }),
      ).toBeInTheDocument()
    },
  )

  it.each([
    ['Chat', ChatMarkdownMessage],
    ['Activity', ActivityMarkdownMessage],
  ] as Array<[string, ComponentType<{ content: string }>]>)(
    '%s clamps deep-first headings without skipping route levels',
    async (_surface, MarkdownMessage) => {
      render(
        <main>
          <h1>Route heading</h1>
          <MarkdownMessage content={'#### Deep start\n\n###### Deeper'} />
        </main>,
      )

      expect(
        screen.getByRole('heading', { level: 2, name: 'Deep start' }),
      ).toBeInTheDocument()
      expect(
        screen.getByRole('heading', { level: 3, name: 'Deeper' }),
      ).toBeInTheDocument()
      const violations = await getA11yViolations()
      expect(violations, formatA11yViolations(violations)).toEqual([])
    },
  )
})
