import { act } from '@testing-library/react'
import axe, { type AxeResults, type Result } from 'axe-core'

/**
 * Run axe against rendered application markup. Color contrast is excluded
 * because JSDOM does not compute the CSS custom properties used by the app;
 * contrast remains part of the documented browser audit.
 */
export async function getA11yViolations(
  context: Element | Document = document,
): Promise<Result[]> {
  let results: AxeResults | undefined
  await act(async () => {
    results = await axe.run(context, {
      rules: {
        'color-contrast': { enabled: false },
      },
    })
  })
  return results?.violations ?? []
}

export function formatA11yViolations(violations: Result[]) {
  return violations
    .map(
      (violation) =>
        `${violation.id}: ${violation.help}\n${violation.nodes
          .map((node) => `  ${node.target.join(' ')} — ${node.failureSummary}`)
          .join('\n')}`,
    )
    .join('\n\n')
}
