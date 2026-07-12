export const REFERENCE_VIEWPORTS = [320, 390, 768, 1024, 1440] as const

export function setTestViewport(width: number, height = 900) {
  Object.defineProperties(window, {
    innerWidth: { configurable: true, value: width },
    outerWidth: { configurable: true, value: width },
    innerHeight: { configurable: true, value: height },
    outerHeight: { configurable: true, value: height },
  })

  Object.defineProperty(window, 'matchMedia', {
    configurable: true,
    value: (query: string): MediaQueryList => {
      const min = /\(min-width:\s*(\d+)px\)/.exec(query)
      const max = /\(max-width:\s*(\d+)px\)/.exec(query)
      const matches =
        (!min || width >= Number(min[1])) && (!max || width <= Number(max[1]))
      return {
        matches,
        media: query,
        onchange: null,
        addEventListener: () => undefined,
        removeEventListener: () => undefined,
        addListener: () => undefined,
        removeListener: () => undefined,
        dispatchEvent: () => true,
      }
    },
  })
  window.dispatchEvent(new Event('resize'))
}

/**
 * Return DOM nodes whose measured box escapes the viewport. JSDOM reports
 * zero-sized boxes unless a component/test supplies measurements, so this is a
 * regression signal for explicit DOM sizing—not a replacement for browser
 * screenshots or manual visual QA.
 */
export function findHorizontalOverflowSignals(
  root: HTMLElement = document.documentElement,
) {
  const tolerance = 1
  const signals: string[] = []
  if (root.scrollWidth > window.innerWidth + tolerance) {
    signals.push(
      `document scrollWidth ${root.scrollWidth}px exceeds ${window.innerWidth}px`,
    )
  }

  for (const element of root.querySelectorAll<HTMLElement>('body *')) {
    const rect = element.getBoundingClientRect()
    if (rect.width <= 0 && rect.height <= 0) continue
    if (rect.left < -tolerance || rect.right > window.innerWidth + tolerance) {
      const identity = [
        element.tagName.toLowerCase(),
        element.id ? `#${element.id}` : '',
        element.className && typeof element.className === 'string'
          ? `.${element.className.trim().split(/\s+/).slice(0, 2).join('.')}`
          : '',
      ].join('')
      signals.push(
        `${identity} spans ${Math.round(rect.left)}px–${Math.round(rect.right)}px`,
      )
    }
  }
  return signals
}
