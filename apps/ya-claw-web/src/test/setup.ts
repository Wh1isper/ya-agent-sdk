import '@testing-library/jest-dom/vitest'
import { afterAll, afterEach, beforeAll } from 'vitest'

import { resetHandledApiRequests } from './handlers'
import { apiServer } from './server'

// Vitest's synthetic document does not load index.html; mirror its document
// metadata so document-level axe rules exercise the production baseline.
document.documentElement.lang = 'en'
document.title = 'YA Claw'
Object.defineProperty(window, 'scrollTo', {
  configurable: true,
  value: () => undefined,
})

class TestResizeObserver implements ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

Object.defineProperty(globalThis, 'ResizeObserver', {
  configurable: true,
  value: TestResizeObserver,
})

beforeAll(() => {
  apiServer.listen({ onUnhandledRequest: 'error' })
})

afterEach(() => {
  apiServer.resetHandlers()
  resetHandledApiRequests()
})

afterAll(() => {
  apiServer.close()
})
