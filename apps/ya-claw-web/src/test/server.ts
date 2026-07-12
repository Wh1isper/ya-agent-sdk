import { setupServer } from 'msw/node'

import { apiHandlers } from './handlers'

export const apiServer = setupServer(...apiHandlers)
