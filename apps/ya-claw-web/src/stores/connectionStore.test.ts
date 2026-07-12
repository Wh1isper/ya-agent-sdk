import { beforeEach, describe, expect, it } from 'vitest'

import { getDefaultBaseUrl, useConnectionStore } from './connectionStore'

describe('connectionStore', () => {
  beforeEach(() => {
    localStorage.clear()
    useConnectionStore.setState({
      baseUrl: getDefaultBaseUrl(),
      apiToken: '',
      connectionScope: 'initial-scope',
      connectionIssue: null,
    })
  })

  it('scrubs an API token from legacy persisted browser storage', async () => {
    localStorage.setItem(
      'ya-claw-connection',
      JSON.stringify({
        state: {
          baseUrl: 'http://legacy-runtime.local/',
          apiToken: 'legacy-secret-token',
          connectionScope: 'legacy-scope',
          connectionIssue: 'legacy issue',
        },
        version: 0,
      }),
    )

    await useConnectionStore.persist.rehydrate()

    expect(useConnectionStore.getState()).toMatchObject({
      baseUrl: 'http://legacy-runtime.local',
      apiToken: '',
      connectionScope: 'initial-scope',
      connectionIssue: null,
    })
    expect(localStorage.getItem('ya-claw-connection')).not.toContain(
      'legacy-secret-token',
    )
  })

  it('keeps the API token out of persisted browser storage', () => {
    useConnectionStore.getState().setConnection({
      baseUrl: 'http://runtime.local/',
      apiToken: 'super-secret-token',
    })

    expect(useConnectionStore.getState().apiToken).toBe('super-secret-token')
    expect(localStorage.getItem('ya-claw-connection')).not.toContain(
      'super-secret-token',
    )
  })

  it('rotates the connection scope after reconnect and logout', () => {
    const firstScope = useConnectionStore.getState().connectionScope
    useConnectionStore.getState().setConnection({
      baseUrl: 'http://runtime.local',
      apiToken: 'token',
    })
    const connectedScope = useConnectionStore.getState().connectionScope
    expect(connectedScope).not.toBe(firstScope)

    useConnectionStore.getState().logout()
    expect(useConnectionStore.getState().apiToken).toBe('')
    expect(useConnectionStore.getState().connectionScope).not.toBe(
      connectedScope,
    )
  })

  it('ignores unauthorized responses from an obsolete connection', () => {
    useConnectionStore.getState().setConnection({
      baseUrl: 'http://runtime.local',
      apiToken: 'current-token',
    })

    useConnectionStore
      .getState()
      .invalidateConnection('Expired', 'obsolete-scope')

    expect(useConnectionStore.getState().apiToken).toBe('current-token')
    expect(useConnectionStore.getState().connectionIssue).toBeNull()
  })
})
