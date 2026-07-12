import { beforeEach, describe, expect, it } from 'vitest'

import { getDefaultBaseUrl, useConnectionStore } from './connectionStore'

describe('connectionStore', () => {
  beforeEach(() => {
    localStorage.clear()
    useConnectionStore.setState({
      baseUrl: getDefaultBaseUrl(),
      apiToken: '',
      rememberConnection: false,
      connectionScope: 'initial-scope',
      connectionIssue: null,
    })
  })

  it('drops credentials from persisted connections created before opt-in', async () => {
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
      rememberConnection: false,
      connectionScope: 'initial-scope',
      connectionIssue: null,
    })
  })

  it('keeps a connection in memory unless persistence is explicitly enabled', () => {
    useConnectionStore.getState().setConnection({
      baseUrl: 'http://runtime.local/',
      apiToken: 'memory-only-token',
    })

    const persisted = JSON.parse(
      localStorage.getItem('ya-claw-connection') ?? '{}',
    ) as { state?: { apiToken?: string; rememberConnection?: boolean } }
    expect(persisted.state).toMatchObject({ rememberConnection: false })
    expect(persisted.state?.apiToken).toBeUndefined()
  })

  it('persists an explicitly remembered connection after a reload', async () => {
    useConnectionStore.getState().setConnection({
      baseUrl: 'http://runtime.local/',
      apiToken: 'super-secret-token',
      rememberConnection: true,
    })

    const savedConnection = localStorage.getItem('ya-claw-connection')
    expect(savedConnection).toContain('super-secret-token')

    useConnectionStore.setState({
      baseUrl: getDefaultBaseUrl(),
      apiToken: '',
      rememberConnection: false,
      connectionScope: 'reloaded-scope',
      connectionIssue: null,
    })
    localStorage.setItem('ya-claw-connection', savedConnection ?? '')
    await useConnectionStore.persist.rehydrate()

    expect(useConnectionStore.getState()).toMatchObject({
      baseUrl: 'http://runtime.local',
      apiToken: 'super-secret-token',
      rememberConnection: true,
      connectionScope: 'reloaded-scope',
      connectionIssue: null,
    })
  })

  it('rotates the connection scope after reconnect and logout', () => {
    const firstScope = useConnectionStore.getState().connectionScope
    useConnectionStore.getState().setConnection({
      baseUrl: 'http://runtime.local',
      apiToken: 'token',
      rememberConnection: true,
    })
    const connectedScope = useConnectionStore.getState().connectionScope
    expect(connectedScope).not.toBe(firstScope)

    useConnectionStore.getState().logout()
    expect(useConnectionStore.getState().apiToken).toBe('')
    const persisted = JSON.parse(
      localStorage.getItem('ya-claw-connection') ?? '{}',
    ) as { state?: { apiToken?: string; rememberConnection?: boolean } }
    expect(persisted.state).toMatchObject({ rememberConnection: false })
    expect(persisted.state?.apiToken).toBeUndefined()
    expect(useConnectionStore.getState().connectionScope).not.toBe(
      connectedScope,
    )
  })

  it('clears the current tab after another tab removes the saved credential', () => {
    useConnectionStore.getState().setConnection({
      baseUrl: 'http://runtime.local',
      apiToken: 'current-token',
      rememberConnection: true,
    })

    window.dispatchEvent(
      new StorageEvent('storage', {
        key: 'ya-claw-connection',
        newValue: JSON.stringify({
          state: {
            baseUrl: 'http://runtime.local',
            rememberConnection: false,
          },
          version: 2,
        }),
        storageArea: localStorage,
      }),
    )

    expect(useConnectionStore.getState().apiToken).toBe('')
    expect(useConnectionStore.getState().rememberConnection).toBe(false)
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
