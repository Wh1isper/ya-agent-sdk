import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export function getDefaultBaseUrl() {
  if (import.meta.env.VITE_CLAW_BASE_URL) {
    return normalizeBaseUrl(import.meta.env.VITE_CLAW_BASE_URL)
  }
  if (typeof window !== 'undefined') {
    return window.location.origin
  }
  return 'http://127.0.0.1:9042'
}

export function normalizeBaseUrl(baseUrl: string) {
  return baseUrl.trim().replace(/\/+$/, '')
}

function createConnectionScope() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`
}

type PersistedConnection = {
  baseUrl?: unknown
  apiToken?: unknown
  rememberConnection?: unknown
}

function normalizedPersistedConnection(persistedState: unknown) {
  const persisted = persistedState as PersistedConnection | null
  const rememberConnection = persisted?.rememberConnection === true
  return {
    ...(typeof persisted?.baseUrl === 'string'
      ? { baseUrl: normalizeBaseUrl(persisted.baseUrl) }
      : {}),
    rememberConnection,
    ...(rememberConnection && typeof persisted?.apiToken === 'string'
      ? { apiToken: persisted.apiToken.trim() }
      : {}),
  }
}

function hasPersistedApiToken(rawValue: string | null) {
  if (!rawValue) return false
  try {
    const persisted = JSON.parse(rawValue) as { state?: PersistedConnection }
    return (
      typeof persisted.state?.apiToken === 'string' &&
      persisted.state.apiToken.trim().length > 0
    )
  } catch {
    return false
  }
}

export type ConnectionState = {
  baseUrl: string
  apiToken: string
  rememberConnection: boolean
  connectionScope: string
  connectionIssue: string | null
  connectionDraftDirty: boolean
  setConnectionDraftDirty: (dirty: boolean) => void
  setConnection: (connection: {
    baseUrl: string
    apiToken: string
    rememberConnection?: boolean
  }) => void
  invalidateConnection: (reason: string, connectionScope?: string) => void
  logout: () => void
}

export const useConnectionStore = create<ConnectionState>()(
  persist(
    (set) => ({
      baseUrl: getDefaultBaseUrl(),
      apiToken: '',
      rememberConnection: false,
      connectionScope: createConnectionScope(),
      connectionIssue: null,
      connectionDraftDirty: false,
      setConnectionDraftDirty: (connectionDraftDirty) =>
        set({ connectionDraftDirty }),
      setConnection: ({ baseUrl, apiToken, rememberConnection }) =>
        set((state) => ({
          baseUrl: normalizeBaseUrl(baseUrl),
          apiToken: apiToken.trim(),
          rememberConnection: rememberConnection ?? state.rememberConnection,
          connectionScope: createConnectionScope(),
          connectionIssue: null,
          connectionDraftDirty: false,
        })),
      invalidateConnection: (reason, connectionScope) =>
        set((state) => {
          if (connectionScope && connectionScope !== state.connectionScope) {
            return state
          }
          return {
            apiToken: '',
            rememberConnection: false,
            connectionScope: createConnectionScope(),
            connectionIssue: reason,
            connectionDraftDirty: false,
          }
        }),
      logout: () =>
        set({
          apiToken: '',
          rememberConnection: false,
          connectionScope: createConnectionScope(),
          connectionIssue: null,
          connectionDraftDirty: false,
        }),
    }),
    {
      name: 'ya-claw-connection',
      version: 2,
      partialize: (state) => ({
        baseUrl: state.baseUrl,
        rememberConnection: state.rememberConnection,
        ...(state.rememberConnection ? { apiToken: state.apiToken } : {}),
      }),
      migrate: (persistedState, version) => {
        if (version < 2) {
          const persisted = persistedState as PersistedConnection | null
          return typeof persisted?.baseUrl === 'string'
            ? { baseUrl: normalizeBaseUrl(persisted.baseUrl) }
            : {}
        }
        return normalizedPersistedConnection(persistedState)
      },
      merge: (persistedState, currentState) => ({
        ...currentState,
        ...normalizedPersistedConnection(persistedState),
      }),
    },
  ),
)

if (typeof window !== 'undefined') {
  window.addEventListener('storage', (event) => {
    if (
      event.storageArea !== window.localStorage ||
      (event.key !== 'ya-claw-connection' && event.key !== null) ||
      hasPersistedApiToken(event.newValue)
    ) {
      return
    }

    const connection = useConnectionStore.getState()
    if (connection.apiToken) connection.logout()
  })
}
