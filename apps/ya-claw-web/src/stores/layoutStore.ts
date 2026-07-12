import { create } from 'zustand'
import { persist } from 'zustand/middleware'

import {
  buildAgencyPath,
  buildChatPath,
  buildProfilePath,
  buildRoutePath,
  parseUrlSelection,
  pushBrowserPath,
  replaceBrowserPath,
} from '../lib/urlState'

export type AppRoute =
  | 'overview'
  | 'chat'
  | 'debug'
  | 'automation'
  | 'agency'
  | 'schedules'
  | 'workflows'
  | 'bridges'
  | 'heartbeat'
  | 'workspace'
  | 'profiles'
  | 'settings'

export type LayoutState = {
  route: AppRoute
  selectedSessionId: string | null
  selectedRunId: string | null
  selectedChatSessionId: string | null
  selectedChatRunId: string | null
  selectedDebugSessionId: string | null
  selectedDebugRunId: string | null
  selectedAgencySessionId: string | null
  selectedProfileName: string | null
  inspectorTab: string
  advancedMode: boolean
  railCollapsed: boolean
  setRoute: (route: AppRoute) => void
  selectSession: (sessionId: string | null) => void
  selectRun: (runId: string | null) => void
  selectProfile: (
    profileName: string | null,
    options?: { replace?: boolean },
  ) => void
  setInspectorTab: (tab: string) => void
  setAdvancedMode: (enabled: boolean) => void
  setRailCollapsed: (collapsed: boolean) => void
  resetConnectionSelection: () => void
  syncFromUrl: () => void
}

const initialUrlSelection = parseUrlSelection()

function persistedLayoutPreferences(persistedState: unknown) {
  const persisted = persistedState as Record<string, unknown> | null
  return {
    ...(typeof persisted?.inspectorTab === 'string'
      ? { inspectorTab: persisted.inspectorTab }
      : {}),
    ...(typeof persisted?.advancedMode === 'boolean'
      ? { advancedMode: persisted.advancedMode }
      : {}),
    ...(typeof persisted?.railCollapsed === 'boolean'
      ? { railCollapsed: persisted.railCollapsed }
      : {}),
  }
}

export const useLayoutStore = create<LayoutState>()(
  persist(
    (set, get) => ({
      route: initialUrlSelection.route,
      selectedSessionId: initialUrlSelection.selectedSessionId,
      selectedRunId: initialUrlSelection.selectedRunId,
      selectedChatSessionId:
        initialUrlSelection.route === 'chat'
          ? initialUrlSelection.selectedSessionId
          : null,
      selectedChatRunId:
        initialUrlSelection.route === 'chat'
          ? initialUrlSelection.selectedRunId
          : null,
      selectedDebugSessionId:
        initialUrlSelection.route === 'debug'
          ? initialUrlSelection.selectedSessionId
          : null,
      selectedDebugRunId:
        initialUrlSelection.route === 'debug'
          ? initialUrlSelection.selectedRunId
          : null,
      selectedAgencySessionId:
        initialUrlSelection.route === 'agency'
          ? initialUrlSelection.selectedSessionId
          : null,
      selectedProfileName: initialUrlSelection.selectedProfileName,
      inspectorTab: 'summary',
      advancedMode: false,
      railCollapsed: false,
      setRoute: (route) => {
        const state = get()
        const selectedSessionId =
          route === 'chat'
            ? state.selectedChatSessionId
            : route === 'debug'
              ? state.selectedDebugSessionId
              : route === 'agency'
                ? state.selectedAgencySessionId
                : null
        const selectedRunId =
          route === 'chat'
            ? state.selectedChatRunId
            : route === 'debug'
              ? state.selectedDebugRunId
              : null
        pushBrowserPath(
          route === 'chat' || route === 'debug'
            ? buildChatPath(selectedSessionId, selectedRunId, route)
            : route === 'agency'
              ? buildAgencyPath(selectedSessionId)
              : buildRoutePath(route),
        )
        set({ route, selectedSessionId, selectedRunId })
      },
      selectSession: (selectedSessionId) => {
        if (get().route === 'agency') {
          pushBrowserPath(buildAgencyPath(selectedSessionId))
          set({
            selectedSessionId,
            selectedRunId: null,
            selectedAgencySessionId: selectedSessionId,
            route: 'agency',
          })
          return
        }
        const route = get().route === 'debug' ? 'debug' : 'chat'
        pushBrowserPath(buildChatPath(selectedSessionId, null, route))
        set(
          route === 'debug'
            ? {
                selectedSessionId,
                selectedRunId: null,
                selectedDebugSessionId: selectedSessionId,
                selectedDebugRunId: null,
                route,
              }
            : {
                selectedSessionId,
                selectedRunId: null,
                selectedChatSessionId: selectedSessionId,
                selectedChatRunId: null,
                route,
              },
        )
      },
      selectRun: (selectedRunId) => {
        const route = get().route === 'debug' ? 'debug' : 'chat'
        const selectedSessionId = get().selectedSessionId
        replaceBrowserPath(
          buildChatPath(selectedSessionId, selectedRunId, route),
        )
        set(
          route === 'debug'
            ? { selectedRunId, selectedDebugRunId: selectedRunId, route }
            : { selectedRunId, selectedChatRunId: selectedRunId, route },
        )
      },
      selectProfile: (selectedProfileName, options) => {
        const path = buildProfilePath(selectedProfileName)
        if (options?.replace) {
          replaceBrowserPath(path)
        } else {
          pushBrowserPath(path)
        }
      },
      setInspectorTab: (inspectorTab) => set({ inspectorTab }),
      setAdvancedMode: (advancedMode) => set({ advancedMode }),
      setRailCollapsed: (railCollapsed) => set({ railCollapsed }),
      resetConnectionSelection: () => {
        replaceBrowserPath(buildRoutePath('overview'), true)
        set({
          route: 'overview',
          selectedSessionId: null,
          selectedRunId: null,
          selectedChatSessionId: null,
          selectedChatRunId: null,
          selectedDebugSessionId: null,
          selectedDebugRunId: null,
          selectedAgencySessionId: null,
          selectedProfileName: null,
        })
      },
      syncFromUrl: () => {
        const next = parseUrlSelection()
        set({
          ...next,
          selectedChatSessionId:
            next.route === 'chat'
              ? next.selectedSessionId
              : get().selectedChatSessionId,
          selectedChatRunId:
            next.route === 'chat'
              ? next.selectedRunId
              : get().selectedChatRunId,
          selectedDebugSessionId:
            next.route === 'debug'
              ? next.selectedSessionId
              : get().selectedDebugSessionId,
          selectedDebugRunId:
            next.route === 'debug'
              ? next.selectedRunId
              : get().selectedDebugRunId,
          selectedAgencySessionId:
            next.route === 'agency'
              ? next.selectedSessionId
              : get().selectedAgencySessionId,
        })
        replaceBrowserPath(
          next.route === 'chat' || next.route === 'debug'
            ? buildChatPath(
                next.selectedSessionId,
                next.selectedRunId,
                next.route,
              )
            : next.route === 'agency'
              ? buildAgencyPath(next.selectedSessionId, next.selectedRunId)
              : next.route === 'profiles'
                ? buildProfilePath(next.selectedProfileName)
                : buildRoutePath(next.route),
        )
      },
    }),
    {
      name: 'ya-claw-layout',
      version: 1,
      partialize: (state) => ({
        inspectorTab: state.inspectorTab,
        advancedMode: state.advancedMode,
        railCollapsed: state.railCollapsed,
      }),
      migrate: (persistedState) => persistedLayoutPreferences(persistedState),
      merge: (persistedState, currentState) => ({
        ...currentState,
        ...persistedLayoutPreferences(persistedState),
      }),
    },
  ),
)
