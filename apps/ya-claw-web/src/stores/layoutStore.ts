import { create } from 'zustand'
import { persist } from 'zustand/middleware'

import {
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
  | 'agency'
  | 'schedules'
  | 'bridges'
  | 'heartbeat'
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
  setRoute: (route: AppRoute) => void
  selectSession: (sessionId: string | null) => void
  selectRun: (runId: string | null) => void
  selectProfile: (profileName: string | null) => void
  setInspectorTab: (tab: string) => void
  syncFromUrl: () => void
}

const initialUrlSelection = parseUrlSelection()

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
            : route === 'agency' && selectedSessionId
              ? `/agency/sessions/${encodeURIComponent(selectedSessionId)}`
              : buildRoutePath(route),
        )
        set({ route, selectedSessionId, selectedRunId })
      },
      selectSession: (selectedSessionId) => {
        if (get().route === 'agency') {
          pushBrowserPath(
            selectedSessionId
              ? `/agency/sessions/${encodeURIComponent(selectedSessionId)}`
              : '/agency',
          )
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
        set((state) => {
          const selectedRunId = selectedSessionId ? state.selectedRunId : null
          return route === 'debug'
            ? {
                selectedSessionId,
                selectedRunId,
                selectedDebugSessionId: selectedSessionId,
                selectedDebugRunId: selectedRunId,
                route,
              }
            : {
                selectedSessionId,
                selectedRunId,
                selectedChatSessionId: selectedSessionId,
                selectedChatRunId: selectedRunId,
                route,
              }
        })
      },
      selectRun: (selectedRunId) => {
        const route = get().route === 'debug' ? 'debug' : 'chat'
        const selectedSessionId = get().selectedSessionId
        pushBrowserPath(buildChatPath(selectedSessionId, selectedRunId, route))
        set(
          route === 'debug'
            ? { selectedRunId, selectedDebugRunId: selectedRunId, route }
            : { selectedRunId, selectedChatRunId: selectedRunId, route },
        )
      },
      selectProfile: (selectedProfileName) => {
        pushBrowserPath(buildProfilePath(selectedProfileName))
        set({ selectedProfileName, route: 'profiles' })
      },
      setInspectorTab: (inspectorTab) => set({ inspectorTab }),
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
            : next.route === 'agency' && next.selectedSessionId
              ? `/agency/sessions/${encodeURIComponent(next.selectedSessionId)}`
              : next.route === 'profiles'
                ? buildProfilePath(next.selectedProfileName)
                : buildRoutePath(next.route),
        )
      },
    }),
    {
      name: 'ya-claw-layout',
      partialize: (state) => ({
        inspectorTab: state.inspectorTab,
      }),
    },
  ),
)
