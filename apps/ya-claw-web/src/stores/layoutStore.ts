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
  | 'schedules'
  | 'bridges'
  | 'heartbeat'
  | 'profiles'
  | 'settings'

export type LayoutState = {
  route: AppRoute
  selectedSessionId: string | null
  selectedRunId: string | null
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
      selectedProfileName: initialUrlSelection.selectedProfileName,
      inspectorTab: 'summary',
      setRoute: (route) => {
        pushBrowserPath(buildRoutePath(route))
        set({
          route,
          selectedSessionId: route === 'chat' ? get().selectedSessionId : null,
          selectedRunId: route === 'chat' ? get().selectedRunId : null,
        })
      },
      selectSession: (selectedSessionId) => {
        pushBrowserPath(buildChatPath(selectedSessionId))
        set((state) => ({
          selectedSessionId,
          selectedRunId: selectedSessionId ? state.selectedRunId : null,
          route: 'chat',
        }))
      },
      selectRun: (selectedRunId) => {
        pushBrowserPath(buildChatPath(get().selectedSessionId, selectedRunId))
        set({ selectedRunId, route: 'chat' })
      },
      selectProfile: (selectedProfileName) => {
        pushBrowserPath(buildProfilePath(selectedProfileName))
        set({ selectedProfileName, route: 'profiles' })
      },
      setInspectorTab: (inspectorTab) => set({ inspectorTab }),
      syncFromUrl: () => {
        const next = parseUrlSelection()
        set(next)
        replaceBrowserPath(
          next.route === 'chat'
            ? buildChatPath(next.selectedSessionId, next.selectedRunId)
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
