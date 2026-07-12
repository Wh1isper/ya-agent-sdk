import {
  createBrowserHistory,
  createRootRoute,
  createRoute,
  createRouter,
  lazyRouteComponent,
  redirect,
  type RouterHistory,
} from '@tanstack/react-router'

import { AuthenticatedAppShell } from './AppShell'
import { registerAppNavigate } from './navigation'
import { NotFoundPage, RouteErrorPage } from './RouteErrorPages'

const rootRoute = createRootRoute({
  component: AuthenticatedAppShell,
  notFoundComponent: NotFoundPage,
  errorComponent: ({ error, reset }) => (
    <RouteErrorPage
      error={error instanceof Error ? error : new Error(String(error))}
      onRetry={reset}
    />
  ),
})

const homeRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: lazyRouteComponent(
    () => import('../features/home/HomePage'),
    'HomePage',
  ),
})

const conversationsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/conversations',
  component: lazyRouteComponent(
    () => import('../features/chat/ChatPage'),
    'ChatPage',
  ),
})

const newConversationRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/conversations/new',
  component: lazyRouteComponent(
    () => import('../features/chat/ChatPage'),
    'ChatPage',
  ),
})

const conversationSessionRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/conversations/sessions/$sessionId',
  component: lazyRouteComponent(
    () => import('../features/chat/ChatPage'),
    'ChatPage',
  ),
})

const conversationRunRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/conversations/sessions/$sessionId/runs/$runId',
  component: lazyRouteComponent(
    () => import('../features/chat/ChatPage'),
    'ChatPage',
  ),
})

const activityRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/activity',
  component: lazyRouteComponent(
    () => import('../features/chat/DebugPage'),
    'DebugPage',
  ),
})

const activitySessionRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/activity/sessions/$sessionId',
  component: lazyRouteComponent(
    () => import('../features/chat/DebugPage'),
    'DebugPage',
  ),
})

const activityRunRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/activity/sessions/$sessionId/runs/$runId',
  component: lazyRouteComponent(
    () => import('../features/chat/DebugPage'),
    'DebugPage',
  ),
})

const automationRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/automation',
  component: lazyRouteComponent(
    () => import('../features/automation/AutomationPage'),
    'AutomationPage',
  ),
})

const schedulesRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/automation/schedules',
  component: lazyRouteComponent(
    () => import('../features/schedules/SchedulesPage'),
    'SchedulesPage',
  ),
})

const scheduleDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/automation/schedules/$scheduleId',
  component: lazyRouteComponent(
    () => import('../features/schedules/SchedulesPage'),
    'SchedulesPage',
  ),
})

const workflowsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/automation/workflows',
  component: lazyRouteComponent(
    () => import('../features/workflows/WorkflowsPage'),
    'WorkflowsPage',
  ),
})

const workflowDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/automation/workflows/$workflowId',
  component: lazyRouteComponent(
    () => import('../features/workflows/WorkflowsPage'),
    'WorkflowsPage',
  ),
})

const agencyRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/automation/agency',
  component: lazyRouteComponent(
    () => import('../features/agency/AgencyPage'),
    'AgencyPage',
  ),
})

const agencySessionRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/automation/agency/sessions/$sessionId',
  component: lazyRouteComponent(
    () => import('../features/agency/AgencyPage'),
    'AgencyPage',
  ),
})

const agencyRunRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/automation/agency/sessions/$sessionId/runs/$runId',
  component: lazyRouteComponent(
    () => import('../features/agency/AgencyPage'),
    'AgencyPage',
  ),
})

const heartbeatRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/automation/heartbeat',
  component: lazyRouteComponent(
    () => import('../features/heartbeat/HeartbeatPage'),
    'HeartbeatPage',
  ),
})

const workspaceRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/workspace',
  component: lazyRouteComponent(
    () => import('../features/workspace/WorkspacePage'),
    'WorkspacePage',
  ),
})

const agentsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/agents',
  component: lazyRouteComponent(
    () => import('../features/profiles/ProfilesPage'),
    'ProfilesPage',
  ),
})

const newAgentRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/agents/new',
  component: lazyRouteComponent(
    () => import('../features/profiles/ProfilesPage'),
    'ProfilesPage',
  ),
})

const agentDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/agents/by-name/$profileName',
  component: lazyRouteComponent(
    () => import('../features/profiles/ProfilesPage'),
    'ProfilesPage',
  ),
})

const integrationsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/integrations',
  component: lazyRouteComponent(
    () => import('../features/bridges/BridgesPage'),
    'BridgesPage',
  ),
})

const integrationSetupRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/integrations/setup',
  component: lazyRouteComponent(
    () => import('../features/bridges/BridgesPage'),
    'BridgesPage',
  ),
})

const integrationConversationRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/integrations/conversations/$conversationId',
  component: lazyRouteComponent(
    () => import('../features/bridges/BridgesPage'),
    'BridgesPage',
  ),
})

const settingsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/settings',
  component: lazyRouteComponent(
    () => import('../features/settings/SettingsPage'),
    'SettingsPage',
  ),
})

const legacyRoutes = [
  ['/chat', '/conversations'],
  ['/debug', '/activity'],
  ['/schedules', '/automation/schedules'],
  ['/workflows', '/automation/workflows'],
  ['/agency', '/automation/agency'],
  ['/automation/background', '/automation/agency'],
  ['/heartbeat', '/automation/heartbeat'],
  ['/profiles', '/agents'],
  ['/bridges', '/integrations'],
] as const

const legacyRouteNodes = legacyRoutes.map(([path, target]) =>
  createRoute({
    getParentRoute: () => rootRoute,
    path,
    beforeLoad: () => {
      throw redirect({ to: target, replace: true })
    },
  }),
)

const legacyChatSessionRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/chat/sessions/$sessionId',
  beforeLoad: ({ params }) => {
    throw redirect({
      to: '/conversations/sessions/$sessionId',
      params: { sessionId: params.sessionId },
      replace: true,
    })
  },
})

const legacyChatRunRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/chat/sessions/$sessionId/runs/$runId',
  beforeLoad: ({ params }) => {
    throw redirect({
      to: '/conversations/sessions/$sessionId/runs/$runId',
      params: { sessionId: params.sessionId, runId: params.runId },
      replace: true,
    })
  },
})

const legacyDebugSessionRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/debug/sessions/$sessionId',
  beforeLoad: ({ params }) => {
    throw redirect({
      to: '/activity/sessions/$sessionId',
      params: { sessionId: params.sessionId },
      replace: true,
    })
  },
})

const legacyDebugRunRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/debug/sessions/$sessionId/runs/$runId',
  beforeLoad: ({ params }) => {
    throw redirect({
      to: '/activity/sessions/$sessionId/runs/$runId',
      params: { sessionId: params.sessionId, runId: params.runId },
      replace: true,
    })
  },
})

const legacyAgencySessionRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/agency/sessions/$sessionId',
  beforeLoad: ({ params }) => {
    throw redirect({
      to: '/automation/agency/sessions/$sessionId',
      params: { sessionId: params.sessionId },
      replace: true,
    })
  },
})

const legacyAgencyRunRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/agency/sessions/$sessionId/runs/$runId',
  beforeLoad: ({ params }) => {
    throw redirect({
      to: '/automation/agency/sessions/$sessionId/runs/$runId',
      params: { sessionId: params.sessionId, runId: params.runId },
      replace: true,
    })
  },
})

const legacyNewProfileRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/profiles/__new__',
  beforeLoad: () => {
    throw redirect({ to: '/agents/new', replace: true })
  },
})

const legacyProfileDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/profiles/$profileName',
  beforeLoad: ({ params }) => {
    throw redirect({
      to: '/agents/by-name/$profileName',
      params: { profileName: params.profileName },
      replace: true,
    })
  },
})

const legacyAgentDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/agents/$profileName',
  beforeLoad: ({ params }) => {
    throw redirect({
      to: '/agents/by-name/$profileName',
      params: { profileName: params.profileName },
      replace: true,
    })
  },
})

const parameterizedLegacyRouteNodes = [
  legacyChatSessionRoute,
  legacyChatRunRoute,
  legacyDebugSessionRoute,
  legacyDebugRunRoute,
  legacyAgencySessionRoute,
  legacyAgencyRunRoute,
  legacyNewProfileRoute,
  legacyProfileDetailRoute,
  legacyAgentDetailRoute,
]

const routeTree = rootRoute.addChildren([
  homeRoute,
  conversationsRoute,
  newConversationRoute,
  conversationSessionRoute,
  conversationRunRoute,
  activityRoute,
  activitySessionRoute,
  activityRunRoute,
  automationRoute,
  schedulesRoute,
  scheduleDetailRoute,
  workflowsRoute,
  workflowDetailRoute,
  agencyRoute,
  agencySessionRoute,
  agencyRunRoute,
  heartbeatRoute,
  workspaceRoute,
  agentsRoute,
  newAgentRoute,
  agentDetailRoute,
  integrationsRoute,
  integrationSetupRoute,
  integrationConversationRoute,
  settingsRoute,
  ...legacyRouteNodes,
  ...parameterizedLegacyRouteNodes,
])

export function createAppRouter(
  history: RouterHistory = createBrowserHistory(),
) {
  return createRouter({ routeTree, history })
}

export type AppRouter = ReturnType<typeof createAppRouter>

export function registerAppRouter(appRouter: AppRouter) {
  return registerAppNavigate((path, replace) => {
    if (replace) appRouter.history.replace(path)
    else appRouter.history.push(path)
  })
}

export const router = createAppRouter()

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}
