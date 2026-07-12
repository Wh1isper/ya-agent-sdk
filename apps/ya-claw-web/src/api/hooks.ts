import {
  keepPreviousData,
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query'
import { useEffect, useMemo } from 'react'
import { toast } from 'sonner'

import { useConnectionStore } from '../stores/connectionStore'
import type {
  BridgeEventStatus,
  InputPart,
  ProfileUpsertRequest,
  ScheduleCreateRequest,
  ScheduleListFilters,
  ScheduleUpdateRequest,
  SessionSubmitRequest,
  WorkflowDefinitionCreateRequest,
  WorkflowDefinitionUpdateRequest,
  WorkflowListFilters,
  WorkflowRunListFilters,
  WorkflowTriggerRequest,
} from '../types'
import { ClawApiClient } from './client'
import { queryKeys } from './queryKeys'

function pollingInterval(milliseconds: number) {
  return import.meta.env.MODE === 'test' ? false : milliseconds
}

export function useApiClient() {
  const baseUrl = useConnectionStore((state) => state.baseUrl)
  const apiToken = useConnectionStore((state) => state.apiToken)
  const connectionScope = useConnectionStore((state) => state.connectionScope)
  const invalidateConnection = useConnectionStore(
    (state) => state.invalidateConnection,
  )
  return useMemo(
    () =>
      new ClawApiClient({
        baseUrl,
        apiToken,
        connectionScope,
        onUnauthorized: (scope) =>
          invalidateConnection('Your API token is invalid or expired.', scope),
      }),
    [apiToken, baseUrl, connectionScope, invalidateConnection],
  )
}

export function useHealthQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: ({ signal }) => api.health({ signal }),
    refetchInterval: pollingInterval(15_000),
    retry: 1,
  })
}

export function useClawInfoQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.clawInfo,
    queryFn: ({ signal }) => api.clawInfo({ signal }),
    staleTime: 60_000,
    retry: 1,
  })
}

export function useWorkspaceRuntimeQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.workspaceRuntime,
    queryFn: ({ signal }) => api.getWorkspaceRuntime({ signal }),
    refetchInterval: pollingInterval(15_000),
    staleTime: 10_000,
    retry: 1,
  })
}

export function useSessionWorkspaceQuery(sessionId: string | null) {
  const api = useApiClient()
  return useQuery({
    queryKey: sessionId
      ? queryKeys.sessionWorkspace(sessionId)
      : ['session-workspace', 'none'],
    queryFn: ({ signal }) =>
      api.getSessionWorkspace(sessionId ?? '', { signal }),
    enabled: Boolean(sessionId),
    refetchInterval: pollingInterval(2_000),
    staleTime: 1_000,
  })
}

type WorkspacePageParam = { cursor?: string; offset?: number }

const WORKSPACE_AUTO_LOAD_MAX_PAGES = 50
const WORKSPACE_AUTO_LOAD_MAX_ITEMS = 25_000

export function useWorkspaceFilesQuery(
  sessionId: string | null,
  path: string | null,
  options: { autoLoadAll?: boolean } = {},
) {
  const api = useApiClient()
  const queryClient = useQueryClient()
  const normalizedPath = path ?? ''
  const queryKey = queryKeys.workspaceFiles(sessionId ?? 'none', normalizedPath)
  const query = useInfiniteQuery({
    queryKey,
    queryFn: ({ pageParam, signal }) =>
      api.listWorkspaceFiles(sessionId ?? '', {
        path,
        cursor: pageParam.cursor,
        offset: pageParam.offset,
        signal,
      }),
    enabled: Boolean(sessionId),
    initialPageParam: {} as WorkspacePageParam,
    getNextPageParam: (
      lastPage,
      allPages,
      _lastPageParam,
      allPageParams,
    ): WorkspacePageParam | undefined => {
      if (
        !(
          lastPage.has_more ||
          lastPage.truncated ||
          lastPage.next_cursor ||
          lastPage.next_offset != null
        )
      ) {
        return undefined
      }
      // A continuation that returned no entries cannot derive a safe fallback
      // offset, even if a buggy server still advertises another page.
      if (lastPage.items.length === 0) return undefined
      if (
        options.autoLoadAll &&
        (allPages.length >= WORKSPACE_AUTO_LOAD_MAX_PAGES ||
          allPages.reduce((count, page) => count + page.items.length, 0) >=
            WORKSPACE_AUTO_LOAD_MAX_ITEMS)
      ) {
        return undefined
      }

      if (lastPage.next_cursor) {
        const cursorWasRequested = allPageParams.some(
          (pageParam) => pageParam.cursor === lastPage.next_cursor,
        )
        return cursorWasRequested ? undefined : { cursor: lastPage.next_cursor }
      }

      // Continue safely against older servers that only exposed offsets or
      // `truncated`; the offset must advance and must not have been requested.
      const currentOffset = lastPage.offset ?? 0
      const nextOffset =
        lastPage.next_offset ?? currentOffset + lastPage.items.length
      const offsetWasRequested = allPageParams.some(
        (pageParam) => pageParam.offset === nextOffset,
      )
      if (nextOffset <= currentOffset || offsetWasRequested) return undefined
      return { offset: nextOffset }
    },
    select: (data) => {
      const firstPage = data.pages[0]
      const lastPage = data.pages[data.pages.length - 1]
      if (!firstPage || !lastPage) return undefined
      const itemsByCanonicalPath = new Map(
        data.pages
          .flatMap((page) => page.items)
          .map((item) => [item.path, item] as const),
      )
      return {
        ...firstPage,
        items: [...itemsByCanonicalPath.values()],
        has_more: lastPage.has_more,
        next_cursor: lastPage.next_cursor ?? null,
        next_offset: lastPage.next_offset,
        truncated: lastPage.truncated,
      }
    },
    // A failed continuation must remain stopped until the user explicitly
    // retries. Automatic query retries would violate that contract.
    retry: false,
    staleTime: 5_000,
  })

  useEffect(() => {
    const activeQueryKey = queryKeys.workspaceFiles(
      sessionId ?? 'none',
      normalizedPath,
    )
    return () => {
      void queryClient.cancelQueries({ queryKey: activeQueryKey, exact: true })
    }
  }, [normalizedPath, queryClient, sessionId])

  const continuationKey =
    query.data?.next_cursor ?? query.data?.next_offset ?? null
  const {
    fetchNextPage,
    hasNextPage,
    isFetchNextPageError,
    isFetchingNextPage,
  } = query
  useEffect(() => {
    if (
      options.autoLoadAll &&
      hasNextPage &&
      !isFetchingNextPage &&
      !isFetchNextPageError
    ) {
      void fetchNextPage()
    }
  }, [
    continuationKey,
    fetchNextPage,
    hasNextPage,
    isFetchNextPageError,
    isFetchingNextPage,
    options.autoLoadAll,
  ])

  return query
}

export function useWorkspaceFileQuery(
  sessionId: string | null,
  path: string | null,
) {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.workspaceFile(sessionId ?? 'none', path ?? ''),
    queryFn: ({ signal }) =>
      api.getWorkspaceFile(sessionId ?? '', path ?? '', { signal }),
    enabled: Boolean(sessionId && path),
    staleTime: 10_000,
  })
}

export function useSessionSandboxMutations(sessionId: string | null) {
  const api = useApiClient()
  const queryClient = useQueryClient()
  const refresh = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: queryKeys.sessions }),
      queryClient.invalidateQueries({ queryKey: queryKeys.workspaceRuntime }),
      sessionId
        ? queryClient.invalidateQueries({
            queryKey: queryKeys.session(sessionId),
          })
        : Promise.resolve(),
      sessionId
        ? queryClient.invalidateQueries({
            queryKey: queryKeys.sessionWorkspace(sessionId),
          })
        : Promise.resolve(),
      sessionId
        ? queryClient.invalidateQueries({
            queryKey: queryKeys.sessionSandbox(sessionId),
          })
        : Promise.resolve(),
    ])
  }
  return {
    prepare: useMutation({
      mutationFn: () => api.prepareSessionSandbox(sessionId ?? ''),
      onSuccess: refresh,
    }),
    stop: useMutation({
      mutationFn: () => api.stopSessionSandbox(sessionId ?? ''),
      onSuccess: refresh,
    }),
  }
}

export function useBridgeConversationsQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.bridgeConversations,
    queryFn: ({ signal }) => api.listBridgeConversations({ signal }),
    placeholderData: keepPreviousData,
    refetchInterval: pollingInterval(10_000),
    staleTime: 5_000,
  })
}

export function useBridgeEventsQuery(filters: {
  conversationId?: string | null
  status?: BridgeEventStatus | 'all'
}) {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.bridgeEvents(filters.conversationId, filters.status),
    queryFn: ({ signal }) => api.listBridgeEvents(filters, { signal }),
    placeholderData: keepPreviousData,
    refetchInterval: pollingInterval(10_000),
    staleTime: 5_000,
  })
}

export function useSessionsQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.sessions,
    queryFn: ({ signal }) => api.listSessions({ signal }),
    placeholderData: keepPreviousData,
    refetchInterval: pollingInterval(5_000),
    staleTime: 2_000,
  })
}

export function useSessionQuery(sessionId: string | null) {
  const api = useApiClient()
  return useQuery({
    queryKey: sessionId ? queryKeys.session(sessionId) : ['session', 'none'],
    queryFn: ({ signal }) => api.getSession(sessionId ?? '', { signal }),
    enabled: Boolean(sessionId),
    staleTime: 5_000,
  })
}

export function useSessionHistoryQuery(
  sessionId: string | null,
  options: { runsLimit?: number } = {},
) {
  const api = useApiClient()
  const runsLimit = options.runsLimit ?? 3
  return useInfiniteQuery({
    queryKey: sessionId
      ? queryKeys.sessionHistory(sessionId, runsLimit)
      : ['session-history', 'none', runsLimit],
    queryFn: ({ pageParam, signal }) =>
      api.getSession(sessionId ?? '', {
        runsLimit,
        beforeSequenceNo: pageParam,
        includeMessage: true,
        includeInputParts: true,
        signal,
      }),
    enabled: Boolean(sessionId),
    initialPageParam: null as number | null,
    getNextPageParam: (lastPage) =>
      lastPage.session.runs_next_before_sequence_no ?? undefined,
    staleTime: 5_000,
  })
}

export function useAgencyConfigQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.agencyConfig,
    queryFn: ({ signal }) => api.getAgencyConfig({ signal }),
    placeholderData: keepPreviousData,
    refetchInterval: pollingInterval(10_000),
    staleTime: 5_000,
  })
}

export function useAgencyStatusQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.agencyStatus,
    queryFn: ({ signal }) => api.getAgencyStatus({ signal }),
    placeholderData: keepPreviousData,
    refetchInterval: pollingInterval(5_000),
    staleTime: 2_000,
  })
}

export function useAgencyFiresQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.agencyFires,
    queryFn: ({ signal }) => api.listAgencyFires({ signal }),
    placeholderData: keepPreviousData,
    refetchInterval: pollingInterval(5_000),
    staleTime: 2_000,
  })
}

export function useAgencyMutations() {
  const api = useApiClient()
  const queryClient = useQueryClient()
  const refresh = async (
    ...agencySessionIds: Array<string | null | undefined>
  ) => {
    const sessionIds = agencySessionIds.filter((value): value is string =>
      Boolean(value),
    )
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: queryKeys.agencyConfig }),
      queryClient.invalidateQueries({ queryKey: queryKeys.agencyStatus }),
      queryClient.invalidateQueries({ queryKey: queryKeys.agencyFires }),
      queryClient.invalidateQueries({ queryKey: queryKeys.sessions }),
      queryClient.invalidateQueries({ queryKey: ['session-history'] }),
      ...sessionIds.flatMap((agencySessionId) => [
        queryClient.invalidateQueries({
          queryKey: queryKeys.session(agencySessionId),
        }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.sessionHistoryBase(agencySessionId),
        }),
      ]),
    ])
  }
  return {
    clear: useMutation({
      mutationFn: () => api.clearAgency(),
      onSuccess: async (response) => {
        toast.success('Agency cleared')
        await refresh(
          response.cleared_session_id,
          response.new_agency_session_id,
        )
      },
    }),
  }
}

export function useRunQuery(runId: string | null) {
  const api = useApiClient()
  return useQuery({
    queryKey: runId ? queryKeys.run(runId) : ['run', 'none'],
    queryFn: ({ signal }) => api.getRun(runId ?? '', { signal }),
    enabled: Boolean(runId),
    staleTime: 5_000,
  })
}

export function useRunTraceQuery(runId: string | null) {
  const api = useApiClient()
  return useQuery({
    queryKey: runId ? queryKeys.runTrace(runId) : ['run-trace', 'none'],
    queryFn: ({ signal }) => api.getRunTrace(runId ?? '', { signal }),
    enabled: Boolean(runId),
    staleTime: 10_000,
  })
}

export function useProfilesQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.profiles,
    queryFn: ({ signal }) => api.listProfiles({ signal }),
    placeholderData: keepPreviousData,
    staleTime: 10_000,
  })
}

export function useProfileQuery(profileName: string | null) {
  const api = useApiClient()
  return useQuery({
    queryKey: profileName
      ? queryKeys.profile(profileName)
      : ['profile', 'none'],
    queryFn: ({ signal }) => api.getProfile(profileName ?? '', { signal }),
    enabled: Boolean(profileName),
    staleTime: 10_000,
  })
}

export function useCreateSessionMutation() {
  const api = useApiClient()
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (payload: {
      profile_name?: string | null
      input_parts: InputPart[]
      metadata?: Record<string, unknown>
    }) => api.createSession(payload),
    onSuccess: async (response) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: queryKeys.sessions }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.session(response.session.id),
        }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.sessionWorkspace(response.session.id),
        }),
      ])
    },
  })
}

export function useSubmitSessionInputMutation(sessionId: string | null) {
  const api = useApiClient()
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (payload: SessionSubmitRequest) =>
      api.submitSessionInput(sessionId ?? '', payload),
    onSuccess: async (response) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: queryKeys.sessions }),
        sessionId
          ? queryClient.invalidateQueries({
              queryKey: queryKeys.session(sessionId),
            })
          : Promise.resolve(),
        sessionId
          ? queryClient.invalidateQueries({
              queryKey: queryKeys.sessionWorkspace(sessionId),
            })
          : Promise.resolve(),
        queryClient.invalidateQueries({
          queryKey: queryKeys.run(response.run_id),
        }),
      ])
    },
  })
}

export function useRunControlMutations(runId: string | null) {
  const api = useApiClient()
  const queryClient = useQueryClient()
  const refresh = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: queryKeys.sessions }),
      runId
        ? queryClient.invalidateQueries({ queryKey: queryKeys.run(runId) })
        : Promise.resolve(),
    ])
  }
  return {
    interrupt: useMutation({
      mutationFn: () => api.interruptRun(runId ?? ''),
      onSuccess: refresh,
    }),
    cancel: useMutation({
      mutationFn: () => api.cancelRun(runId ?? ''),
      onSuccess: refresh,
    }),
  }
}

export function useUpsertProfileMutation(profileName: string | null) {
  const api = useApiClient()
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      name,
      payload,
    }: {
      name: string
      payload: ProfileUpsertRequest
    }) => api.upsertProfile(name, payload),
    onSuccess: async (profile) => {
      toast.success(`Saved profile ${profile.name}`)
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: queryKeys.profiles }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.profile(profile.name),
        }),
        profileName && profileName !== profile.name
          ? queryClient.invalidateQueries({
              queryKey: queryKeys.profile(profileName),
            })
          : Promise.resolve(),
      ])
    },
  })
}

export function useDeleteProfileMutation() {
  const api = useApiClient()
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (profileName: string) => api.deleteProfile(profileName),
    onSuccess: async (_, profileName) => {
      toast.success(`Deleted profile ${profileName}`)
      await queryClient.invalidateQueries({ queryKey: queryKeys.profiles })
    },
  })
}

export function useSeedProfilesMutation() {
  const api = useApiClient()
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (pruneMissing: boolean) => api.seedProfiles(pruneMissing),
    onSuccess: async (response) => {
      toast.success(`Seeded ${response.seeded_names.length} profiles`)
      await queryClient.invalidateQueries({ queryKey: queryKeys.profiles })
    },
  })
}

function stableFiltersKey(value: unknown) {
  return JSON.stringify(value)
}

export function useWorkflowsQuery(filters: WorkflowListFilters = {}) {
  const api = useApiClient()
  const key = stableFiltersKey(filters)
  return useQuery({
    queryKey: queryKeys.workflows(key),
    queryFn: ({ signal }) => api.listWorkflows(filters, { signal }),
    placeholderData: keepPreviousData,
    staleTime: 10_000,
    refetchInterval: pollingInterval(15_000),
  })
}

export function useWorkflowQuery(workflowId: string | null) {
  const api = useApiClient()
  return useQuery({
    queryKey: workflowId
      ? queryKeys.workflow(workflowId)
      : ['workflow', 'none'],
    queryFn: ({ signal }) => api.getWorkflow(workflowId ?? '', { signal }),
    enabled: Boolean(workflowId),
    staleTime: 10_000,
  })
}

export function useWorkflowRunsQuery(filters: WorkflowRunListFilters = {}) {
  const api = useApiClient()
  const key = stableFiltersKey(filters)
  return useQuery({
    queryKey: queryKeys.workflowRuns(key),
    queryFn: ({ signal }) => api.listWorkflowRuns(filters, { signal }),
    placeholderData: keepPreviousData,
    staleTime: 5_000,
    refetchInterval: pollingInterval(5_000),
  })
}

export function useWorkflowRunQuery(workflowRunId: string | null) {
  const api = useApiClient()
  return useQuery({
    queryKey: workflowRunId
      ? queryKeys.workflowRun(workflowRunId)
      : ['workflow-run', 'none'],
    queryFn: ({ signal }) =>
      api.getWorkflowRun(workflowRunId ?? '', { signal }),
    enabled: Boolean(workflowRunId),
    staleTime: 3_000,
    refetchInterval: pollingInterval(5_000),
  })
}

export function useWorkflowEventsQuery(workflowRunId: string | null) {
  const api = useApiClient()
  return useQuery({
    queryKey: workflowRunId
      ? queryKeys.workflowEvents(workflowRunId)
      : ['workflow-events', 'none'],
    queryFn: ({ signal }) =>
      api.listWorkflowEvents(workflowRunId ?? '', { signal }),
    enabled: Boolean(workflowRunId),
    staleTime: 3_000,
    refetchInterval: pollingInterval(5_000),
  })
}

export function useCreateWorkflowMutation() {
  const api = useApiClient()
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (payload: WorkflowDefinitionCreateRequest) =>
      api.createWorkflow(payload),
    onSuccess: async (workflow) => {
      toast.success(`Created workflow ${workflow.name}`)
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['workflows'] }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.workflow(workflow.id),
        }),
      ])
    },
  })
}

export function useUpdateWorkflowMutation() {
  const api = useApiClient()
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workflowId,
      payload,
    }: {
      workflowId: string
      payload: WorkflowDefinitionUpdateRequest
    }) => api.updateWorkflow(workflowId, payload),
    onSuccess: async (workflow) => {
      toast.success(`Saved workflow ${workflow.name}`)
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['workflows'] }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.workflow(workflow.id),
        }),
      ])
    },
  })
}

export function useArchiveWorkflowMutation() {
  const api = useApiClient()
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (workflowId: string) => api.archiveWorkflow(workflowId),
    onSuccess: async (workflow) => {
      toast.success(`Archived workflow ${workflow.name}`)
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['workflows'] }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.workflow(workflow.id),
        }),
      ])
    },
  })
}

export function useTriggerWorkflowMutation() {
  const api = useApiClient()
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      workflowId,
      payload,
    }: {
      workflowId: string
      payload: WorkflowTriggerRequest
    }) => api.triggerWorkflow(workflowId, payload),
    onSuccess: async (run) => {
      toast.success(`Started workflow run ${run.id.slice(0, 8)}`)
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['workflows'] }),
        queryClient.invalidateQueries({ queryKey: ['workflow-runs'] }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.workflow(run.workflow_id),
        }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.workflowRun(run.id),
        }),
      ])
    },
  })
}

export function useWorkflowRunMutations(workflowRunId: string | null) {
  const api = useApiClient()
  const queryClient = useQueryClient()
  const refresh = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ['workflows'] }),
      queryClient.invalidateQueries({ queryKey: ['workflow-runs'] }),
      workflowRunId
        ? queryClient.invalidateQueries({
            queryKey: queryKeys.workflowRun(workflowRunId),
          })
        : Promise.resolve(),
      workflowRunId
        ? queryClient.invalidateQueries({
            queryKey: queryKeys.workflowEvents(workflowRunId),
          })
        : Promise.resolve(),
    ])
  }
  return {
    cancel: useMutation({
      mutationFn: (reason?: string | null) =>
        api.cancelWorkflowRun(workflowRunId ?? '', reason),
      onSuccess: refresh,
    }),
    steerNode: useMutation({
      mutationFn: ({ nodeId, prompt }: { nodeId: string; prompt: string }) =>
        api.steerWorkflowNode(workflowRunId ?? '', nodeId, {
          prompt,
          input_parts: [],
        }),
      onSuccess: refresh,
    }),
  }
}

export function useSchedulesQuery(filters: ScheduleListFilters = {}) {
  const api = useApiClient()
  const key = stableFiltersKey(filters)
  return useQuery({
    queryKey: queryKeys.schedules(key),
    queryFn: ({ signal }) => api.listSchedules(filters, { signal }),
    placeholderData: keepPreviousData,
    staleTime: 10_000,
    refetchInterval: pollingInterval(30_000),
  })
}

export function useScheduleQuery(scheduleId: string | null) {
  const api = useApiClient()
  return useQuery({
    queryKey: scheduleId
      ? queryKeys.schedule(scheduleId)
      : ['schedule', 'none'],
    queryFn: ({ signal }) => api.getSchedule(scheduleId ?? '', { signal }),
    enabled: Boolean(scheduleId),
    staleTime: 10_000,
    refetchInterval: pollingInterval(30_000),
  })
}

export function useScheduleFiresQuery(scheduleId: string | null) {
  const api = useApiClient()
  return useQuery({
    queryKey: scheduleId
      ? queryKeys.scheduleFires(scheduleId)
      : ['schedule-fires', 'none'],
    queryFn: ({ signal }) =>
      api.listScheduleFires(scheduleId ?? '', { signal }),
    enabled: Boolean(scheduleId),
    placeholderData: keepPreviousData,
    staleTime: 10_000,
  })
}

export function useCreateScheduleMutation() {
  const api = useApiClient()
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (payload: ScheduleCreateRequest) => api.createSchedule(payload),
    onSuccess: async (schedule) => {
      toast.success(`Created schedule ${schedule.name}`)
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['schedules'] }),
        queryClient.invalidateQueries({ queryKey: ['workflows'] }),
      ])
    },
  })
}

export function useUpdateScheduleMutation() {
  const api = useApiClient()
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      scheduleId,
      payload,
    }: {
      scheduleId: string
      payload: ScheduleUpdateRequest
    }) => api.updateSchedule(scheduleId, payload),
    onSuccess: async (schedule) => {
      toast.success(`Saved schedule ${schedule.name}`)
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['schedules'] }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.schedule(schedule.id),
        }),
        queryClient.invalidateQueries({ queryKey: ['workflows'] }),
      ])
    },
  })
}

export function useDeleteScheduleMutation() {
  const api = useApiClient()
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (scheduleId: string) => api.deleteSchedule(scheduleId),
    onSuccess: async (schedule) => {
      toast.success(`Deleted schedule ${schedule.name}`)
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['schedules'] }),
        queryClient.invalidateQueries({ queryKey: ['workflows'] }),
      ])
    },
  })
}

export function useTriggerScheduleMutation() {
  const api = useApiClient()
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      scheduleId,
      promptOverride,
    }: {
      scheduleId: string
      promptOverride?: string | null
    }) => api.triggerSchedule(scheduleId, promptOverride),
    onSuccess: async (fire) => {
      toast.success(`Triggered schedule ${fire.schedule_id.slice(0, 8)}`)
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['schedules'] }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.scheduleFires(fire.schedule_id),
        }),
        queryClient.invalidateQueries({ queryKey: queryKeys.sessions }),
        queryClient.invalidateQueries({ queryKey: ['workflows'] }),
        queryClient.invalidateQueries({ queryKey: ['workflow-runs'] }),
      ])
    },
  })
}

export function useHeartbeatConfigQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.heartbeatConfig,
    queryFn: ({ signal }) => api.getHeartbeatConfig({ signal }),
    staleTime: 10_000,
  })
}

export function useHeartbeatStatusQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.heartbeatStatus,
    queryFn: ({ signal }) => api.getHeartbeatStatus({ signal }),
    refetchInterval: pollingInterval(15_000),
    staleTime: 10_000,
  })
}

export function useHeartbeatFiresQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.heartbeatFires,
    queryFn: ({ signal }) => api.listHeartbeatFires({ signal }),
    placeholderData: keepPreviousData,
    staleTime: 10_000,
  })
}

export function useTriggerHeartbeatMutation() {
  const api = useApiClient()
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () => api.triggerHeartbeat(),
    onSuccess: async () => {
      toast.success('Triggered heartbeat')
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: queryKeys.heartbeatStatus }),
        queryClient.invalidateQueries({ queryKey: queryKeys.heartbeatFires }),
        queryClient.invalidateQueries({ queryKey: queryKeys.sessions }),
      ])
    },
  })
}
