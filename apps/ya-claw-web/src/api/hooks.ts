import {
  keepPreviousData,
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query'
import { useMemo } from 'react'
import { toast } from 'sonner'

import { useConnectionStore } from '../stores/connectionStore'
import type {
  AgencyTriggerRequest,
  BridgeEventStatus,
  InputPart,
  ProfileUpsertRequest,
  ScheduleCreateRequest,
  ScheduleUpdateRequest,
  SessionRunCreateRequest,
} from '../types'
import { ClawApiClient } from './client'
import { queryKeys } from './queryKeys'

export function useApiClient() {
  const baseUrl = useConnectionStore((state) => state.baseUrl)
  const apiToken = useConnectionStore((state) => state.apiToken)
  return useMemo(
    () => new ClawApiClient({ baseUrl, apiToken }),
    [apiToken, baseUrl],
  )
}

export function useHealthQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: () => api.health(),
    refetchInterval: 15_000,
    retry: 1,
  })
}

export function useClawInfoQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.clawInfo,
    queryFn: () => api.clawInfo(),
    staleTime: 60_000,
    retry: 1,
  })
}

export function useWorkspaceRuntimeQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.workspaceRuntime,
    queryFn: () => api.getWorkspaceRuntime(),
    refetchInterval: 15_000,
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
    queryFn: () => api.getSessionWorkspace(sessionId ?? ''),
    enabled: Boolean(sessionId),
    placeholderData: keepPreviousData,
    refetchInterval: 2_000,
    staleTime: 1_000,
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
    queryFn: () => api.listBridgeConversations(),
    placeholderData: keepPreviousData,
    refetchInterval: 10_000,
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
    queryFn: () => api.listBridgeEvents(filters),
    placeholderData: keepPreviousData,
    refetchInterval: 10_000,
    staleTime: 5_000,
  })
}

export function useSessionsQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.sessions,
    queryFn: () => api.listSessions(),
    placeholderData: keepPreviousData,
    refetchInterval: 5_000,
    staleTime: 2_000,
  })
}

export function useSessionQuery(sessionId: string | null) {
  const api = useApiClient()
  return useQuery({
    queryKey: sessionId ? queryKeys.session(sessionId) : ['session', 'none'],
    queryFn: () => api.getSession(sessionId ?? ''),
    enabled: Boolean(sessionId),
    placeholderData: keepPreviousData,
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
    queryFn: ({ pageParam }: { pageParam: number | null }) =>
      api.getSession(sessionId ?? '', {
        runsLimit,
        beforeSequenceNo: pageParam,
        includeMessage: true,
        includeInputParts: true,
      }),
    enabled: Boolean(sessionId),
    initialPageParam: null as number | null,
    getNextPageParam: (lastPage) =>
      lastPage.session.runs_next_before_sequence_no ?? undefined,
    placeholderData: keepPreviousData,
    staleTime: 5_000,
  })
}

export function useAgencyConfigQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.agencyConfig,
    queryFn: () => api.getAgencyConfig(),
    placeholderData: keepPreviousData,
    refetchInterval: 10_000,
    staleTime: 5_000,
  })
}

export function useAgencyStatusQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.agencyStatus,
    queryFn: () => api.getAgencyStatus(),
    placeholderData: keepPreviousData,
    refetchInterval: 5_000,
    staleTime: 2_000,
  })
}

export function useAgencyFiresQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.agencyFires,
    queryFn: () => api.listAgencyFires(),
    placeholderData: keepPreviousData,
    refetchInterval: 5_000,
    staleTime: 2_000,
  })
}

export function useAgencyMutations() {
  const api = useApiClient()
  const queryClient = useQueryClient()
  const refresh = async (agencySessionId?: string | null) => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: queryKeys.agencyConfig }),
      queryClient.invalidateQueries({ queryKey: queryKeys.agencyStatus }),
      queryClient.invalidateQueries({ queryKey: queryKeys.agencyFires }),
      queryClient.invalidateQueries({ queryKey: queryKeys.sessions }),
      agencySessionId
        ? queryClient.invalidateQueries({
            queryKey: queryKeys.session(agencySessionId),
          })
        : Promise.resolve(),
      agencySessionId
        ? queryClient.invalidateQueries({
            queryKey: queryKeys.sessionHistoryBase(agencySessionId),
          })
        : Promise.resolve(),
    ])
  }
  return {
    trigger: useMutation({
      mutationFn: (payload: AgencyTriggerRequest) => api.triggerAgency(payload),
      onSuccess: async (response) => {
        toast.success(`Agency fire ${response.delivery}`)
        await refresh(response.agency_session_id)
      },
    }),
  }
}

export function useRunQuery(runId: string | null) {
  const api = useApiClient()
  return useQuery({
    queryKey: runId ? queryKeys.run(runId) : ['run', 'none'],
    queryFn: () => api.getRun(runId ?? ''),
    enabled: Boolean(runId),
    placeholderData: keepPreviousData,
    staleTime: 5_000,
  })
}

export function useRunTraceQuery(runId: string | null) {
  const api = useApiClient()
  return useQuery({
    queryKey: runId ? queryKeys.runTrace(runId) : ['run-trace', 'none'],
    queryFn: () => api.getRunTrace(runId ?? ''),
    enabled: Boolean(runId),
    placeholderData: keepPreviousData,
    staleTime: 10_000,
  })
}

export function useProfilesQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.profiles,
    queryFn: () => api.listProfiles(),
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
    queryFn: () => api.getProfile(profileName ?? ''),
    enabled: Boolean(profileName),
    placeholderData: keepPreviousData,
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

export function useCreateSessionRunMutation(sessionId: string | null) {
  const api = useApiClient()
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (payload: SessionRunCreateRequest) =>
      api.createSessionRun(sessionId ?? '', payload),
    onSuccess: async (run) => {
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
        queryClient.invalidateQueries({ queryKey: queryKeys.run(run.id) }),
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
    steer: useMutation({
      mutationFn: (inputParts: InputPart[]) =>
        api.steerRun(runId ?? '', inputParts),
      onSuccess: refresh,
    }),
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

export function useSchedulesQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.schedules,
    queryFn: () => api.listSchedules(),
    placeholderData: keepPreviousData,
    staleTime: 10_000,
  })
}

export function useScheduleQuery(scheduleId: string | null) {
  const api = useApiClient()
  return useQuery({
    queryKey: scheduleId
      ? queryKeys.schedule(scheduleId)
      : ['schedule', 'none'],
    queryFn: () => api.getSchedule(scheduleId ?? ''),
    enabled: Boolean(scheduleId),
    placeholderData: keepPreviousData,
    staleTime: 10_000,
  })
}

export function useScheduleFiresQuery(scheduleId: string | null) {
  const api = useApiClient()
  return useQuery({
    queryKey: scheduleId
      ? queryKeys.scheduleFires(scheduleId)
      : ['schedule-fires', 'none'],
    queryFn: () => api.listScheduleFires(scheduleId ?? ''),
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
      await queryClient.invalidateQueries({ queryKey: queryKeys.schedules })
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
        queryClient.invalidateQueries({ queryKey: queryKeys.schedules }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.schedule(schedule.id),
        }),
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
      await queryClient.invalidateQueries({ queryKey: queryKeys.schedules })
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
        queryClient.invalidateQueries({ queryKey: queryKeys.schedules }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.scheduleFires(fire.schedule_id),
        }),
        queryClient.invalidateQueries({ queryKey: queryKeys.sessions }),
      ])
    },
  })
}

export function useHeartbeatConfigQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.heartbeatConfig,
    queryFn: () => api.getHeartbeatConfig(),
    staleTime: 10_000,
  })
}

export function useHeartbeatStatusQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.heartbeatStatus,
    queryFn: () => api.getHeartbeatStatus(),
    refetchInterval: 15_000,
    staleTime: 10_000,
  })
}

export function useHeartbeatFiresQuery() {
  const api = useApiClient()
  return useQuery({
    queryKey: queryKeys.heartbeatFires,
    queryFn: () => api.listHeartbeatFires(),
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
