import type {
  ClawInfo,
  HealthStatus,
  HeartbeatStatus,
  ScheduleListResponse,
  SessionSummary,
  WorkspaceRuntimeStatus,
} from '../types'

export const TEST_API_TOKEN = 'integration-test-token'

export const healthFixture = {
  status: 'ok',
  database: 'ok',
  runtime_state: 'ready',
} satisfies HealthStatus

export const clawInfoFixture = {
  name: 'YA Claw test runtime',
  environment: 'test',
  version: '0.0.0-test',
  service_version: '0.0.0-test',
  service_commit: 'test-commit',
  service_revision: 'test-revision',
  service_build: 'test-build',
  service_image: null,
  public_base_url: 'http://localhost:3000',
  instance_id: 'test-instance',
  auth: 'bearer',
  surfaces: ['web'],
  workspace_provider_backend: 'local',
  storage_model: 'ephemeral',
  features: {
    session_events: true,
    run_events: true,
    notifications: true,
    profiles: true,
    schedules: true,
    heartbeat: true,
    workflows: true,
  },
} satisfies ClawInfo

export const workspaceRuntimeFixture = {
  backend: 'local',
  status: 'ready',
  execution_location: 'test',
  workspace: {
    service_path: '/workspace',
    docker_host_path: null,
    virtual_path: '/workspace',
    exists: true,
    writable: true,
  },
  capabilities: {
    file_browse: true,
    shell: true,
    sandbox_prepare: true,
    sandbox_stop: true,
  },
  checks: [],
  docker: null,
  updated_at: '2026-01-01T00:00:00Z',
} satisfies WorkspaceRuntimeStatus

export const sessionsFixture = [] satisfies SessionSummary[]

export const schedulesFixture = {
  schedules: [],
} satisfies ScheduleListResponse

export const heartbeatStatusFixture = {
  enabled: false,
  next_fire_at: null,
  last_fire: null,
} satisfies HeartbeatStatus
