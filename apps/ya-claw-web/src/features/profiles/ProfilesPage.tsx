import {
  Bot,
  ChevronDown,
  ChevronLeft,
  CopyPlus,
  DatabaseZap,
  RefreshCcw,
  Save,
  Search,
  SlidersHorizontal,
  Trash2,
} from 'lucide-react'
import { useBlocker } from '@tanstack/react-router'
import { useEffect, useMemo, useRef, useState } from 'react'
import { type UseFormRegisterReturn, useForm } from 'react-hook-form'
import { toast } from 'sonner'
import {
  useDeleteProfileMutation,
  useProfileQuery,
  useProfilesQuery,
  useSeedProfilesMutation,
  useUpsertProfileMutation,
} from '../../api/hooks'
import { EmptyState } from '../../components/EmptyState'
import { JsonView } from '../../components/JsonView'
import { StatusBadge } from '../../components/StatusBadge'
import { ConfirmDialog, QueryError } from '../../components/ui'
import { cn, parseJsonObject, safeJsonStringify } from '../../lib/utils'
import { buildProfilePath } from '../../lib/urlState'
import { useLayoutStore } from '../../stores/layoutStore'
import type {
  ClawProfileHostConfig,
  NativeAgentSpec,
  NativeSubagentSpec,
  ProfileDetail,
  ProfileSummary,
  ProfileUpsertRequest,
} from '../../types'

type ProfileFormValues = {
  name: string
  enabled: boolean
  source_type: string
  source_version: string
  source_checksum: string
  agent: string
  host: string
  subagents: string
}

const blankAgent: NativeAgentSpec = {
  model: 'openai:gpt-4.1-mini',
  name: '',
  instructions: 'Be concise, accurate, and workspace-aware.',
  capabilities: [
    'FilesystemCapability',
    'ShellCapability',
    'WebSearchCapability',
    'WebContentCapability',
  ],
}

const blankHost: ClawProfileHostConfig = {
  tool_groups: ['session'],
  need_user_approve_tools: [],
  need_user_approve_mcps: [],
  enabled_mcps: [],
  disabled_mcps: [],
  mcp_servers: {},
}

const blankProfile: ProfileFormValues = {
  name: '',
  enabled: true,
  source_type: 'web',
  source_version: '',
  source_checksum: '',
  agent: safeJsonStringify(blankAgent),
  host: safeJsonStringify(blankHost),
  subagents: '[]',
}

export function ProfilesPage() {
  const profiles = useProfilesQuery()
  const selectedProfileName = useLayoutStore(
    (state) => state.selectedProfileName,
  )
  const selectProfile = useLayoutStore((state) => state.selectProfile)
  const [search, setSearch] = useState('')
  const effectiveProfileName =
    selectedProfileName ?? profiles.data?.[0]?.name ?? null
  const mobileDetailOpen = selectedProfileName !== null
  const filteredProfiles = useMemo(() => {
    const needle = search.trim().toLowerCase()
    const rows = profiles.data ?? []
    if (!needle) return rows
    return rows.filter((profile) =>
      [
        profile.name,
        profile.model,
        profile.workspace_backend_hint ?? '',
        profile.source_type ?? '',
      ]
        .join(' ')
        .toLowerCase()
        .includes(needle),
    )
  }, [profiles.data, search])

  return (
    <div className="flex h-full min-h-0 flex-col overflow-auto bg-slate-100 lg:flex-row lg:overflow-hidden">
      <h1 className="sr-only">Agents</h1>
      <aside
        aria-label="Agent list"
        className={cn(
          'max-h-none w-full shrink-0 flex-col border-b border-slate-200 bg-white lg:flex lg:w-80 lg:border-b-0 lg:border-r',
          mobileDetailOpen ? 'hidden' : 'flex',
        )}
      >
        <div className="border-b border-slate-200 p-4">
          <div className="flex items-center justify-between gap-2">
            <div>
              <p className="text-sm font-medium text-blue-600">
                Native agent configuration
              </p>
              <h2 className="mt-1 text-xl font-semibold tracking-tight text-slate-950">
                Agents
              </h2>
            </div>
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-3 py-2 text-xs font-semibold text-white shadow-sm hover:bg-blue-700"
              onClick={() => selectProfile('__new__')}
            >
              <CopyPlus className="h-3.5 w-3.5" />
              New
            </button>
          </div>
          <div className="relative mt-4">
            <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
            <input
              className="w-full rounded-xl border border-slate-200 bg-slate-50 py-2 pl-9 pr-3 text-sm outline-none ring-blue-600 focus:bg-white focus:ring-2"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search agents"
              aria-label="Search agents"
            />
          </div>
        </div>
        <div className="scrollbar-thin min-h-0 flex-1 overflow-auto p-3">
          {profiles.isError ? (
            <QueryError
              title="Could not load agents"
              error={profiles.error}
              onRetry={() => void profiles.refetch()}
            />
          ) : null}
          {!profiles.isLoading &&
          !profiles.isError &&
          filteredProfiles.length === 0 ? (
            <EmptyState
              title={profiles.data?.length ? 'No matching agents' : 'No agents'}
              description="Create a version 2 native profile or seed the configured profile document."
            />
          ) : null}
          <div className="space-y-2">
            {filteredProfiles.map((profile) => (
              <ProfileListItem
                key={profile.name}
                profile={profile}
                active={effectiveProfileName === profile.name}
                onClick={() => selectProfile(profile.name)}
              />
            ))}
          </div>
        </div>
        <SeedPanel />
      </aside>
      <section
        aria-label="Agent editor"
        className={cn(
          'min-h-0 w-full min-w-0 flex-1 flex-col overflow-auto lg:flex lg:overflow-hidden',
          mobileDetailOpen ? 'flex' : 'hidden',
        )}
      >
        <div className="shrink-0 p-4 pb-0 lg:hidden">
          <button
            type="button"
            className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700"
            onClick={() => selectProfile(null, { replace: true })}
          >
            <ChevronLeft className="h-4 w-4" />
            Back to agents
          </button>
        </div>
        <ProfileEditor
          profileName={effectiveProfileName}
          profiles={profiles.data ?? []}
        />
      </section>
    </div>
  )
}

function ProfileListItem({
  profile,
  active,
  onClick,
}: {
  profile: ProfileSummary
  active: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      className={cn(
        'w-full rounded-2xl border p-3 text-left transition',
        active
          ? 'border-blue-200 bg-blue-50 shadow-sm'
          : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50',
      )}
      onClick={onClick}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-slate-900">
            {profile.name}
          </p>
          <p className="mt-1 truncate mono text-xs text-slate-500">
            {profile.model}
          </p>
        </div>
        <StatusBadge status={profile.enabled ? 'enabled' : 'disabled'} />
      </div>
      <div className="mt-3 flex items-center justify-between text-xs text-slate-500">
        <span>{profile.workspace_backend_hint ?? 'workspace auto'}</span>
        <span>{profile.source_type ?? 'manual'}</span>
      </div>
    </button>
  )
}

function SeedPanel() {
  const seed = useSeedProfilesMutation()
  const [pruneMissing, setPruneMissing] = useState(false)
  const trigger = (
    <button
      type="button"
      className="mt-3 inline-flex w-full items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm hover:bg-slate-50 disabled:opacity-60"
      disabled={seed.isPending}
      onClick={pruneMissing ? undefined : () => void seed.mutateAsync(false)}
    >
      <DatabaseZap className="h-4 w-4" />
      {seed.isPending ? 'Seeding profiles…' : 'Seed profiles'}
    </button>
  )
  return (
    <div className="border-t border-slate-200 p-4">
      <label className="flex items-center justify-between gap-3 text-xs font-medium text-slate-600">
        Prune missing seeded profiles
        <input
          type="checkbox"
          checked={pruneMissing}
          onChange={(event) => setPruneMissing(event.target.checked)}
        />
      </label>
      {pruneMissing ? (
        <ConfirmDialog
          title="Seed profiles and prune missing profiles?"
          description="This removes seeded profiles that no longer exist in the version 2 seed document."
          confirmLabel="Seed and prune"
          danger
          pending={seed.isPending}
          onConfirm={async () => {
            await seed.mutateAsync(true)
          }}
          trigger={trigger}
        />
      ) : (
        trigger
      )}
    </div>
  )
}

function ProfileEditor({
  profileName,
  profiles,
}: {
  profileName: string | null
  profiles: ProfileSummary[]
}) {
  const isNew = profileName === '__new__'
  const profile = useProfileQuery(profileName && !isNew ? profileName : null)
  const selectProfile = useLayoutStore((state) => state.selectProfile)
  const upsert = useUpsertProfileMutation(
    profileName && !isNew ? profileName : null,
  )
  const remove = useDeleteProfileMutation()
  const form = useForm<ProfileFormValues>({
    defaultValues: blankProfile,
    mode: 'onBlur',
  })
  const loadedProfileRef = useRef<string | null>(null)
  const allowedSavedPathRef = useRef<string | null>(null)
  const [previewOpen, setPreviewOpen] = useState(false)
  const [remoteUpdateAvailable, setRemoteUpdateAvailable] = useState(false)
  const [pendingRemote, setPendingRemote] = useState<ProfileDetail | null>(null)
  const isDirty = form.formState.isDirty
  const blocker = useBlocker({
    shouldBlockFn: ({ current, next }) => {
      if (
        current.pathname === '/agents/new' &&
        next.pathname === allowedSavedPathRef.current
      ) {
        allowedSavedPathRef.current = null
        return false
      }
      return isDirty
    },
    disabled: !isDirty,
    enableBeforeUnload: isDirty,
    withResolver: true,
  })

  useEffect(() => {
    const key = isNew ? '__new__' : profileName
    if (!key) return
    if (isNew) {
      if (loadedProfileRef.current !== key) {
        form.reset(blankProfile)
        loadedProfileRef.current = key
      }
      return
    }
    if (!profile.data || profile.data.name !== profileName) return
    if (loadedProfileRef.current === key && isDirty) {
      setPendingRemote(profile.data)
      setRemoteUpdateAvailable(true)
      return
    }
    form.reset(formValuesFromProfile(profile.data))
    loadedProfileRef.current = key
    setPendingRemote(null)
    setRemoteUpdateAvailable(false)
  }, [form, isDirty, isNew, profile.data, profileName])

  const values = form.watch()
  const preview = useMemo(() => {
    try {
      return payloadFromForm(values)
    } catch (error) {
      return { error: error instanceof Error ? error.message : String(error) }
    }
  }, [values])

  if (!profileName) {
    return (
      <div className="h-full p-6">
        <EmptyState
          title="Select a profile"
          headingLevel={2}
          description="Create or select a native AgentSpec profile."
        />
      </div>
    )
  }
  if (!isNew && profile.isError) {
    return (
      <div className="h-full p-6">
        <QueryError
          title="Could not load this agent"
          error={profile.error}
          onRetry={() => void profile.refetch()}
        />
      </div>
    )
  }
  if (!isNew && profile.data?.name !== profileName) {
    return <div className="h-full animate-pulse bg-slate-50" role="status" />
  }

  async function submit(values: ProfileFormValues) {
    const name = values.name.trim()
    try {
      const saved = await upsert.mutateAsync({
        name,
        payload: payloadFromForm(values),
      })
      form.reset(formValuesFromProfile(saved))
      loadedProfileRef.current = saved.name
      allowedSavedPathRef.current = buildProfilePath(saved.name)
      selectProfile(saved.name)
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : 'Failed to save profile',
      )
    }
  }

  async function deleteSelected() {
    if (!profileName || isNew) return
    const index = profiles.findIndex((item) => item.name === profileName)
    await remove.mutateAsync(profileName)
    form.reset(form.getValues())
    const next = profiles[index + 1] ?? profiles[index - 1] ?? null
    selectProfile(next?.name ?? null)
  }

  return (
    <form
      className="flex min-h-0 flex-1 flex-col lg:h-full"
      onSubmit={form.handleSubmit(submit)}
    >
      <fieldset className="contents" disabled={upsert.isPending}>
        <header className="border-b border-slate-200 bg-white px-4 py-4 sm:px-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-sm font-medium text-blue-600">
                Profile editor
              </p>
              <h2 className="mt-1 text-xl font-semibold text-slate-950">
                {isNew ? 'New native profile' : profileName}
              </h2>
              <p className="mt-1 text-xs text-slate-500">
                AgentSpec + Claw host policy + SubagentSpec list
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              {!isNew ? (
                <button
                  type="button"
                  className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700"
                  onClick={() => void profile.refetch()}
                >
                  <RefreshCcw className="h-4 w-4" /> Reload
                </button>
              ) : null}
              {!isNew ? (
                <ConfirmDialog
                  title={`Delete ${profileName}?`}
                  description="Sessions and automations referencing this profile may no longer start."
                  confirmLabel="Delete agent"
                  danger
                  pending={remove.isPending}
                  onConfirm={deleteSelected}
                  trigger={
                    <button
                      type="button"
                      className="inline-flex items-center gap-2 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-sm font-medium text-rose-700"
                    >
                      <Trash2 className="h-4 w-4" /> Delete
                    </button>
                  }
                />
              ) : null}
              <button
                type="submit"
                className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white disabled:bg-slate-300"
              >
                <Save className="h-4 w-4" /> Save agent
              </button>
            </div>
          </div>
        </header>

        {remoteUpdateAvailable ? (
          <div className="border-b border-amber-200 bg-amber-50 px-6 py-3 text-sm text-amber-900">
            A newer server version is available. Unsaved JSON remains intact.
            <button
              type="button"
              className="ml-3 font-semibold underline"
              onClick={() => {
                if (!pendingRemote) return
                form.reset(formValuesFromProfile(pendingRemote))
                setPendingRemote(null)
                setRemoteUpdateAvailable(false)
              }}
            >
              Discard edits and load it
            </button>
          </div>
        ) : null}

        <ConfirmDialog
          open={blocker.status === 'blocked'}
          onOpenChange={(open) => {
            if (!open && blocker.status === 'blocked') blocker.reset()
          }}
          title="Discard unsaved agent changes?"
          description="Your AgentSpec, host policy, and subagent edits have not been saved."
          confirmLabel="Discard and leave"
          danger
          onConfirm={() => {
            if (blocker.status === 'blocked') blocker.proceed()
          }}
        />

        <div className="scrollbar-thin min-h-0 flex-1 overflow-auto p-4 lg:p-6">
          <div className="grid grid-cols-1 gap-6 2xl:grid-cols-[minmax(0,1fr)_360px]">
            <div className="space-y-6">
              <Section
                title="Profile identity"
                description="Stable path identity and source provenance."
                icon={SlidersHorizontal}
                defaultOpen
              >
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <TextField
                    label="Name"
                    registration={form.register('name', {
                      validate: (value) =>
                        Boolean(value.trim()) || 'Profile name is required',
                    })}
                    error={form.formState.errors.name?.message}
                    disabled={!isNew}
                  />
                  <label className="flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-700">
                    Enabled
                    <input type="checkbox" {...form.register('enabled')} />
                  </label>
                  <TextField
                    label="Source type"
                    registration={form.register('source_type')}
                  />
                  <TextField
                    label="Source version"
                    registration={form.register('source_version')}
                  />
                  <TextField
                    label="Source checksum"
                    registration={form.register('source_checksum')}
                  />
                </div>
              </Section>

              <Section
                title="Native AgentSpec"
                description="Model, settings, instructions, output schema, metadata, and exact feature capabilities."
                icon={Bot}
                defaultOpen
              >
                <JsonField
                  label="AgentSpec JSON"
                  registration={form.register('agent')}
                />
              </Section>

              <Section
                title="Claw host policy"
                description="Workspace hint, Claw control-plane groups, approvals, MCPs, and model context policy."
                icon={DatabaseZap}
                defaultOpen
              >
                <JsonField
                  label="Host policy JSON"
                  registration={form.register('host')}
                />
              </Section>

              <Section
                title="Portable subagents"
                description="A JSON list of versioned SubagentSpec documents with nested native AgentSpec definitions."
                icon={Bot}
                defaultOpen
              >
                <JsonField
                  label="SubagentSpec list JSON"
                  registration={form.register('subagents')}
                  tall
                />
              </Section>
            </div>

            <aside className="space-y-4">
              <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                <button
                  type="button"
                  className="flex w-full items-center justify-between text-left"
                  onClick={() => setPreviewOpen((current) => !current)}
                >
                  <span className="text-sm font-semibold text-slate-900">
                    Version 2 payload
                  </span>
                  <ChevronDown
                    className={cn(
                      'h-4 w-4 text-slate-400 transition',
                      previewOpen && 'rotate-180',
                    )}
                  />
                </button>
                {previewOpen ? (
                  <div className="mt-4">
                    <JsonView value={preview} height="600px" />
                  </div>
                ) : null}
              </div>
              {profile.data ? (
                <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                  <p className="text-sm font-semibold text-slate-900">
                    Stored profile
                  </p>
                  <div className="mt-4">
                    <JsonView value={profile.data} height="600px" />
                  </div>
                </div>
              ) : null}
            </aside>
          </div>
        </div>
      </fieldset>
    </form>
  )
}

function Section({
  title,
  description,
  icon: Icon,
  children,
  defaultOpen = false,
}: {
  title: string
  description: string
  icon: typeof Bot
  children: React.ReactNode
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
      <button
        type="button"
        className="flex w-full items-center gap-3 p-5 text-left hover:bg-slate-50"
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
      >
        <span className="inline-flex h-8 w-8 items-center justify-center rounded-xl bg-blue-50 text-blue-600">
          <Icon className="h-4 w-4" />
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-sm font-semibold text-slate-900">
            {title}
          </span>
          <span className="mt-1 block text-xs leading-5 text-slate-500">
            {description}
          </span>
        </span>
        <ChevronDown
          className={cn(
            'h-4 w-4 text-slate-400 transition',
            open && 'rotate-180',
          )}
        />
      </button>
      {open ? (
        <div className="border-t border-slate-100 p-5">{children}</div>
      ) : null}
    </section>
  )
}

function TextField({
  label,
  registration,
  error,
  disabled,
}: {
  label: string
  registration: UseFormRegisterReturn
  error?: string
  disabled?: boolean
}) {
  return (
    <label className="block min-w-0">
      <span className="text-sm font-medium text-slate-700">{label}</span>
      <input
        className={cn(
          'mt-2 w-full rounded-xl border bg-slate-50 px-3 py-2 text-sm outline-none ring-blue-600 focus:bg-white focus:ring-2',
          error ? 'border-rose-400' : 'border-slate-200',
        )}
        disabled={disabled}
        aria-label={label}
        {...registration}
      />
      {error ? (
        <span className="mt-1 block text-xs text-rose-600">{error}</span>
      ) : null}
    </label>
  )
}

function JsonField({
  label,
  registration,
  tall = false,
}: {
  label: string
  registration: UseFormRegisterReturn
  tall?: boolean
}) {
  return (
    <label className="block min-w-0">
      <span className="text-sm font-medium text-slate-700">{label}</span>
      <textarea
        className={cn(
          'mt-2 w-full rounded-xl border border-slate-200 bg-slate-50 p-3 mono text-xs leading-5 text-slate-900 outline-none ring-blue-600 focus:bg-white focus:ring-2',
          tall ? 'min-h-96' : 'min-h-72',
        )}
        spellCheck={false}
        {...registration}
      />
    </label>
  )
}

function formValuesFromProfile(profile: ProfileDetail): ProfileFormValues {
  return {
    name: profile.name,
    enabled: profile.enabled,
    source_type: profile.source_type ?? '',
    source_version: profile.source_version ?? '',
    source_checksum: profile.source_checksum ?? '',
    agent: safeJsonStringify(profile.agent),
    host: safeJsonStringify(profile.host),
    subagents: safeJsonStringify(profile.subagents),
  }
}

function payloadFromForm(values: ProfileFormValues): ProfileUpsertRequest {
  const name = values.name.trim()
  const agent = parseJsonObject(values.agent) as NativeAgentSpec | null
  if (!agent) throw new Error('AgentSpec JSON is required.')
  agent.name = name
  const host = parseJsonObject(values.host) as ClawProfileHostConfig | null
  if (!host) throw new Error('Claw host policy JSON is required.')
  const subagents = parseSubagentSpecs(values.subagents)
  return {
    schema_version: 2,
    agent,
    host,
    subagents,
    enabled: values.enabled,
    source_type: nullableText(values.source_type),
    source_version: nullableText(values.source_version),
    source_checksum: nullableText(values.source_checksum),
  }
}

function parseSubagentSpecs(value: string): NativeSubagentSpec[] {
  const normalized = value.trim()
  if (!normalized) return []
  const parsed: unknown = JSON.parse(normalized)
  if (!Array.isArray(parsed)) {
    throw new Error('SubagentSpec JSON must be an array.')
  }
  if (
    parsed.some(
      (item) => !item || typeof item !== 'object' || Array.isArray(item),
    )
  ) {
    throw new Error('Every SubagentSpec entry must be an object.')
  }
  return parsed as NativeSubagentSpec[]
}

function nullableText(value: string): string | null {
  const normalized = value.trim()
  return normalized || null
}
