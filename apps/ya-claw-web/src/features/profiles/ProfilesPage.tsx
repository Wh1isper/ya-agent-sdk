import {
  Bot,
  CopyPlus,
  DatabaseZap,
  RefreshCcw,
  Save,
  Search,
  SlidersHorizontal,
  Trash2,
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import {
  Controller,
  type UseFormRegisterReturn,
  useFieldArray,
  useForm,
} from 'react-hook-form'
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
import {
  cn,
  joinCsv,
  parseJsonObject,
  safeJsonStringify,
  splitCsv,
} from '../../lib/utils'
import { useLayoutStore } from '../../stores/layoutStore'
import type {
  ProfileDetail,
  ProfileMCPServer,
  ProfileShellReviewConfig,
  ProfileSummary,
  ProfileUpsertRequest,
} from '../../types'

type ProfileFormSubagent = {
  name: string
  description: string
  system_prompt: string
  model: string
  model_settings_preset: string
  model_settings_override: string
  model_config_preset: string
  model_config_override: string
}

type ProfileFormValues = {
  name: string
  model: string
  enabled: boolean
  workspace_backend_hint: string
  source_type: string
  source_version: string
  source_checksum: string
  system_prompt: string
  builtin_toolsets: string
  include_builtin_subagents: boolean
  unified_subagents: boolean
  need_user_approve_tools: string
  need_user_approve_mcps: string
  enabled_mcps: string
  disabled_mcps: string
  mcp_servers: string
  model_settings_preset: string
  model_settings_override: string
  model_config_preset: string
  model_config_override: string
  shell_review_enabled: boolean
  shell_review_model: string
  shell_review_model_settings: string
  shell_review_on_needs_approval: 'deny' | 'defer'
  shell_review_risk_threshold: 'low' | 'medium' | 'high' | 'extra_high'
  shell_review_system_prompt: string
  subagents: ProfileFormSubagent[]
}

const blankProfile: ProfileFormValues = {
  name: '',
  model: 'openai:gpt-4.1-mini',
  enabled: true,
  workspace_backend_hint: '',
  source_type: 'web',
  source_version: '',
  source_checksum: '',
  system_prompt: '',
  builtin_toolsets: 'session',
  include_builtin_subagents: true,
  unified_subagents: true,
  need_user_approve_tools: '',
  need_user_approve_mcps: '',
  enabled_mcps: '',
  disabled_mcps: '',
  mcp_servers: '',
  model_settings_preset: '',
  model_settings_override: '',
  model_config_preset: '',
  model_config_override: '',
  shell_review_enabled: false,
  shell_review_model: '',
  shell_review_model_settings: 'openai_responses_low',
  shell_review_on_needs_approval: 'deny',
  shell_review_risk_threshold: 'extra_high',
  shell_review_system_prompt: '',
  subagents: [],
}

export function ProfilesPage() {
  const profiles = useProfilesQuery()
  const selectedProfileName = useLayoutStore(
    (state) => state.selectedProfileName,
  )
  const selectProfile = useLayoutStore((state) => state.selectProfile)
  const [search, setSearch] = useState('')

  useEffect(() => {
    if (!selectedProfileName && profiles.data?.[0]) {
      selectProfile(profiles.data[0].name)
    }
  }, [profiles.data, selectProfile, selectedProfileName])

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
    <div className="flex h-full min-h-0 bg-slate-100">
      <aside className="flex w-80 shrink-0 flex-col border-r border-slate-200 bg-white">
        <div className="border-b border-slate-200 p-4">
          <div className="flex items-center justify-between gap-2">
            <div>
              <p className="text-sm font-medium text-blue-600">AgentProfile</p>
              <h1 className="mt-1 text-xl font-semibold tracking-tight text-slate-950">
                Profiles
              </h1>
            </div>
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-3 py-2 text-xs font-semibold text-white shadow-sm transition hover:bg-blue-700"
              onClick={() => selectProfile('__new__')}
            >
              <CopyPlus className="h-3.5 w-3.5" />
              New
            </button>
          </div>
          <div className="relative mt-4">
            <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
            <input
              className="w-full rounded-xl border border-slate-200 bg-slate-50 py-2 pl-9 pr-3 text-sm outline-none ring-blue-600 transition focus:bg-white focus:ring-2"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search profiles"
            />
          </div>
        </div>
        <div className="scrollbar-thin min-h-0 flex-1 overflow-auto p-3">
          {profiles.isLoading ? <ProfileListSkeleton /> : null}
          {!profiles.isLoading && filteredProfiles.length === 0 ? (
            <EmptyState
              title="No profiles"
              description="Seed defaults or create a profile."
            />
          ) : null}
          <div className="space-y-2">
            {filteredProfiles.map((profile) => (
              <ProfileListItem
                key={profile.name}
                profile={profile}
                active={selectedProfileName === profile.name}
                onClick={() => selectProfile(profile.name)}
              />
            ))}
          </div>
        </div>
        <SeedPanel />
      </aside>
      <main className="min-w-0 flex-1 overflow-hidden">
        <ProfileEditor
          profileName={selectedProfileName}
          profiles={profiles.data ?? []}
        />
      </main>
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

function ProfileListSkeleton() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 5 }).map((_, index) => (
        <div
          key={index}
          className="rounded-2xl border border-slate-200 bg-white p-3"
        >
          <div className="h-4 w-28 animate-pulse rounded bg-slate-100" />
          <div className="mt-3 h-3 w-full animate-pulse rounded bg-slate-100" />
          <div className="mt-3 h-3 w-20 animate-pulse rounded bg-slate-100" />
        </div>
      ))}
    </div>
  )
}

function SeedPanel() {
  const seed = useSeedProfilesMutation()
  const [pruneMissing, setPruneMissing] = useState(false)
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
      <button
        type="button"
        className="mt-3 inline-flex w-full items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:bg-slate-50 disabled:opacity-60"
        disabled={seed.isPending}
        onClick={() => seed.mutate(pruneMissing)}
      >
        <DatabaseZap className="h-4 w-4" />
        Seed profiles
      </button>
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
  const subagents = useFieldArray({ control: form.control, name: 'subagents' })
  const [previewOpen, setPreviewOpen] = useState(false)
  const [expandedSubagents, setExpandedSubagents] = useState<
    Record<number, boolean>
  >({})

  useEffect(() => {
    if (isNew) {
      form.reset(blankProfile)
      return
    }
    if (profile.data) {
      form.reset(formValuesFromProfile(profile.data))
    }
  }, [form, isNew, profile.data])

  async function submit(values: ProfileFormValues) {
    const profileNameValue = values.name.trim()
    if (!profileNameValue) {
      toast.error('Profile name is required')
      return
    }
    if (!values.model.trim()) {
      toast.error('Model is required')
      return
    }
    try {
      const payload = payloadFromForm(values)
      const saved = await upsert.mutateAsync({
        name: profileNameValue,
        payload,
      })
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
    const next = profiles[index + 1] ?? profiles[index - 1] ?? null
    selectProfile(next?.name ?? null)
  }

  const values = form.watch()
  const payloadPreview = useMemo(() => {
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
          description="Create, edit, seed, and inspect AgentProfiles from this workspace."
        />
      </div>
    )
  }

  const saveProfile = form.handleSubmit(submit)

  return (
    <form className="flex h-full min-h-0 flex-col" onSubmit={saveProfile}>
      <div className="border-b border-slate-200 bg-white px-6 py-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-sm font-medium text-blue-600">Profile editor</p>
            <h2 className="mt-1 text-xl font-semibold tracking-tight text-slate-950">
              {isNew ? 'New profile' : profileName}
            </h2>
            {profile.data ? (
              <p className="mt-1 text-xs text-slate-500">
                Updated {profile.data.updated_at}
              </p>
            ) : null}
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:bg-slate-50"
              onClick={() => profile.refetch()}
            >
              <RefreshCcw className="h-4 w-4" />
              Reload
            </button>
            {!isNew ? (
              <button
                type="button"
                className="inline-flex items-center gap-2 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-sm font-medium text-rose-700 transition hover:bg-rose-100"
                onClick={() => void deleteSelected()}
                disabled={remove.isPending}
              >
                <Trash2 className="h-4 w-4" />
                Delete
              </button>
            ) : null}
            <button
              type="button"
              className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 disabled:bg-slate-300"
              disabled={upsert.isPending}
              onClick={() => void saveProfile()}
            >
              <Save className="h-4 w-4" />
              Save profile
            </button>
          </div>
        </div>
      </div>

      <div className="scrollbar-thin min-h-0 flex-1 overflow-auto p-6">
        {profile.isLoading && !isNew ? (
          <div className="h-40 animate-pulse rounded-2xl bg-white" />
        ) : null}
        <div className="grid min-w-0 grid-cols-1 gap-6 2xl:grid-cols-[minmax(0,1fr)_360px]">
          <div className="space-y-6">
            <Section title="Basic" icon={SlidersHorizontal}>
              <div className="grid grid-cols-2 gap-4">
                <TextField
                  label="Name"
                  registration={form.register('name')}
                  error={form.formState.errors.name?.message}
                  disabled={!isNew}
                />
                <TextField
                  label="Model"
                  registration={form.register('model')}
                  error={form.formState.errors.model?.message}
                />
                <TextField
                  label="Workspace backend hint"
                  registration={form.register('workspace_backend_hint')}
                  placeholder="local or docker"
                />
                <TextField
                  label="Source type"
                  registration={form.register('source_type')}
                  placeholder="web, seed, manual"
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
              <div className="mt-4 grid grid-cols-3 gap-3">
                <SwitchField
                  label="Enabled"
                  control={
                    <Controller
                      control={form.control}
                      name="enabled"
                      render={({ field }) => (
                        <input
                          type="checkbox"
                          checked={field.value}
                          onChange={field.onChange}
                        />
                      )}
                    />
                  }
                />
                <SwitchField
                  label="Builtin subagents"
                  control={
                    <Controller
                      control={form.control}
                      name="include_builtin_subagents"
                      render={({ field }) => (
                        <input
                          type="checkbox"
                          checked={field.value}
                          onChange={field.onChange}
                        />
                      )}
                    />
                  }
                />
                <SwitchField
                  label="Unified subagents"
                  control={
                    <Controller
                      control={form.control}
                      name="unified_subagents"
                      render={({ field }) => (
                        <input
                          type="checkbox"
                          checked={field.value}
                          onChange={field.onChange}
                        />
                      )}
                    />
                  }
                />
              </div>
            </Section>

            <Section title="Prompt" icon={Bot}>
              <textarea
                className="min-h-56 w-full rounded-xl border border-slate-200 bg-slate-50 p-3 text-sm leading-6 text-slate-900 outline-none ring-blue-600 transition focus:bg-white focus:ring-2"
                {...form.register('system_prompt')}
                placeholder="System prompt"
              />
            </Section>

            <Section title="Tools and MCP" icon={DatabaseZap}>
              <div className="grid grid-cols-2 gap-4">
                <TextField
                  label="Builtin toolsets"
                  registration={form.register('builtin_toolsets')}
                  placeholder="core, web, multimodal, document"
                  helper="Comma-separated"
                />
                <TextField
                  label="Tools requiring approval"
                  registration={form.register('need_user_approve_tools')}
                  helper="Comma-separated"
                />
                <TextField
                  label="Enabled MCPs"
                  registration={form.register('enabled_mcps')}
                  helper="Comma-separated"
                />
                <TextField
                  label="Disabled MCPs"
                  registration={form.register('disabled_mcps')}
                  helper="Comma-separated"
                />
                <TextField
                  label="MCPs requiring approval"
                  registration={form.register('need_user_approve_mcps')}
                  helper="Comma-separated"
                />
              </div>
              <div className="mt-4">
                <JsonField
                  label="MCP servers"
                  registration={form.register('mcp_servers')}
                />
                <p className="mt-1 text-xs text-slate-400">
                  JSON object keyed by namespace. Use streamable_http transport.
                </p>
              </div>
            </Section>

            <Section title="Model advanced" icon={SlidersHorizontal}>
              <div className="grid grid-cols-2 gap-4">
                <TextField
                  label="Model settings preset"
                  registration={form.register('model_settings_preset')}
                />
                <TextField
                  label="Model config preset"
                  registration={form.register('model_config_preset')}
                />
                <JsonField
                  label="Model settings override"
                  registration={form.register('model_settings_override')}
                />
                <JsonField
                  label="Model config override"
                  registration={form.register('model_config_override')}
                />
              </div>
            </Section>

            <Section title="Shell review" icon={SlidersHorizontal}>
              <div className="grid grid-cols-2 gap-4">
                <SwitchField
                  label="Enable shell review"
                  control={
                    <Controller
                      control={form.control}
                      name="shell_review_enabled"
                      render={({ field }) => (
                        <input
                          type="checkbox"
                          checked={field.value}
                          onChange={field.onChange}
                        />
                      )}
                    />
                  }
                />
                <TextField
                  label="Reviewer model"
                  registration={form.register('shell_review_model')}
                  placeholder="gateway@openai-responses:gpt-5.4-mini"
                />
                <TextField
                  label="Reviewer model settings"
                  registration={form.register('shell_review_model_settings')}
                  helper="Preset name or JSON object"
                />
                <SelectField
                  label="Needs approval action"
                  registration={form.register('shell_review_on_needs_approval')}
                  helper="Claw coerces defer to deny at runtime"
                  options={[
                    { value: 'deny', label: 'Deny' },
                    { value: 'defer', label: 'Defer' },
                  ]}
                />
                <SelectField
                  label="Risk threshold"
                  registration={form.register('shell_review_risk_threshold')}
                  options={[
                    { value: 'low', label: 'Low' },
                    { value: 'medium', label: 'Medium' },
                    { value: 'high', label: 'High' },
                    { value: 'extra_high', label: 'Extra high' },
                  ]}
                />
                <div className="col-span-2">
                  <label className="block min-w-0">
                    <span className="text-sm font-medium text-slate-700">
                      Reviewer system prompt override
                    </span>
                    <textarea
                      className="mt-2 min-h-28 w-full rounded-xl border border-slate-200 bg-slate-50 p-3 text-sm leading-6 text-slate-900 outline-none ring-blue-600 transition focus:bg-white focus:ring-2"
                      {...form.register('shell_review_system_prompt')}
                      placeholder="Optional"
                    />
                  </label>
                </div>
              </div>
            </Section>

            <Section title="Subagents" icon={Bot}>
              <div className="space-y-4">
                {subagents.fields.map((field, index) => {
                  const expanded = expandedSubagents[index] ?? false
                  const subagent = values.subagents[index]
                  return (
                    <div
                      key={field.id}
                      className="rounded-2xl border border-slate-200 bg-slate-50 p-4"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <button
                          type="button"
                          className="min-w-0 flex-1 text-left"
                          onClick={() =>
                            setExpandedSubagents((current) => ({
                              ...current,
                              [index]: !expanded,
                            }))
                          }
                        >
                          <p className="truncate text-sm font-semibold text-slate-900">
                            {subagent?.name || `Subagent #${index + 1}`}
                          </p>
                          <p className="mt-1 truncate text-xs text-slate-500">
                            {subagent?.description ||
                              subagent?.model ||
                              'Click to edit details'}
                          </p>
                        </button>
                        <div className="flex shrink-0 items-center gap-2">
                          <button
                            type="button"
                            className="rounded-lg border border-slate-200 bg-white px-2 py-1 text-xs font-medium text-slate-600"
                            onClick={() =>
                              setExpandedSubagents((current) => ({
                                ...current,
                                [index]: !expanded,
                              }))
                            }
                          >
                            {expanded ? 'Collapse' : 'Edit'}
                          </button>
                          <button
                            type="button"
                            className="rounded-lg border border-rose-200 bg-rose-50 px-2 py-1 text-xs font-medium text-rose-700"
                            onClick={() => subagents.remove(index)}
                          >
                            Remove
                          </button>
                        </div>
                      </div>
                      {expanded ? (
                        <div className="mt-4 grid grid-cols-2 gap-4">
                          <TextField
                            label="Name"
                            registration={form.register(
                              `subagents.${index}.name`,
                            )}
                          />
                          <TextField
                            label="Model"
                            registration={form.register(
                              `subagents.${index}.model`,
                            )}
                          />
                          <TextField
                            label="Settings preset"
                            registration={form.register(
                              `subagents.${index}.model_settings_preset`,
                            )}
                          />
                          <TextField
                            label="Config preset"
                            registration={form.register(
                              `subagents.${index}.model_config_preset`,
                            )}
                          />
                          <div className="col-span-2">
                            <TextField
                              label="Description"
                              registration={form.register(
                                `subagents.${index}.description`,
                              )}
                            />
                          </div>
                          <div className="col-span-2">
                            <label className="block min-w-0">
                              <span className="text-sm font-medium text-slate-700">
                                System prompt
                              </span>
                              <textarea
                                className="mt-2 min-h-32 w-full rounded-xl border border-slate-200 bg-white p-3 text-sm leading-6 outline-none ring-blue-600 focus:ring-2"
                                {...form.register(
                                  `subagents.${index}.system_prompt`,
                                )}
                              />
                            </label>
                          </div>
                        </div>
                      ) : null}
                    </div>
                  )
                })}
                <button
                  type="button"
                  className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:bg-slate-50"
                  onClick={() =>
                    subagents.append({
                      name: '',
                      description: '',
                      system_prompt: '',
                      model: '',
                      model_settings_preset: '',
                      model_settings_override: '',
                      model_config_preset: '',
                      model_config_override: '',
                    })
                  }
                >
                  <CopyPlus className="h-4 w-4" />
                  Add subagent
                </button>
              </div>
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
                  Payload preview
                </span>
                <ChevronIcon open={previewOpen} />
              </button>
              {previewOpen ? (
                <div className="mt-4">
                  <JsonView value={payloadPreview} height="520px" />
                </div>
              ) : null}
            </div>
            {profile.data ? (
              <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                <p className="text-sm font-semibold text-slate-900">
                  Stored profile
                </p>
                <div className="mt-4">
                  <JsonView value={profile.data} height="520px" />
                </div>
              </div>
            ) : null}
          </aside>
        </div>
      </div>
    </form>
  )
}

function Section({
  title,
  icon: Icon,
  children,
}: {
  title: string
  icon: typeof Bot
  children: React.ReactNode
}) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-center gap-2">
        <span className="inline-flex h-8 w-8 items-center justify-center rounded-xl bg-blue-50 text-blue-600">
          <Icon className="h-4 w-4" />
        </span>
        <h3 className="text-sm font-semibold text-slate-900">{title}</h3>
      </div>
      {children}
    </section>
  )
}

function TextField({
  label,
  registration,
  error,
  helper,
  placeholder,
  disabled,
}: {
  label: string
  registration: UseFormRegisterReturn
  error?: string
  helper?: string
  placeholder?: string
  disabled?: boolean
}) {
  return (
    <label className="block min-w-0">
      <span className="text-sm font-medium text-slate-700">{label}</span>
      <input
        className="mt-2 w-full min-w-0 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900 outline-none ring-blue-600 transition focus:bg-white focus:ring-2 disabled:text-slate-400"
        placeholder={placeholder}
        disabled={disabled}
        {...registration}
      />
      {helper ? (
        <span className="mt-1 block text-xs text-slate-400">{helper}</span>
      ) : null}
      {error ? (
        <span className="mt-1 block text-xs text-rose-600">{error}</span>
      ) : null}
    </label>
  )
}

function JsonField({
  label,
  registration,
}: {
  label: string
  registration: UseFormRegisterReturn
}) {
  return (
    <label className="block min-w-0">
      <span className="text-sm font-medium text-slate-700">{label}</span>
      <textarea
        className="mt-2 min-h-36 w-full min-w-0 rounded-xl border border-slate-200 bg-slate-50 p-3 mono text-xs leading-5 text-slate-900 outline-none ring-blue-600 transition focus:bg-white focus:ring-2"
        placeholder="{}"
        {...registration}
      />
    </label>
  )
}

function SelectField({
  label,
  registration,
  options,
  helper,
}: {
  label: string
  registration: UseFormRegisterReturn
  options: Array<{ value: string; label: string }>
  helper?: string
}) {
  return (
    <label className="block min-w-0">
      <span className="text-sm font-medium text-slate-700">{label}</span>
      <select
        className="mt-2 w-full min-w-0 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900 outline-none ring-blue-600 transition focus:bg-white focus:ring-2"
        {...registration}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      {helper ? (
        <span className="mt-1 block text-xs text-slate-400">{helper}</span>
      ) : null}
    </label>
  )
}

function SwitchField({
  label,
  control,
}: {
  label: string
  control: React.ReactNode
}) {
  return (
    <label className="flex items-center justify-between gap-3 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-700">
      {label}
      {control}
    </label>
  )
}

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <span className={cn('text-slate-400 transition', open && 'rotate-180')}>
      ⌄
    </span>
  )
}

function formValuesFromProfile(profile: ProfileDetail): ProfileFormValues {
  return {
    name: profile.name,
    model: profile.model,
    enabled: profile.enabled,
    workspace_backend_hint: profile.workspace_backend_hint ?? '',
    source_type: profile.source_type ?? '',
    source_version: profile.source_version ?? '',
    source_checksum: profile.source_checksum ?? '',
    system_prompt: profile.system_prompt ?? '',
    builtin_toolsets: joinCsv(
      profile.builtin_toolsets.length
        ? profile.builtin_toolsets
        : profile.toolsets,
    ),
    include_builtin_subagents: profile.include_builtin_subagents,
    unified_subagents: profile.unified_subagents,
    need_user_approve_tools: joinCsv(profile.need_user_approve_tools),
    need_user_approve_mcps: joinCsv(profile.need_user_approve_mcps),
    enabled_mcps: joinCsv(profile.enabled_mcps),
    disabled_mcps: joinCsv(profile.disabled_mcps),
    mcp_servers: Object.keys(profile.mcp_servers).length
      ? safeJsonStringify(profile.mcp_servers)
      : '',
    model_settings_preset: profile.model_settings_preset ?? '',
    model_settings_override: profile.model_settings_override
      ? safeJsonStringify(profile.model_settings_override)
      : '',
    model_config_preset: profile.model_config_preset ?? '',
    model_config_override: modelConfigOverrideText(
      profile.model_config_override,
    ),
    ...shellReviewFormValues(profile.model_config_override),
    subagents: profile.subagents.map((subagent) => ({
      name: subagent.name,
      description: subagent.description,
      system_prompt: subagent.system_prompt,
      model: subagent.model ?? '',
      model_settings_preset: subagent.model_settings_preset ?? '',
      model_settings_override: subagent.model_settings_override
        ? safeJsonStringify(subagent.model_settings_override)
        : '',
      model_config_preset: subagent.model_config_preset ?? '',
      model_config_override: subagent.model_config_override
        ? safeJsonStringify(subagent.model_config_override)
        : '',
    })),
  }
}

function payloadFromForm(values: ProfileFormValues): ProfileUpsertRequest {
  return {
    model: values.model.trim(),
    model_settings_preset: nullableText(values.model_settings_preset),
    model_settings_override: parseJsonObject(values.model_settings_override),
    model_config_preset: nullableText(values.model_config_preset),
    model_config_override: buildModelConfigOverride(values),
    system_prompt: nullableText(values.system_prompt),
    builtin_toolsets: splitCsv(values.builtin_toolsets),
    subagents: values.subagents.map((subagent) => ({
      name: subagent.name.trim(),
      description: subagent.description,
      system_prompt: subagent.system_prompt,
      model: nullableText(subagent.model),
      model_settings_preset: nullableText(subagent.model_settings_preset),
      model_settings_override: parseJsonObject(
        subagent.model_settings_override,
      ),
      model_config_preset: nullableText(subagent.model_config_preset),
      model_config_override: parseJsonObject(subagent.model_config_override),
    })),
    include_builtin_subagents: values.include_builtin_subagents,
    unified_subagents: values.unified_subagents,
    need_user_approve_tools: splitCsv(values.need_user_approve_tools),
    need_user_approve_mcps: splitCsv(values.need_user_approve_mcps),
    enabled_mcps: splitCsv(values.enabled_mcps),
    disabled_mcps: splitCsv(values.disabled_mcps),
    mcp_servers: parseMcpServers(values.mcp_servers),
    workspace_backend_hint: nullableText(values.workspace_backend_hint),
    enabled: values.enabled,
    source_type: nullableText(values.source_type),
    source_version: nullableText(values.source_version),
    source_checksum: nullableText(values.source_checksum),
  }
}

function modelConfigOverrideText(
  value: Record<string, unknown> | null | undefined,
): string {
  const cleaned = withoutShellReviewConfig(value)
  return cleaned ? safeJsonStringify(cleaned) : ''
}

function shellReviewFormValues(
  value: Record<string, unknown> | null | undefined,
): Pick<
  ProfileFormValues,
  | 'shell_review_enabled'
  | 'shell_review_model'
  | 'shell_review_model_settings'
  | 'shell_review_on_needs_approval'
  | 'shell_review_risk_threshold'
  | 'shell_review_system_prompt'
> {
  const config = extractShellReviewConfig(value)
  return {
    shell_review_enabled: Boolean(config?.enabled),
    shell_review_model: typeof config?.model === 'string' ? config.model : '',
    shell_review_model_settings:
      typeof config?.model_settings === 'string'
        ? config.model_settings
        : config?.model_settings && typeof config.model_settings === 'object'
          ? safeJsonStringify(config.model_settings)
          : 'openai_responses_low',
    shell_review_on_needs_approval:
      config?.on_needs_approval === 'defer' ? 'defer' : 'deny',
    shell_review_risk_threshold: shellReviewRiskLevel(config?.risk_threshold),
    shell_review_system_prompt:
      typeof config?.system_prompt === 'string' ? config.system_prompt : '',
  }
}

function buildModelConfigOverride(
  values: ProfileFormValues,
): Record<string, unknown> | null {
  const override = withoutShellReviewConfig(
    parseJsonObject(values.model_config_override),
  )
  if (!values.shell_review_enabled) return override
  const shellReview: ProfileShellReviewConfig = {
    enabled: true,
    model: nullableText(values.shell_review_model),
    model_settings: parseShellReviewModelSettings(
      values.shell_review_model_settings,
    ),
    on_needs_approval: values.shell_review_on_needs_approval,
    risk_threshold: values.shell_review_risk_threshold,
    system_prompt: nullableText(values.shell_review_system_prompt),
  }
  return {
    ...(override ?? {}),
    security: {
      ...securityObject(override),
      shell_review: removeNullish(shellReview),
    },
  }
}

function parseShellReviewModelSettings(
  value: string,
): string | Record<string, unknown> | null {
  const normalized = value.trim()
  if (!normalized) return null
  if (!normalized.startsWith('{')) return normalized
  return parseJsonObject(normalized)
}

function extractShellReviewConfig(
  value: Record<string, unknown> | null | undefined,
): ProfileShellReviewConfig | null {
  const security = securityObject(value)
  const shellReview = security.shell_review
  if (
    !shellReview ||
    typeof shellReview !== 'object' ||
    Array.isArray(shellReview)
  ) {
    return null
  }
  return shellReview as ProfileShellReviewConfig
}

function withoutShellReviewConfig(
  value: Record<string, unknown> | null | undefined,
): Record<string, unknown> | null {
  if (!value) return null
  const next = { ...value }
  const security = securityObject(value)
  if ('shell_review' in security) {
    const nextSecurity = { ...security }
    delete nextSecurity.shell_review
    if (Object.keys(nextSecurity).length > 0) {
      next.security = nextSecurity
    } else {
      delete next.security
    }
  }
  return Object.keys(next).length > 0 ? next : null
}

function securityObject(
  value: Record<string, unknown> | null | undefined,
): Record<string, unknown> {
  const security = value?.security
  if (!security || typeof security !== 'object' || Array.isArray(security)) {
    return {}
  }
  return security as Record<string, unknown>
}

function shellReviewRiskLevel(
  value: unknown,
): 'low' | 'medium' | 'high' | 'extra_high' {
  if (
    value === 'low' ||
    value === 'medium' ||
    value === 'high' ||
    value === 'extra_high'
  ) {
    return value
  }
  return 'extra_high'
}

function removeNullish<T extends Record<string, unknown>>(value: T) {
  return Object.fromEntries(
    Object.entries(value).filter(([, item]) => item !== null && item !== ''),
  )
}

function parseMcpServers(value: string): Record<string, ProfileMCPServer> {
  const parsed = parseJsonObject(value) ?? {}
  const servers: Record<string, ProfileMCPServer> = {}
  for (const [name, rawServer] of Object.entries(parsed)) {
    if (
      !rawServer ||
      typeof rawServer !== 'object' ||
      Array.isArray(rawServer)
    ) {
      throw new Error(`MCP server ${name} must be a JSON object`)
    }
    const server = rawServer as Record<string, unknown>
    if (server.transport !== 'streamable_http') {
      throw new Error(`MCP server ${name} must use streamable_http transport`)
    }
    if (typeof server.url !== 'string' || !server.url.trim()) {
      throw new Error(`MCP server ${name} requires url`)
    }
    const headers = server.headers
    if (headers && (typeof headers !== 'object' || Array.isArray(headers))) {
      throw new Error(`MCP server ${name} headers must be an object`)
    }
    servers[name] = {
      transport: 'streamable_http',
      url: server.url.trim(),
      headers: (headers ?? {}) as Record<string, string>,
      description:
        typeof server.description === 'string' ? server.description : '',
      required: typeof server.required === 'boolean' ? server.required : true,
    }
  }
  return servers
}

function nullableText(value: string) {
  const normalized = value.trim()
  return normalized ? normalized : null
}
