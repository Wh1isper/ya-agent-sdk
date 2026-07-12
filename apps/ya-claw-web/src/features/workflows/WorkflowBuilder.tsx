import { AlertTriangle, CheckCircle2, Plus, Trash2 } from 'lucide-react'
import { useEffect, useMemo } from 'react'

import { cn } from '../../lib/utils'

type WorkflowNodeDraft = {
  id: string
  profile: string
  mode: 'isolate' | 'continue'
  needs: string[]
  prompt: string
}

type WorkflowTemplate = {
  id: string
  name: string
  description: string
  definition: Record<string, unknown>
}

const templates: WorkflowTemplate[] = [
  {
    id: 'single-step',
    name: 'Single step',
    description: 'One agent handles the request from start to finish.',
    definition: {
      schema: 'ya-claw.workflow.v1',
      policy: { max_concurrency: 1 },
      nodes: {
        work: {
          profile: 'Self',
          mode: 'isolate',
          prompt:
            'Complete {{ inputs.topic | default("the requested task") }}.',
        },
      },
      result: { from_node: 'work' },
    },
  },
  {
    id: 'research-pipeline',
    name: 'Research pipeline',
    description: 'Research in one step, then synthesize the result.',
    definition: {
      schema: 'ya-claw.workflow.v1',
      policy: { max_concurrency: 2 },
      nodes: {
        research: {
          profile: 'Self',
          mode: 'isolate',
          prompt: 'Research {{ inputs.topic }} and collect evidence.',
        },
        synthesize: {
          profile: 'Self',
          mode: 'continue',
          needs: ['research'],
          prompt:
            'Synthesize {{ nodes.research.output_text }} into a concise answer for {{ inputs.topic }}.',
        },
      },
      result: { from_node: 'synthesize' },
    },
  },
  {
    id: 'draft-review',
    name: 'Draft and review',
    description: 'Create a draft, review it, then produce a final revision.',
    definition: {
      schema: 'ya-claw.workflow.v1',
      policy: { max_concurrency: 1 },
      nodes: {
        draft: {
          profile: 'Self',
          mode: 'isolate',
          prompt: 'Draft a response for {{ inputs.topic }}.',
        },
        review: {
          profile: 'Self',
          mode: 'isolate',
          needs: ['draft'],
          prompt:
            'Review this draft and identify improvements: {{ nodes.draft.output_text }}',
        },
        finalize: {
          profile: 'Self',
          mode: 'continue',
          needs: ['draft', 'review'],
          prompt:
            'Revise {{ nodes.draft.output_text }} using {{ nodes.review.output_text }}.',
        },
      },
      result: { from_node: 'finalize' },
    },
  },
]

export function WorkflowBuilder({
  value,
  onChange,
  onValidationChange,
}: {
  value: string
  onChange: (value: string) => void
  onValidationChange?: (issues: string[]) => void
}) {
  const parsed = useMemo(() => parseWorkflowDefinition(value), [value])
  const nodes = parsed.nodes

  useEffect(() => {
    onValidationChange?.(parsed.issues)
  }, [onValidationChange, parsed.issues])

  function updateNodes(
    nextNodes: WorkflowNodeDraft[],
    renamedNode?: { from: string; to: string },
  ) {
    const definition = parsed.definition ?? {
      schema: 'ya-claw.workflow.v1',
    }
    const rawNodes = isRecord(definition.nodes) ? definition.nodes : {}
    const nextDefinition: Record<string, unknown> = {
      ...definition,
      schema: 'ya-claw.workflow.v1',
      nodes: Object.fromEntries(
        nextNodes.map((node) => {
          const sourceId =
            renamedNode?.to === node.id ? renamedNode.from : node.id
          const existingNode = isRecord(rawNodes[sourceId])
            ? rawNodes[sourceId]
            : {}
          const nextNode: Record<string, unknown> = {
            ...existingNode,
            profile: node.profile || 'Self',
            mode: node.mode,
            prompt: node.prompt,
          }
          if (node.needs.length) nextNode.needs = node.needs
          else delete nextNode.needs
          return [node.id, nextNode]
        }),
      ),
    }
    const existingResult = isRecord(definition.result) ? definition.result : {}
    const currentResultNode = existingResult.from_node
    const renamedResultNode =
      renamedNode && currentResultNode === renamedNode.from
        ? renamedNode.to
        : currentResultNode
    const resultNode =
      typeof renamedResultNode === 'string' &&
      nextNodes.some((node) => node.id === renamedResultNode)
        ? renamedResultNode
        : (nextNodes[nextNodes.length - 1]?.id ?? '')
    nextDefinition.result = {
      ...existingResult,
      ...(resultNode ? { from_node: resultNode } : {}),
    }
    if (!resultNode)
      delete (nextDefinition.result as Record<string, unknown>).from_node
    onChange(JSON.stringify(nextDefinition, null, 2))
  }

  function updateNode(index: number, patch: Partial<WorkflowNodeDraft>) {
    const current = nodes[index]
    if (!current) return
    const next = nodes.map((node, nodeIndex) =>
      nodeIndex === index ? { ...node, ...patch } : node,
    )
    if (patch.id && patch.id !== current.id) {
      for (const node of next) {
        node.needs = node.needs.map((need) =>
          need === current.id ? (patch.id ?? need) : need,
        )
      }
    }
    updateNodes(
      next,
      patch.id && patch.id !== current.id
        ? { from: current.id, to: patch.id }
        : undefined,
    )
  }

  function applyTemplate(template: WorkflowTemplate) {
    const definition = parsed.definition ?? {}
    const existingNodes = isRecord(definition.nodes) ? definition.nodes : {}
    const templateNodes = isRecord(template.definition.nodes)
      ? template.definition.nodes
      : {}
    const mergedNodes = Object.fromEntries(
      Object.entries(templateNodes).map(([id, templateNode]) => [
        id,
        {
          ...(isRecord(existingNodes[id]) ? existingNodes[id] : {}),
          ...(isRecord(templateNode) ? templateNode : {}),
        },
      ]),
    )
    const nextDefinition = {
      ...definition,
      ...template.definition,
      nodes: mergedNodes,
      result: {
        ...(isRecord(definition.result) ? definition.result : {}),
        ...(isRecord(template.definition.result)
          ? template.definition.result
          : {}),
      },
    }
    onChange(JSON.stringify(nextDefinition, null, 2))
  }

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
        <div>
          <p className="text-sm font-medium text-blue-600">Guided builder</p>
          <h3 className="mt-1 text-lg font-semibold text-slate-950">
            Workflow steps
          </h3>
          <p className="mt-1 text-sm text-slate-500">
            Start from a template, then describe each agent step and its
            dependencies.
          </p>
        </div>
        <button
          type="button"
          className="inline-flex items-center justify-center gap-2 rounded-xl border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"
          onClick={() =>
            updateNodes([
              ...nodes,
              {
                id: nextNodeId(nodes),
                profile: 'Self',
                mode: nodes.length ? 'continue' : 'isolate',
                needs: nodes.length ? [nodes[nodes.length - 1]!.id] : [],
                prompt: '',
              },
            ])
          }
        >
          <Plus className="h-4 w-4" />
          Add step
        </button>
      </div>

      <div className="mt-4 grid gap-2 md:grid-cols-3">
        {templates.map((template) => (
          <button
            key={template.id}
            type="button"
            className="rounded-xl border border-slate-200 p-3 text-left transition hover:border-blue-300 hover:bg-blue-50"
            onClick={() => applyTemplate(template)}
          >
            <span className="block text-sm font-semibold text-slate-900">
              {template.name}
            </span>
            <span className="mt-1 block text-xs leading-5 text-slate-500">
              {template.description}
            </span>
          </button>
        ))}
      </div>

      <ValidationSummary issues={parsed.issues} nodeCount={nodes.length} />

      <div className="mt-4 space-y-3">
        {nodes.map((node, index) => (
          <article
            key={`${node.id}-${index}`}
            className="rounded-xl border border-slate-200 bg-slate-50 p-4"
          >
            <div className="flex items-center justify-between gap-3">
              <p className="text-sm font-semibold text-slate-900">
                Step {index + 1}
              </p>
              <button
                type="button"
                className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-rose-600 hover:bg-rose-50 disabled:opacity-40"
                onClick={() =>
                  updateNodes(
                    nodes.filter((_, itemIndex) => itemIndex !== index),
                  )
                }
                disabled={nodes.length === 1}
                aria-label={`Remove ${node.id}`}
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </div>
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <BuilderField label="Step ID">
                <input
                  className={inputClass}
                  value={node.id}
                  onChange={(event) =>
                    updateNode(index, {
                      id: normalizeNodeId(event.target.value),
                    })
                  }
                />
              </BuilderField>
              <BuilderField label="Agent profile">
                <input
                  className={inputClass}
                  value={node.profile}
                  onChange={(event) =>
                    updateNode(index, { profile: event.target.value })
                  }
                  placeholder="Self"
                />
              </BuilderField>
              <BuilderField label="Execution mode">
                <select
                  className={inputClass}
                  value={node.mode}
                  onChange={(event) =>
                    updateNode(index, {
                      mode: event.target.value as WorkflowNodeDraft['mode'],
                    })
                  }
                >
                  <option value="isolate">New isolated session</option>
                  <option value="continue">Continue workflow context</option>
                </select>
              </BuilderField>
              <BuilderField
                label="Depends on"
                hint="Comma-separated step IDs; leave empty for an entry step."
              >
                <input
                  className={inputClass}
                  value={node.needs.join(', ')}
                  onChange={(event) =>
                    updateNode(index, {
                      needs: splitNodeIds(event.target.value),
                    })
                  }
                />
              </BuilderField>
            </div>
            <BuilderField label="Prompt">
              <textarea
                className={`${inputClass} min-h-28`}
                value={node.prompt}
                onChange={(event) =>
                  updateNode(index, { prompt: event.target.value })
                }
                placeholder="Describe what this step should produce. Inputs are available as {{ inputs.name }}."
              />
            </BuilderField>
          </article>
        ))}
      </div>

      <div className="mt-5">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          Graph preview
        </p>
        <div className="mt-2 grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
          {nodes.map((node) => (
            <div
              key={`graph-${node.id}`}
              className="rounded-xl border border-slate-200 bg-white p-3"
            >
              <p className="mono text-sm font-semibold text-slate-900">
                {node.id}
              </p>
              <p className="mt-1 text-xs text-slate-500">
                {node.needs.length
                  ? `After ${node.needs.join(', ')}`
                  : 'Entry step'}
              </p>
              <p className="mt-2 line-clamp-2 text-xs leading-5 text-slate-600">
                {node.prompt || 'Prompt not configured'}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

function ValidationSummary({
  issues,
  nodeCount,
}: {
  issues: string[]
  nodeCount: number
}) {
  const valid = issues.length === 0
  return (
    <div
      className={cn(
        'mt-4 rounded-xl border p-3',
        valid
          ? 'border-emerald-200 bg-emerald-50 text-emerald-900'
          : 'border-amber-200 bg-amber-50 text-amber-950',
      )}
      role={valid ? 'status' : 'alert'}
    >
      <div className="flex items-start gap-2">
        {valid ? (
          <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" />
        ) : (
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
        )}
        <div>
          <p className="text-sm font-semibold">
            {valid
              ? `${nodeCount} workflow step${nodeCount === 1 ? '' : 's'} ready`
              : 'Resolve workflow validation issues'}
          </p>
          {issues.length ? (
            <ul className="mt-1 list-disc space-y-1 pl-5 text-xs leading-5">
              {issues.map((issue) => (
                <li key={issue}>{issue}</li>
              ))}
            </ul>
          ) : (
            <p className="mt-1 text-xs">
              All dependencies and output references are valid.
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

function BuilderField({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: React.ReactNode
}) {
  return (
    <label className="mt-3 block text-sm font-medium text-slate-700">
      {label}
      {children}
      {hint ? (
        <span className="mt-1 block text-xs font-normal text-slate-500">
          {hint}
        </span>
      ) : null}
    </label>
  )
}

const inputClass =
  'mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm outline-none ring-blue-600 focus:ring-2'

function parseWorkflowDefinition(value: string): {
  definition: Record<string, unknown> | null
  nodes: WorkflowNodeDraft[]
  issues: string[]
} {
  let definition: Record<string, unknown>
  try {
    const parsed: unknown = JSON.parse(value)
    if (!isRecord(parsed)) throw new Error('Definition must be a JSON object.')
    definition = parsed
  } catch (error) {
    return {
      definition: null,
      nodes: [],
      issues: [
        error instanceof Error
          ? error.message
          : 'Definition is not valid JSON.',
      ],
    }
  }

  const issues: string[] = []
  if (definition.schema !== 'ya-claw.workflow.v1') {
    issues.push('Schema must be ya-claw.workflow.v1.')
  }
  const rawNodes = isRecord(definition.nodes) ? definition.nodes : {}
  const nodes: WorkflowNodeDraft[] = Object.entries(rawNodes).map(
    ([id, rawNode]): WorkflowNodeDraft => {
      const node = isRecord(rawNode) ? rawNode : {}
      const mode: WorkflowNodeDraft['mode'] =
        node.mode === 'continue' ? 'continue' : 'isolate'
      const needs = Array.isArray(node.needs)
        ? node.needs.filter((item): item is string => typeof item === 'string')
        : []
      return {
        id,
        profile: typeof node.profile === 'string' ? node.profile : 'Self',
        mode,
        needs,
        prompt: typeof node.prompt === 'string' ? node.prompt : '',
      }
    },
  )
  if (!nodes.length) issues.push('Add at least one workflow step.')
  const ids = new Set(nodes.map((node) => node.id))
  for (const node of nodes) {
    if (!/^[A-Za-z][A-Za-z0-9_-]*$/.test(node.id)) {
      issues.push(
        `Step ID “${node.id}” must start with a letter and use letters, numbers, _ or -.`,
      )
    }
    if (!node.prompt.trim()) issues.push(`Step “${node.id}” needs a prompt.`)
    for (const dependency of node.needs) {
      if (!ids.has(dependency)) {
        issues.push(
          `Step “${node.id}” depends on unknown step “${dependency}”.`,
        )
      }
      if (dependency === node.id) {
        issues.push(`Step “${node.id}” cannot depend on itself.`)
      }
    }
  }
  if (hasDependencyCycle(nodes))
    issues.push('Step dependencies contain a cycle.')
  const resultNode = isRecord(definition.result)
    ? definition.result.from_node
    : null
  if (typeof resultNode !== 'string' || !ids.has(resultNode)) {
    issues.push('Result must reference an existing final step.')
  }
  return { definition, nodes, issues: [...new Set(issues)] }
}

function hasDependencyCycle(nodes: WorkflowNodeDraft[]) {
  const dependencies = new Map(nodes.map((node) => [node.id, node.needs]))
  const visiting = new Set<string>()
  const visited = new Set<string>()
  function visit(id: string): boolean {
    if (visiting.has(id)) return true
    if (visited.has(id)) return false
    visiting.add(id)
    for (const dependency of dependencies.get(id) ?? []) {
      if (dependencies.has(dependency) && visit(dependency)) return true
    }
    visiting.delete(id)
    visited.add(id)
    return false
  }
  return nodes.some((node) => visit(node.id))
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function normalizeNodeId(value: string) {
  return value.replace(/\s+/g, '_').replace(/[^A-Za-z0-9_-]/g, '')
}

function splitNodeIds(value: string) {
  return [
    ...new Set(
      value
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean),
    ),
  ]
}

function nextNodeId(nodes: WorkflowNodeDraft[]) {
  const ids = new Set(nodes.map((node) => node.id))
  let index = nodes.length + 1
  while (ids.has(`step_${index}`)) index += 1
  return `step_${index}`
}
