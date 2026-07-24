import type {
  ApiDashboard,
  ApiResults,
  ApiRunDetail,
  ApiRunsPage,
  ApiSummary,
  DataState,
  InstrumentMatch,
  PreflightReport,
} from './types'
import type { MandateSubmission, WorkflowMandate } from './mandate-requests'
import { RequestCoordinator } from './http'

const API_URL = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '')
const API_TOKEN = import.meta.env.VITE_API_TOKEN as string | undefined
const WORKFLOW_API_URL = (import.meta.env.VITE_WORKFLOW_API_URL as string | undefined)?.replace(/\/$/, '')
const INVESTOR_WORKFLOW_TOKEN = import.meta.env.VITE_INVESTOR_WORKFLOW_TOKEN as string | undefined
const OPS_WORKFLOW_TOKEN = import.meta.env.VITE_OPS_WORKFLOW_TOKEN as string | undefined
const coordinator = new RequestCoordinator(10_000, 5_000)
const workflowCoordinator = new RequestCoordinator(20_000, 5_000)

/** True when no API URL is configured — pages use cached workspace data. */
export const OFFLINE_MODE = !API_URL
export const WORKFLOW_OFFLINE = !WORKFLOW_API_URL || !INVESTOR_WORKFLOW_TOKEN
/** @deprecated Use OFFLINE_MODE */
export const DEMO_MODE = OFFLINE_MODE

function assertShape(condition: boolean, message: string): asserts condition {
  if (!condition) throw new Error(`Unexpected API response: ${message}`)
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers)
  if (API_TOKEN) headers.set('Authorization', `Bearer ${API_TOKEN}`)
  const signal = init?.signal ?? undefined
  const fetchInit = { ...init, headers }
  delete fetchInit.signal
  return coordinator.request<T>(`${API_URL}${path}`, fetchInit, signal)
}

async function workflowRequest<T>(
  path: string,
  init?: RequestInit,
  audience: 'investor' | 'operator' = 'investor',
): Promise<T> {
  const token = audience === 'operator' ? OPS_WORKFLOW_TOKEN : INVESTOR_WORKFLOW_TOKEN
  if (!WORKFLOW_API_URL || !token) throw new Error(`Authenticated ${audience} workflow API is not configured`)
  const headers = new Headers(init?.headers)
  headers.set('Authorization', `Bearer ${token}`)
  const signal = init?.signal ?? undefined
  const fetchInit = { ...init, headers }
  delete fetchInit.signal
  return workflowCoordinator.request<T>(`${WORKFLOW_API_URL}${path}`, fetchInit, signal)
}

/**
 * Fetch wrapper producing an explicit DataState. A configured API that fails
 * yields `error` (with retry available to callers) — never silent cached data.
 */
async function toState<T>(loader: () => Promise<T>, offline = OFFLINE_MODE): Promise<DataState<T>> {
  if (offline) return { kind: 'offline' }
  try {
    return { kind: 'live', data: await loader() }
  } catch (error) {
    if (
      (error instanceof Error && (error.name === 'AbortError' || error.message === 'Request cancelled'))
      || (typeof error === 'object' && error !== null && 'kind' in error && (error as { kind: string }).kind === 'aborted')
    ) {
      return { kind: 'loading' }
    }
    const message = error instanceof Error ? error.message : String(error)
    if (message.includes('Failed to fetch') || message.includes('Network or CORS')) {
      return {
        kind: 'error',
        message: `${message} Start the local services: python scripts/frontend_api.py --port 8787 and python scripts/workflow_api.py --port 8790.`,
      }
    }
    return { kind: 'error', message }
  }
}

export function fetchSummary(): Promise<DataState<ApiSummary>> {
  return toState(async () => {
    const data = await request<ApiSummary>('/api/summary')
    assertShape(typeof data.total_runs === 'number', 'summary.total_runs missing')
    return data
  })
}

export type RunQuery = {
  prefix?: string
  search?: string
  status?: '' | 'completed' | 'active' | 'interrupted'
  offset?: number
  limit?: number
}

export function fetchRuns(query: RunQuery = {}, signal?: AbortSignal): Promise<DataState<ApiRunsPage>> {
  return toState(async () => {
    const params = new URLSearchParams()
    if (query.prefix) params.set('prefix', query.prefix)
    if (query.search) params.set('search', query.search)
    if (query.status) params.set('status', query.status)
    params.set('offset', String(query.offset ?? 0))
    params.set('limit', String(query.limit ?? 50))
    const data = await request<ApiRunsPage>(`/api/runs?${params}`, { signal })
    assertShape(Array.isArray(data.runs), 'runs is not an array')
    assertShape(typeof data.total === 'number' && typeof data.counts?.all === 'number', 'runs pagination missing')
    return data
  })
}

export function fetchDashboard(signal?: AbortSignal): Promise<DataState<ApiDashboard>> {
  return toState(async () => {
    const data = await request<ApiDashboard>('/api/dashboard', { signal })
    assertShape(typeof data.summary?.total_runs === 'number', 'dashboard.summary missing')
    assertShape(Array.isArray(data.recent_runs), 'dashboard.recent_runs is not an array')
    assertShape(Array.isArray(data.window_sharpes), 'dashboard.window_sharpes is not an array')
    return data
  })
}

export function fetchRunDetail(runId: string, signal?: AbortSignal): Promise<DataState<ApiRunDetail>> {
  return toState(async () => {
    const data = await request<ApiRunDetail>(`/api/runs/${encodeURIComponent(runId)}`, { signal })
    assertShape(typeof data.run_id === 'string', 'run detail missing run_id')
    return data
  })
}

export function fetchResults(cohort = '', signal?: AbortSignal): Promise<DataState<ApiResults>> {
  return toState(async () => {
    const data = await request<ApiResults>(
      `/api/results${cohort ? `?cohort=${encodeURIComponent(cohort)}` : ''}`,
      { signal },
    )
    assertShape(Array.isArray(data.rows), 'results.rows is not an array')
    return data
  })
}

/** Engine-backed preflight. Throws when no API is configured — callers must branch first. */
export async function runPreflight(yamlText: string): Promise<PreflightReport> {
  if (OFFLINE_MODE) throw new Error('Preflight requires an active research API connection')
  const data = await request<PreflightReport>('/api/preflight', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ yaml_text: yamlText }),
  })
  assertShape(typeof data.ok === 'boolean' && Array.isArray(data.errors), 'preflight report malformed')
  return data
}

function parseInstrumentMatch(data: unknown): InstrumentMatch {
  assertShape(typeof data === 'object' && data !== null, 'instrument match malformed')
  const row = data as Record<string, unknown>
  assertShape(typeof row.symbol === 'string' && typeof row.name === 'string', 'instrument match fields missing')
  return {
    found: Boolean(row.found),
    symbol: row.symbol,
    name: row.name,
    group: (typeof row.group === 'string' ? row.group : 'Alternative') as InstrumentMatch['group'],
    exchange: typeof row.exchange === 'string' ? row.exchange : null,
    currency: typeof row.currency === 'string' ? row.currency : null,
  }
}

export async function searchInstruments(query: string, signal?: AbortSignal): Promise<InstrumentMatch[]> {
  if (WORKFLOW_OFFLINE) throw new Error('Instrument search requires the authenticated workflow API')
  const data = await workflowRequest<{ results: unknown[] }>(
    `/api/instruments/search?q=${encodeURIComponent(query.trim())}&limit=8`,
    { signal },
  )
  assertShape(Array.isArray(data.results), 'instrument search results missing')
  return data.results.map(parseInstrumentMatch)
}

export function fetchMandates(
  signal?: AbortSignal,
  audience: 'investor' | 'operator' = 'investor',
): Promise<DataState<WorkflowMandate[]>> {
  const offline = !WORKFLOW_API_URL || (audience === 'operator' ? !OPS_WORKFLOW_TOKEN : !INVESTOR_WORKFLOW_TOKEN)
  return toState(async () => {
    const data = await workflowRequest<{ mandates: WorkflowMandate[] }>('/api/mandates', { signal }, audience)
    assertShape(Array.isArray(data.mandates), 'mandates missing')
    return data.mandates
  }, offline)
}

export async function fetchMandateDetail(
  id: string,
  signal?: AbortSignal,
  audience: 'investor' | 'operator' = 'investor',
): Promise<WorkflowMandate> {
  return workflowRequest<WorkflowMandate>(`/api/mandates/${encodeURIComponent(id)}`, { signal }, audience)
}

export async function submitMandate(submission: MandateSubmission): Promise<WorkflowMandate> {
  return workflowRequest<WorkflowMandate>('/api/mandates', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Idempotency-Key': crypto.randomUUID(),
    },
    body: JSON.stringify(submission),
  })
}

export async function performMandateAction(
  id: string,
  action: string,
  detail: Record<string, unknown> = {},
  audience: 'investor' | 'operator' = 'operator',
): Promise<WorkflowMandate> {
  return workflowRequest<WorkflowMandate>(`/api/mandates/${encodeURIComponent(id)}/actions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, detail }),
  }, audience)
}

export async function cancelMandate(id: string): Promise<WorkflowMandate> {
  return performMandateAction(id, 'cancel', {}, 'investor')
}
