import type {
  ApiDashboard,
  ApiForward,
  ApiResults,
  ApiRun,
  ApiRunDetail,
  ApiRunsPage,
  ApiSummary,
  DataState,
  InstrumentMatch,
  PreflightReport,
} from './types'
import type { EligibilityResult, MandateSubmission, WorkflowMandate } from './mandate-requests'
import { RequestCoordinator } from './http'
import {
  clearStaticDataCache,
  loadStaticDashboard,
  loadStaticForward,
  loadStaticResults,
  loadStaticRunDetail,
  loadStaticRuns,
  loadStaticSummary,
  normalizeRun,
} from './static-data'

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String) : []
}

function normalizeEligibility(row: Partial<EligibilityResult> | Record<string, unknown>): EligibilityResult {
  const raw = row as Record<string, unknown>
  const bars = Number(raw.historyBars ?? raw.history_bars ?? 0)
  const symbolFound = Boolean(raw.symbolFound ?? raw.symbol_found)
  const approvedPolicy = Boolean(raw.approvedPolicy ?? raw.approved_policy)
  const sufficientHistory = Boolean(raw.sufficientHistory ?? raw.sufficient_history ?? bars >= 2_500)
  const eligible = Boolean(
    raw.eligible ?? (symbolFound && approvedPolicy && sufficientHistory),
  )
  return {
    ticker: String(raw.ticker ?? ''),
    symbolFound,
    historyBars: Number.isFinite(bars) ? bars : 0,
    firstDate: (raw.firstDate ?? raw.first_date ?? null) as string | null,
    lastDate: (raw.lastDate ?? raw.last_date ?? null) as string | null,
    approvedPolicy,
    panelCompatible: Boolean(raw.panelCompatible ?? raw.panel_compatible ?? symbolFound),
    sufficientHistory,
    eligible,
  }
}

function normalizeMandate(mandate: WorkflowMandate): WorkflowMandate {
  return {
    ...mandate,
    instruments: Array.isArray(mandate.instruments) ? mandate.instruments : [],
    eligibility: Array.isArray(mandate.eligibility)
      ? mandate.eligibility.map((row) => normalizeEligibility(row as EligibilityResult))
      : [],
    allowedActions: asStringArray(mandate.allowedActions),
    auditLog: Array.isArray(mandate.auditLog) ? mandate.auditLog : [],
  }
}

/**
 * Research data source:
 * - ``static`` (default) → Vite ``/data/*.json`` snapshots (milliseconds)
 * - ``api`` → live research API on :8787 (slow on iCloud Desktop)
 * - ``offline`` → labeled synthetic sandbox (no Runs/)
 *
 * Resolution: explicit ``VITE_DATA_SOURCE`` wins. Otherwise ``VITE_API_URL``
 * unset/offline → offline; anything else (including empty proxy URL) → static.
 * Force the live API with ``VITE_DATA_SOURCE=api``.
 */
const rawApiUrl = import.meta.env.VITE_API_URL as string | undefined
const rawDataSource = (import.meta.env.VITE_DATA_SOURCE as string | undefined)?.toLowerCase()

function resolveDataSource(): 'static' | 'api' | 'offline' {
  if (rawDataSource === 'static' || rawDataSource === 'api' || rawDataSource === 'offline') {
    return rawDataSource
  }
  if (rawApiUrl === undefined || rawApiUrl === 'offline') return 'offline'
  return 'static'
}

const DATA_SOURCE = resolveDataSource()
export const OFFLINE_MODE = DATA_SOURCE === 'offline'
export const STATIC_DATA_MODE = DATA_SOURCE === 'static'
const API_URL = DATA_SOURCE === 'api' ? String(rawApiUrl ?? '').replace(/\/$/, '') : undefined
const API_TOKEN = import.meta.env.VITE_API_TOKEN as string | undefined
const WORKFLOW_API_URL = (import.meta.env.VITE_WORKFLOW_API_URL as string | undefined)?.replace(/\/$/, '')
const INVESTOR_WORKFLOW_TOKEN = import.meta.env.VITE_INVESTOR_WORKFLOW_TOKEN as string | undefined
const OPS_WORKFLOW_TOKEN = import.meta.env.VITE_OPS_WORKFLOW_TOKEN as string | undefined
const coordinator = new RequestCoordinator(12_000, 8_000)
const workflowCoordinator = new RequestCoordinator(20_000, 5_000)
/** Forward live MTM can wait on a Yahoo pull; allow longer than the default. */
const FORWARD_TIMEOUT_MS = 45_000
/** Manual / hard reload waits for Yahoo 5m history to refresh server-side. */
const FORCE_FORWARD_TIMEOUT_MS = 95_000

export const WORKFLOW_OFFLINE = !WORKFLOW_API_URL || !INVESTOR_WORKFLOW_TOKEN
/** @deprecated Use OFFLINE_MODE */
export const DEMO_MODE = OFFLINE_MODE

function assertShape(condition: boolean, message: string): asserts condition {
  if (!condition) throw new Error(`Unexpected API response: ${message}`)
}

async function request<T>(path: string, init?: RequestInit, timeoutMs?: number): Promise<T> {
  if (API_URL === undefined) throw new Error('Research API is not configured')
  const headers = new Headers(init?.headers)
  if (API_TOKEN) headers.set('Authorization', `Bearer ${API_TOKEN}`)
  const signal = init?.signal ?? undefined
  const fetchInit = { ...init, headers }
  delete fetchInit.signal
  return coordinator.request<T>(`${API_URL}${path}`, fetchInit, signal, timeoutMs)
}

async function workflowRequest<T>(
  path: string,
  init?: RequestInit,
  audience: 'investor' | 'operator' = 'investor',
): Promise<T> {
  const token = audience === 'operator' ? OPS_WORKFLOW_TOKEN : INVESTOR_WORKFLOW_TOKEN
  if (!WORKFLOW_API_URL || !token) throw new Error(`Authenticated ${audience} workflow API is not configured`)
  const headers = new Headers(headersInit(init))
  headers.set('Authorization', `Bearer ${token}`)
  const signal = init?.signal ?? undefined
  const fetchInit = { ...init, headers }
  delete fetchInit.signal
  return workflowCoordinator.request<T>(`${WORKFLOW_API_URL}${path}`, fetchInit, signal)
}

function headersInit(init?: RequestInit): Headers {
  return new Headers(init?.headers)
}

/**
 * Fetch wrapper producing an explicit DataState. A configured API that fails
 * yields `error` (with retry available to callers) — never silent cached data.
 */
async function toState<T>(
  loader: () => Promise<T>,
  offline = OFFLINE_MODE,
  hint: 'research' | 'workflow' = 'research',
): Promise<DataState<T>> {
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
    if (
      message.includes('Failed to fetch')
      || message.includes('Network or CORS')
      || message.includes('HTTP 502')
      || message.includes('HTTP 504')
      || message.includes('Static data HTTP')
      || message.includes('timed out')
    ) {
      if (hint === 'workflow') {
        return {
          kind: 'error',
          message:
            `${message} Workflow API on :8790 is down or restarting. ` +
            `From the repo root: python scripts/workflow_api.py --port 8790 ` +
            `(or restart with bash scripts/start_ui.sh). Confirm VITE_WORKFLOW_API_URL and ` +
            `VITE_OPS_WORKFLOW_TOKEN in frontend/.env.local.`,
        }
      }
      return {
        kind: 'error',
        message: STATIC_DATA_MODE
          ? `${message} Missing or stale snapshots under public/data/. From the repo root run: python3 scripts/publish_frontend_data.py --with-details`
          : `${message} The research API on :8787 is down or restarting. ` +
            `From the repo root run: python scripts/frontend_api_lite.py --port 8787 ` +
            `(or restart with cd frontend && npm run dev).`,
      }
    }
    return { kind: 'error', message }
  }
}

export function fetchSummary(): Promise<DataState<ApiSummary>> {
  return toState(async () => {
    if (STATIC_DATA_MODE) {
      const data = await loadStaticSummary()
      assertShape(typeof data.total_runs === 'number', 'summary.total_runs missing')
      return data
    }
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
    if (STATIC_DATA_MODE) {
      // Drop memoized snapshot so auto-refresh / Retry sees republished OOS.
      clearStaticDataCache('runs.json')
      const data = await loadStaticRuns(query, signal)
      assertShape(Array.isArray(data.runs), 'runs is not an array')
      assertShape(typeof data.total === 'number' && typeof data.counts?.all === 'number', 'runs pagination missing')
      return data
    }
    const params = new URLSearchParams()
    if (query.prefix) params.set('prefix', query.prefix)
    if (query.search) params.set('search', query.search)
    if (query.status) params.set('status', query.status)
    params.set('offset', String(query.offset ?? 0))
    params.set('limit', String(query.limit ?? 50))
    const data = await request<ApiRunsPage>(`/api/runs?${params}`, { signal })
    assertShape(Array.isArray(data.runs), 'runs is not an array')
    assertShape(typeof data.total === 'number' && typeof data.counts?.all === 'number', 'runs pagination missing')
    return { ...data, runs: data.runs.map(normalizeRun) }
  })
}

export function fetchDashboard(signal?: AbortSignal): Promise<DataState<ApiDashboard>> {
  return toState(async () => {
    if (STATIC_DATA_MODE) {
      clearStaticDataCache('dashboard.json')
      clearStaticDataCache('summary.json')
      const data = await loadStaticDashboard(signal)
      assertShape(typeof data.summary?.total_runs === 'number', 'dashboard.summary missing')
      assertShape(Array.isArray(data.recent_runs), 'dashboard.recent_runs is not an array')
      assertShape(Array.isArray(data.window_sharpes), 'dashboard.window_sharpes is not an array')
      return data
    }
    const data = await request<ApiDashboard>('/api/dashboard', { signal })
    assertShape(typeof data.summary?.total_runs === 'number', 'dashboard.summary missing')
    assertShape(Array.isArray(data.recent_runs), 'dashboard.recent_runs is not an array')
    assertShape(Array.isArray(data.window_sharpes), 'dashboard.window_sharpes is not an array')
    return { ...data, recent_runs: data.recent_runs.map(normalizeRun) }
  })
}

export function fetchRunDetail(runId: string, signal?: AbortSignal): Promise<DataState<ApiRunDetail>> {
  return toState(async () => {
    if (STATIC_DATA_MODE) {
      const data = await loadStaticRunDetail(runId, signal)
      assertShape(typeof data.run_id === 'string', 'run detail missing run_id')
      return data
    }
    const data = await request<ApiRunDetail>(`/api/runs/${encodeURIComponent(runId)}`, { signal })
    assertShape(typeof data.run_id === 'string', 'run detail missing run_id')
    return {
      ...data,
      audit: data.audit ? normalizeRun(data.audit) : null,
    }
  })
}

export function fetchResults(cohort = '', signal?: AbortSignal): Promise<DataState<ApiResults>> {
  return toState(async () => {
    if (STATIC_DATA_MODE) {
      const data = await loadStaticResults(cohort, signal)
      assertShape(Array.isArray(data.rows), 'results.rows is not an array')
      return data
    }
    const data = await request<ApiResults>(
      `/api/results${cohort ? `?cohort=${encodeURIComponent(cohort)}` : ''}`,
      { signal },
    )
    assertShape(Array.isArray(data.rows), 'results.rows is not an array')
    return data
  })
}

export function fetchForward(
  runId = '',
  signal?: AbortSignal,
  opts: { forceRefresh?: boolean } = {},
): Promise<DataState<ApiForward>> {
  return toState(async () => {
    // Force refresh still needs the research API (Yahoo pull). Soft loads use snapshots.
    if (STATIC_DATA_MODE && !opts.forceRefresh) {
      const data = await loadStaticForward(signal)
      assertShape(typeof data.available === 'boolean', 'forward.available missing')
      if (runId && data.run_id && data.run_id !== runId) {
        // Snapshot is for the active LIVE run; ignore mismatched ids.
      }
      return data
    }
    if (STATIC_DATA_MODE && opts.forceRefresh) {
      // Best-effort: hit same-origin proxy if lite API is up; else return static.
      try {
        const params = new URLSearchParams()
        if (runId) params.set('run_id', runId)
        params.set('live', '1')
        params.set('force_refresh', '1')
        params.set('_', String(Date.now()))
        const data = await coordinator.request<ApiForward>(
          `/api/forward?${params.toString()}`,
          {},
          signal,
          FORCE_FORWARD_TIMEOUT_MS,
        )
        assertShape(typeof data.available === 'boolean', 'forward.available missing')
        return data
      } catch {
        const data = await loadStaticForward(signal)
        assertShape(typeof data.available === 'boolean', 'forward.available missing')
        return data
      }
    }
    const params = new URLSearchParams()
    if (runId) params.set('run_id', runId)
    params.set('live', '1')
    if (opts.forceRefresh) params.set('force_refresh', '1')
    params.set('_', String(Date.now()))
    const timeoutMs = opts.forceRefresh ? FORCE_FORWARD_TIMEOUT_MS : FORWARD_TIMEOUT_MS
    const data = await request<ApiForward>(`/api/forward?${params.toString()}`, { signal }, timeoutMs)
    assertShape(typeof data.available === 'boolean', 'forward.available missing')
    return data
  })
}

/** Engine-backed preflight. Throws when no API is configured — callers must branch first. */
export async function runPreflight(yamlText: string): Promise<PreflightReport> {
  if (OFFLINE_MODE) throw new Error('Preflight requires an active research API connection')
  const data = await coordinator.request<PreflightReport>('/api/preflight', {
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
    // Avoid serving a failed/empty in-flight cache while the API was still booting.
    workflowCoordinator.clear()
    const data = await workflowRequest<{ mandates: WorkflowMandate[] }>('/api/mandates', { signal }, audience)
    assertShape(Array.isArray(data.mandates), 'mandates missing')
    return data.mandates.map(normalizeMandate)
  }, offline, 'workflow')
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

// Re-export for callers that already imported normalize helpers via api.
export type { ApiRun }
