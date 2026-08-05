/**
 * Millisecond-fast reads from Vite-served ``public/data/*.json`` snapshots.
 *
 * Snapshots are produced by ``scripts/publish_frontend_data.py`` (also invoked
 * from ``frontend/scripts/dev.mjs``). In-memory memoization makes subsequent
 * page navigations free.
 */

import type {
  ApiDashboard,
  ApiForward,
  ApiResults,
  ApiRun,
  ApiRunDetail,
  ApiRunsPage,
  ApiSummary,
} from './types'

const DATA_BASE = (import.meta.env.VITE_STATIC_DATA_URL as string | undefined)?.replace(/\/$/, '') || '/data'

type CacheEntry<T> = { at: number; data: T; promise?: Promise<T> }

const memory = new Map<string, CacheEntry<unknown>>()
/**
 * Soft navigations share one in-flight fetch via ``promise``. TTL is long
 * enough that 60s auto-refresh is the usual refresh path (not every mount).
 * Call ``clearStaticDataCache`` after an explicit republish / force refresh.
 * Forward marks use a shorter TTL so the page does not stick on yesterday's snapshot.
 */
const MEMORY_TTL_MS = 45_000
const FORWARD_MEMORY_TTL_MS = 8_000

function ttlFor(relPath: string): number {
  return relPath === 'forward.json' ? FORWARD_MEMORY_TTL_MS : MEMORY_TTL_MS
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String) : []
}

export function normalizeRun(run: ApiRun): ApiRun {
  return {
    ...run,
    labels: asStringArray(run.labels),
    warnings: asStringArray(run.warnings),
  }
}

async function loadJson<T>(
  relPath: string,
  signal?: AbortSignal,
  opts: { bypassCache?: boolean } = {},
): Promise<T> {
  const key = relPath
  const now = Date.now()
  const ttl = ttlFor(relPath)
  const hit = memory.get(key) as CacheEntry<T> | undefined
  if (
    !opts.bypassCache &&
    hit &&
    hit.data !== undefined &&
    now - hit.at < ttl &&
    !hit.promise
  ) {
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError')
    return hit.data
  }

  // Shared in-flight fetch must NOT be tied to any one caller's AbortSignal —
  // React Strict Mode / tab changes abort the first subscriber and would cancel
  // the download for everyone waiting on the same promise (Runs "All" stuck loading).
  let promise = !opts.bypassCache ? hit?.promise : undefined
  if (!promise) {
    promise = (async () => {
      const bust = opts.bypassCache ? now : Math.floor(now / ttl)
      const url = `${DATA_BASE}/${relPath}?v=${bust}`
      const response = await fetch(url, { cache: 'no-store' })
      if (!response.ok) {
        throw new Error(`Static data HTTP ${response.status} for /data/${relPath}`)
      }
      return (await response.json()) as T
    })()
    memory.set(key, { at: now, data: hit?.data as T, promise })
  }

  try {
    const data = await promise
    memory.set(key, { at: Date.now(), data })
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError')
    return data
  } catch (error) {
    const current = memory.get(key)
    if (current?.promise === promise) memory.delete(key)
    throw error
  }
}

export function clearStaticDataCache(path?: string) {
  if (path) memory.delete(path)
  else memory.clear()
}

export async function loadStaticSummary(signal?: AbortSignal): Promise<ApiSummary> {
  return loadJson<ApiSummary>('summary.json', signal)
}

export async function loadStaticDashboard(signal?: AbortSignal): Promise<ApiDashboard> {
  const data = await loadJson<ApiDashboard>('dashboard.json', signal)
  return {
    ...data,
    recent_runs: (data.recent_runs ?? []).map(normalizeRun),
  }
}

export async function loadStaticResults(cohort = '', signal?: AbortSignal): Promise<ApiResults> {
  const data = await loadJson<ApiResults>('results.json', signal)
  const rows = cohort ? data.rows.filter((row) => row.cohort === cohort) : data.rows
  return { ...data, rows }
}

export async function loadStaticForward(
  signal?: AbortSignal,
  opts: { bypassCache?: boolean } = {},
): Promise<ApiForward> {
  return loadJson<ApiForward>('forward.json', signal, opts)
}

export type StaticRunQuery = {
  prefix?: string
  search?: string
  status?: '' | 'completed' | 'active' | 'interrupted'
  offset?: number
  limit?: number
}

export async function loadStaticRuns(
  query: StaticRunQuery = {},
  signal?: AbortSignal,
): Promise<ApiRunsPage> {
  const data = await loadJson<ApiRunsPage>('runs.json', signal)
  let rows = (data.runs ?? []).map(normalizeRun)
  const counts = data.counts ?? {
    all: rows.length,
    completed: rows.filter((r) => r.training_status === 'completed').length,
    active: rows.filter((r) => r.training_status !== 'completed' && r.training_status !== 'interrupted').length,
    interrupted: rows.filter((r) => r.training_status === 'interrupted').length,
    with_backtest: rows.filter((r) => r.has_backtest).length,
  }
  if (query.prefix) rows = rows.filter((r) => r.run_id.startsWith(query.prefix!))
  if (query.search) {
    const needle = query.search.toLowerCase()
    rows = rows.filter((r) => r.run_id.toLowerCase().includes(needle))
  }
  if (query.status === 'completed') {
    rows = rows.filter((r) => r.training_status === 'completed')
  } else if (query.status === 'interrupted') {
    rows = rows.filter((r) => r.training_status === 'interrupted')
  } else if (query.status === 'active') {
    rows = rows.filter((r) => r.training_status !== 'completed' && r.training_status !== 'interrupted')
  }
  const offset = query.offset ?? 0
  const limit = query.limit ?? 50
  return {
    runs: rows.slice(offset, offset + limit),
    total: rows.length,
    offset,
    limit,
    counts,
  }
}

export async function loadStaticRunDetail(
  runId: string,
  signal?: AbortSignal,
): Promise<ApiRunDetail> {
  const synthesizeBacktest = (audit: ApiRun | null): ApiRunDetail['backtest'] => {
    if (!audit) return null
    if (!audit.has_backtest && audit.oos_sharpe == null && audit.oos_return == null) return null
    return {
      checkpoint_label: audit.labels[0] ?? null,
      oos_window: audit.window,
      total_return: audit.oos_return,
      sharpe: audit.oos_sharpe,
      excess_sharpe: null,
      max_drawdown: audit.oos_max_drawdown,
      deflated_sharpe: audit.oos_deflated_sharpe,
      deflated_sharpe_excess: null,
      oos_trials_for_window: null,
      oos_trials_conservative: null,
      equal_weight_daily_return: null,
      excess_return_vs_equal_weight: audit.ew_excess_return,
      hash_drift: null,
      n_bars: null,
      portfolio_diagnostics: null,
    }
  }

  try {
    const data = await loadJson<ApiRunDetail>(`details/${encodeURIComponent(runId)}.json`, signal)
    const audit = data.audit ? normalizeRun(data.audit) : null
    return {
      ...data,
      audit,
      // Stale publish stubs left backtest=null while audit already had OOS Sharpe.
      backtest: data.backtest ?? synthesizeBacktest(audit),
    }
  } catch {
    // Fallback: synthesize a minimal detail from the runs index (still ms-fast).
    const page = await loadStaticRuns({ search: runId, limit: 200 }, signal)
    const audit = page.runs.find((r) => r.run_id === runId) ?? null
    if (!audit) throw new Error(`No static detail for run ${runId}`)
    return {
      run_id: runId,
      audit,
      provenance: {
        git_commit: null,
        git_dirty: audit.git_dirty,
        config_hash: null,
        data_cache_hash: null,
        started_at_utc: audit.started_at_utc,
        finished_at_utc: audit.finished_at_utc,
      },
      holdout: null,
      universe: null,
      backtest: synthesizeBacktest(audit),
    }
  }
}
