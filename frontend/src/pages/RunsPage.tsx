import { ChevronDown, Search } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Badge, Card, EmptyPanel, ErrorPanel, Input, Skeleton } from '../components/ui'
import { fetchRunDetail, fetchRuns } from '../lib/api'
import { sampleRuns } from '../lib/demo-data'
import { fmtDate, fmtNum, fmtPct, fmtSteps, statusLabel, statusTone } from '../lib/format'
import type { ApiRun, ApiRunDetail, ApiRunsPage, DataState } from '../lib/types'
import { cn } from '../lib/utils'

const filters = ['All', 'Completed', 'In progress', 'Interrupted'] as const
type Filter = (typeof filters)[number]
const PAGE_SIZE = 25

function matchesFilter(run: ApiRun, filter: Filter): boolean {
  if (filter === 'All') return true
  return statusLabel(run.training_status) === filter
}

function filterToStatus(filter: Filter): '' | 'completed' | 'active' | 'interrupted' {
  if (filter === 'Completed') return 'completed'
  if (filter === 'In progress') return 'active'
  if (filter === 'Interrupted') return 'interrupted'
  return ''
}

/** Only operational blockers belong in the list “needs attention” path. */
function blockingRunWarnings(run: ApiRun): string[] {
  return run.warnings.filter((warning) =>
    warning.startsWith('curriculum_preflight_failed')
    || warning.includes('hash_drift')
    || warning.includes('missing_vec_normalize'),
  )
}

function ProvenanceRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-line py-2 last:border-0">
      <span className="shrink-0 text-[11px] text-ink/60">{label}</span>
      <span className="truncate font-mono text-[11px] text-ink/80" title={value}>{value}</span>
    </div>
  )
}

function RunDetailPanel({ runId }: { runId: string }) {
  const [detail, setDetail] = useState<DataState<ApiRunDetail>>({ kind: 'loading' })

  useEffect(() => {
    const controller = new AbortController()
    fetchRunDetail(runId, controller.signal).then((state) => {
      if (!controller.signal.aborted) setDetail(state)
    })
    return () => controller.abort()
  }, [runId])

  if (detail.kind === 'loading') return <Skeleton className="h-32" />
  if (detail.kind === 'offline') {
    return <p className="text-xs text-ink/60">Run detail is available when the research API is connected.</p>
  }
  if (detail.kind === 'error') return <p role="alert" className="text-xs text-red-900">Could not load run detail: {detail.message}</p>

  const { provenance, backtest, holdout } = detail.data
  const auditWarnings = detail.data.audit?.warnings ?? []
  const blockers = [
    ...(detail.data.audit ? blockingRunWarnings(detail.data.audit) : []),
    ...(detail.data.audit?.training_status === 'interrupted' ? ['Training was interrupted'] : []),
    ...(backtest?.hash_drift ? ['Backtest artifact hash drift detected'] : []),
  ]
  const auditNotes = [
    ...auditWarnings.filter((warning) => !blockers.some((blocker) => blocker.includes(warning) || warning.includes(blocker))),
    ...(provenance.git_dirty ? ['Training source was dirty; preserved provenance is required for release.'] : []),
  ]
  const releaseGatePassed = (
    detail.data.audit?.training_status === 'completed'
    && blockers.length === 0
    && provenance.git_dirty !== true
    && backtest != null
  )
  return (
    <div>
      {blockers.length > 0 && (
        <section className="mb-5 rounded-xl border border-amber-700/15 bg-amber-50 p-4">
          <p className="text-xs font-semibold text-amber-950">Operational blockers</p>
          <ul className="mt-2 space-y-1">
            {blockers.map((warning) => <li key={warning} className="text-[11px] text-amber-900">{warning}</li>)}
          </ul>
        </section>
      )}
      {auditNotes.length > 0 && (
        <section className="mb-5 rounded-xl border border-line bg-white/55 p-4">
          <p className="text-xs font-semibold">Historical audit notes</p>
          <ul className="mt-2 space-y-1">
            {auditNotes.map((warning) => <li key={warning} className="text-[11px] text-ink/60">{warning.replaceAll('_', ' ')}</li>)}
          </ul>
        </section>
      )}
      <div className="grid gap-5 lg:grid-cols-3">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[.15em] text-ink/55">Provenance</p>
          <div className="mt-2">
            <ProvenanceRow label="Git commit" value={provenance.git_commit ?? 'unavailable'} />
            <ProvenanceRow label="Source dirty" value={provenance.git_dirty == null ? 'unavailable' : String(provenance.git_dirty)} />
            <ProvenanceRow label="Config hash" value={provenance.config_hash ?? 'unavailable'} />
            <ProvenanceRow label="Data-cache hash" value={provenance.data_cache_hash ?? 'unavailable'} />
            <ProvenanceRow label="Started" value={fmtDate(provenance.started_at_utc)} />
            <ProvenanceRow label="Finished" value={fmtDate(provenance.finished_at_utc)} />
            {holdout && (
              <ProvenanceRow
                label="Holdout"
                value={`${String(holdout.holdout_start ?? '?')} → ${String(holdout.holdout_end ?? '?')}`}
              />
            )}
          </div>
        </div>
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[.15em] text-ink/55">Research evidence</p>
          {backtest ? (
            <div className="mt-2">
              <ProvenanceRow label="Checkpoint" value={backtest.checkpoint_label ?? 'unavailable'} />
              <ProvenanceRow label="Total return" value={fmtPct(backtest.total_return)} />
              <ProvenanceRow label="Sharpe / DSR" value={`${fmtNum(backtest.sharpe)} / ${fmtNum(backtest.deflated_sharpe)}`} />
              <ProvenanceRow label="Max drawdown" value={fmtPct(backtest.max_drawdown)} />
              <ProvenanceRow label="Excess vs equal-weight" value={fmtPct(backtest.excess_return_vs_equal_weight)} />
              <ProvenanceRow label="OOS trials (window)" value={String(backtest.oos_trials_for_window ?? 'unavailable')} />
              <ProvenanceRow
                label="Hash drift"
                value={backtest.hash_drift ? JSON.stringify(backtest.hash_drift) : 'none detected'}
              />
            </div>
          ) : (
            <p className="mt-2 text-xs text-ink/60">
              No OOS backtest recorded for this run. Backtests spend holdout budget; launch them deliberately via
              scripts/backtest.py.
            </p>
          )}
        </div>
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[.15em] text-ink/55">Release gate</p>
          <div className="mt-2 rounded-xl bg-ink/[.035] p-4">
            <Badge tone={releaseGatePassed ? 'success' : 'warning'}>
              {releaseGatePassed ? 'Evidence checks passed' : 'Not release-ready'}
            </Badge>
            <ul className="mt-3 space-y-2 text-[11px] text-ink/65">
              <li>Training completed: {detail.data.audit?.training_status === 'completed' ? 'yes' : 'no'}</li>
              <li>Operational blockers absent: {blockers.length === 0 ? 'yes' : 'no'}</li>
              <li>Clean or fully preserved source: {provenance.git_dirty === true ? 'no' : 'yes'}</li>
              <li>Governed OOS evidence present: {backtest ? 'yes' : 'no'}</li>
            </ul>
            <p className="mt-3 text-[10px] leading-4 text-ink/50">This view reports evidence only; release authorization belongs to the paid immutable mandate workflow.</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export function RunsPage() {
  const [runsState, setRunsState] = useState<DataState<ApiRunsPage>>({ kind: 'loading' })
  const [refreshing, setRefreshing] = useState(false)
  const [refreshError, setRefreshError] = useState<string | null>(null)
  const [filter, setFilter] = useState<Filter>('All')
  const [search, setSearch] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [offset, setOffset] = useState(0)
  const [expanded, setExpanded] = useState<string | null>(null)
  /** Query key of the data currently shown in the UI (updated only after a successful fetch). */
  const loadedQueryRef = useRef<string | null>(null)
  const requestIdRef = useRef(0)

  const status = filterToStatus(filter)
  const queryKey = `${status}|${debouncedSearch}|${offset}`

  const load = useCallback((signal?: AbortSignal) => {
    const requestId = ++requestIdRef.current
    setRefreshing(true)
    setRefreshError(null)
    setRunsState((current) => (
      current.kind === 'live' && loadedQueryRef.current === queryKey
        ? current
        : { kind: 'loading' }
    ))
    fetchRuns({ search: debouncedSearch, status, offset, limit: PAGE_SIZE }, signal).then((next) => {
      if (signal?.aborted || requestId !== requestIdRef.current) return
      setRefreshing(false)
      if (next.kind === 'loading') return
      if (next.kind === 'error') {
        setRefreshError(next.message)
        setRunsState((current) => (
          current.kind === 'live' && loadedQueryRef.current === queryKey ? current : next
        ))
        return
      }
      loadedQueryRef.current = queryKey
      setRunsState(next)
    })
  }, [debouncedSearch, offset, queryKey, status])

  useEffect(() => {
    const controller = new AbortController()
    load(controller.signal)
    return () => controller.abort()
  }, [load])

  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedSearch(search.trim()), 250)
    return () => window.clearTimeout(timer)
  }, [search])

  useEffect(() => {
    setOffset(0)
    setExpanded(null)
  }, [debouncedSearch, filter])

  const allRuns = useMemo(
    () => (runsState.kind === 'live' ? runsState.data.runs : runsState.kind === 'offline' ? sampleRuns : []),
    [runsState],
  )
  const runs = useMemo(
    () =>
      runsState.kind === 'offline'
        ? allRuns.filter(
            (run) => matchesFilter(run, filter) && run.run_id.toLowerCase().includes(debouncedSearch.toLowerCase()),
          )
        : allRuns,
    [allRuns, debouncedSearch, filter, runsState.kind],
  )

  const counts = runsState.kind === 'live'
    ? runsState.data.counts
    : {
        all: allRuns.length,
        completed: allRuns.filter((run) => run.training_status === 'completed').length,
        with_backtest: allRuns.filter((run) => run.has_backtest).length,
      }
  const total = runsState.kind === 'live' ? runsState.data.total : runs.length
  const groupedRuns = useMemo(() => {
    const groups = new Map<string, ApiRun[]>()
    for (const run of runs) {
      const match = /^W\d+_([^_]+)/i.exec(run.run_id)
      const cohort = match?.[1] ?? 'ungrouped'
      if (!groups.has(cohort)) groups.set(cohort, [])
      groups.get(cohort)!.push(run)
    }
    return Array.from(groups.entries())
      .sort(([a], [b]) => b.localeCompare(a, undefined, { numeric: true }))
      .map(([cohort, cohortRuns]) => ({
        cohort,
        runs: cohortRuns,
        completed: cohortRuns.filter((run) => run.training_status === 'completed').length,
        blocked: cohortRuns.filter((run) => run.training_status === 'interrupted' || blockingRunWarnings(run).length > 0).length,
      }))
  }, [runs])

  return (
    <div className="px-5 py-7 sm:px-8 lg:px-10 lg:py-9">
      <header className="flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
        <div>
          <div className="flex items-center gap-2">
            <p className="font-mono text-[11px] uppercase tracking-[.2em] text-ink/55">Research operations</p>
            <Badge tone="success">Training pipeline</Badge>
          </div>
          <h1 className="font-display mt-2 text-4xl tracking-[-.035em] sm:text-5xl">Runs</h1>
          <p className="mt-2 text-xs text-ink/60">
            Every row reflects a training run with status, progress, and out-of-sample evidence on expand.
          </p>
          {runsState.kind === 'live' && refreshing && <p className="mt-2 text-[11px] text-ink/55">Refreshing runs…</p>}
          {runsState.kind === 'live' && refreshError && (
            <button type="button" onClick={() => load()} className="mt-2 text-[11px] font-semibold text-amber-800">
              Refresh failed: {refreshError}. Retry
            </button>
          )}
        </div>
      </header>

      <section className="mt-8 grid gap-3 sm:grid-cols-3" aria-label="Run counts">
        {[
          ['Runs discovered', String(counts.all)],
          ['Completed', String(counts.completed)],
          ['With OOS backtest', String(counts.with_backtest)],
        ].map(([label, value]) => (
          <Card key={label} className="p-5">
            <p className="text-[11px] text-ink/60">{label}</p>
            <p className="font-display mt-3 text-3xl">{runsState.kind === 'loading' ? '…' : value}</p>
          </Card>
        ))}
      </section>

      <section className="mt-5 overflow-hidden rounded-[24px] border border-line bg-paper/85">
        <div className="flex flex-col gap-4 border-b border-line p-5 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex overflow-x-auto" role="tablist" aria-label="Filter runs by status">
            {filters.map((item) => (
              <button
                key={item}
                type="button"
                role="tab"
                aria-selected={filter === item}
                onClick={() => setFilter(item)}
                className={cn(
                  'whitespace-nowrap rounded-full px-3.5 py-2 text-[11px] font-semibold',
                  filter === item ? 'bg-pine text-mint' : 'text-ink/60 hover:text-ink',
                )}
              >
                {item}
              </button>
            ))}
          </div>
          <div className="relative">
            <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-ink/60" aria-hidden="true" />
            <Input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search run id"
              aria-label="Search runs by id"
              className="h-9 w-full pl-8 sm:w-56"
            />
          </div>
        </div>

        {runsState.kind === 'loading' && (
          <div className="space-y-3 p-5">{Array.from({ length: 5 }).map((_, index) => <Skeleton key={index} className="h-16" />)}</div>
        )}
        {runsState.kind === 'error' && <div className="p-5"><ErrorPanel message={runsState.message} onRetry={() => load()} /></div>}
        {runsState.kind !== 'loading' && runsState.kind !== 'error' && runs.length === 0 && (
          <div className="p-5">
            <EmptyPanel
              title={counts.all === 0 ? 'No runs found' : 'No runs match this view'}
              body={
                counts.all === 0
                  ? 'Runs/ contains no run directories with manifests yet. Start one with scripts/train.py.'
                  : 'Adjust the status filter or the search text.'
              }
            />
          </div>
        )}

        <div className="divide-y divide-line">
          {groupedRuns.map((group) => (
            <section key={group.cohort}>
              <div className="flex flex-wrap items-center justify-between gap-3 bg-ink/[.035] px-5 py-3">
                <div>
                  <h2 className="text-xs font-semibold">
                    {group.cohort === 'ungrouped' ? 'Ungrouped runs' : `Cohort ${group.cohort}`}
                  </h2>
                  <p className="mt-0.5 text-[11px] text-ink/55">
                    {group.runs.length} jobs on this page · {group.completed} complete
                    {group.blocked > 0 ? ` · ${group.blocked} blocked` : ''}
                  </p>
                </div>
                {group.blocked > 0 && <Badge tone="warning">Needs attention</Badge>}
              </div>
              <div className="divide-y divide-line">
                {group.runs.map((run) => {
                  const isOpen = expanded === run.run_id
                  const blockers = blockingRunWarnings(run)
                  return (
                    <article key={run.run_id}>
                      <button
                        type="button"
                        onClick={() => setExpanded(isOpen ? null : run.run_id)}
                        aria-expanded={isOpen}
                        aria-label={`${isOpen ? 'Collapse' : 'Expand'} details for run ${run.run_id}`}
                        className="grid w-full gap-4 p-5 text-left hover:bg-white/55 lg:grid-cols-[1.2fr_.7fr_1.1fr_.8fr_28px] lg:items-center"
                      >
                        <div className="min-w-0">
                          <span className="block truncate text-xs font-semibold">{run.run_id}</span>
                          <span className="mt-1 block text-[11px] text-ink/55">
                            {run.window ?? '—'}
                            {run.labels.length > 0 && ` · ${run.labels.slice(0, 2).join(', ')}`}
                          </span>
                        </div>
                        <div>
                          <Badge tone={statusTone(run.training_status)}>{statusLabel(run.training_status)}</Badge>
                          {blockers.length > 0 && (
                            <span className="mt-1.5 block text-[11px] text-amber-800">
                              {blockers.length} operational issue{blockers.length > 1 ? 's' : ''}
                            </span>
                          )}
                        </div>
                        <div>
                          {run.progress_pct == null ? (
                            <span className="text-[11px] text-ink/50">Progress unavailable</span>
                          ) : (
                            <>
                              <div className="mb-1.5 flex items-center justify-between text-[11px]">
                                <span className="text-ink/55">{fmtSteps(run.elapsed_timesteps)} / {fmtSteps(run.nominal_timesteps)}</span>
                                <span className="font-mono">{run.progress_pct}%</span>
                              </div>
                              <div
                                className="h-1.5 overflow-hidden rounded-full bg-ink/8"
                                role="progressbar"
                                aria-valuenow={run.progress_pct}
                                aria-valuemin={0}
                                aria-valuemax={100}
                                aria-label={`${run.run_id} training progress`}
                              >
                                <div className="h-full rounded-full bg-pine" style={{ width: `${run.progress_pct}%` }} />
                              </div>
                            </>
                          )}
                        </div>
                        <div className="flex gap-6 text-[11px] lg:block">
                          <div>
                            <span className="text-ink/55">OOS Sharpe</span>
                            <span className="ml-2 font-mono lg:ml-0 lg:mt-0.5 lg:block">{fmtNum(run.oos_sharpe)}</span>
                          </div>
                          <div className="lg:mt-1.5">
                            <span className="text-ink/55">DSR</span>
                            <span className="ml-2 font-mono lg:ml-0 lg:mt-0.5 lg:block">{fmtNum(run.oos_deflated_sharpe)}</span>
                          </div>
                        </div>
                        <ChevronDown size={16} aria-hidden="true" className={cn('text-ink/60 transition-transform', isOpen && 'rotate-180')} />
                      </button>
                      {isOpen && (
                        <div className="overflow-hidden">
                          <div className="border-t border-line bg-white/45 p-5">
                            <RunDetailPanel runId={run.run_id} />
                          </div>
                        </div>
                      )}
                    </article>
                  )
                })}
              </div>
            </section>
          ))}
        </div>
        {runsState.kind === 'live' && total > 0 && (
          <div className="flex items-center justify-between border-t border-line px-5 py-4">
            <p className="text-[11px] text-ink/55">
              {offset + 1}–{Math.min(offset + PAGE_SIZE, total)} of {total} matching runs
              {filter !== 'All' ? ` · filter: ${filter}` : ''}
            </p>
            <div className="flex gap-2">
              <button
                type="button"
                disabled={offset === 0}
                onClick={() => setOffset((current) => Math.max(0, current - PAGE_SIZE))}
                className="rounded-full border border-line px-3 py-1.5 text-[11px] font-semibold disabled:opacity-40"
              >
                Previous
              </button>
              <button
                type="button"
                disabled={offset + PAGE_SIZE >= total}
                onClick={() => setOffset((current) => current + PAGE_SIZE)}
                className="rounded-full border border-line px-3 py-1.5 text-[11px] font-semibold disabled:opacity-40"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </section>

      <p className="mt-5 text-[11px] text-ink/55">
        Run launch and cancellation are operator actions; this console is read-only by design.
      </p>
    </div>
  )
}
