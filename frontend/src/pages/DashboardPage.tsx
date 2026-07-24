import { Activity, AlertTriangle, Clock3, CreditCard, FileCheck2, Plus } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Badge, Card, EmptyPanel, ErrorPanel, Skeleton } from '../components/ui'
import { fetchDashboard, fetchMandates } from '../lib/api'
import { sampleResultRows, sampleRuns, sampleSummary } from '../lib/demo-data'
import { fmtNum, fmtSteps, statusLabel, statusTone } from '../lib/format'
import type { ApiDashboard, ApiResultRow, DataState } from '../lib/types'
import type { WorkflowMandate } from '../lib/mandate-requests'

function useDashboardData() {
  const [state, setState] = useState<DataState<ApiDashboard>>({ kind: 'loading' })
  const [refreshing, setRefreshing] = useState(false)
  const [refreshError, setRefreshError] = useState<string | null>(null)

  const load = useCallback((signal?: AbortSignal) => {
    setRefreshing(true)
    setRefreshError(null)
    setState((current) => current.kind === 'live' ? current : { kind: 'loading' })
    fetchDashboard(signal).then((next) => {
      if (signal?.aborted) return
      setRefreshing(false)
      if (next.kind === 'error') {
        setRefreshError(next.message)
        setState((current) => current.kind === 'live' ? current : next)
      } else {
        setState(next)
      }
    })
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    load(controller.signal)
    return () => controller.abort()
  }, [load])
  return { state, refreshing, refreshError, reload: () => load() }
}

/** Median model Sharpe per window, computed from the served cohort table. */
function windowSharpes(rows: ApiResultRow[]): { window: string; sharpe: number }[] {
  const byWindow = new Map<string, number[]>()
  for (const row of rows) {
    if (!byWindow.has(row.window)) byWindow.set(row.window, [])
    byWindow.get(row.window)!.push(row.model_sh)
  }
  return Array.from(byWindow.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([window, values]) => {
      const sorted = [...values].sort((a, b) => a - b)
      const mid = Math.floor(sorted.length / 2)
      const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
      return { window, sharpe: Number(median.toFixed(2)) }
    })
}

export function DashboardPage() {
  const { state, refreshing, refreshError, reload } = useDashboardData()
  const [workflowState, setWorkflowState] = useState<DataState<WorkflowMandate[]>>({ kind: 'loading' })

  useEffect(() => {
    const controller = new AbortController()
    fetchMandates(controller.signal, 'operator').then((next) => {
      if (!controller.signal.aborted) setWorkflowState(next)
    })
    return () => controller.abort()
  }, [])

  const summaryData = state.kind === 'live' ? state.data.summary : state.kind === 'offline' ? sampleSummary : null
  const runData = state.kind === 'live' ? state.data.recent_runs : state.kind === 'offline' ? sampleRuns : null
  const chartRows = state.kind === 'live' ? state.data.window_sharpes : state.kind === 'offline' ? windowSharpes(sampleResultRows) : []
  const mandates = workflowState.kind === 'live' ? workflowState.data : []
  const awaitingReview = mandates.filter((mandate) => mandate.state === 'draft').length
  const awaitingPayment = mandates.filter((mandate) => ['quote_issued', 'checkout'].includes(mandate.state)).length
  const activeBuilds = mandates.filter((mandate) => ['queued', 'training', 'validation', 'governed_oos_evaluation'].includes(mandate.state)).length
  const blockingWarnings = (warnings: string[]) => warnings.filter((warning) =>
    warning.startsWith('curriculum_preflight_failed')
    || warning.includes('hash_drift')
    || warning.includes('missing_vec_normalize'),
  )
  const needsIntervention = runData?.filter((run) => (
    run.training_status === 'interrupted' || blockingWarnings(run.warnings).length > 0
  )).length ?? 0
  const reportsReady = mandates.filter((mandate) => mandate.state === 'governed_oos_evaluation').length
  const systemAlerts = (runData ?? []).flatMap((run) => [
    ...(run.training_status === 'interrupted' ? [`${run.run_id}: training interrupted`] : []),
    ...blockingWarnings(run.warnings).map((warning) => `${run.run_id}: ${warning}`),
  ]).slice(0, 6)

  return (
    <div className="px-5 py-7 sm:px-8 lg:px-10 lg:py-9">
      <header className="flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
        <div>
          <div className="flex items-center gap-2">
            <p className="font-mono text-[11px] uppercase tracking-[.2em] text-ink/55">Overview</p>
            <Badge tone={state.kind === 'live' ? 'success' : state.kind === 'error' ? 'danger' : 'warning'}>
              {state.kind === 'live' ? 'API connected' : state.kind === 'error' ? 'API unavailable' : state.kind === 'offline' ? 'API not configured' : 'Connecting'}
            </Badge>
          </div>
          <h1 className="font-display mt-2 text-4xl tracking-[-.035em] sm:text-5xl">Operations overview</h1>
          <p className="mt-2 text-xs text-ink/60">
            Training progress, out-of-sample performance, and walk-forward evidence across your research runs.
          </p>
          {state.kind === 'live' && refreshing && <p className="mt-2 text-[11px] text-ink/55">Refreshing current data…</p>}
          {state.kind === 'live' && refreshError && (
            <button type="button" onClick={reload} className="mt-2 text-[11px] font-semibold text-amber-800">
              Refresh failed: {refreshError}. Retry
            </button>
          )}
        </div>
        <Link
          to="/ops/mandates/new"
          className="inline-flex h-11 items-center justify-center gap-2 self-start rounded-full bg-pine px-5 text-xs font-semibold text-cream shadow-[0_10px_25px_rgba(16,43,35,.16)] hover:bg-[#173c31] sm:self-auto"
        >
          <Plus size={15} aria-hidden="true" /> New mandate
        </Link>
      </header>

      {state.kind === 'error' ? (
        <div className="mt-8"><ErrorPanel message={state.message} onRetry={reload} /></div>
      ) : (
        <section className="mt-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-5" aria-label="Operations work queue">
          {summaryData ? (
            [
              { label: 'Mandates awaiting review', value: workflowState.kind === 'live' ? String(awaitingReview) : '—', note: 'require data and policy preflight', icon: Clock3 },
              { label: 'Quotes awaiting payment', value: workflowState.kind === 'live' ? String(awaitingPayment) : '—', note: 'payment advances by webhook only', icon: CreditCard },
              { label: 'Active builds', value: workflowState.kind === 'live' ? String(activeBuilds) : '—', note: `${summaryData.active_runs} active training runs`, icon: Activity },
              { label: 'Needs intervention', value: String(needsIntervention), note: 'interrupted jobs or operational blockers', icon: AlertTriangle },
              { label: 'Reports ready for release', value: workflowState.kind === 'live' ? String(reportsReady) : '—', note: 'governed OOS completed', icon: FileCheck2 },
            ].map((item) => (
              <Card key={item.label} className="p-5">
                <span className="grid h-9 w-9 place-items-center rounded-xl bg-pine text-mint"><item.icon size={16} aria-hidden="true" /></span>
                <p className="font-display mt-6 text-3xl tracking-[-.03em]">{item.value}</p>
                <p className="mt-1 text-xs font-semibold">{item.label}</p>
                <p className="mt-2 text-[11px] text-ink/60">{item.note}</p>
              </Card>
            ))
          ) : (
            Array.from({ length: 5 }).map((_, index) => <Skeleton key={index} className="h-40" />)
          )}
        </section>
      )}

      {systemAlerts.length > 0 && (
        <Card className="mt-4 border-amber-700/15 bg-amber-50/70 p-5">
          <div className="flex items-center gap-2">
            <AlertTriangle size={15} className="text-amber-800" aria-hidden="true" />
            <h2 className="text-sm font-semibold text-amber-950">Recent system alerts</h2>
          </div>
          <ul className="mt-3 grid gap-2 sm:grid-cols-2">
            {systemAlerts.map((alert) => <li key={alert} className="font-mono text-[11px] text-amber-900">{alert}</li>)}
          </ul>
        </Card>
      )}

      <section className="mt-4 grid gap-4 xl:grid-cols-[1.35fr_.85fr]">
        <Card className="overflow-hidden">
          <div className="flex items-center justify-between border-b border-line px-6 py-5">
            <div>
              <p className="text-sm font-semibold">Training pipeline</p>
              <p className="mt-1 text-[11px] text-ink/60">Most recent training runs</p>
            </div>
            <Link to="/ops/runs" className="text-[11px] font-semibold text-pine">View all →</Link>
          </div>
          {state.kind === 'loading' && <div className="space-y-3 p-6">{Array.from({ length: 4 }).map((_, index) => <Skeleton key={index} className="h-12" />)}</div>}
          {state.kind === 'error' && <div className="p-6"><ErrorPanel message={state.message} onRetry={reload} /></div>}
          {runData && runData.length === 0 && (
            <div className="p-6">
              <EmptyPanel
                title="No runs yet"
                body="Train your first model with scripts/train.py; completed runs appear here automatically from their manifests."
              />
            </div>
          )}
          {runData && runData.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] text-left">
                <caption className="sr-only">Recent training runs with status, progress, and out-of-sample Sharpe</caption>
                <thead className="border-b border-line bg-ink/[.015] font-mono text-[11px] uppercase tracking-wider text-ink/55">
                  <tr>
                    <th scope="col" className="px-6 py-3 font-normal">Run</th>
                    <th scope="col" className="px-4 font-normal">Status</th>
                    <th scope="col" className="px-4 font-normal">Progress</th>
                    <th scope="col" className="px-4 font-normal">Best step</th>
                    <th scope="col" className="px-6 font-normal">OOS Sharpe</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {runData.slice(0, 6).map((run) => (
                    <tr key={run.run_id} className="text-[12px] hover:bg-white/50">
                      <td className="px-6 py-4">
                        <span className="font-semibold">{run.run_id}</span>
                        <span className="mt-1 block text-[11px] text-ink/55">{run.window ?? '—'}</span>
                      </td>
                      <td className="px-4"><Badge tone={statusTone(run.training_status)}>{statusLabel(run.training_status)}</Badge></td>
                      <td className="px-4">
                        {run.progress_pct == null ? (
                          <span className="text-ink/50">—</span>
                        ) : (
                          <div className="flex items-center gap-3">
                            <div className="h-1.5 w-20 overflow-hidden rounded-full bg-ink/8" role="progressbar" aria-valuenow={run.progress_pct} aria-valuemin={0} aria-valuemax={100} aria-label={`${run.run_id} progress`}>
                              <div className="h-full rounded-full bg-pine" style={{ width: `${run.progress_pct}%` }} />
                            </div>
                            <span className="font-mono text-[11px] text-ink/60">{run.progress_pct}%</span>
                          </div>
                        )}
                      </td>
                      <td className="px-4 font-mono text-[11px] text-ink/70">{fmtSteps(run.best_eval_step)}</td>
                      <td className="px-6 font-mono text-[11px]">{fmtNum(run.oos_sharpe)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card className="p-6">
          <p className="text-sm font-semibold">Median OOS Sharpe by window</p>
          <p className="mt-1 text-[11px] text-ink/60">Median out-of-sample Sharpe by walk-forward window</p>
          {state.kind === 'loading' && <Skeleton className="mt-5 h-[260px]" />}
          {state.kind === 'error' && <p className="mt-5 text-xs text-ink/60">Unavailable — cohort table could not be loaded.</p>}
          {chartRows.length > 0 && (
            <>
              <div className="mt-6 space-y-3" role="img" aria-label="Median out-of-sample Sharpe by walk-forward window">
                {chartRows.map((row) => {
                  const max = Math.max(1, ...chartRows.map((item) => Math.abs(item.sharpe)))
                  const width = Math.max(2, Math.abs(row.sharpe) / max * 100)
                  return (
                    <div key={row.window} className="grid grid-cols-[34px_1fr_42px] items-center gap-3">
                      <span className="font-mono text-[11px] text-ink/60">{row.window}</span>
                      <div className="h-3 overflow-hidden rounded-full bg-ink/[.06]">
                        <div
                          className={`h-full rounded-full ${row.sharpe >= 0 ? 'bg-pine' : 'bg-coral'}`}
                          style={{ width: `${width}%` }}
                        />
                      </div>
                      <span className="text-right font-mono text-[11px]">{row.sharpe.toFixed(2)}</span>
                    </div>
                  )
                })}
              </div>
              <p className="sr-only">
                Median out-of-sample Sharpe per walk-forward window: {chartRows.map((row) => `${row.window}: ${row.sharpe}`).join(', ')}
              </p>
            </>
          )}
          {state.kind === 'live' && chartRows.length === 0 && (
            <p className="mt-5 text-xs text-ink/60">Unavailable — no cohort comparison table has been generated yet.</p>
          )}
        </Card>
      </section>

      <p className="mt-5 text-[11px] leading-4 text-ink/55">
        OOS metrics are measurements on reserved holdouts, not investment recommendations.
      </p>
    </div>
  )
}
