import { Download } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { Badge, Button, Card, EmptyPanel, ErrorPanel, Select, Skeleton } from '../components/ui'
import { fetchResults } from '../lib/api'
import { sampleResultRows } from '../lib/demo-data'
import { fmtNum, fmtPct } from '../lib/format'
import type { ApiResultRow, ApiResults, DataState } from '../lib/types'
import { useAutoRefresh } from '../lib/use-auto-refresh'

function median(values: Array<number | null | undefined>): number | null {
  const finite = values.filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
  if (finite.length === 0) return null
  const sorted = [...finite].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

function downloadJson(rows: ApiResultRow[], cohort: string) {
  const blob = new Blob([JSON.stringify({ cohort: cohort || 'all', rows }, null, 2)], { type: 'application/json' })
  const href = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = href
  link.download = `results-${cohort || 'all'}.json`
  link.click()
  URL.revokeObjectURL(href)
}

function ComparisonBars({
  label,
  model,
  equalWeight,
  spy,
  scale,
}: {
  label: string
  model: number | null
  equalWeight: number | null
  spy: number | null
  scale: number
}) {
  const bars = [
    { key: 'Model', value: model, className: 'bg-pine' },
    { key: 'Equal-weight', value: equalWeight, className: 'bg-moss' },
    { key: 'SPY', value: spy, className: 'bg-ink/35' },
  ]
  return (
    <div className="grid grid-cols-[34px_1fr] items-center gap-3">
      <span className="font-mono text-[11px] text-ink/60">{label}</span>
      <div className="space-y-1.5">
        {bars.map((bar) => {
          const value = bar.value ?? 0
          const width = Math.max(2, (Math.abs(value) / scale) * 100)
          return (
            <div key={bar.key} className="grid grid-cols-[88px_1fr_40px] items-center gap-2">
              <span className="truncate text-[10px] text-ink/55">{bar.key}</span>
              <div className="h-2 overflow-hidden rounded-full bg-ink/[.06]">
                <div className={`h-full rounded-full ${bar.className}`} style={{ width: `${width}%` }} />
              </div>
              <span className="text-right font-mono text-[10px]">{fmtNum(bar.value)}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export function ResultsPage() {
  const [state, setState] = useState<DataState<ApiResults>>({ kind: 'loading' })
  const [cohort, setCohort] = useState('')
  const [refreshing, setRefreshing] = useState(false)
  const [refreshError, setRefreshError] = useState<string | null>(null)

  const load = useCallback((signal?: AbortSignal) => {
    setRefreshing(true)
    setRefreshError(null)
    setState((current) => current.kind === 'live' ? current : { kind: 'loading' })
    fetchResults(cohort, signal).then((next) => {
      if (signal?.aborted) return
      setRefreshing(false)
      if (next.kind === 'error') {
        setRefreshError(next.message)
        setState((current) => current.kind === 'live' ? current : next)
      } else {
        setState(next)
      }
    })
  }, [cohort])
  useEffect(() => {
    const controller = new AbortController()
    load(controller.signal)
    return () => controller.abort()
  }, [load])

  const reload = useCallback(() => load(), [load])
  useAutoRefresh(reload, { enabled: state.kind === 'live' || state.kind === 'error', refreshing })

  const rows = useMemo(
    () => (state.kind === 'live' ? state.data.rows : state.kind === 'offline' ? sampleResultRows : []),
    [state],
  )
  const cohorts = state.kind === 'live' ? state.data.cohorts : ['725']

  const perWindow = useMemo(() => {
    const byWindow = new Map<string, ApiResultRow[]>()
    for (const row of rows) {
      if (!byWindow.has(row.window)) byWindow.set(row.window, [])
      byWindow.get(row.window)!.push(row)
    }
    return Array.from(byWindow.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([window, windowRows]) => ({
        window,
        n: windowRows.length,
        sharpe: median(windowRows.map((row) => row.model_sh)),
        ewSharpe: median(windowRows.map((row) => row.ew_sh)),
        spySharpe: median(windowRows.map((row) => row.spy_sh)),
        return: median(windowRows.map((row) => row.model_ret)),
        ewExcess: median(windowRows.map((row) => (
          row.ew_ret == null ? null : row.model_ret - row.ew_ret
        ))),
        spyExcess: median(windowRows.map((row) => (
          row.spy_ret == null ? null : row.model_ret - row.spy_ret
        ))),
        ewSharpeExcess: median(windowRows.map((row) => (
          row.ew_sh == null ? null : row.model_sh - row.ew_sh
        ))),
        spySharpeExcess: median(windowRows.map((row) => (
          row.spy_sh == null ? null : row.model_sh - row.spy_sh
        ))),
      }))
  }, [rows])

  const medianSharpe = median(rows.map((row) => row.model_sh))
  const medianEwSharpe = median(rows.map((row) => row.ew_sh))
  const medianSpySharpe = median(rows.map((row) => row.spy_sh))
  const medianEwExcess = median(rows.map((row) => (
    row.ew_ret == null ? null : row.model_ret - row.ew_ret
  )))
  const medianEwSharpeExcess = median(rows.map((row) => (
    row.ew_sh == null ? null : row.model_sh - row.ew_sh
  )))
  const coverage = state.kind === 'live' ? state.data.coverage : null
  const scoredRuns = coverage?.published_runs ?? new Set(rows.map((row) => row.run_id)).size
  const withBenchmarks = coverage?.runs_with_benchmarks
  const sharpeScale = Math.max(
    1,
    ...perWindow.flatMap((row) => [Math.abs(row.sharpe ?? 0), Math.abs(row.ewSharpe ?? 0), Math.abs(row.spySharpe ?? 0)]),
  )

  return (
    <div className="px-5 py-7 sm:px-8 lg:px-10 lg:py-9">
      <header className="flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
        <div>
          <div className="flex items-center gap-2">
            <p className="font-mono text-[11px] uppercase tracking-[.2em] text-ink/55">Research Operations</p>
            <Badge tone="success">Out-of-sample results</Badge>
          </div>
          <h1 className="font-display mt-2 text-4xl tracking-[-.035em] sm:text-5xl">OOS results</h1>
          <p className="mt-2 text-xs text-ink/60">
            Aggregated holdout performance across walk-forward windows with benchmark comparisons.
          </p>
          {state.kind === 'live' && refreshing && <p className="mt-2 text-[11px] text-ink/55">Refreshing evidence…</p>}
          {state.kind === 'live' && refreshError && (
            <button type="button" onClick={reload} className="mt-2 text-[11px] font-semibold text-amber-800">
              Refresh failed: {refreshError}. Retry
            </button>
          )}
        </div>
        <div className="flex items-center gap-2">
          {cohorts.length > 1 && (
            <Select value={cohort} onChange={(event) => setCohort(event.target.value)} aria-label="Filter by cohort" className="h-10 w-40">
              <option value="">All cohorts</option>
              {cohorts.map((item) => <option key={item} value={item}>Cohort {item}</option>)}
            </Select>
          )}
          <Button variant="secondary" onClick={() => downloadJson(rows, cohort)} disabled={rows.length === 0}>
            <Download size={14} aria-hidden="true" /> Download JSON
          </Button>
        </div>
      </header>

      {state.kind === 'error' && <div className="mt-8"><ErrorPanel message={state.message} onRetry={reload} /></div>}
      {state.kind === 'loading' && (
        <div className="mt-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {Array.from({ length: 4 }).map((_, index) => <Skeleton key={index} className="h-32" />)}
        </div>
      )}
      {state.kind === 'live' && rows.length === 0 && (
        <div className="mt-8">
          <EmptyPanel
            title="No OOS results yet"
            body="No backtest summaries found under Runs/*/ (backtest_summary.json, or final/latest fallbacks). Score holdouts with scripts/backtest.py — post-train backtests write the canonical file automatically."
          />
        </div>
      )}

      {rows.length > 0 && (
        <>
          <section className="mt-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-4" aria-label="Headline metrics">
            {[
              {
                label: 'Scored OOS rows',
                value: String(rows.length),
                note: `${scoredRuns} runs · ${cohorts.length} cohorts available`,
              },
              {
                label: 'Median model Sharpe',
                value: fmtNum(medianSharpe),
                note: `EW ${fmtNum(medianEwSharpe)} · SPY ${fmtNum(medianSpySharpe)}`,
              },
              {
                label: 'Median Sharpe vs EW',
                value: fmtNum(medianEwSharpeExcess),
                note: 'model Sharpe − equal-weight Sharpe',
              },
              {
                label: 'Median excess return vs EW',
                value: fmtPct(medianEwExcess),
                note: 'per-window total return',
              },
            ].map((metric) => (
              <Card key={metric.label} className="p-5">
                <p className="text-[11px] text-ink/60">{metric.label}</p>
                <p className="font-display mt-4 text-3xl">{metric.value}</p>
                <p className="mt-3 text-[11px] text-ink/55">{metric.note}</p>
              </Card>
            ))}
          </section>
          <p className="mt-3 text-[11px] leading-5 text-ink/55">
            Cohort filter lists every cohort with a local backtest summary. Rows are rebuilt from{' '}
            <span className="font-mono">Runs/*/backtest_summary*.json</span>
            {' '}(best preferred; final/latest used when best is absent) and refresh automatically.
            {withBenchmarks != null && withBenchmarks < rows.length
              ? `; ${rows.length - withBenchmarks} older summaries lack EW/SPY sleeves.`
              : '.'}
          </p>

          <section className="mt-4 overflow-hidden rounded-[24px] border border-line bg-paper/85">
            <div className="border-b border-line px-6 py-5">
              <p className="text-sm font-semibold">Per-window medians</p>
              <p className="mt-1 text-[11px] text-ink/60">
                Model Sharpe and return versus equal-weight and SPY sleeves
              </p>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[760px] text-left text-[12px]">
                <caption className="sr-only">Median out-of-sample metrics per walk-forward window</caption>
                <thead className="bg-ink/[.02] font-mono text-[11px] uppercase tracking-wider text-ink/55">
                  <tr>
                    <th scope="col" className="px-6 py-3 font-normal">Window</th>
                    <th scope="col" className="font-normal">Runs</th>
                    <th scope="col" className="font-normal">Model Sharpe</th>
                    <th scope="col" className="font-normal">EW Sharpe</th>
                    <th scope="col" className="font-normal">SPY Sharpe</th>
                    <th scope="col" className="font-normal">vs EW Sharpe</th>
                    <th scope="col" className="font-normal">vs SPY Sharpe</th>
                    <th scope="col" className="font-normal">Return</th>
                    <th scope="col" className="font-normal">vs EW ret</th>
                    <th scope="col" className="px-6 font-normal">vs SPY ret</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {perWindow.map((row) => (
                    <tr key={row.window}>
                      <td className="px-6 py-4 font-semibold">{row.window}</td>
                      <td className="font-mono">{row.n}</td>
                      <td className="font-mono">{fmtNum(row.sharpe)}</td>
                      <td className="font-mono">{fmtNum(row.ewSharpe)}</td>
                      <td className="font-mono">{fmtNum(row.spySharpe)}</td>
                      <td className={`font-mono ${row.ewSharpeExcess != null && row.ewSharpeExcess >= 0 ? 'text-emerald-700' : 'text-red-800'}`}>
                        {fmtNum(row.ewSharpeExcess)}
                      </td>
                      <td className={`font-mono ${row.spySharpeExcess != null && row.spySharpeExcess >= 0 ? 'text-emerald-700' : 'text-red-800'}`}>
                        {fmtNum(row.spySharpeExcess)}
                      </td>
                      <td className="font-mono">{fmtPct(row.return)}</td>
                      <td className={`font-mono ${row.ewExcess != null && row.ewExcess >= 0 ? 'text-emerald-700' : 'text-red-800'}`}>{fmtPct(row.ewExcess)}</td>
                      <td className={`px-6 font-mono ${row.spyExcess != null && row.spyExcess >= 0 ? 'text-emerald-700' : 'text-red-800'}`}>{fmtPct(row.spyExcess)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="mt-4 grid gap-4 xl:grid-cols-2">
            <Card className="p-6">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold">Sharpe comparison by window</p>
                  <p className="mt-1 text-[11px] text-ink/60">Model vs equal-weight vs SPY medians</p>
                </div>
                <div className="flex gap-3 text-[10px] text-ink/55">
                  <span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-pine" /> Model</span>
                  <span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-moss" /> Equal-weight</span>
                  <span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-ink/35" /> SPY</span>
                </div>
              </div>
              <div className="mt-6 space-y-5" role="img" aria-label="Median Sharpe comparison by walk-forward window">
                {perWindow.map((row) => (
                  <ComparisonBars
                    key={row.window}
                    label={row.window}
                    model={row.sharpe}
                    equalWeight={row.ewSharpe}
                    spy={row.spySharpe}
                    scale={sharpeScale}
                  />
                ))}
              </div>
            </Card>

            <Card className="p-6">
              <p className="text-sm font-semibold">Sharpe edge vs benchmarks</p>
              <p className="mt-1 text-[11px] text-ink/60">Model Sharpe minus equal-weight / SPY Sharpe</p>
              <div className="mt-6 space-y-4" role="img" aria-label="Median Sharpe excess versus benchmarks">
                {perWindow.map((row) => {
                  const max = Math.max(
                    1,
                    ...perWindow.flatMap((item) => [
                      Math.abs(item.ewSharpeExcess ?? 0),
                      Math.abs(item.spySharpeExcess ?? 0),
                    ]),
                  )
                  return (
                    <div key={row.window} className="grid grid-cols-[34px_1fr] items-center gap-3">
                      <span className="font-mono text-[11px] text-ink/60">{row.window}</span>
                      <div className="space-y-1.5">
                        {[
                          { label: 'vs EW', value: row.ewSharpeExcess },
                          { label: 'vs SPY', value: row.spySharpeExcess },
                        ].map((item) => {
                          const value = item.value ?? 0
                          const width = Math.max(2, (Math.abs(value) / max) * 100)
                          return (
                            <div key={item.label} className="grid grid-cols-[52px_1fr_44px] items-center gap-2">
                              <span className="text-[10px] text-ink/55">{item.label}</span>
                              <div className="h-2 overflow-hidden rounded-full bg-ink/[.06]">
                                <div
                                  className={`h-full rounded-full ${value >= 0 ? 'bg-pine' : 'bg-coral'}`}
                                  style={{ width: `${width}%` }}
                                />
                              </div>
                              <span className={`text-right font-mono text-[10px] ${value >= 0 ? 'text-emerald-700' : 'text-red-800'}`}>
                                {fmtNum(item.value)}
                              </span>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )
                })}
              </div>
            </Card>
          </section>

          <section className="mt-4 rounded-[24px] bg-pine p-6 text-cream">
            <p className="text-sm font-semibold">Methodology</p>
            <p className="mt-3 max-w-3xl text-[12px] leading-6 text-cream/70">
              Each result row is a single out-of-sample backtest of a best-checkpoint model on a reserved
              chronological holdout (two-year walk-forward windows W1–W5). Equal-weight and SPY Sharpes come
              from the detailed backtest benchmark sleeves when present. Checkpoints are selected without
              touching the holdout; every holdout read is logged in the global OOS ledger.
            </p>
          </section>
        </>
      )}
    </div>
  )
}
