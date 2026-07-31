import { Braces, RefreshCw, Server, SlidersHorizontal } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { Badge, Button, Card, EmptyPanel, ErrorPanel, Skeleton } from '../components/ui'
import { fetchMandates, performMandateAction } from '../lib/api'
import { fmtDate } from '../lib/format'
import { formatTradingCapital, type WorkflowMandate } from '../lib/mandate-requests'
import type { DataState } from '../lib/types'

const actionLabels: Record<string, string> = {
  run_preflight: 'Run data preflight',
  issue_quote: 'Issue server quote',
  create_checkout: 'Create checkout',
  queue_training: 'Queue training',
  start_training: 'Mark training started',
  start_validation: 'Start validation',
  authorize_oos_evaluation: 'Authorize governed OOS',
  release_report: 'Release report',
  cancel: 'Cancel mandate',
}

export function DeveloperRequestsPage() {
  const [state, setState] = useState<DataState<WorkflowMandate[]>>({ kind: 'loading' })
  const [acting, setActing] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)

  const load = useCallback((signal?: AbortSignal) => {
    setRefreshing(true)
    setState((current) => current.kind === 'live' ? current : { kind: 'loading' })
    fetchMandates(signal, 'operator').then((next) => {
      if (signal?.aborted || next.kind === 'loading') return
      setRefreshing(false)
      setState((current) => next.kind === 'error' && current.kind === 'live' ? current : next)
      if (next.kind === 'error') setActionError(next.message)
      else setActionError(null)
    })
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    load(controller.signal)
    return () => controller.abort()
  }, [load])

  const requests = state.kind === 'live'
    ? state.data.filter((mandate) => mandate.state !== 'cancelled')
    : []

  const performAction = async (request: WorkflowMandate, action: string) => {
    setActing(`${request.id}:${action}`)
    setActionError(null)
    try {
      const updated = await performMandateAction(request.id, action)
      setState((current) => current.kind === 'live'
        ? { kind: 'live', data: current.data.map((item) => item.id === updated.id ? updated : item) }
        : current)
    } catch (error) {
      setActionError(error instanceof Error ? error.message : 'Workflow action failed')
    } finally {
      setActing(null)
    }
  }

  return (
    <div className="px-5 py-7 sm:px-8 lg:px-10 lg:py-9">
      <header className="flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
        <div>
          <div className="flex items-center gap-2">
            <p className="font-mono text-[11px] uppercase tracking-[.2em] text-ink/55">Research Operations</p>
            <Badge tone={state.kind === 'live' ? 'success' : state.kind === 'error' ? 'danger' : 'warning'}>
              <Server size={11} aria-hidden="true" />
              {state.kind === 'live' ? 'API connected' : state.kind === 'error' ? 'API unavailable' : 'Connecting'}
            </Badge>
          </div>
          <h1 className="font-display mt-2 text-4xl tracking-[-.035em] sm:text-5xl">Mandate requests</h1>
          <p className="mt-2 max-w-2xl text-xs leading-5 text-ink/60">
            Investor-submitted mandates and their requested model configurations, loaded from the backend API.
          </p>
        </div>
        <Button variant="secondary" onClick={() => load()} disabled={state.kind === 'loading' || refreshing}>
          <RefreshCw size={14} aria-hidden="true" /> {refreshing && state.kind === 'live' ? 'Refreshing…' : 'Refresh'}
        </Button>
      </header>
      {actionError && (
        <p role="alert" className="mt-5 rounded-xl bg-red-50 p-3 text-xs text-red-900">{actionError}</p>
      )}

      {state.kind === 'loading' && (
        <div className="mt-8 space-y-4">
          {Array.from({ length: 3 }).map((_, index) => <Skeleton key={index} className="h-52" />)}
        </div>
      )}
      {state.kind === 'error' && (
        <div className="mt-8">
          <ErrorPanel
            title="Workflow API unavailable"
            message={state.message}
            hint="Start python scripts/workflow_api.py --port 8790, confirm VITE_WORKFLOW_API_URL and VITE_OPS_WORKFLOW_TOKEN in frontend/.env.local, then retry."
            onRetry={() => load()}
          />
        </div>
      )}
      {state.kind === 'offline' && (
        <div className="mt-8">
          <EmptyPanel
            title="Workflow API not configured"
            body="Set VITE_WORKFLOW_API_URL and VITE_OPS_WORKFLOW_TOKEN, then start scripts/workflow_api.py."
          />
        </div>
      )}
      {state.kind === 'live' && requests.length === 0 && (
        <div className="mt-8">
          <EmptyPanel title="No model requests" body="Investor-submitted mandates will appear here as soon as the API accepts them." />
        </div>
      )}

      {requests.length > 0 && (
        <div className="mt-8 space-y-4">
          {requests.map((request) => (
            <Card key={request.id} className="overflow-hidden">
              <div className="flex flex-col justify-between gap-4 border-b border-line p-6 sm:flex-row sm:items-start">
                <div>
                  <div className="flex flex-wrap items-center gap-2">
                    <h2 className="text-lg font-semibold">{request.name}</h2>
                    <Badge tone={request.state === 'released' ? 'success' : request.state === 'draft' ? 'warning' : 'neutral'}>
                      {request.state.replaceAll('_', ' ')}
                    </Badge>
                    {request.immutable && <Badge tone="dark">Immutable v{request.version}</Badge>}
                  </div>
                  <p className="mt-2 font-mono text-[11px] text-ink/55">
                    {request.id} · submitted {fmtDate(request.createdAt)}
                  </p>
                  <p className="mt-1 text-[11px] text-ink/55">
                    Organization {request.organizationId} · owner {request.ownerId} · operator {request.assignedOperator ?? 'unassigned'}
                  </p>
                </div>
                <div className="flex items-center gap-2 rounded-xl bg-ink/[.035] px-3 py-2">
                  <Braces size={14} className="text-pine" aria-hidden="true" />
                  <span className="font-mono text-[11px]">{request.instruments.length} assets</span>
                </div>
              </div>
              <div className="grid gap-6 p-6 lg:grid-cols-[.8fr_1.2fr]">
                <section>
                  <div className="flex items-center gap-2">
                    <SlidersHorizontal size={14} className="text-pine" aria-hidden="true" />
                    <h3 className="text-xs font-semibold">Requested configuration</h3>
                  </div>
                  <dl className="mt-4 space-y-3 text-xs">
                    <div className="flex justify-between gap-4"><dt className="text-ink/55">Risk preference</dt><dd className="capitalize">{request.configuration.riskPreference}</dd></div>
                    <div className="flex justify-between gap-4"><dt className="text-ink/55">Max asset weight</dt><dd className="font-mono">{request.configuration.maxWeight}%</dd></div>
                    <div className="flex justify-between gap-4">
                      <dt className="text-ink/55">Approx. trading size</dt>
                      <dd className="font-mono">{formatTradingCapital(request.configuration.approximateTradingCapital)}</dd>
                    </div>
                    <div className="flex justify-between gap-4"><dt className="text-ink/55">Mandate</dt><dd className="font-mono">Long-only · daily</dd></div>
                    <div className="flex justify-between gap-4"><dt className="text-ink/55">Payment</dt><dd className="font-mono">{request.paymentState}</dd></div>
                    <div className="flex justify-between gap-4"><dt className="text-ink/55">Quote</dt><dd className="font-mono">{request.quoteAmount == null ? 'Not issued' : `$${request.quoteAmount.toLocaleString()}`}</dd></div>
                    {request.runPlan && (
                      <>
                        <div className="flex justify-between gap-4"><dt className="text-ink/55">Cohort</dt><dd className="font-mono">{request.runPlan.cohortId}</dd></div>
                        <div className="flex justify-between gap-4"><dt className="text-ink/55">Controlled run plan</dt><dd className="font-mono">{request.runPlan.totalJobs} jobs</dd></div>
                      </>
                    )}
                  </dl>
                </section>
                <section>
                  <h3 className="text-xs font-semibold">Instrument universe</h3>
                  <div className="mt-4 flex flex-wrap gap-2">
                    {request.instruments.map((instrument) => (
                      <span key={instrument.ticker} className="rounded-lg border border-line bg-white/70 px-2.5 py-2">
                        <span className="block font-mono text-[11px] font-semibold">{instrument.ticker}</span>
                        <span className="mt-0.5 block text-[10px] text-ink/55">{instrument.name} · {instrument.group}</span>
                      </span>
                    ))}
                  </div>
                </section>
              </div>
              {(request.eligibility?.length ?? 0) > 0 && (
                <section className="border-t border-line px-6 py-5">
                  <h3 className="text-xs font-semibold">Instrument and data eligibility</h3>
                  <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                    {request.eligibility.map((result) => {
                      const blockReason = !result.symbolFound
                        ? 'symbol not found'
                        : !result.approvedPolicy
                          ? 'outside product universe'
                          : !result.sufficientHistory
                            ? `need ≥2,500 daily bars (have ${result.historyBars.toLocaleString()})`
                            : 'not eligible'
                      return (
                      <div key={result.ticker} className="rounded-xl bg-ink/[.035] p-3 text-[11px]">
                        <div className="flex items-center justify-between">
                          <span className="font-mono font-semibold">{result.ticker}</span>
                          <Badge tone={result.eligible ? 'success' : 'danger'}>{result.eligible ? 'Eligible' : 'Blocked'}</Badge>
                        </div>
                        <p className="mt-2 text-ink/60">
                          {result.historyBars.toLocaleString()} bars · policy{' '}
                          {result.approvedPolicy ? 'approved' : 'not approved'}
                          {!result.eligible && (
                            <>
                              {' · '}
                              {blockReason}
                            </>
                          )}
                        </p>
                      </div>
                      )
                    })}
                  </div>
                </section>
              )}
              <section className="grid gap-5 border-t border-line px-6 py-5 lg:grid-cols-[1fr_.8fr]">
                <div>
                  <h3 className="text-xs font-semibold">Audit timeline</h3>
                  <ol className="mt-3 space-y-2">
                    {(request.auditLog ?? []).map((event, index) => (
                      <li key={`${event.createdAt}-${index}`} className="flex justify-between gap-4 text-[11px]">
                        <span><span className="font-semibold">{event.eventType.replaceAll('_', ' ')}</span> · {event.actorId}</span>
                        <time className="shrink-0 font-mono text-ink/55">{fmtDate(event.createdAt)}</time>
                      </li>
                    ))}
                  </ol>
                </div>
                <div>
                  <h3 className="text-xs font-semibold">Allowed next action</h3>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {(request.allowedActions ?? []).map((action) => (
                      <Button
                        key={action}
                        variant={action === 'authorize_oos_evaluation' || action === 'release_report' ? 'secondary' : 'primary'}
                        disabled={acting !== null}
                        onClick={() => performAction(request, action)}
                      >
                        {acting === `${request.id}:${action}` ? 'Working…' : (actionLabels[action] ?? action)}
                      </Button>
                    ))}
                    {(request.allowedActions?.length ?? 0) === 0 && <p className="text-[11px] text-ink/55">No transition is available in this state.</p>}
                  </div>
                  {request.state === 'checkout' && (
                    <p className="mt-3 text-[11px] leading-4 text-ink/55">
                      Payment can only advance through the verified provider webhook; no operator override is exposed.
                    </p>
                  )}
                </div>
              </section>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
