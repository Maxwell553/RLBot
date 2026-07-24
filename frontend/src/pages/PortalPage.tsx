import { ArrowRight, Clock3, FileCheck2, FolderKanban, SlidersHorizontal } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Badge, Button, Card, ErrorPanel, Skeleton } from '../components/ui'
import { cancelMandate, fetchMandates } from '../lib/api'
import type { WorkflowMandate } from '../lib/mandate-requests'
import { fmtDate } from '../lib/format'
import type { DataState } from '../lib/types'

const stages = [
  'Draft submitted',
  'Preflight passed',
  'Quote issued',
  'Checkout',
  'Verified payment',
  'Queued',
  'Training',
  'Validation',
  'Governed OOS evaluation',
  'Released',
]

const stateLabels: Record<WorkflowMandate['state'], string> = {
  draft: 'Awaiting preflight',
  preflight_passed: 'Preflight passed',
  quote_issued: 'Quote issued',
  checkout: 'Checkout',
  payment_verified: 'Payment verified',
  queued: 'Queued',
  training: 'Training',
  validation: 'Validation',
  governed_oos_evaluation: 'Governed OOS evaluation',
  released: 'Released',
  cancelled: 'Cancelled',
}

const WORKFLOW_HINT = 'Start the workflow API with python scripts/workflow_api.py --port 8790, confirm VITE_WORKFLOW_API_URL in frontend/.env.local, then retry.'

function useMandates() {
  const [state, setState] = useState<DataState<WorkflowMandate[]>>({ kind: 'loading' })
  const load = useCallback((signal?: AbortSignal) => {
    setState((current) => current.kind === 'live' ? current : { kind: 'loading' })
    fetchMandates(signal).then((next) => {
      if (signal?.aborted || next.kind === 'loading') return
      setState((current) => (next.kind === 'error' && current.kind === 'live' ? current : next))
    })
  }, [])
  useEffect(() => {
    const controller = new AbortController()
    load(controller.signal)
    return () => controller.abort()
  }, [load])
  return { state, load, setState }
}

export function PortalPage() {
  const { state } = useMandates()
  const activeBuilds = state.kind === 'live'
    ? state.data.filter((mandate) => mandate.state !== 'released' && mandate.state !== 'cancelled').length
    : 0

  return (
    <div className="px-5 py-7 sm:px-8 lg:px-10 lg:py-9">
      <header>
        <div className="flex items-center gap-2">
          <p className="font-mono text-[11px] uppercase tracking-[.2em] text-ink/55">Investor portal</p>
          {activeBuilds > 0 && <Badge tone="success">{activeBuilds} active build{activeBuilds > 1 ? 's' : ''}</Badge>}
        </div>
        <h1 className="font-display mt-2 max-w-2xl text-4xl tracking-[-.035em] sm:text-5xl">
          Your mandates and released research.
        </h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-ink/60">
          Create mandates, track build progress, and access release-approved performance reports for your
          portfolio models.
        </p>
      </header>

      <section className="mt-8 grid gap-4 lg:grid-cols-3">
        <Card className="p-6 lg:col-span-2">
          <span className="grid h-10 w-10 place-items-center rounded-xl bg-pine text-mint">
            <SlidersHorizontal size={18} aria-hidden="true" />
          </span>
          <h2 className="mt-6 text-xl font-semibold tracking-[-.025em]">Create a mandate</h2>
          <p className="mt-2 max-w-xl text-sm leading-6 text-ink/60">
            Choose instruments, set concentration and risk preferences, and submit a build request with a
            server-issued quote after data eligibility and product-policy review.
          </p>
          <Link
            to="/portal/mandates/new"
            className="mt-6 inline-flex h-10 items-center gap-2 rounded-full bg-pine px-4 text-xs font-semibold text-cream"
          >
            Create mandate <ArrowRight size={14} aria-hidden="true" />
          </Link>
        </Card>
        <Card className="p-6">
          <FolderKanban size={18} className="text-pine" aria-hidden="true" />
          <h2 className="mt-5 text-sm font-semibold">Build pipeline</h2>
          <p className="mt-2 text-xs leading-5 text-ink/60">
            Every build follows a governed sequence from quote acceptance through training, validation, and
            report release.
          </p>
          <Link to="/portal/builds" className="mt-4 inline-flex text-xs font-semibold text-pine">
            View build status →
          </Link>
        </Card>
      </section>

      <section className="mt-4 grid gap-4 xl:grid-cols-[1.2fr_.8fr]">
        <Card className="p-6">
          <div className="flex items-center gap-2">
            <FolderKanban size={17} className="text-pine" aria-hidden="true" />
            <h2 className="text-sm font-semibold">Build timeline</h2>
          </div>
          <ol className="mt-6 grid gap-3 sm:grid-cols-2">
            {stages.map((stage, index) => (
              <li key={stage} className="flex items-center gap-3 rounded-xl bg-ink/[.035] p-3">
                <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full border border-line bg-paper font-mono text-[10px]">
                  {index + 1}
                </span>
                <span className="text-xs font-medium">{stage}</span>
              </li>
            ))}
          </ol>
        </Card>
        <Card className="p-6">
          <FileCheck2 size={17} className="text-pine" aria-hidden="true" />
          <h2 className="mt-5 text-sm font-semibold">Released reports</h2>
          <p className="mt-2 text-xs leading-5 text-ink/60">
            Only release-approved packages appear in Delivered models. Cancel is available before payment
            locks the mandate.
          </p>
          <Link to="/portal/reports" className="mt-4 inline-flex text-xs font-semibold text-pine">
            Open delivered models →
          </Link>
        </Card>
      </section>
    </div>
  )
}

export function PortalBuildsPage() {
  const { state, load, setState } = useMandates()
  const [confirmCancelId, setConfirmCancelId] = useState<string | null>(null)
  const [cancellingId, setCancellingId] = useState<string | null>(null)
  const [cancelError, setCancelError] = useState<string | null>(null)
  const requests = state.kind === 'live'
    ? state.data.filter((mandate) => mandate.state !== 'cancelled')
    : []

  const handleCancel = async (mandate: WorkflowMandate) => {
    if (!mandate.allowedActions.includes('cancel')) return
    setCancellingId(mandate.id)
    setCancelError(null)
    try {
      await cancelMandate(mandate.id)
      setState((current) => (
        current.kind === 'live'
          ? { kind: 'live', data: current.data.filter((item) => item.id !== mandate.id) }
          : current
      ))
      setConfirmCancelId(null)
    } catch (error) {
      setCancelError(error instanceof Error ? error.message : 'Unable to cancel mandate')
    } finally {
      setCancellingId(null)
    }
  }

  if (state.kind === 'loading') {
    return (
      <div className="px-5 py-7 sm:px-8 lg:px-10 lg:py-9">
        <Skeleton className="h-12 w-72" />
        <div className="mt-8 space-y-4">{Array.from({ length: 2 }).map((_, index) => <Skeleton key={index} className="h-40" />)}</div>
      </div>
    )
  }

  if (state.kind === 'error') {
    return (
      <div className="px-5 py-7 sm:px-8 lg:px-10 lg:py-9">
        <ErrorPanel
          title="Workflow API unavailable"
          message={state.message}
          hint={WORKFLOW_HINT}
          onRetry={() => load()}
        />
      </div>
    )
  }

  if (state.kind === 'offline') {
    return (
      <div className="px-5 py-7 sm:px-8 lg:px-10 lg:py-9">
        <Card className="max-w-2xl border-dashed p-10 text-center">
          <p className="text-sm font-semibold">Secure workflow unavailable</p>
          <p className="mx-auto mt-2 max-w-md text-xs leading-5 text-ink/60">
            This portal requires an authenticated workflow service. No browser-local or shared fallback records are shown.
          </p>
        </Card>
      </div>
    )
  }

  if (requests.length === 0) {
    return (
      <div className="px-5 py-7 sm:px-8 lg:px-10 lg:py-9">
        <p className="font-mono text-[11px] uppercase tracking-[.2em] text-ink/55">Build status</p>
        <h1 className="font-display mt-2 text-4xl tracking-[-.035em] sm:text-5xl">Your model builds</h1>
        <Card className="mt-8 max-w-2xl border-dashed p-10 text-center">
          <Clock3 size={22} className="mx-auto text-pine" aria-hidden="true" />
          <p className="mt-4 text-sm font-semibold text-ink">No builds yet</p>
          <p className="mx-auto mt-2 max-w-md text-xs leading-5 text-ink/60">
            Submit a mandate to start a model build. Progress and delivery estimates appear here once your
            request is accepted.
          </p>
          <Link
            to="/portal/mandates/new"
            className="mt-6 inline-flex h-10 items-center gap-2 rounded-full bg-pine px-4 text-xs font-semibold text-cream"
          >
            Create mandate <ArrowRight size={14} aria-hidden="true" />
          </Link>
        </Card>
      </div>
    )
  }

  return (
    <div className="px-5 py-7 sm:px-8 lg:px-10 lg:py-9">
      <header>
        <p className="font-mono text-[11px] uppercase tracking-[.2em] text-ink/55">Build status</p>
        <h1 className="font-display mt-2 text-4xl tracking-[-.035em] sm:text-5xl">Your model builds</h1>
        <p className="mt-3 text-sm text-ink/60">Track quote review, queue position, and delivery estimates.</p>
        {cancelError && (
          <p role="alert" className="mt-3 rounded-xl bg-red-50 px-3 py-2 text-xs text-red-900">{cancelError}</p>
        )}
      </header>
      <div className="mt-8 space-y-4">
        {requests.map((request) => {
          const canCancel = request.allowedActions.includes('cancel')
          const confirming = confirmCancelId === request.id
          return (
            <Card key={request.id} className="p-6">
              <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-start">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <h2 className="text-lg font-semibold">{request.name}</h2>
                    <Badge tone={request.state === 'released' ? 'success' : request.state === 'draft' ? 'warning' : 'neutral'}>
                      {stateLabels[request.state]}
                    </Badge>
                  </div>
                  <p className="mt-2 text-xs text-ink/60">
                    Submitted {fmtDate(request.createdAt)} · version {request.version} · {request.instruments.length} instruments
                  </p>
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {request.instruments.map((instrument) => (
                      <Badge key={instrument.ticker}>{instrument.ticker}</Badge>
                    ))}
                  </div>
                  {canCancel && !confirming && (
                    <Button
                      variant="secondary"
                      className="mt-4"
                      onClick={() => {
                        setCancelError(null)
                        setConfirmCancelId(request.id)
                      }}
                    >
                      Cancel mandate
                    </Button>
                  )}
                  {confirming && (
                    <div
                      role="region"
                      aria-label={`Confirm cancellation of ${request.name}`}
                      className="mt-4 max-w-lg rounded-2xl border border-line bg-ink/[.03] p-4"
                    >
                      <p className="text-xs font-semibold text-ink">Cancel this mandate?</p>
                      <p className="mt-1.5 text-[11px] leading-5 text-ink/60">
                        This stops the request before payment. It will leave your build list, and you can
                        submit a new mandate anytime.
                      </p>
                      <div className="mt-4 flex flex-wrap gap-2">
                        <Button
                          variant="secondary"
                          disabled={cancellingId === request.id}
                          onClick={() => setConfirmCancelId(null)}
                        >
                          Keep mandate
                        </Button>
                        <Button
                          disabled={cancellingId === request.id}
                          onClick={() => handleCancel(request)}
                        >
                          {cancellingId === request.id ? 'Cancelling…' : 'Yes, cancel'}
                        </Button>
                      </div>
                    </div>
                  )}
                </div>
                <div className="rounded-2xl bg-ink/[.035] px-5 py-4 text-right">
                  <p className="font-mono text-[11px] uppercase tracking-wider text-ink/55">
                    {request.quoteAmount == null ? 'Quote status' : 'Server-issued quote'}
                  </p>
                  <p className="font-display mt-1 text-3xl">
                    {request.quoteAmount == null ? 'Pending' : `$${request.quoteAmount.toLocaleString()}`}
                  </p>
                  <p className="mt-1 text-[11px] text-ink/55">
                    {request.quoteAmount == null
                      ? 'Eligibility review must pass before pricing'
                      : request.paymentState === 'verified' ? 'Payment verified' : 'Payment not yet verified'}
                  </p>
                </div>
              </div>
            </Card>
          )
        })}
      </div>
    </div>
  )
}

export function PortalReportsPage() {
  const { state, load } = useMandates()
  const released = state.kind === 'live' ? state.data.filter((mandate) => mandate.state === 'released') : []

  if (state.kind === 'loading') {
    return <div className="px-5 py-7 sm:px-8 lg:px-10 lg:py-9"><Skeleton className="h-48 max-w-2xl" /></div>
  }
  if (state.kind === 'error') {
    return (
      <div className="px-5 py-7 sm:px-8 lg:px-10 lg:py-9">
        <ErrorPanel
          title="Workflow API unavailable"
          message={state.message}
          hint={WORKFLOW_HINT}
          onRetry={() => load()}
        />
      </div>
    )
  }

  return (
    <div className="px-5 py-7 sm:px-8 lg:px-10 lg:py-9">
      <p className="font-mono text-[11px] uppercase tracking-[.2em] text-ink/55">Delivered models</p>
      <h1 className="font-display mt-2 text-4xl tracking-[-.035em] sm:text-5xl">Released reports</h1>
      {released.length === 0 ? (
        <Card className="mt-8 max-w-2xl border-dashed p-10 text-center">
          <FileCheck2 size={22} className="mx-auto text-pine" aria-hidden="true" />
          <p className="mt-4 text-sm font-semibold text-ink">No released reports yet</p>
          <p className="mx-auto mt-2 max-w-md text-xs leading-5 text-ink/60">
            Only mandates that pass the governed OOS and operator release gates appear here.
          </p>
        </Card>
      ) : (
        <div className="mt-8 space-y-4">
          {released.map((mandate) => (
            <Card key={mandate.id} className="p-6">
              <div className="flex items-start justify-between gap-5">
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-lg font-semibold">{mandate.name}</h2>
                    <Badge tone="success">Released</Badge>
                  </div>
                  <p className="mt-2 text-xs text-ink/60">
                    Immutable version {mandate.version} · released {fmtDate(mandate.updatedAt)}
                  </p>
                </div>
                <FileCheck2 size={20} className="text-emerald-700" aria-hidden="true" />
              </div>
              <p className="mt-5 rounded-xl bg-ink/[.035] p-3 text-[11px] leading-5 text-ink/65">
                Release approval is recorded.
              </p>
              {mandate.release?.artifactUrl ? (
                <a
                  href={mandate.release.artifactUrl}
                  className="mt-4 inline-flex h-10 items-center rounded-full bg-pine px-4 text-xs font-semibold text-cream"
                >
                  Download entitled package
                </a>
              ) : (
                <p className="mt-3 text-[11px] text-ink/55">
                  The entitled package is being attached; no unauthenticated artifact link is exposed.
                </p>
              )}
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
