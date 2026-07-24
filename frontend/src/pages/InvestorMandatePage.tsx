import { ArrowRight, Check, CheckCircle2, Info, Plus, Search } from 'lucide-react'
import { useCallback, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AddInstrumentModal, type AddedInstrument } from '../components/AddInstrumentModal'
import { Badge, Button, Card, Field, Input, Select, SwitchControl } from '../components/ui'
import { submitMandate } from '../lib/api'
import { defaultAssets } from '../lib/demo-data'
import {
  TRADING_SIZE_OPTIONS,
  createMandateSubmission,
  formatTradingCapital,
} from '../lib/mandate-requests'
import type { Asset } from '../lib/types'
import { cn } from '../lib/utils'

type RiskPreference = 'balanced' | 'defensive' | 'growth'

type Instrument = {
  id: string
  name: string
  ticker: string
  group: Asset['group']
}

export function InvestorMandatePage() {
  const navigate = useNavigate()
  const [name, setName] = useState('My portfolio mandate')
  const [search, setSearch] = useState('')
  const [instruments, setInstruments] = useState<Instrument[]>(() =>
    defaultAssets.map((asset) => ({
      id: asset.id,
      name: asset.name,
      ticker: asset.ticker,
      group: asset.group,
    })),
  )
  const [selected, setSelected] = useState(() => new Set(defaultAssets.map((asset) => asset.id)))
  const [showAddInstrument, setShowAddInstrument] = useState(false)
  const [maxWeight, setMaxWeight] = useState(20)
  const [riskPreference, setRiskPreference] = useState<RiskPreference>('balanced')
  const [tradingCapital, setTradingCapital] = useState(1_000_000)
  const [accepted, setAccepted] = useState(false)
  const [submitted, setSubmitted] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const marginEnabled = false
  const leverageCap = '1.0'
  const shortsEnabled = false

  const filteredInstruments = useMemo(
    () =>
      instruments.filter((instrument) =>
        `${instrument.name} ${instrument.ticker} ${instrument.group}`.toLowerCase().includes(search.toLowerCase()),
      ),
    [instruments, search],
  )

  const selectedCount = selected.size
  const canSubmit = accepted && name.trim().length > 0 && selectedCount >= 5 && selectedCount <= 55 && selectedCount * maxWeight >= 100
  const summary = useMemo(
    () => instruments.filter((instrument) => selected.has(instrument.id)).map((instrument) => instrument.ticker),
    [instruments, selected],
  )

  const modelConstraints = useMemo(() => {
    const items = [
      'Daily portfolio decisions',
      shortsEnabled ? 'Long and short exposure requested' : 'Long-only base allocation',
      marginEnabled ? `Margin financing · up to ${leverageCap}× leverage` : 'Cash is allowed · no margin by default',
      `Maximum ${maxWeight}% single-asset concentration`,
      '5–55 supported assets',
      'New symbols validated via yfinance before inclusion',
    ]
    return items
  }, [marginEnabled, leverageCap, shortsEnabled, maxWeight])

  const closeAddInstrument = useCallback(() => setShowAddInstrument(false), [])

  const toggleInstrument = (instrumentIdValue: string, enabled: boolean) => {
    setSelected((current) => {
      const next = new Set(current)
      if (enabled) next.add(instrumentIdValue)
      else next.delete(instrumentIdValue)
      return next
    })
  }

  const handleAddInstrument = useCallback(
    (instrument: AddedInstrument) => {
      if (instruments.length >= 55) return
      const next: Instrument = {
        id: instrument.id,
        name: instrument.name,
        ticker: instrument.ticker,
        group: instrument.group,
      }
      setInstruments((current) => [...current, next])
      setSelected((current) => new Set([...current, next.id]))
      setShowAddInstrument(false)
    },
    [instruments.length],
  )

  const submitRequest = async () => {
    if (!canSubmit) return
    const payload = createMandateSubmission({
      name: name.trim(),
      instruments: instruments
        .filter((instrument) => selected.has(instrument.id))
        .map((instrument) => ({ name: instrument.name, ticker: instrument.ticker, group: instrument.group })),
      maxWeight,
      riskPreference,
      approximateTradingCapital: tradingCapital,
    })
    setSubmitting(true)
    setSubmitError(null)
    try {
      await submitMandate(payload)
      setSubmitted(true)
      window.setTimeout(() => navigate('/portal/builds'), 1200)
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : 'Unable to submit model request')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="px-5 py-7 sm:px-8 lg:px-10 lg:py-9">
      <header>
        <div className="flex items-center gap-2">
          <p className="font-mono text-[11px] uppercase tracking-[.2em] text-ink/55">Create mandate</p>
          <Badge tone="success">Draft</Badge>
        </div>
        <h1 className="font-display mt-2 max-w-2xl text-4xl tracking-[-.035em] sm:text-5xl">
          Describe the portfolio you want researched.
        </h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-ink/60">
          Select instruments and set risk preferences. After submission, the server checks data eligibility and
          product policy before it can issue a quote.
        </p>
      </header>

      <div className="mt-8 grid gap-5 xl:grid-cols-[1.25fr_.75fr]">
        <div className="space-y-5">
          <Card className="p-6">
            <h2 className="text-lg font-semibold tracking-[-.02em]">1. Mandate and instruments</h2>
            <p className="mt-1 text-xs text-ink/60">
              Search the instrument catalog or add any symbol to your universe. New instruments are
              validated for data eligibility before your quote is finalized.
            </p>
            <div className="mt-6">
              <Field label="Mandate name">
                <Input value={name} onChange={(event) => setName(event.target.value)} />
              </Field>
            </div>
            <div className="mt-6 flex gap-2">
              <div className="relative flex-1">
                <Search size={14} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-ink/60" aria-hidden="true" />
                <Input
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Search by name, ticker, or asset class"
                  aria-label="Search instruments"
                  className="pl-9"
                />
              </div>
              <Button variant="secondary" onClick={() => setShowAddInstrument(true)}>
                <Plus size={14} aria-hidden="true" /> Add instrument
              </Button>
            </div>
            <div className="mt-4 overflow-hidden rounded-2xl border border-line">
              <div className="grid grid-cols-[1fr_90px_60px] bg-ink/[.025] px-4 py-2.5 font-mono text-[11px] uppercase tracking-wider text-ink/55">
                <span>Instrument</span><span>Group</span><span>Use</span>
              </div>
              <div className="max-h-[420px] divide-y divide-line overflow-y-auto">
                {filteredInstruments.map((instrument) => (
                  <div
                    key={instrument.id}
                    className={cn(
                      'grid grid-cols-[1fr_90px_60px] items-center px-4 py-3',
                      !selected.has(instrument.id) && 'opacity-50',
                    )}
                  >
                    <div>
                      <p className="text-xs font-semibold">{instrument.name}</p>
                      <p className="mt-1 font-mono text-[11px] text-ink/55">{instrument.ticker}</p>
                    </div>
                    <span className="text-[11px] text-ink/60">{instrument.group}</span>
                    <SwitchControl
                      checked={selected.has(instrument.id)}
                      onChange={(enabled) => toggleInstrument(instrument.id, enabled)}
                      label={`Include ${instrument.name}`}
                    />
                  </div>
                ))}
                {filteredInstruments.length === 0 && (
                  <p className="px-4 py-8 text-center text-xs text-ink/55">
                    No instruments match this search. Add a new symbol to include it in your mandate.
                  </p>
                )}
              </div>
            </div>
            <p className="mt-3 text-[11px] text-ink/55">
              {instruments.length} instruments in universe · {selectedCount} selected
            </p>
          </Card>

          <Card className="p-6">
            <h2 className="text-lg font-semibold tracking-[-.02em]">2. Portfolio preferences</h2>
            <div className="mt-6 grid gap-7 sm:grid-cols-2">
              <div>
                <div className="flex items-center justify-between">
                  <label htmlFor="max-weight" className="text-xs font-semibold text-ink/80">Maximum single-asset concentration</label>
                  <span className="rounded-lg bg-pine px-2.5 py-1 font-mono text-[11px] text-mint">{maxWeight}%</span>
                </div>
                <input
                  id="max-weight"
                  type="range"
                  min={10}
                  max={40}
                  value={maxWeight}
                  onChange={(event) => setMaxWeight(Number(event.target.value))}
                  className="mt-4 w-full"
                />
              </div>
              <Field label="Risk preference">
                <Select value={riskPreference} onChange={(event) => setRiskPreference(event.target.value as RiskPreference)}>
                  <option value="defensive">Defensive</option>
                  <option value="balanced">Balanced</option>
                  <option value="growth">Growth-oriented</option>
                </Select>
              </Field>
              <div className="sm:col-span-2">
                <Field label="Approximate trading size">
                  <Select
                    value={String(tradingCapital)}
                    onChange={(event) => setTradingCapital(Number(event.target.value))}
                    aria-label="Approximate trading size"
                  >
                    {TRADING_SIZE_OPTIONS.map((option) => (
                      <option key={option.value} value={option.value}>{option.label}</option>
                    ))}
                  </Select>
                </Field>
                <p className="mt-2 text-[11px] leading-5 text-ink/55">
                  An estimate is enough. This sizes slippage and transaction-cost assumptions for your mandate.
                </p>
              </div>
            </div>
            {selectedCount > 0 && selectedCount < 5 && (
              <p role="alert" className="mt-5 rounded-xl bg-amber-50 p-3 text-xs leading-5 text-amber-900">
                Select at least five instruments to request a model build.
              </p>
            )}
            {selectedCount >= 5 && selectedCount * maxWeight < 100 && (
              <p role="alert" className="mt-5 rounded-xl bg-amber-50 p-3 text-xs leading-5 text-amber-900">
                Raise the concentration cap or add instruments so the portfolio can reach full investment.
              </p>
            )}
          </Card>

        </div>

        <div className="space-y-5">
          <Card className="p-6">
            <h2 className="text-sm font-semibold">Standard model constraints</h2>
            <ul className="mt-5 space-y-3">
              {modelConstraints.map((item) => (
                <li key={item} className="flex items-center gap-2 text-xs text-ink/70">
                  <span className="grid h-5 w-5 place-items-center rounded-full bg-emerald-100 text-emerald-700">
                    <Check size={11} aria-hidden="true" />
                  </span>
                  {item}
                </li>
              ))}
            </ul>
          </Card>

          <Card className="p-6">
            <p className="font-mono text-[11px] uppercase tracking-[.15em] text-ink/55">Mandate summary</p>
            <dl className="mt-5 space-y-4 text-xs">
              <div className="flex justify-between gap-4"><dt className="text-ink/55">Mandate</dt><dd className="text-right font-semibold">{name || 'Untitled'}</dd></div>
              <div className="flex justify-between"><dt className="text-ink/55">Instruments</dt><dd className="font-mono">{selectedCount}</dd></div>
              <div className="flex justify-between"><dt className="text-ink/55">Risk preference</dt><dd className="capitalize">{riskPreference}</dd></div>
              <div className="flex justify-between"><dt className="text-ink/55">Trading size</dt><dd className="font-mono">{formatTradingCapital(tradingCapital)}</dd></div>
              <div className="flex justify-between border-t border-line pt-4">
                <dt className="text-ink/55">Quote</dt>
                <dd className="text-right text-ink/65">Issued after eligibility review</dd>
              </div>
            </dl>
            <div className="mt-5 flex flex-wrap gap-1.5">
              {summary.map((ticker) => <Badge key={ticker}>{ticker}</Badge>)}
            </div>
          </Card>

          <div className="rounded-[22px] border border-pine/10 bg-pine p-5 text-cream">
            <Info size={16} className="text-mint" aria-hidden="true" />
            <p className="mt-4 text-xs font-semibold">Included in your build</p>
            <p className="mt-2 text-[11px] leading-5 text-cream/70">
              Walk-forward validation across all applicable windows, governed OOS evaluation, benchmark-relative
              reporting, deflated Sharpe, and a release-approved research package.
            </p>
          </div>

          <Card className="p-6">
            <label className="flex items-start gap-3 text-xs leading-5 text-ink/70">
              <input
                type="checkbox"
                checked={accepted}
                onChange={(event) => setAccepted(event.target.checked)}
                className="mt-1"
              />
              I accept the methodology, data, and risk disclosures. I understand this is investment research,
              not investment advice, and that the paid mandate version becomes immutable after verified payment.
            </label>
            <Button className="mt-5 w-full" disabled={!canSubmit || submitted || submitting} onClick={submitRequest}>
              {submitted ? (
                <>
                  <CheckCircle2 size={14} aria-hidden="true" /> Request submitted
                </>
              ) : (
                <>
                  {submitting ? 'Submitting request…' : 'Request model build'} {!submitting && <ArrowRight size={14} aria-hidden="true" />}
                </>
              )}
            </Button>
            {submitError && (
              <p role="alert" className="mt-3 rounded-lg bg-red-50 p-2.5 text-xs leading-5 text-red-900">
                {submitError}. Confirm the authenticated workflow service is available and try again.
              </p>
            )}
            {!accepted && (
              <p className="mt-3 text-center text-[11px] leading-4 text-ink/55">
                Accept the disclosures to submit your build request.
              </p>
            )}
            {accepted && !canSubmit && (
              <p className="mt-3 text-center text-[11px] leading-4 text-ink/55">
                Complete the mandate requirements above to submit.
              </p>
            )}
          </Card>
        </div>
      </div>

      <AddInstrumentModal
        open={showAddInstrument}
        existingTickers={instruments.map((instrument) => instrument.ticker)}
        existingIds={instruments.map((instrument) => instrument.id)}
        onClose={closeAddInstrument}
        onAdd={handleAddInstrument}
      />
    </div>
  )
}
