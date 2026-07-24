import {
  ArrowLeft,
  ArrowRight,
  Check,
  CheckCircle2,
  Clipboard,
  Download,
  Info,
  Plus,
  Search,
  ShieldAlert,
  ShieldCheck,
  X,
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { AddInstrumentModal, type AddedInstrument } from '../components/AddInstrumentModal'
import { Badge, Button, Card, Field, Input, Select, SwitchControl, Toggle } from '../components/ui'
import { OFFLINE_MODE, runPreflight } from '../lib/api'
import { defaultAssets } from '../lib/demo-data'
import {
  MAX_ASSETS,
  buildConfigYaml,
  buildRunPlan,
  configSha256,
  downloadConfig,
  validateDraft,
} from '../lib/integration'
import { formatBps } from '../lib/transaction-costs'
import type { MandateDraft, PreflightReport } from '../lib/types'
import { cn } from '../lib/utils'

const DRAFT_KEY = 'markettrainer.mandate-draft.v3'

const initialDraft: MandateDraft = {
  name: 'My first mandate',
  description: 'Multi-asset mandate focused on benchmark-relative consistency and drawdown control.',
  capital: 1_000_000,
  baseCurrency: 'USD',
  assets: defaultAssets,
  maxAssetWeight: 20,
  stopLoss: 45,
  minCash: 0,
  maxTurnover: 150,
  objective: 'balanced',
  benchmarkAssetId: 'sp500',
  trainingBudget: 50_000_000,
  seedCount: 3,
  reproducible: true,
}

function loadDraft(): MandateDraft {
  try {
    const raw = localStorage.getItem(DRAFT_KEY)
    if (!raw) return initialDraft
    const parsed = JSON.parse(raw) as MandateDraft
    if (!Array.isArray(parsed.assets) || parsed.assets.some((asset) => typeof asset.holdingCostBps !== 'number')) {
      return initialDraft
    }
    return { ...initialDraft, ...parsed }
  } catch {
    return initialDraft
  }
}

const steps = ['Universe', 'Constraints', 'Training design', 'Review'] as const

function SliderField({ label, value, min, max, step = 1, suffix = '%', onChange, hint }: {
  label: string; value: number; min: number; max: number; step?: number; suffix?: string; hint?: string; onChange: (value: number) => void
}) {
  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <span className="text-xs font-semibold text-ink/80">{label}</span>
        <span className="rounded-lg bg-pine px-2.5 py-1 font-mono text-[11px] text-mint">{value.toLocaleString()}{suffix}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-label={label}
        onChange={(event) => onChange(Number(event.target.value))}
        className="w-full"
      />
      {hint && <p className="mt-2 text-[11px] leading-4 text-ink/60">{hint}</p>}
    </div>
  )
}

export function ConfigurePage() {
  const [step, setStep] = useState(0)
  const [draft, setDraft] = useState<MandateDraft>(loadDraft)
  const [search, setSearch] = useState('')
  const [showAddAsset, setShowAddAsset] = useState(false)
  const [copied, setCopied] = useState(false)
  const [saved, setSaved] = useState(false)
  const [configHash, setConfigHash] = useState<string | null>(null)
  const [preflight, setPreflight] = useState<{ state: 'idle' | 'running' | 'done' | 'failed'; report?: PreflightReport; error?: string }>({ state: 'idle' })

  const update = <K extends keyof MandateDraft>(key: K, value: MandateDraft[K]) => {
    setDraft((current) => ({ ...current, [key]: value }))
    setPreflight({ state: 'idle' })
    setConfigHash(null)
  }

  const enabledAssets = draft.assets.filter((asset) => asset.enabled)
  const filteredAssets = draft.assets.filter((asset) =>
    `${asset.name} ${asset.ticker} ${asset.group}`.toLowerCase().includes(search.toLowerCase()),
  )
  const validationErrors = useMemo(() => validateDraft(draft), [draft])
  const universeErrors = validationErrors.filter((error) => !error.includes('benchmark'))
  const plan = useMemo(() => buildRunPlan(draft), [draft])
  const yamlText = useMemo(() => (validationErrors.length === 0 ? buildConfigYaml(draft) : null), [draft, validationErrors])

  useEffect(() => {
    let cancelled = false
    if (yamlText) {
      configSha256(yamlText).then((hash) => {
        if (!cancelled) setConfigHash(hash)
      })
    }
    return () => {
      cancelled = true
    }
  }, [yamlText])

  // Keep the benchmark valid when the selection changes.
  useEffect(() => {
    if (enabledAssets.length > 0 && !enabledAssets.some((asset) => asset.id === draft.benchmarkAssetId)) {
      update('benchmarkAssetId', enabledAssets[0].id)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft.assets])

  const toggleAsset = (id: string, enabled: boolean) => {
    if (enabled && enabledAssets.length >= MAX_ASSETS) return
    update('assets', draft.assets.map((asset) => (asset.id === id ? { ...asset, enabled } : asset)))
  }

  const addAsset = (instrument: AddedInstrument) => {
    if (enabledAssets.length >= MAX_ASSETS) return
    update('assets', [
      ...draft.assets,
      {
        id: instrument.id,
        name: instrument.name,
        ticker: instrument.ticker,
        group: instrument.group,
        enabled: true,
        feeBps: instrument.feeBps,
        slippageBps: instrument.slippageBps,
        holdingCostBps: instrument.holdingCostBps,
      },
    ])
    setShowAddAsset(false)
  }

  const saveDraft = () => {
    localStorage.setItem(DRAFT_KEY, JSON.stringify(draft))
    setSaved(true)
    window.setTimeout(() => setSaved(false), 1800)
  }

  const copyPlan = async () => {
    await navigator.clipboard.writeText(plan.commands.join('\n'))
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1500)
  }

  const startPreflight = async () => {
    if (!yamlText) return
    setPreflight({ state: 'running' })
    try {
      const report = await runPreflight(yamlText)
      setPreflight({ state: 'done', report })
    } catch (error) {
      setPreflight({ state: 'failed', error: error instanceof Error ? error.message : String(error) })
    }
  }

  const canAdvance = step === 0 ? universeErrors.length === 0 : validationErrors.length === 0

  return (
    <div className="min-h-screen">
      <header className="border-b border-line bg-paper/70 px-5 py-6 backdrop-blur sm:px-8 lg:px-10">
        <div className="flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-[.2em] text-ink/55">Operator config builder</p>
            <h1 className="font-display mt-2 text-4xl tracking-[-.035em]">Materialize a research configuration.</h1>
          </div>
          <button onClick={saveDraft} className="inline-flex items-center gap-2 self-start text-xs font-semibold text-pine">
            {saved ? <CheckCircle2 size={15} aria-hidden="true" /> : null}
            {saved ? 'Draft saved' : 'Save draft'}
          </button>
        </div>
        <nav aria-label="Builder steps" className="mt-8 max-w-3xl">
          <ol className="flex items-center">
            {steps.map((title, index) => (
              <li key={title} className={cn('flex items-center', index < steps.length - 1 && 'flex-1')}>
                <button
                  onClick={() => index < step && setStep(index)}
                  disabled={index > step}
                  aria-current={index === step ? 'step' : undefined}
                  aria-label={`Step ${index + 1}: ${title}${index < step ? ' (completed)' : index === step ? ' (current)' : ''}`}
                  className="flex items-center gap-2 text-left"
                >
                  <span className={cn(
                    'grid h-7 w-7 shrink-0 place-items-center rounded-full border font-mono text-[11px]',
                    index < step && 'border-pine bg-pine text-mint',
                    index === step && 'border-pine bg-mint text-pine',
                    index > step && 'border-line text-ink/60',
                  )}>
                    {index < step ? <Check size={12} aria-hidden="true" /> : index + 1}
                  </span>
                  <span className={cn('hidden text-[11px] sm:block', index === step ? 'font-semibold text-ink' : 'text-ink/55')}>{title}</span>
                </button>
                {index < steps.length - 1 && <span aria-hidden="true" className={cn('mx-3 h-px flex-1', index < step ? 'bg-pine' : 'bg-line')} />}
              </li>
            ))}
          </ol>
        </nav>
      </header>

      <div className="mx-auto max-w-[1120px] px-5 py-8 sm:px-8 lg:px-10 lg:py-10">
          <div key={step}>
            {step === 0 && (
              <div className="grid gap-5 xl:grid-cols-[1.55fr_.75fr]">
                <Card className="p-6">
                  <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
                    <div>
                      <h2 className="text-lg font-semibold tracking-[-.02em]">Investable universe</h2>
                      <p className="mt-1 text-xs text-ink/60">
                        Select 5–55 unique instruments. Fee and holding (bps) export as per-asset cost arrays in enabled order.
                      </p>
                    </div>
                    <Badge tone={enabledAssets.length >= 5 && enabledAssets.length <= MAX_ASSETS ? 'success' : 'warning'}>
                      {enabledAssets.length} selected
                    </Badge>
                  </div>
                  <div className="mt-6 flex gap-2">
                    <div className="relative flex-1">
                      <Search size={14} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-ink/60" aria-hidden="true" />
                      <Input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search assets" aria-label="Search assets" className="pl-9" />
                    </div>
                    <Button variant="secondary" onClick={() => setShowAddAsset(true)}>
                      <Plus size={14} aria-hidden="true" /> <span className="hidden sm:inline">Add instrument</span>
                      <span className="sr-only sm:hidden">Add instrument</span>
                    </Button>
                  </div>
                  <div className="mt-4 overflow-hidden rounded-2xl border border-line">
                    <div className="grid grid-cols-[1fr_82px_60px] bg-ink/[.025] px-4 py-2.5 font-mono text-[11px] uppercase tracking-wider text-ink/55 sm:grid-cols-[1fr_100px_80px_80px_80px]">
                      <span>Instrument</span><span>Group</span><span className="hidden sm:block">Fee</span><span className="hidden sm:block">Holding</span><span>Use</span>
                    </div>
                    <div className="max-h-[460px] divide-y divide-line overflow-y-auto">
                      {filteredAssets.map((asset) => (
                        <div key={asset.id} className={cn('grid grid-cols-[1fr_82px_60px] items-center px-4 py-3 sm:grid-cols-[1fr_100px_80px_80px_80px]', !asset.enabled && 'opacity-50')}>
                          <div className="min-w-0">
                            <p className="truncate text-xs font-semibold">{asset.name}</p>
                            <p className="mt-1 font-mono text-[11px] text-ink/55">{asset.ticker}</p>
                          </div>
                          <span className="text-[11px] text-ink/60">{asset.group}</span>
                          <span className="hidden font-mono text-[11px] text-ink/60 sm:block">{formatBps(asset.feeBps)} bps</span>
                          <span className="hidden font-mono text-[11px] text-ink/60 sm:block">{formatBps(asset.holdingCostBps)} bps</span>
                          <SwitchControl
                            checked={asset.enabled}
                            onChange={(enabled) => toggleAsset(asset.id, enabled)}
                            label={`Include ${asset.name} in universe`}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                  {universeErrors.length > 0 && (
                    <div role="alert" className="mt-4 rounded-2xl border border-amber-700/15 bg-amber-50 p-4">
                      {universeErrors.map((error) => (
                        <p key={error} className="text-xs leading-5 text-amber-900">{error}</p>
                      ))}
                    </div>
                  )}
                </Card>

                <div className="space-y-5">
                  <Card className="p-5">
                    <h3 className="text-sm font-semibold">Mandate details</h3>
                    <div className="mt-5 space-y-4">
                      <Field label="Mandate name"><Input value={draft.name} onChange={(event) => update('name', event.target.value)} /></Field>
                      <Field label="Base currency" hint="metadata only">
                        <Select value={draft.baseCurrency} onChange={(event) => update('baseCurrency', event.target.value)}>
                          <option>USD</option><option>EUR</option><option>GBP</option><option>JPY</option>
                        </Select>
                      </Field>
                    </div>
                  </Card>
                  <div className="rounded-[22px] border border-pine/10 bg-pine p-5 text-cream">
                    <Info size={16} className="text-mint" aria-hidden="true" />
                    <p className="mt-4 text-xs font-semibold">Engine compatibility</p>
                    <p className="mt-2 text-[11px] leading-5 text-cream/70">
                      1 bps = 0.01%. Fee (plus slippage) is charged on traded notional at each rebalance;
                      holding is annual carry converted to a daily drag (÷252). New instruments inherit the
                      asset-class schedule — no per-ticker cost entry in the UI. Macro series feed observations
                      only and are not tradeable.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {step === 1 && (
              <div className="grid gap-5 xl:grid-cols-[1.25fr_.75fr]">
                <div className="space-y-5">
                  <Card className="p-6">
                    <div className="flex items-center gap-2">
                      <h2 className="text-lg font-semibold tracking-[-.02em]">Enforced constraints</h2>
                      <Badge tone="success">Engine behavior</Badge>
                    </div>
                    <p className="mt-1 text-xs text-ink/60">These fields are written into the exported config and enforced during training and execution.</p>
                    <div className="mt-7 grid gap-4 sm:grid-cols-2">
                      <Toggle checked onChange={() => {}} disabled label="Long-only portfolio" description="Structural: the action projection produces a long-only simplex." />
                      <Toggle checked onChange={() => {}} disabled label="No leverage" description="Structural: gross exposure is bounded at 100% of NAV." />
                    </div>
                    <div className="mt-8 grid gap-x-10 gap-y-8 sm:grid-cols-2">
                      <SliderField
                        label="Max weight per asset (global)"
                        value={draft.maxAssetWeight}
                        min={10}
                        max={40}
                        onChange={(value) => update('maxAssetWeight', value)}
                        hint="One global cap applied to every risky asset via clip-and-redistribute. Per-asset caps are not yet supported by the engine."
                      />
                      <SliderField
                        label="Portfolio stop-loss"
                        value={draft.stopLoss}
                        min={10}
                        max={60}
                        onChange={(value) => update('stopLoss', value)}
                        hint="Episode terminates when NAV falls this far below its starting value."
                      />
                    </div>
                    {draft.maxAssetWeight * enabledAssets.length < 100 && (
                      <div role="alert" className="mt-7 flex gap-3 rounded-2xl border border-amber-700/15 bg-amber-50 p-4 text-amber-900">
                        <ShieldAlert size={17} className="shrink-0" aria-hidden="true" />
                        <p className="text-xs leading-5">
                          A {draft.maxAssetWeight}% cap across {enabledAssets.length} assets cannot reach full
                          investment ({draft.maxAssetWeight * enabledAssets.length}% max). Raise the cap or add assets.
                        </p>
                      </div>
                    )}
                  </Card>

                  <Card className="p-6">
                    <div className="flex items-center gap-2">
                      <h2 className="text-lg font-semibold tracking-[-.02em]">Advisory targets</h2>
                      <Badge tone="warning">Metadata only</Badge>
                    </div>
                    <p className="mt-1 text-xs text-ink/60">
                      Recorded in the exported file header for your operator and future policy layer. They do{' '}
                      <strong>not</strong> change engine behavior and never block this workflow.
                    </p>
                    <div className="mt-7 grid gap-x-10 gap-y-8 sm:grid-cols-2">
                      <SliderField label="Minimum cash target" value={draft.minCash} min={0} max={30} onChange={(value) => update('minCash', value)} hint="A hard cash floor requires an environment extension." />
                      <SliderField label="Annual turnover budget" value={draft.maxTurnover} min={25} max={400} step={25} onChange={(value) => update('maxTurnover', value)} hint="The engine currently applies a soft turnover penalty, not a hard budget." />
                    </div>
                  </Card>
                </div>

                <div className="space-y-5">
                  <Card className="p-6">
                    <p className="font-mono text-[11px] uppercase tracking-[.15em] text-ink/55">Configuration summary</p>
                    <div className="mt-6 space-y-5">
                      {[
                        ['Assets', `${enabledAssets.length}`, '5–55 supported'],
                        ['Max invested', `${Math.min(100, enabledAssets.length * draft.maxAssetWeight)}%`, 'under the global cap'],
                        ['Action space', `${enabledAssets.length + 1}`, 'cash + risky assets'],
                        ['Observation size', `${10 * enabledAssets.length + 32}`, 'self-state features enabled'],
                      ].map(([label, value, note]) => (
                        <div key={label} className="flex items-end justify-between border-b border-line pb-4 last:border-0 last:pb-0">
                          <span>
                            <span className="block text-xs font-semibold">{label}</span>
                            <span className="mt-1 block text-[11px] text-ink/55">{note}</span>
                          </span>
                          <span className="font-display text-2xl">{value}</span>
                        </div>
                      ))}
                    </div>
                  </Card>
                  <div className="rounded-[22px] border border-coral/20 bg-[#fff4ed] p-5">
                    <p className="text-xs font-semibold text-ink">Hard vs. advisory controls</p>
                    <p className="mt-2 text-[11px] leading-5 text-ink/70">
                      Only the "Enforced constraints" card changes engine behavior. Advisory targets travel as
                      commented metadata in the exported YAML so nothing is silently promised.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {step === 2 && (
              <div className="grid gap-5 xl:grid-cols-[1.25fr_.75fr]">
                <Card className="p-6">
                  <h2 className="text-lg font-semibold tracking-[-.02em]">Training design</h2>
                  <p className="mt-1 text-xs text-ink/60">Budget, benchmark, and seed coverage are engine-enforced. The objective profile is advisory metadata.</p>
                  <div className="mt-7 grid gap-x-10 gap-y-8 sm:grid-cols-2">
                    <Field label="Evaluation benchmark" hint="written to universe.benchmark">
                      <Select value={draft.benchmarkAssetId} onChange={(event) => update('benchmarkAssetId', event.target.value)}>
                        {enabledAssets.map((asset) => <option key={asset.id} value={asset.id}>{asset.name} ({asset.ticker})</option>)}
                      </Select>
                    </Field>
                    <Field label="Training budget" hint="global timesteps">
                      <Select value={draft.trainingBudget} onChange={(event) => update('trainingBudget', Number(event.target.value))}>
                        <option value={10_000_000}>10M · Screening</option>
                        <option value={50_000_000}>50M · Standard</option>
                        <option value={120_000_000}>120M · Extended</option>
                      </Select>
                    </Field>
                    <Field label="Seed coverage" hint="runs per window">
                      <Select value={draft.seedCount} onChange={(event) => update('seedCount', Number(event.target.value))}>
                        <option value={1}>1 seed · Exploratory</option>
                        <option value={3}>3 seeds · Standard</option>
                        <option value={5}>5 seeds · High confidence</option>
                      </Select>
                    </Field>
                    <Field label="Objective profile" hint="advisory metadata only">
                      <Select value={draft.objective} onChange={(event) => update('objective', event.target.value as MandateDraft['objective'])}>
                        <option value="balanced">Balanced</option>
                        <option value="return">Return seeking</option>
                        <option value="drawdown">Capital preservation</option>
                        <option value="benchmark">Benchmark aware</option>
                      </Select>
                    </Field>
                  </div>
                  <p className="mt-3 rounded-xl bg-ink/[.035] p-3 text-[11px] leading-5 text-ink/60">
                    The objective profile does not modify reward terms. Reward changes are research decisions
                    made with your operator through the spec workflow, where they can be A/B tested honestly.
                  </p>
                  <div className="mt-6">
                    <Toggle
                      checked={draft.reproducible}
                      onChange={(value) => update('reproducible', value)}
                      label="Reproducible episode resets"
                      description="Deterministic per-environment seed streams; same-seed runs reproduce."
                    />
                  </div>
                </Card>
                <Card className="overflow-hidden">
                  <div className="bg-pine p-6 text-cream">
                    <p className="font-mono text-[11px] uppercase tracking-[.15em] text-mint/70">Planned workload</p>
                    <p className="font-display mt-4 text-5xl">{plan.totalJobs}</p>
                    <p className="mt-1 text-xs text-cream/70">training jobs · {plan.windows.length} windows × {plan.seeds.length} seeds</p>
                  </div>
                  <div className="p-6">
                    <div className="space-y-4 text-[11px]">
                      <div className="flex justify-between"><span className="text-ink/60">Windows</span><span className="font-mono">W1–W5</span></div>
                      <div className="flex justify-between"><span className="text-ink/60">Seeds</span><span className="font-mono">{plan.seeds.join(', ')}</span></div>
                      <div className="flex justify-between"><span className="text-ink/60">Steps per run</span><span className="font-mono">{draft.trainingBudget / 1_000_000}M</span></div>
                      <div className="flex justify-between border-t border-line pt-4"><span className="text-ink/60">Terminal window</span><span className="font-mono">W6 embargoed</span></div>
                    </div>
                    <p className="mt-6 rounded-xl bg-ink/[.035] p-3 text-[11px] leading-4 text-ink/60">
                      Wall-clock time depends on hardware and backend; it is deliberately not estimated in the browser.
                    </p>
                  </div>
                </Card>
              </div>
            )}

            {step === 3 && (
              <div className="grid gap-5 xl:grid-cols-[1.15fr_.85fr]">
                <Card className="p-6">
                  <div className="flex items-start justify-between gap-5">
                    <div>
                      {preflight.state === 'done' && preflight.report?.ok ? (
                        <Badge tone="success"><ShieldCheck size={11} className="mr-1.5" aria-hidden="true" />Engine preflight passed</Badge>
                      ) : preflight.state === 'done' ? (
                        <Badge tone="danger">Preflight rejected</Badge>
                      ) : (
                        <Badge tone="warning">Draft — not yet validated by the engine</Badge>
                      )}
                      <h2 className="mt-4 text-2xl font-semibold tracking-[-.03em]">{draft.name}</h2>
                      <p className="mt-2 max-w-xl text-xs leading-5 text-ink/60">{draft.description}</p>
                    </div>
                    <button onClick={() => setStep(0)} className="text-[11px] font-semibold text-pine">Edit</button>
                  </div>

                  <div className="mt-7 grid grid-cols-2 gap-3 sm:grid-cols-4">
                    {[
                      ['Assets', enabledAssets.length],
                      ['Global cap', `${draft.maxAssetWeight}%`],
                      ['Budget', `${draft.trainingBudget / 1_000_000}M`],
                      ['Jobs planned', plan.totalJobs],
                    ].map(([label, value]) => (
                      <div key={label} className="rounded-2xl bg-ink/[.035] p-4">
                        <span className="font-display text-2xl">{value}</span>
                        <span className="mt-1 block text-[11px] text-ink/55">{label}</span>
                      </div>
                    ))}
                  </div>

                  <div className="mt-7">
                    <p className="text-xs font-semibold">Selected universe</p>
                    <div className="mt-3 flex flex-wrap gap-2">{enabledAssets.map((asset) => <Badge key={asset.id}>{asset.ticker}</Badge>)}</div>
                  </div>

                  {configHash && (
                    <div className="mt-6 rounded-2xl border border-line bg-white/60 p-4">
                      <p className="font-mono text-[11px] uppercase tracking-wider text-ink/55">Config SHA-256</p>
                      <p className="mt-2 break-all font-mono text-[11px] text-ink/80">{configHash}</p>
                      <p className="mt-2 text-[11px] text-ink/55">The downloaded filename embeds the first 8 characters so the export can be matched to this exact draft.</p>
                    </div>
                  )}

                  <div className="mt-6 rounded-2xl border border-line bg-ink p-4">
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-[11px] uppercase tracking-wider text-mint/70">Run plan · {plan.windows.length} windows × {plan.seeds.length} seeds</span>
                      <button onClick={copyPlan} className="inline-flex items-center gap-1.5 text-[11px] font-semibold text-mint" aria-label="Copy run plan to clipboard">
                        <Clipboard size={13} aria-hidden="true" /> {copied ? 'Copied' : 'Copy'}
                      </button>
                    </div>
                    <pre className="mt-3 overflow-x-auto font-mono text-[11px] leading-5 text-cream/80">{plan.commands.join('\n')}</pre>
                  </div>

                  <div className="mt-6 flex flex-wrap gap-3">
                    <Button onClick={() => downloadConfig(draft, configHash ?? undefined)} disabled={validationErrors.length > 0}>
                      <Download size={15} aria-hidden="true" /> Download YAML
                    </Button>
                    <Button variant="secondary" onClick={startPreflight} disabled={!yamlText || OFFLINE_MODE || preflight.state === 'running'}>
                      <ShieldCheck size={15} aria-hidden="true" />
                      {preflight.state === 'running' ? 'Validating…' : 'Run engine preflight'}
                    </Button>
                  </div>
                  {OFFLINE_MODE && (
                    <p className="mt-3 text-[11px] leading-5 text-ink/60">
                      Engine preflight requires an active research API connection.
                    </p>
                  )}
                </Card>

                <div className="space-y-5">
                  <Card className="p-6">
                    <p className="text-sm font-semibold">Browser checks</p>
                    <p className="mt-1 text-[11px] text-ink/60">Structural checks only — not a substitute for engine preflight.</p>
                    <div className="mt-5 space-y-4">
                      {[
                        [enabledAssets.length >= 5 && enabledAssets.length <= MAX_ASSETS, `Universe has 5–${MAX_ASSETS} unique assets`],
                        [validationErrors.every((error) => !error.startsWith('Duplicate')), 'No duplicate ids or tickers'],
                        [draft.maxAssetWeight * enabledAssets.length >= 100, 'Global cap allows full investment'],
                        [validationErrors.length === 0, 'All structural checks pass'],
                      ].map(([valid, label]) => (
                        <div key={String(label)} className="flex items-center gap-3 text-xs">
                          <span className={cn('grid h-5 w-5 shrink-0 place-items-center rounded-full', valid ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-800')}>
                            {valid ? <Check size={11} aria-hidden="true" /> : <X size={11} aria-hidden="true" />}
                          </span>
                          <span className="text-ink/70">{label}</span>
                        </div>
                      ))}
                    </div>
                    {validationErrors.length > 0 && (
                      <div role="alert" className="mt-4 rounded-xl bg-amber-50 p-3">
                        {validationErrors.map((error) => <p key={error} className="text-[11px] leading-5 text-amber-900">{error}</p>)}
                      </div>
                    )}
                  </Card>

                  {preflight.state !== 'idle' && (
                    <Card className="p-6">
                      <p className="text-sm font-semibold">Engine preflight report</p>
                      {preflight.state === 'running' && <p className="mt-3 text-xs text-ink/60">Validating with the live config parser…</p>}
                      {preflight.state === 'failed' && <p role="alert" className="mt-3 text-xs text-red-900">Preflight request failed: {preflight.error}</p>}
                      {preflight.state === 'done' && preflight.report && (
                        <div className="mt-3 space-y-3">
                          <p className="text-[11px] text-ink/55">Validated with: {preflight.report.validated_with}</p>
                          {preflight.report.n_assets != null && <p className="text-xs text-ink/70">Parsed universe: {preflight.report.n_assets} assets</p>}
                          {preflight.report.errors.map((error) => (
                            <p key={error} role="alert" className="rounded-lg bg-red-50 p-2.5 font-mono text-[11px] leading-4 text-red-900">{error}</p>
                          ))}
                          {preflight.report.warnings.map((warning) => (
                            <p key={warning} className="rounded-lg bg-amber-50 p-2.5 text-[11px] leading-4 text-amber-900">{warning}</p>
                          ))}
                          {preflight.report.ok && preflight.report.warnings.length === 0 && (
                            <p className="rounded-lg bg-emerald-50 p-2.5 text-[11px] text-emerald-800">Config parses cleanly with no curriculum warnings.</p>
                          )}
                        </div>
                      )}
                    </Card>
                  )}

                  <div className="rounded-[22px] border border-amber-700/15 bg-amber-50 p-5">
                    <ShieldAlert size={17} className="text-amber-700" aria-hidden="true" />
                    <p className="mt-3 text-xs font-semibold text-amber-950">Operator handoff required</p>
                    <p className="mt-2 text-[11px] leading-5 text-amber-900/80">
                      The browser exports configuration and a run plan; it never starts compute or reads OOS
                      holdouts. Launching runs remains a deliberate, audited CLI action.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>

        <div className="mt-7 flex items-center justify-between border-t border-line pt-6">
          <Button variant="ghost" onClick={() => setStep((current) => Math.max(0, current - 1))} disabled={step === 0}>
            <ArrowLeft size={15} aria-hidden="true" /> Back
          </Button>
          {step < 3 && (
            <Button onClick={() => setStep((current) => Math.min(3, current + 1))} disabled={!canAdvance}>
              Continue <ArrowRight size={15} aria-hidden="true" />
            </Button>
          )}
        </div>
      </div>

      <AddInstrumentModal
        open={showAddAsset}
        existingTickers={draft.assets.map((asset) => asset.ticker)}
        existingIds={draft.assets.map((asset) => asset.id)}
        onClose={() => setShowAddAsset(false)}
        onAdd={addAsset}
        title="Add instrument"
        description="Search market data by ticker. Fee, slippage, and holding are applied from the cost schedule — not entered by hand."
        maxInstruments={MAX_ASSETS}
      />
    </div>
  )
}
