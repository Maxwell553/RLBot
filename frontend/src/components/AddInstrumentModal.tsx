import { LoaderCircle, Plus } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { Badge, Button, Field, Input, Modal, Select } from './ui'
import { searchInstruments } from '../lib/api'
import { costsForInstrument, formatBps } from '../lib/transaction-costs'
import type { Asset, InstrumentMatch } from '../lib/types'
import { cn } from '../lib/utils'

type InstrumentDraft = {
  name: string
  ticker: string
  group: Asset['group']
}

const emptyDraft: InstrumentDraft = { name: '', ticker: '', group: 'Alternative' }

function instrumentId(name: string, existingIds: string[]): string {
  const base = name.trim().toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '') || 'asset'
  if (!existingIds.includes(base)) return base
  let n = 2
  while (existingIds.includes(`${base}_${n}`)) n += 1
  return `${base}_${n}`
}

export type AddedInstrument = {
  id: string
  name: string
  ticker: string
  group: Asset['group']
  feeBps: number
  slippageBps: number
  holdingCostBps: number
}

export function AddInstrumentModal({
  open,
  existingTickers,
  existingIds,
  onClose,
  onAdd,
  title = 'Add instrument',
  description = 'Search market data by ticker. Costs are applied from the asset-class schedule once a match is selected.',
  maxInstruments = 55,
}: {
  open: boolean
  existingTickers: string[]
  existingIds: string[]
  onClose: () => void
  onAdd: (instrument: AddedInstrument) => void
  title?: string
  description?: string
  maxInstruments?: number
}) {
  const [draft, setDraft] = useState<InstrumentDraft>(emptyDraft)
  const [error, setError] = useState<string | null>(null)
  const [lookupState, setLookupState] = useState<'idle' | 'loading' | 'found' | 'missing' | 'error'>('idle')
  const [lookupMessage, setLookupMessage] = useState<string | null>(null)
  const [suggestions, setSuggestions] = useState<InstrumentMatch[]>([])
  const [searching, setSearching] = useState(false)
  const appliedCosts = useMemo(
    () => costsForInstrument(draft.ticker, draft.group),
    [draft.ticker, draft.group],
  )

  const reset = useCallback(() => {
    setDraft(emptyDraft)
    setError(null)
    setLookupState('idle')
    setLookupMessage(null)
    setSuggestions([])
    setSearching(false)
  }, [])

  useEffect(() => {
    if (!open) reset()
  }, [open, reset])

  const applyMatch = (match: InstrumentMatch) => {
    setDraft({
      name: match.name,
      ticker: match.symbol,
      group: match.group,
    })
    setSuggestions([])
    setLookupState(match.found ? 'found' : 'missing')
    setLookupMessage(
      match.found
        ? `Resolved via market data${match.exchange ? ` · ${match.exchange}` : ''}${match.currency ? ` · ${match.currency}` : ''}`
        : 'Symbol not found in market data.',
    )
  }

  useEffect(() => {
    if (!open) return
    const query = draft.ticker.trim()
    if (query.length < 2) {
      setSuggestions([])
      setSearching(false)
      return
    }

    const controller = new AbortController()
    const timer = window.setTimeout(async () => {
      setSearching(true)
      setLookupState('loading')
      setLookupMessage('Checking this symbol against market data…')
      try {
        const results = await searchInstruments(query, controller.signal)
        if (controller.signal.aborted) return
        setSuggestions(results)
        const exact = results.find((result) => result.symbol.toUpperCase() === query.toUpperCase())
        if (exact) {
          applyMatch(exact)
        } else {
          setLookupState('missing')
          setLookupMessage('Select a market match. A symbol match does not yet confirm final training eligibility.')
        }
      } catch (searchError) {
        if (!controller.signal.aborted) {
          setSuggestions([])
          setLookupState('error')
          setLookupMessage(searchError instanceof Error ? searchError.message : 'Market lookup failed')
        }
      } finally {
        if (!controller.signal.aborted) setSearching(false)
      }
    }, 350)

    return () => {
      controller.abort()
      window.clearTimeout(timer)
    }
  }, [draft.ticker, open])

  const handleClose = () => {
    reset()
    onClose()
  }

  const handleAdd = () => {
    const label = draft.name.trim()
    const ticker = draft.ticker.trim().toUpperCase()
    if (!label) return setError('A display name is required.')
    if (!ticker) return setError('A ticker or symbol is required.')
    if (lookupState !== 'found') {
      return setError('Enter a valid ticker recognized by market data before adding the instrument.')
    }
    if (existingTickers.some((existing) => existing.toUpperCase() === ticker)) {
      return setError(`Symbol ${ticker} is already in your universe.`)
    }
    if (existingIds.length >= maxInstruments) {
      return setError(`A mandate supports at most ${maxInstruments} instruments.`)
    }
    const costs = costsForInstrument(ticker, draft.group)
    onAdd({
      id: instrumentId(label, existingIds),
      name: label,
      ticker,
      group: draft.group,
      ...costs,
    })
    reset()
  }

  return (
    <Modal open={open} onClose={handleClose} title={title} description={description}>
      <div className="mt-6 space-y-4">
        <Field label="Symbol or ticker">
          <div className="relative">
            <Input
              value={draft.ticker}
              onChange={(event) => {
                setDraft((current) => ({ ...current, ticker: event.target.value.toUpperCase() }))
                setError(null)
                setLookupState('idle')
              }}
              autoComplete="off"
              spellCheck={false}
              placeholder="Search e.g. AAPL, QQQ, BTC-USD"
            />
            {searching && (
              <LoaderCircle
                size={16}
                className="absolute right-3 top-1/2 -translate-y-1/2 animate-spin text-ink/45"
                aria-hidden="true"
              />
            )}
          </div>
        </Field>

        {suggestions.length > 0 && (
          <div className="overflow-hidden rounded-xl border border-line bg-white">
            <p className="border-b border-line px-3 py-2 font-mono text-[10px] uppercase tracking-wider text-ink/55">
              Market matches
            </p>
            <ul className="max-h-44 divide-y divide-line overflow-y-auto">
              {suggestions.map((match) => (
                <li key={match.symbol}>
                  <button
                    type="button"
                    onClick={() => applyMatch(match)}
                    className="flex w-full items-center justify-between gap-3 px-3 py-2.5 text-left hover:bg-ink/[.03]"
                  >
                    <span>
                      <span className="block text-xs font-semibold">{match.name}</span>
                      <span className="mt-0.5 block font-mono text-[11px] text-ink/55">{match.symbol}</span>
                    </span>
                    <Badge tone="success">{match.group}</Badge>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}

        <Field label="Display name">
          <Input
            value={draft.name}
            onChange={(event) => {
              setDraft((current) => ({ ...current, name: event.target.value }))
              setError(null)
            }}
          />
        </Field>

        <Field label="Asset class">
          <Select
            value={draft.group}
            onChange={(event) => setDraft((current) => ({ ...current, group: event.target.value as Asset['group'] }))}
          >
            {['Equity', 'Commodity', 'FX', 'Rates', 'Alternative'].map((group) => (
              <option key={group}>{group}</option>
            ))}
          </Select>
        </Field>

        <div className="rounded-xl border border-line bg-ink/[.025] p-3">
          <p className="font-mono text-[10px] uppercase tracking-wider text-ink/55">Applied cost schedule</p>
          <p className="mt-1 text-[11px] leading-4 text-ink/60">
            Auto-filled from the asset-class schedule (or the published default-universe row when the
            ticker is known). 1 bps = 0.01%. Fee and slippage apply to traded notional; holding is annual carry.
          </p>
          <dl className="mt-3 grid grid-cols-3 gap-2 text-center">
            <div className="rounded-lg bg-paper px-2 py-2">
              <dt className="text-[10px] text-ink/55">Fee</dt>
              <dd className="mt-1 font-mono text-xs font-semibold">{formatBps(appliedCosts.feeBps)} bps</dd>
            </div>
            <div className="rounded-lg bg-paper px-2 py-2">
              <dt className="text-[10px] text-ink/55">Slippage</dt>
              <dd className="mt-1 font-mono text-xs font-semibold">{formatBps(appliedCosts.slippageBps)} bps</dd>
            </div>
            <div className="rounded-lg bg-paper px-2 py-2">
              <dt className="text-[10px] text-ink/55">Holding / yr</dt>
              <dd className="mt-1 font-mono text-xs font-semibold">{formatBps(appliedCosts.holdingCostBps)} bps</dd>
            </div>
          </dl>
        </div>

        {lookupMessage && (
          <p
            role="status"
            className={cn(
              'rounded-lg p-2.5 text-[11px] leading-4',
              lookupState === 'found' && 'bg-emerald-50 text-emerald-900',
              lookupState === 'missing' && 'bg-amber-50 text-amber-900',
              lookupState === 'error' && 'bg-red-50 text-red-900',
              lookupState === 'loading' && 'bg-ink/[.035] text-ink/65',
            )}
          >
            {lookupState === 'loading' ? 'Checking symbol with market data…' : lookupMessage}
          </p>
        )}

        {error && (
          <p role="alert" className="rounded-lg bg-red-50 p-2.5 text-xs text-red-900">{error}</p>
        )}
      </div>
      <div className="mt-6 flex justify-end gap-2">
        <Button variant="ghost" onClick={handleClose}>Cancel</Button>
        <Button onClick={handleAdd} disabled={lookupState === 'loading'}>
          <Plus size={14} aria-hidden="true" /> Add instrument
        </Button>
      </div>
    </Modal>
  )
}
