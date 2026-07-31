import { Activity, RefreshCw, TrendingUp } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState, type MouseEvent } from 'react'
import { Badge, Card, ErrorPanel, Skeleton } from '../components/ui'
import { fetchForward } from '../lib/api'
import { fmtDate, fmtNum, fmtPct } from '../lib/format'
import type {
  ApiForward,
  ApiForwardCandle,
  ApiForwardMark,
  ApiForwardStats,
  DataState,
} from '../lib/types'
import { useAutoRefresh } from '../lib/use-auto-refresh'

const SERIES = [
  { key: 'model' as const, label: 'Model (best)', color: '#1f3d34' },
  { key: 'equal_weight' as const, label: 'Equal-weight 10', color: '#b0892e' },
  { key: 'spy' as const, label: 'S&P (SPY)', color: '#c45c3e' },
]

const RANGE_OPTIONS = [
  { id: '1d', label: '1D' },
  { id: '1w', label: '1W' },
  { id: '1m', label: '1M' },
  { id: 'mtd', label: 'MTD' },
  { id: '1y', label: '1Y' },
  { id: '5y', label: '5Y' },
] as const

type RangeId = (typeof RANGE_OPTIONS)[number]['id']
type SeriesKey = (typeof SERIES)[number]['key']

/** UI poll; Yahoo 5m NAV marks refresh about every 5 minutes server-side. */
const FORWARD_POLL_MS = 60_000
const BARS_PER_YEAR_5M = 78 * 252
/** Hide annualized Sharpe until we have ~1 trading month of daily closes. */
const MIN_TRADING_DAYS_FOR_SHARPE = 21
const BARS_PER_YEAR_DAILY = 252
const MIN_BARS_FOR_SHARPE_5M = 156 // legacy gate; daily-resample path is authoritative
const MIN_BARS_FOR_SHARPE_DAILY = 20

function fmtNav(value: number): string {
  if (!Number.isFinite(value)) return '—'
  if (Math.abs(value) >= 1_000_000) return `$${(value / 1_000_000).toFixed(2)}M`
  return `$${Math.round(value).toLocaleString()}`
}

function parseTs(iso: string): Date {
  if (/^\d{4}-\d{2}-\d{2}$/.test(iso)) return new Date(`${iso}T00:00:00`)
  const d = new Date(iso)
  if (!Number.isNaN(d.getTime())) return d
  return new Date(`${iso}T00:00:00`)
}

function rangeStart(range: RangeId, asOf: Date): Date {
  const d = new Date(asOf)
  if (range === '1d') {
    return new Date(asOf.getFullYear(), asOf.getMonth(), asOf.getDate())
  }
  if (range === '1w') {
    d.setDate(d.getDate() - 7)
    return d
  }
  if (range === '1m') {
    d.setMonth(d.getMonth() - 1)
    return d
  }
  if (range === 'mtd') {
    return new Date(asOf.getFullYear(), asOf.getMonth(), 1)
  }
  if (range === '1y') {
    d.setFullYear(d.getFullYear() - 1)
    return d
  }
  d.setFullYear(d.getFullYear() - 5)
  return d
}

function asNumArray(value: unknown): number[] {
  return Array.isArray(value) ? value.filter((v): v is number => typeof v === 'number' && Number.isFinite(v)) : []
}

function maxDrawdown(navs: number[]): number | null {
  if (!navs || navs.length < 2) return navs?.length ? 0 : null
  let peak = navs[0]
  let worst = 0
  for (const v of navs) {
    peak = Math.max(peak, v)
    worst = Math.min(worst, v / Math.max(peak, 1e-12) - 1)
  }
  return worst
}

function dailyCloses(navs: number[], stamps: string[]): number[] {
  if (!navs?.length || !stamps?.length || navs.length !== stamps.length) return []
  const out: number[] = []
  let lastDay = ''
  for (let i = 0; i < navs.length; i++) {
    const day = parseTs(stamps[i]).toDateString()
    if (day !== lastDay) {
      out.push(navs[i])
      lastDay = day
    } else {
      out[out.length - 1] = navs[i]
    }
  }
  return out
}

function annualizedSharpe(
  navs: number[],
  barsPerYear: number,
  stamps?: string[],
): number | null {
  // Intraday: never annualize raw 5m returns — that yields absurd 5–10+ Sharpes
  // on a few calm sessions. Resample to daily closes and require ~1 month.
  const useDaily = barsPerYear > 1000 && stamps && stamps.length === navs.length
  const series = useDaily ? dailyCloses(navs, stamps) : navs
  const annFactor = useDaily ? BARS_PER_YEAR_DAILY : barsPerYear > 1000 ? BARS_PER_YEAR_DAILY : barsPerYear
  const minBars = useDaily
    ? MIN_TRADING_DAYS_FOR_SHARPE
    : barsPerYear > 1000
      ? MIN_BARS_FOR_SHARPE_5M
      : MIN_BARS_FOR_SHARPE_DAILY
  if (series.length < minBars) return null
  const rets: number[] = []
  for (let i = 1; i < series.length; i++) {
    const a = series[i - 1]
    const b = series[i]
    if (!Number.isFinite(a) || !Number.isFinite(b) || a <= 0 || b <= 0) continue
    rets.push(Math.log(b / a))
  }
  if (rets.length < minBars - 1) return null
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length
  const var_ = rets.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(rets.length - 1, 1)
  const std = Math.sqrt(var_)
  // Flat / near-flat books → hide rather than explode.
  if (!(std > 1e-5)) return null
  const sharpe = (mean / std) * Math.sqrt(annFactor)
  if (!Number.isFinite(sharpe)) return null
  if (Math.abs(sharpe) > 5) return null
  return sharpe
}

function windowStats(navs: number[] | undefined, barsPerYear: number, stamps?: string[]): ApiForwardStats {
  if (!navs?.length) return { total_return: null, sharpe: null, max_drawdown: null, nav: null }
  const start = navs[0]
  const end = navs[navs.length - 1]
  return {
    total_return: start > 0 ? end / start - 1 : null,
    sharpe: annualizedSharpe(navs, barsPerYear, stamps),
    max_drawdown: maxDrawdown(navs),
    nav: end,
  }
}

function rebaseCandles(rows: ApiForwardCandle[], scale: number): ApiForwardCandle[] {
  return rows.map((r) => ({
    t: r.t,
    o: r.o * scale,
    h: r.h * scale,
    l: r.l * scale,
    c: r.c * scale,
  }))
}

function markTimestamps(mark: ApiForwardMark): string[] {
  if (mark.timestamps?.length) return mark.timestamps
  if (mark.candles?.model?.length) return mark.candles.model.map((c) => c.t)
  return Array.isArray(mark.dates) ? mark.dates : []
}

/** Slice mark to the selected lookback and rebase to initial_cash at window start. */
function sliceMark(mark: ApiForwardMark, range: RangeId): {
  mark: ApiForwardMark
  clipped: boolean
  availableBars: number
} {
  const stamps = markTimestamps(mark)
  if (!stamps.length) return { mark, clipped: false, availableBars: 0 }
  const asOf = parseTs(stamps[stamps.length - 1])
  const start = rangeStart(range, asOf)
  let i0 = stamps.findIndex((d) => parseTs(d) >= start)
  if (i0 < 0) i0 = 0
  if (range === '1d' && i0 === stamps.length - 1 && stamps.length >= 2) {
    i0 = Math.max(0, stamps.length - 2)
  }
  const n = stamps.length - i0
  const barsPerYear = mark.bar_interval === '5m' || mark.bar_interval === '30m' ? BARS_PER_YEAR_5M : 252

  const sliceNav = (vals: number[] | undefined) => {
    const raw = (vals ?? []).slice(i0)
    if (!raw.length) return []
    const scale = mark.initial_cash / Math.max(raw[0], 1e-12)
    return raw.map((v) => v * scale)
  }

  const sliceCandleSeries = (rows: ApiForwardCandle[] | undefined) => {
    if (!rows?.length) return []
    const raw = rows.slice(i0)
    if (!raw.length) return []
    const scale = mark.initial_cash / Math.max(raw[0].o, 1e-12)
    return rebaseCandles(raw, scale)
  }

  const modelCandles = sliceCandleSeries(mark.candles?.model)
  const spyCandles = sliceCandleSeries(mark.candles?.spy)
  const ewCandles = sliceCandleSeries(mark.candles?.equal_weight)

  const model =
    modelCandles.length > 0 ? modelCandles.map((c) => c.c) : sliceNav(mark.nav?.model)
  const spy = spyCandles.length > 0 ? spyCandles.map((c) => c.c) : sliceNav(mark.nav?.spy)
  const equal_weight =
    ewCandles.length > 0 ? ewCandles.map((c) => c.c) : sliceNav(mark.nav?.equal_weight)

  const windowStamps = stamps.slice(i0)
  const sliced: ApiForwardMark = {
    ...mark,
    n_bars: n,
    dates: windowStamps,
    timestamps: windowStamps,
    nav: { model, spy, equal_weight },
    candles:
      modelCandles.length || spyCandles.length || ewCandles.length
        ? {
            model: modelCandles,
            spy: spyCandles,
            equal_weight: ewCandles,
          }
        : mark.candles,
    stats: {
      model: windowStats(model, barsPerYear, windowStamps),
      spy: windowStats(spy, barsPerYear, windowStamps),
      equal_weight: windowStats(equal_weight, barsPerYear, windowStamps),
    },
    weights: mark.weights ? mark.weights.slice(i0) : mark.weights,
  }
  return {
    mark: sliced,
    clipped: i0 === 0 && parseTs(stamps[0]) > start,
    availableBars: stamps.length,
  }
}

function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0]
  if (max <= min) return [min]
  const span = max - min
  const raw = span / Math.max(count - 1, 1)
  const pow = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * pow).find((s) => s >= raw) ?? raw
  const start = Math.floor(min / step) * step
  const ticks: number[] = []
  // Keep labels inside the plot domain — floor(start) can sit below yMin and
  // otherwise renders under the x-axis (e.g. a stray $98k tick).
  for (let v = start; v <= max + step * 1e-9; v += step) {
    if (v + step * 1e-6 < min) continue
    if (v - step * 1e-6 > max) break
    ticks.push(Number(v.toPrecision(12)))
    if (ticks.length > 12) break
  }
  // Never emit a tick outside [min, max] — stray lows render under the x-axis.
  const clipped = ticks.filter((v) => v >= min - 1e-9 && v <= max + 1e-9)
  return clipped.length ? clipped : [min, max]
}

function dateTickIndexes(stamps: string[], maxTicks = 5): number[] {
  const n = stamps.length
  if (n <= 0) return []
  if (n === 1) return [0]
  const times = stamps.map(parseTs)
  const t0 = times[0].getTime()
  const t1 = times[n - 1].getTime()
  const spanMs = Math.max(t1 - t0, 1)
  const intraday = spanMs < 36 * 3_600_000 && stamps.some((s) => s.includes('T') || s.length > 10)

  if (intraday) {
    // Snap labels to round clock marks (15m / 1h) so axes don't look jittery.
    const stepMin = spanMs <= 4 * 3_600_000 ? 15 : 60
    const stepMs = stepMin * 60_000
    const firstAligned = Math.ceil(t0 / stepMs) * stepMs
    const targets: number[] = [t0]
    for (let t = firstAligned; t < t1; t += stepMs) targets.push(t)
    targets.push(t1)
    const stride = Math.max(1, Math.ceil((targets.length - 2) / Math.max(maxTicks - 2, 1)))
    const kept = targets.filter((_, i) => i === 0 || i === targets.length - 1 || i % stride === 0)
    const idxs: number[] = []
    for (const target of kept) {
      let best = 0
      let bestDist = Infinity
      for (let i = 0; i < times.length; i++) {
        const dist = Math.abs(times[i].getTime() - target)
        if (dist < bestDist) {
          bestDist = dist
          best = i
        }
      }
      if (!idxs.includes(best)) idxs.push(best)
    }
    return idxs.sort((a, b) => a - b)
  }

  const count = Math.min(maxTicks, n)
  const out: number[] = []
  for (let i = 0; i < count; i++) {
    out.push(Math.round((i * (n - 1)) / (count - 1)))
  }
  return [...new Set(out)]
}

function shortDate(iso: string, opts: { timeOnly?: boolean } = {}): string {
  const d = parseTs(iso)
  if (Number.isNaN(d.getTime())) return iso.slice(5, 16)
  const hasTime = iso.includes('T') || iso.length > 10
  if (hasTime && opts.timeOnly) {
    return d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })
  }
  if (hasTime) {
    return d.toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    })
  }
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

function fmtAxisNav(value: number): string {
  if (!Number.isFinite(value)) return '—'
  // Compact whole-dollar labels keep the y-axis readable at ~$100k.
  if (Math.abs(value) >= 10_000) return `$${Math.round(value).toLocaleString()}`
  return `$${value.toFixed(0)}`
}

type HoverPoint = {
  index: number
  date: string
  model: number | null
  equal_weight: number | null
  spy: number | null
}

function NavChart({ mark }: { mark: ApiForwardMark }) {
  const [hover, setHover] = useState<HoverPoint | null>(null)

  const plot = useMemo(() => {
    const stamps = markTimestamps(mark)
    const nav = mark.nav ?? { model: [], spy: [], equal_weight: [] }
    const series = SERIES.map((s) => ({
      ...s,
      values: asNumArray(nav[s.key]),
    })).filter((s) => s.values.length > 0)
    const lengths = series.map((s) => s.values.length)
    const n = Math.max(stamps.length, ...(lengths.length ? lengths : [0]), 0)
    const all = series.flatMap((s) => s.values)
    const cash = Number.isFinite(mark.initial_cash) ? mark.initial_cash : 100_000
    const dataMin = all.length ? Math.min(...all) : cash
    const dataMax = all.length ? Math.max(...all) : cash
    // Floor span so a near-flat cash book still has readable vertical room
    // without the SPY/EW range crushing the model line into the midline.
    const span = Math.max(dataMax - dataMin, cash * 0.0015, 50)
    const pad = Math.max(span * 0.15, 25)
    const yMin = dataMin - pad
    const yMax = dataMax + pad
    const sameDay =
      stamps.length >= 2 &&
      parseTs(stamps[0]).toDateString() === parseTs(stamps[stamps.length - 1]).toDateString()
    return {
      stamps,
      series,
      n,
      yMin,
      yMax,
      yTicks: niceTicks(yMin, yMax, 5),
      timeOnly: sameDay,
    }
  }, [mark])

  const width = 760
  const height = 320
  const margin = { top: 16, right: 20, bottom: 44, left: 64 }
  const innerW = width - margin.left - margin.right
  const innerH = height - margin.top - margin.bottom

  const xAt = (i: number) => {
    if (plot.n <= 1) return margin.left + innerW / 2
    return margin.left + (i / (plot.n - 1)) * innerW
  }
  const yAt = (v: number) => {
    const span = Math.max(plot.yMax - plot.yMin, 1e-9)
    return margin.top + innerH - ((v - plot.yMin) / span) * innerH
  }

  const pathFor = (values: number[]) => {
    if (!values.length) return ''
    return values
      .map((v, i) => `${i === 0 ? 'M' : 'L'}${xAt(i).toFixed(2)},${yAt(v).toFixed(2)}`)
      .join(' ')
  }

  const onMove = (event: MouseEvent<SVGSVGElement>) => {
    if (plot.n === 0) return
    const rect = event.currentTarget.getBoundingClientRect()
    const scaleX = width / rect.width
    const localX = (event.clientX - rect.left) * scaleX
    let idx = 0
    if (plot.n > 1) {
      idx = Math.round(((localX - margin.left) / Math.max(innerW, 1e-9)) * (plot.n - 1))
      idx = Math.max(0, Math.min(plot.n - 1, idx))
    }
    const at = (key: SeriesKey) => {
      const vals = mark.nav?.[key]
      return vals && vals[idx] != null ? vals[idx] : null
    }
    setHover({
      index: idx,
      date: plot.stamps[idx] ?? `bar ${idx + 1}`,
      model: at('model'),
      equal_weight: at('equal_weight'),
      spy: at('spy'),
    })
  }

  const xTicks = dateTickIndexes(plot.stamps)

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="h-auto w-full"
        role="img"
        aria-label="Forward 5-minute NAV: model vs equal-weight and SPY"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        <defs>
          <clipPath id="forward-nav-clip">
            <rect x={margin.left} y={margin.top} width={innerW} height={innerH} />
          </clipPath>
        </defs>

        {plot.yTicks.map((tick) => {
          const y = yAt(tick)
          // Skip any tick that would paint on/under the x-axis gutter.
          if (y < margin.top - 0.5 || y > margin.top + innerH + 0.5) return null
          return (
            <g key={`y-${tick}`}>
              <line
                x1={margin.left}
                x2={width - margin.right}
                y1={y}
                y2={y}
                stroke="currentColor"
                className="text-ink/10"
                strokeWidth={1}
              />
              <text
                x={margin.left - 8}
                y={y + 3}
                textAnchor="end"
                className="fill-ink/55"
                style={{ fontSize: 10, fontFamily: 'var(--font-mono), ui-monospace, monospace' }}
              >
                {fmtAxisNav(tick)}
              </text>
            </g>
          )
        })}

        <line
          x1={margin.left}
          x2={margin.left}
          y1={margin.top}
          y2={margin.top + innerH}
          stroke="currentColor"
          className="text-ink/25"
          strokeWidth={1.2}
        />
        <line
          x1={margin.left}
          x2={width - margin.right}
          y1={margin.top + innerH}
          y2={margin.top + innerH}
          stroke="currentColor"
          className="text-ink/25"
          strokeWidth={1.2}
        />

        <text
          x={14}
          y={margin.top + innerH / 2}
          textAnchor="middle"
          transform={`rotate(-90 14 ${margin.top + innerH / 2})`}
          className="fill-ink/45"
          style={{ fontSize: 10, fontFamily: 'var(--font-sans), sans-serif' }}
        >
          NAV ($)
        </text>
        <text
          x={margin.left + innerW / 2}
          y={height - 6}
          textAnchor="middle"
          className="fill-ink/45"
          style={{ fontSize: 10, fontFamily: 'var(--font-sans), sans-serif' }}
        >
          Time (5m marks)
        </text>

        {xTicks.map((i) => {
          const x = xAt(i)
          const label = plot.stamps[i]
            ? shortDate(plot.stamps[i], { timeOnly: plot.timeOnly })
            : String(i + 1)
          return (
            <g key={`x-${i}`}>
              <line
                x1={x}
                x2={x}
                y1={margin.top + innerH}
                y2={margin.top + innerH + 4}
                stroke="currentColor"
                className="text-ink/30"
                strokeWidth={1}
              />
              <text
                x={x}
                y={margin.top + innerH + 18}
                textAnchor="middle"
                className="fill-ink/55"
                style={{ fontSize: 10, fontFamily: 'var(--font-mono), ui-monospace, monospace' }}
              >
                {label}
              </text>
            </g>
          )
        })}

        <g clipPath="url(#forward-nav-clip)">
          {plot.series.map((s) => (
            <g key={s.key}>
              <path
                d={pathFor(s.values)}
                fill="none"
                stroke={s.color}
                strokeWidth={s.key === 'model' ? 2.4 : 1.7}
                strokeLinejoin="round"
                strokeLinecap="round"
                opacity={s.key === 'model' ? 1 : 0.9}
              />
              {plot.n <= 40 &&
                s.values.map((v, i) => (
                  <circle
                    key={`${s.key}-${i}`}
                    cx={xAt(i)}
                    cy={yAt(v)}
                    r={s.key === 'model' ? 2.4 : 1.8}
                    fill={s.color}
                  />
                ))}
            </g>
          ))}
        </g>

        {hover && (
          <line
            x1={xAt(hover.index)}
            x2={xAt(hover.index)}
            y1={margin.top}
            y2={margin.top + innerH}
            stroke="currentColor"
            className="text-ink/25"
            strokeWidth={1}
            strokeDasharray="3 3"
          />
        )}
      </svg>

      {hover && (
        <div className="pointer-events-none absolute right-3 top-3 min-w-[11rem] rounded-xl border border-line bg-paper/95 px-3 py-2 shadow-[var(--shadow-card)] backdrop-blur-sm">
          <p className="font-mono text-[10px] text-ink/55">{shortDate(hover.date)}</p>
          <dl className="mt-1.5 space-y-1 text-[11px]">
            {SERIES.map((s) => (
              <div key={s.key} className="flex items-center justify-between gap-4">
                <dt className="inline-flex items-center gap-1.5 text-ink/65">
                  <span className="h-2 w-2 rounded-full" style={{ backgroundColor: s.color }} aria-hidden="true" />
                  {s.label}
                </dt>
                <dd className="font-mono text-ink/90">{fmtNav(hover[s.key] ?? Number.NaN)}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}

      <div className="mt-3 flex flex-wrap gap-4 text-[11px]">
        {SERIES.map((s) => (
          <span key={s.key} className="inline-flex items-center gap-2 text-ink/70">
            <span className="h-2 w-4 rounded-sm" style={{ backgroundColor: s.color }} aria-hidden="true" />
            {s.label}
          </span>
        ))}
      </div>
      <p className="mt-1.5 text-[10px] text-ink/45">
        5-minute MTM marks · prices refresh about every 5 minutes (Yahoo intraday history tops out
        near 60 days).
      </p>
    </div>
  )
}

function StatsCard({
  title,
  color,
  stats,
}: {
  title: string
  color: string
  stats: ApiForwardStats
}) {
  return (
    <div className="rounded-2xl border border-line bg-white/60 p-4">
      <div className="flex items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} aria-hidden="true" />
        <p className="text-xs font-semibold">{title}</p>
      </div>
      <dl className="mt-3 grid grid-cols-2 gap-3 text-[11px]">
        <div>
          <dt className="text-ink/55">Return</dt>
          <dd className="mt-0.5 font-mono text-ink/90">{fmtPct(stats.total_return, 2)}</dd>
        </div>
        <div>
          <dt className="text-ink/55">Sharpe</dt>
          <dd className="mt-0.5 font-mono text-ink/90">
            {stats.sharpe == null ? '—' : fmtNum(stats.sharpe)}
          </dd>
        </div>
        <div>
          <dt className="text-ink/55">Max DD</dt>
          <dd className="mt-0.5 font-mono text-ink/90">{fmtPct(stats.max_drawdown, 2)}</dd>
        </div>
        <div>
          <dt className="text-ink/55">NAV</dt>
          <dd className="mt-0.5 font-mono text-ink/90">
            {stats.nav == null ? '—' : fmtNav(stats.nav)}
          </dd>
        </div>
      </dl>
    </div>
  )
}

function WeightsPanel({ mark }: { mark: ApiForwardMark }) {
  const positions = mark.positions
  const latest = mark.latest_weights
  const rows =
    positions && positions.length > 0
      ? positions.map((p) => ({
          label: p.label,
          weight: Number(p.weight),
          value: Number(p.value_usd),
          price: p.price,
          ticker: p.ticker,
        }))
      : latest
        ? Object.entries(latest)
            .map(([label, weight]) => ({
              label,
              weight: Number(weight),
              value: Number(weight) * Number(mark.stats?.model?.nav ?? mark.initial_cash),
              price: null as number | null,
              ticker: label,
            }))
            .filter((r) => Number.isFinite(r.weight))
            .sort((a, b) => {
              if (a.label.toLowerCase() === 'cash') return -1
              if (b.label.toLowerCase() === 'cash') return 1
              return b.weight - a.weight
            })
        : []
  if (!rows.length) return null

  const cashRow = rows.find((r) => r.label.toLowerCase() === 'cash')
  const risky = rows.filter((r) => r.label.toLowerCase() !== 'cash').sort((a, b) => b.weight - a.weight)
  const cashW = cashRow?.weight ?? 0
  const investedW = Math.max(0, 1 - cashW)
  const nav = mark.stats?.model?.nav ?? mark.initial_cash
  const riskyTotal = risky.reduce((s, r) => s + r.value, 0)

  return (
    <div>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold">Current positions</p>
          <p className="mt-1 text-[11px] text-ink/60">
            Marked to latest closes · book NAV {fmtNav(nav)}
            {mark.live?.as_of_bar ? ` · prices through ${mark.live.as_of_bar}` : ''}
          </p>
        </div>
        {cashW >= 0.6 ? (
          <Badge tone="warning">Cash park · {(cashW * 100).toFixed(1)}% cash</Badge>
        ) : (
          <Badge tone="success">{(investedW * 100).toFixed(1)}% invested</Badge>
        )}
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <div className="rounded-xl border border-line bg-white/50 p-3">
          <p className="text-[10px] uppercase tracking-wide text-ink/50">Cash</p>
          <p className="mt-1 font-mono text-lg text-ink/90">{(cashW * 100).toFixed(1)}%</p>
          <p className="mt-0.5 font-mono text-[11px] text-ink/55">{fmtNav(cashRow?.value ?? 0)}</p>
        </div>
        <div className="rounded-xl border border-line bg-white/50 p-3">
          <p className="text-[10px] uppercase tracking-wide text-ink/50">Risky assets</p>
          <p className="mt-1 font-mono text-lg text-ink/90">{(investedW * 100).toFixed(1)}%</p>
          <p className="mt-0.5 font-mono text-[11px] text-ink/55">{fmtNav(riskyTotal)}</p>
        </div>
      </div>

      <div className="mt-4 flex h-3 overflow-hidden rounded-full bg-ink/8">
        <div
          title={`Cash ${(cashW * 100).toFixed(1)}%`}
          style={{ width: `${Math.max(cashW * 100, 0)}%`, backgroundColor: '#8a9a92' }}
        />
        <div
          title={`Invested ${(investedW * 100).toFixed(1)}%`}
          style={{ width: `${Math.max(investedW * 100, 0)}%`, backgroundColor: '#1f3d34' }}
        />
      </div>

      <p className="mt-5 text-[11px] font-semibold text-ink/70">
        Risky sleeve{investedW < 0.05 ? ' (tiny — model is almost fully in cash)' : ''}
      </p>
      {investedW < 1e-6 ? (
        <p className="mt-2 text-[11px] text-ink/55">No risky holdings right now.</p>
      ) : (
        <>
          <div className="mt-2 flex h-2.5 overflow-hidden rounded-full bg-ink/8">
            {risky.map((row, i) => (
              <div
                key={row.label}
                title={`${row.label}: ${((row.weight / Math.max(investedW, 1e-12)) * 100).toFixed(1)}% of risky`}
                style={{
                  width: `${(row.weight / Math.max(investedW, 1e-12)) * 100}%`,
                  backgroundColor: SERIES[i % SERIES.length]?.color ?? '#5c6b7a',
                }}
              />
            ))}
          </div>
          <div className="mt-3 grid gap-1.5 sm:grid-cols-2">
            {risky.map((row) => (
              <div key={row.label} className="flex justify-between gap-3 text-[11px]">
                <span className="truncate text-ink/65">
                  {row.label}
                  {row.price != null ? (
                    <span className="text-ink/40"> · px {row.price.toFixed(2)}</span>
                  ) : null}
                </span>
                <span className="shrink-0 font-mono text-ink/85">
                  {(row.weight * 100).toFixed(2)}% · {fmtNav(row.value)}
                </span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

function RangeToggle({
  value,
  onChange,
}: {
  value: RangeId
  onChange: (next: RangeId) => void
}) {
  return (
    <div className="inline-flex flex-wrap gap-1 rounded-full border border-line bg-paper p-1" role="group" aria-label="Chart range">
      {RANGE_OPTIONS.map((opt) => {
        const active = opt.id === value
        return (
          <button
            key={opt.id}
            type="button"
            onClick={() => onChange(opt.id)}
            className={`rounded-full px-2.5 py-1 font-mono text-[10px] font-semibold transition ${
              active ? 'bg-pine text-paper' : 'text-ink/60 hover:bg-ink/[.04] hover:text-ink/85'
            }`}
            aria-pressed={active}
          >
            {opt.label}
          </button>
        )
      })}
    </div>
  )
}

export function ForwardPage() {
  const [state, setState] = useState<DataState<ApiForward>>({ kind: 'loading' })
  const [refreshing, setRefreshing] = useState(false)
  const [refreshError, setRefreshError] = useState<string | null>(null)
  const [range, setRange] = useState<RangeId>('1d')

  const reload = useCallback(async (mode: 'hard' | 'poll' | 'force' = 'hard') => {
    const force = mode === 'hard' || mode === 'force'
    if (mode === 'hard') {
      setState((current) => (current.kind === 'live' ? current : { kind: 'loading' }))
      setRefreshing(true)
    } else {
      setRefreshing(true)
    }
    const next = await fetchForward('', undefined, { forceRefresh: force })
    if (next.kind === 'live') {
      setState(next)
      setRefreshError(null)
    } else if (next.kind === 'error') {
      setRefreshError(next.message)
      // Keep prior chart on soft poll errors; hard/force surfaces the failure.
      if (mode === 'hard' || mode === 'force') {
        setState((current) => (current.kind === 'live' ? current : next))
      }
    } else {
      setState(next)
    }
    setRefreshing(false)
  }, [])

  useEffect(() => {
    void reload('hard')
  }, [reload])

  useAutoRefresh(() => void reload('poll'), {
    enabled: state.kind === 'live' || state.kind === 'error',
    refreshing,
    intervalMs: FORWARD_POLL_MS,
  })

  const mark = state.kind === 'live' && state.data.available ? state.data.mark : null
  const emptyMessage = state.kind === 'live' ? state.data.message : null

  const windowed = useMemo(() => (mark ? sliceMark(mark, range) : null), [mark, range])
  const view = windowed?.mark ?? null

  const asOf = useMemo(() => {
    if (!mark) return null
    const stamps = markTimestamps(mark)
    if (!stamps.length) return null
    return stamps[stamps.length - 1]
  }, [mark])

  return (
    <div className="px-5 py-6 sm:px-8 lg:px-10">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[.2em] text-ink/55">Forward test</p>
          <h1 className="mt-2 font-display text-3xl text-ink">Live book vs benchmarks</h1>
          <p className="mt-2 max-w-2xl text-sm text-ink/60">
            Five-minute NAV marks from the live book vs equal-weight and SPY. Prices refresh about
            every 5 minutes.
          </p>
          {state.kind === 'live' && refreshing && (
            <p className="mt-2 text-[11px] text-ink/55">Refreshing forward mark…</p>
          )}
          {refreshError && <p className="mt-2 text-[11px] text-red-800">{refreshError}</p>}
        </div>
        <button
          type="button"
          onClick={() => void reload('force')}
          className="inline-flex h-10 items-center gap-2 rounded-full border border-line bg-paper px-4 text-xs font-semibold"
        >
          <RefreshCw size={14} aria-hidden="true" /> Refresh
        </button>
      </header>

      {state.kind === 'loading' && <Skeleton className="mt-6 h-80" />}
      {state.kind === 'offline' && (
        <p className="mt-6 text-sm text-ink/60">Connect the research API to load forward marks.</p>
      )}
      {state.kind === 'error' && (
        <div className="mt-6">
          <ErrorPanel message={state.message} onRetry={() => void reload('hard')} />
        </div>
      )}

      {state.kind === 'live' && !mark && (
        <Card className="mt-6 p-6">
          <Badge tone="warning">No forward mark</Badge>
          <p className="mt-3 text-sm text-ink/70">{emptyMessage}</p>
          <pre className="mt-4 overflow-x-auto rounded-xl bg-ink/[.04] p-4 text-[11px] leading-5 text-ink/80">
{`# Export / refresh the live NAV series (also runs via daily launchd)
python scripts/forward_mark.py --run-id LIVE_MODEL --refresh-data
# or the full daily loop:
bash scripts/daily_live_forward.sh --run-id LIVE_MODEL`}
          </pre>
        </Card>
      )}

      {mark && view && (
        <>
          <div className="mt-6 flex flex-wrap items-center gap-3 text-[11px] text-ink/60">
            <Badge tone="success">{mark.bar_interval ? `${mark.bar_interval} marks` : 'Live MTM'}</Badge>
            <span className="font-mono text-ink/80">{mark.run_id}</span>
            <span>· checkpoint {mark.checkpoint_label}</span>
            <span>· as of {asOf}</span>
            <span>· start {fmtNav(mark.initial_cash)}</span>
            <span>· {mark.n_bars} bars total</span>
            {mark.live?.as_of_utc ? (
              <span>· marked {fmtDate(mark.live.as_of_utc)}</span>
            ) : (
              <span>· generated {fmtDate(mark.generated_at_utc)}</span>
            )}
          </div>

          <section className="mt-5 grid gap-5 lg:grid-cols-[1.45fr_.55fr]">
            <Card className="p-6">
              <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <TrendingUp size={16} className="text-pine" aria-hidden="true" />
                  <p className="text-sm font-semibold">NAV · 5m marks</p>
                </div>
                <RangeToggle value={range} onChange={setRange} />
              </div>
              {windowed?.clipped ? (
                <p className="mb-3 text-[10px] text-ink/50">
                  Only {windowed.availableBars} forward bar{windowed.availableBars === 1 ? '' : 's'} since
                  deploy — showing all available history for this range.
                </p>
              ) : (
                <p className="mb-3 text-[10px] text-ink/50">
                  {(() => {
                    const stamps = markTimestamps(view)
                    return (
                      <>
                        Window {stamps[0] ?? '—'} → {stamps[stamps.length - 1] ?? '—'} ·{' '}
                        {view.n_bars} bar{view.n_bars === 1 ? '' : 's'} · NAVs rebased at window start
                        {(mark.bar_interval === '5m' || mark.bar_interval === '30m')
                          ? ' · Sharpe hidden until ~1 month of daily closes'
                          : ''}
                      </>
                    )
                  })()}
                </p>
              )}
              <NavChart mark={view} />
              {mark.note ? <p className="mt-4 text-[10px] leading-4 text-ink/50">{mark.note}</p> : null}
            </Card>
            <div className="space-y-4">
              {SERIES.map((s) => (
                <StatsCard
                  key={s.key}
                  title={s.label}
                  color={s.color}
                  stats={view.stats?.[s.key] ?? {
                    total_return: null,
                    sharpe: null,
                    max_drawdown: null,
                    nav: null,
                  }}
                />
              ))}
            </div>
          </section>

          <Card className="mt-5 p-6">
            <div className="mb-2 flex items-center gap-2">
              <Activity size={16} className="text-pine" aria-hidden="true" />
              <p className="text-sm font-semibold">Allocation</p>
            </div>
            <WeightsPanel mark={mark} />
          </Card>
        </>
      )}
    </div>
  )
}
