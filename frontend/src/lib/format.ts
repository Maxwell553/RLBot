export function fmtSteps(steps: number | null | undefined): string {
  if (steps == null) return '—'
  if (steps >= 1_000_000) return `${(steps / 1_000_000).toFixed(1)}M`
  if (steps >= 1_000) return `${(steps / 1_000).toFixed(0)}k`
  return String(steps)
}

export function fmtPct(fraction: number | null | undefined, digits = 1): string {
  if (fraction == null) return '—'
  const value = fraction * 100
  return `${value > 0 ? '+' : ''}${value.toFixed(digits)}%`
}

export function fmtNum(value: number | null | undefined, digits = 2): string {
  return value == null ? '—' : value.toFixed(digits)
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return '—'
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
}

/** Honest training-status label straight from the manifest. */
export function statusLabel(status: string | null): string {
  if (status === 'completed') return 'Completed'
  if (status === 'interrupted') return 'Interrupted'
  if (status === 'running') return 'In progress'
  if (status === 'queued') return 'Queued'
  if (!status) return 'In progress'
  return status
}

export function statusTone(status: string | null): 'success' | 'warning' | 'danger' | 'neutral' {
  if (status === 'completed') return 'success'
  if (status === 'interrupted') return 'danger'
  if (status === 'queued' || !status) return 'neutral'
  return 'warning'
}
