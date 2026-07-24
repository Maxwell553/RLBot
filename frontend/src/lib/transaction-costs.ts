import type { Asset } from './types'

export type TransactionCostSchedule = {
  feeBps: number
  slippageBps: number
  holdingCostBps: number
}

/**
 * Asset-class cost schedule used when an instrument is added in the UI.
 * Values are in basis points (1 bps = 0.01% = 0.0001 as a fraction).
 *
 * - feeBps → transaction_costs.tx_fee (charged on traded notional at rebalance)
 * - slippageBps → transaction_costs.slippage (charged on traded notional at rebalance)
 * - holdingCostBps → transaction_costs.annual_holding_cost (annual; env uses /252 daily)
 *
 * The published default-universe arrays in config/config.yaml remain the research
 * source of truth for those ten tickers. New symbols take this class schedule so
 * operators do not hand-enter fee/holding per add.
 */
export const GROUP_COST_SCHEDULE: Record<Asset['group'], TransactionCostSchedule> = {
  Equity: { feeBps: 2, slippageBps: 2, holdingCostBps: 20 },
  Commodity: { feeBps: 3, slippageBps: 5, holdingCostBps: 60 },
  FX: { feeBps: 0.5, slippageBps: 1, holdingCostBps: 0 },
  Rates: { feeBps: 1, slippageBps: 1, holdingCostBps: 15 },
  Alternative: { feeBps: 5, slippageBps: 5, holdingCostBps: 40 },
}

/** Research-calibrated costs for the default config universe (config/config.yaml). */
export const DEFAULT_UNIVERSE_COSTS: Record<string, TransactionCostSchedule> = {
  SPY: { feeBps: 1, slippageBps: 1, holdingCostBps: 9 },
  GLD: { feeBps: 2, slippageBps: 2, holdingCostBps: 40 },
  USO: { feeBps: 2, slippageBps: 3, holdingCostBps: 83 },
  'EURUSD=X': { feeBps: 0.5, slippageBps: 1, holdingCostBps: 0 },
  'USDJPY=X': { feeBps: 0.5, slippageBps: 1, holdingCostBps: 0 },
  '^N225': { feeBps: 10, slippageBps: 5, holdingCostBps: 10 },
  '^FTSE': { feeBps: 10, slippageBps: 5, holdingCostBps: 10 },
  IEF: { feeBps: 1, slippageBps: 1, holdingCostBps: 15 },
  'HG=F': { feeBps: 5, slippageBps: 8, holdingCostBps: 60 },
  EEM: { feeBps: 2, slippageBps: 2, holdingCostBps: 67 },
}

export function costsForGroup(group: Asset['group']): TransactionCostSchedule {
  return { ...GROUP_COST_SCHEDULE[group] }
}

/** Prefer ticker-specific research costs when known; otherwise the class schedule. */
export function costsForInstrument(
  ticker: string,
  group: Asset['group'],
): TransactionCostSchedule {
  const known = DEFAULT_UNIVERSE_COSTS[ticker.trim().toUpperCase()]
    ?? DEFAULT_UNIVERSE_COSTS[ticker.trim()]
  return known ? { ...known } : costsForGroup(group)
}

export function withGroupCosts<T extends { group: Asset['group']; ticker?: string }>(
  instrument: T,
): T & TransactionCostSchedule {
  const costs = instrument.ticker
    ? costsForInstrument(instrument.ticker, instrument.group)
    : costsForGroup(instrument.group)
  return { ...instrument, ...costs }
}

export function formatBps(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1)
}
