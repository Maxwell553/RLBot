export type Asset = {
  id: string
  name: string
  ticker: string
  group: 'Equity' | 'Commodity' | 'FX' | 'Rates' | 'Alternative'
  enabled: boolean
  feeBps: number
  slippageBps: number
  holdingCostBps: number
}

export type MandateDraft = {
  name: string
  description: string
  capital: number
  baseCurrency: string
  assets: Asset[]
  /** Global per-asset cap enforced by the engine (percent). */
  maxAssetWeight: number
  /** Stop-loss fraction enforced by the engine (percent drawdown). */
  stopLoss: number
  /** Advisory only — recorded as mandate metadata, not engine behavior. */
  minCash: number
  /** Advisory only — recorded as mandate metadata, not engine behavior. */
  maxTurnover: number
  /** Advisory only — recorded as mandate metadata, not engine behavior. */
  objective: 'balanced' | 'return' | 'drawdown' | 'benchmark'
  /** Asset id (not ticker) used for universe.benchmark. */
  benchmarkAssetId: string
  trainingBudget: number
  seedCount: number
  reproducible: boolean
}

// ---------------------------------------------------------------------------
// API DTOs — must match scripts/frontend_api.py responses.
// ---------------------------------------------------------------------------

export type ApiRun = {
  run_id: string
  window: string | null
  training_status: string | null
  progress_pct: number | null
  elapsed_timesteps: number | null
  nominal_timesteps: number | null
  best_eval_step: number | null
  best_eval_score: number | null
  curriculum_stage_at_best: string | null
  early_stop_reason: string | null
  started_at_utc: string | null
  finished_at_utc: string | null
  oos_sharpe: number | null
  oos_deflated_sharpe: number | null
  oos_return: number | null
  oos_max_drawdown: number | null
  ew_excess_return: number | null
  has_backtest: boolean
  labels: string[]
  warnings: string[]
  comparable: boolean
  git_dirty: boolean | null
}

export type ApiSummary = {
  generated_at_utc: string
  total_runs: number
  completed_runs: number
  active_runs: number
  runs_with_backtest: number
  best_oos: { run_id: string; sharpe: number; deflated_sharpe: number | null; window: string | null } | null
}

export type ApiRunsPage = {
  runs: ApiRun[]
  total: number
  offset: number
  limit: number
  counts: {
    all: number
    completed: number
    active: number
    interrupted: number
    with_backtest: number
  }
}

export type ApiDashboard = {
  generated_at_utc: string
  summary: ApiSummary
  recent_runs: ApiRun[]
  window_sharpes: { window: string; sharpe: number }[]
}

export type ApiRunDetail = {
  run_id: string
  audit: ApiRun | null
  provenance: {
    git_commit: string | null
    git_dirty: boolean | null
    config_hash: string | null
    data_cache_hash: string | null
    started_at_utc: string | null
    finished_at_utc: string | null
  }
  holdout: Record<string, unknown> | null
  universe: Record<string, unknown> | null
  backtest: {
    checkpoint_label: string | null
    oos_window: string | null
    total_return: number | null
    sharpe: number | null
    excess_sharpe: number | null
    max_drawdown: number | null
    deflated_sharpe: number | null
    deflated_sharpe_excess: number | null
    oos_trials_for_window: number | null
    oos_trials_conservative: number | null
    equal_weight_daily_return: number | null
    excess_return_vs_equal_weight: number | null
    hash_drift: unknown
    n_bars: number | null
    portfolio_diagnostics: Record<string, number> | null
  } | null
}

export type ApiResultRow = {
  run_id: string
  cohort: string
  window: string
  model_ret: number
  model_sh: number
  ew_ret: number | null
  ew_sh: number | null
  spy_ret: number | null
  spy_sh: number | null
  has_benchmarks?: boolean
}

export type ApiResultsCoverage = {
  source: string
  published_rows: number
  published_runs: number
  runs_with_backtest: number
  runs_with_benchmarks?: number
  total_runs: number
}

export type ApiResults = {
  generated_at_utc: string
  available: boolean
  cohorts: string[]
  rows: ApiResultRow[]
  coverage?: ApiResultsCoverage
}

export type PreflightReport = {
  ok: boolean
  errors: string[]
  warnings: string[]
  n_assets: number | null
  milestones: Record<string, unknown> | null
  validated_with: string
}

export type InstrumentMatch = {
  found: boolean
  symbol: string
  name: string
  group: Asset['group']
  exchange: string | null
  currency: string | null
}

/**
 * Every data fetch resolves to one of four explicit states. There is no
 * silent fallback from a configured-but-failing API to cached workspace data.
 */
export type DataState<T> =
  | { kind: 'loading' }
  | { kind: 'offline' }
  | { kind: 'live'; data: T }
  | { kind: 'error'; message: string }
