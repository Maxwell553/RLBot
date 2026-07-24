import type { ApiResultRow, ApiRun, ApiSummary, Asset } from './types'
import { withGroupCosts } from './transaction-costs'

/** Default supported universe — costs come from the asset-class schedule. */
export const defaultAssets: Asset[] = [
  withGroupCosts({ id: 'sp500', name: 'S&P 500', ticker: 'SPY', group: 'Equity', enabled: true }),
  withGroupCosts({ id: 'gold', name: 'Gold', ticker: 'GLD', group: 'Commodity', enabled: true }),
  withGroupCosts({ id: 'oil', name: 'Crude Oil', ticker: 'USO', group: 'Commodity', enabled: true }),
  withGroupCosts({ id: 'eurusd', name: 'Euro / US Dollar', ticker: 'EURUSD=X', group: 'FX', enabled: true }),
  withGroupCosts({ id: 'usdjpy', name: 'US Dollar / Yen', ticker: 'USDJPY=X', group: 'FX', enabled: true }),
  withGroupCosts({ id: 'nikkei', name: 'Nikkei 225', ticker: '^N225', group: 'Equity', enabled: true }),
  withGroupCosts({ id: 'ftse', name: 'FTSE 100', ticker: '^FTSE', group: 'Equity', enabled: true }),
  withGroupCosts({ id: 'bond10y', name: 'US 7–10Y Treasury', ticker: 'IEF', group: 'Rates', enabled: true }),
  withGroupCosts({ id: 'copper', name: 'Copper', ticker: 'HG=F', group: 'Commodity', enabled: true }),
  withGroupCosts({ id: 'em', name: 'Emerging Markets', ticker: 'EEM', group: 'Equity', enabled: true }),
]

export const sampleSummary: ApiSummary = {
  generated_at_utc: new Date().toISOString(),
  total_runs: 4,
  completed_runs: 2,
  active_runs: 1,
  runs_with_backtest: 2,
  best_oos: { run_id: 'W4_725', sharpe: 1.0, deflated_sharpe: 0.8, window: 'W4' },
}

const sampleRunBase = {
  best_eval_score: null,
  curriculum_stage_at_best: null,
  early_stop_reason: null,
  started_at_utc: null,
  finished_at_utc: null,
  ew_excess_return: null,
  labels: [] as string[],
  warnings: [] as string[],
  comparable: true,
  git_dirty: null,
}

export const sampleRuns: ApiRun[] = [
  { ...sampleRunBase, run_id: 'W5_725', window: 'W5', training_status: 'running', progress_pct: 70, elapsed_timesteps: 35_000_000, nominal_timesteps: 50_000_000, best_eval_step: 31_000_000, oos_sharpe: null, oos_deflated_sharpe: null, oos_return: null, oos_max_drawdown: null, has_backtest: false },
  { ...sampleRunBase, run_id: 'W4_725', window: 'W4', training_status: 'completed', progress_pct: 100, elapsed_timesteps: 50_000_000, nominal_timesteps: 50_000_000, best_eval_step: 39_000_000, oos_sharpe: 1.0, oos_deflated_sharpe: 0.8, oos_return: 0.2, oos_max_drawdown: -0.1, has_backtest: true },
  { ...sampleRunBase, run_id: 'W3_725', window: 'W3', training_status: 'completed', progress_pct: 100, elapsed_timesteps: 50_000_000, nominal_timesteps: 50_000_000, best_eval_step: 35_000_000, oos_sharpe: 0.8, oos_deflated_sharpe: 0.6, oos_return: 0.15, oos_max_drawdown: -0.15, has_backtest: true },
  { ...sampleRunBase, run_id: 'W2_725', window: 'W2', training_status: 'queued', progress_pct: 0, elapsed_timesteps: null, nominal_timesteps: 50_000_000, best_eval_step: null, oos_sharpe: null, oos_deflated_sharpe: null, oos_return: null, oos_max_drawdown: null, has_backtest: false },
]

export const sampleResultRows: ApiResultRow[] = [
  { run_id: 'W1_725', cohort: '725', window: 'W1', model_ret: 0.1, model_sh: 1.0, ew_ret: 0.08, ew_sh: 0.9, spy_ret: 0.12, spy_sh: 1.1 },
  { run_id: 'W2_725', cohort: '725', window: 'W2', model_ret: 0.05, model_sh: 0.5, ew_ret: 0.04, ew_sh: 0.4, spy_ret: 0.06, spy_sh: 0.6 },
  { run_id: 'W3_725', cohort: '725', window: 'W3', model_ret: 0.15, model_sh: 0.8, ew_ret: 0.1, ew_sh: 0.7, spy_ret: 0.2, spy_sh: 0.9 },
  { run_id: 'W4_725', cohort: '725', window: 'W4', model_ret: 0.2, model_sh: 1.0, ew_ret: 0.12, ew_sh: 0.8, spy_ret: 0.1, spy_sh: 0.5 },
  { run_id: 'W5_725', cohort: '725', window: 'W5', model_ret: 0.1, model_sh: 0.9, ew_ret: 0.09, ew_sh: 0.8, spy_ret: 0.15, spy_sh: 1.0 },
]

/** @deprecated Use sampleResultRows */
export const demoResultRows = sampleResultRows
