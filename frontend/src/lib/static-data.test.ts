import { afterEach, describe, expect, it, vi } from 'vitest'
import { clearStaticDataCache, loadStaticRuns } from './static-data'

afterEach(() => {
  clearStaticDataCache()
  vi.unstubAllGlobals()
})

describe('loadStaticRuns', () => {
  it('filters and paginates from the published runs snapshot', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
          runs: [
            {
              run_id: 'W1_804',
              window: 'W1',
              training_status: 'completed',
              progress_pct: 100,
              elapsed_timesteps: 1,
              nominal_timesteps: 1,
              best_eval_step: null,
              best_eval_score: null,
              curriculum_stage_at_best: null,
              early_stop_reason: null,
              started_at_utc: null,
              finished_at_utc: null,
              oos_sharpe: 1,
              oos_deflated_sharpe: null,
              oos_return: 0.1,
              oos_max_drawdown: -0.1,
              ew_excess_return: null,
              has_backtest: true,
              labels: null,
              warnings: null,
              comparable: true,
              git_dirty: null,
            },
            {
              run_id: 'W2_804',
              window: 'W2',
              training_status: 'active',
              progress_pct: 50,
              elapsed_timesteps: 1,
              nominal_timesteps: 2,
              best_eval_step: null,
              best_eval_score: null,
              curriculum_stage_at_best: null,
              early_stop_reason: null,
              started_at_utc: null,
              finished_at_utc: null,
              oos_sharpe: null,
              oos_deflated_sharpe: null,
              oos_return: null,
              oos_max_drawdown: null,
              ew_excess_return: null,
              has_backtest: false,
              labels: [],
              warnings: [],
              comparable: true,
              git_dirty: null,
            },
          ],
          total: 2,
          offset: 0,
          limit: 2,
          counts: { all: 2, completed: 1, active: 1, interrupted: 0, with_backtest: 1 },
        }),
      })),
    )

    const page = await loadStaticRuns({ status: 'completed', limit: 10 })
    expect(page.total).toBe(1)
    expect(page.runs).toHaveLength(1)
    expect(page.runs[0].run_id).toBe('W1_804')
    expect(page.runs[0].labels).toEqual([])
    expect(page.counts.all).toBe(2)
  })
})
