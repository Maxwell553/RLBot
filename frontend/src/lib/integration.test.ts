import { describe, expect, it } from 'vitest'
import { parse } from 'yaml'
import { defaultAssets } from './demo-data'
import {
  MAX_ASSETS,
  assetIdFor,
  buildConfigYaml,
  buildRunPlan,
  slugify,
  validateDraft,
} from './integration'
import type { Asset, MandateDraft } from './types'

const baseDraft: MandateDraft = {
  name: 'Test Mandate',
  description: '',
  capital: 500_000,
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

type ParsedConfig = {
  universe: { benchmark: string; assets: Record<string, string> }
  environment: { initial_cash: number; max_single_asset_weight: number; stop_loss_fraction: number }
  reward: { benchmark_cap_weights: number[] }
  transaction_costs: { slippage: number[]; tx_fee: number[]; annual_holding_cost: number[] }
  training: { timesteps: number; reproducible: boolean }
}

function parsedConfig(draft: MandateDraft): ParsedConfig {
  return parse(buildConfigYaml(draft)) as ParsedConfig
}

describe('buildConfigYaml', () => {
  it('exports the full default universe with aligned cost arrays', () => {
    const config = parsedConfig(baseDraft)
    expect(Object.keys(config.universe.assets)).toHaveLength(10)
    expect(config.transaction_costs.slippage).toHaveLength(10)
    expect(config.transaction_costs.tx_fee).toHaveLength(10)
    expect(config.transaction_costs.annual_holding_cost).toHaveLength(10)
    expect(config.reward.benchmark_cap_weights).toHaveLength(10)
  })

  it('preserves per-asset annual holding costs instead of zeroing them', () => {
    const config = parsedConfig(baseDraft)
    // GOLD is the second asset: 40 bps/year in the base config.
    expect(config.transaction_costs.annual_holding_cost[1]).toBeCloseTo(0.004, 10)
    // EM is last: 67 bps/year.
    expect(config.transaction_costs.annual_holding_cost[9]).toBeCloseTo(0.0067, 10)
  })

  it('maps the benchmark by asset id, not ticker', () => {
    const config = parsedConfig({ ...baseDraft, benchmarkAssetId: 'gold' })
    expect(config.universe.benchmark).toBe('GOLD')
    expect(config.universe.assets.GOLD).toBe('GLD')
  })

  it('shrinks all arrays consistently when assets are deselected', () => {
    const assets = baseDraft.assets.map((asset, index) => ({ ...asset, enabled: index < 5 }))
    const config = parsedConfig({ ...baseDraft, assets, maxAssetWeight: 25 })
    expect(Object.keys(config.universe.assets)).toHaveLength(5)
    expect(config.transaction_costs.annual_holding_cost).toHaveLength(5)
    const weights = config.reward.benchmark_cap_weights
    expect(weights).toHaveLength(5)
    expect(weights.reduce((sum, weight) => sum + weight, 0)).toBeCloseTo(1, 10)
  })

  it('writes enforced environment and training fields', () => {
    const config = parsedConfig({ ...baseDraft, capital: 250_000, maxAssetWeight: 30, stopLoss: 40, trainingBudget: 10_000_000, reproducible: false })
    expect(config.environment.initial_cash).toBe(250_000)
    expect(config.environment.max_single_asset_weight).toBeCloseTo(0.3, 10)
    expect(config.environment.stop_loss_fraction).toBeCloseTo(0.4, 10)
    expect(config.training.timesteps).toBe(10_000_000)
    expect(config.training.reproducible).toBe(false)
  })
})

describe('validateDraft', () => {
  it('accepts the default draft', () => {
    expect(validateDraft(baseDraft)).toEqual([])
  })

  it('rejects fewer than five assets', () => {
    const assets = baseDraft.assets.map((asset, index) => ({ ...asset, enabled: index < 4 }))
    expect(validateDraft({ ...baseDraft, assets, maxAssetWeight: 40, benchmarkAssetId: 'sp500' }).join(' ')).toMatch(/at least 5/)
  })

  it('rejects more than the engine asset ceiling', () => {
    const extra: Asset[] = Array.from({ length: MAX_ASSETS + 1 }, (_, index) => ({
      id: `asset_${index}`,
      name: `Asset ${index}`,
      ticker: `TK${index}`,
      group: 'Equity',
      enabled: true,
      feeBps: 1,
      slippageBps: 1,
      holdingCostBps: 0,
    }))
    const errors = validateDraft({ ...baseDraft, assets: extra, benchmarkAssetId: 'asset_0' })
    expect(errors.join(' ')).toMatch(/at most 55/)
  })

  it('rejects duplicate ids and duplicate tickers', () => {
    const duplicate = { ...baseDraft.assets[0] }
    const errors = validateDraft({ ...baseDraft, assets: [...baseDraft.assets, duplicate] })
    expect(errors.join(' ')).toMatch(/Duplicate asset id/)
    expect(errors.join(' ')).toMatch(/Duplicate ticker/)
  })

  it('rejects invalid tickers', () => {
    const assets = baseDraft.assets.map((asset) =>
      asset.id === 'em' ? { ...asset, ticker: 'bad ticker!' } : asset,
    )
    expect(validateDraft({ ...baseDraft, assets }).join(' ')).toMatch(/not a valid data-source symbol/)
  })

  it('rejects an infeasible global cap', () => {
    const assets = baseDraft.assets.map((asset, index) => ({ ...asset, enabled: index < 5 }))
    const errors = validateDraft({ ...baseDraft, assets, maxAssetWeight: 15 })
    expect(errors.join(' ')).toMatch(/cannot reach full investment/)
  })

  it('never gates on advisory fields like minimum cash', () => {
    expect(validateDraft({ ...baseDraft, minCash: 30, maxTurnover: 25 })).toEqual([])
  })

  it('requires the benchmark to be a selected asset', () => {
    expect(validateDraft({ ...baseDraft, benchmarkAssetId: 'missing' }).join(' ')).toMatch(/benchmark/)
  })
})

describe('buildRunPlan', () => {
  it('covers all five canonical windows and every requested seed', () => {
    const plan = buildRunPlan(baseDraft)
    expect(plan.windows).toEqual([1, 2, 3, 4, 5])
    expect(plan.seeds).toHaveLength(3)
    expect(plan.totalJobs).toBe(15)
    for (const window of plan.windows) {
      expect(plan.commands.some((command) => command.includes(`--window ${window}`))).toBe(true)
    }
    expect(plan.commands.join('\n')).toContain(`--seeds "${plan.seeds.join(' ')}"`)
  })
})

describe('helpers', () => {
  it('slugifies names safely', () => {
    expect(slugify('  Global Defensive #1 ')).toBe('global_defensive_1')
    expect(slugify('!!!')).toBe('mandate')
  })

  it('deduplicates generated asset ids', () => {
    const existing = [{ id: 'bitcoin' }, { id: 'bitcoin_2' }] as Asset[]
    expect(assetIdFor('Bitcoin', existing)).toBe('bitcoin_3')
  })
})
