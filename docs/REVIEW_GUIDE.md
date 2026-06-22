# Reviewer Guide

This guide is for a professor or collaborator who wants to understand
MarketTrainer without digging through local experiment artifacts first.

## What This Repository Claims

MarketTrainer is a reproducible research stack for testing a recurrent PPO/LSTM
portfolio allocator under strict chronological out-of-sample evaluation. The
main contribution is the evaluation and artifact discipline, not a claim of a
deployable trading edge.

The core safeguards are:

- chronological holdout reservation before any train/eval split;
- causal feature construction with bounded preroll in independent split mode;
- next-open execution and next-close mark-to-market;
- robust benchmark-relative checkpoint selection using in-training validation
blocks only;
- frozen `VecNormalize` statistics for OOS inference;
- per-run config/data snapshots and OOS burn-ledger accounting.

## Fast Reading Order

1. [README.md](../README.md) for the system overview and canonical commands.
2. [docs/RESEARCH.md](RESEARCH.md) for the currently published walk-forward
results and caveats.
3. [docs/TRAINING.md](TRAINING.md) for the exact training/backtest protocol.
4. [AGENTS.md](../AGENTS.md) for invariants that span modules.
5. Code entry points:
   - [scripts/train.py](../scripts/train.py)
   - [scripts/backtest.py](../scripts/backtest.py)
   - [rlbot/trading_env.py](../rlbot/trading_env.py)
   - [rlbot/data_utils.py](../rlbot/data_utils.py)
   - [rlbot/eval_selection.py](../rlbot/eval_selection.py)
   - [rlbot/research/oos_ledger.py](../rlbot/research/oos_ledger.py)

## Reproducibility Boundaries

Large run artifacts live under `Runs/` and are intentionally gitignored. A local
run directory contains the model weights, matched normalization state, config
snapshot, data snapshot, manifest, logs, plots, and backtest summary for that
specific run. Tracked docs copy only small durable result tables.

For a publication handoff, the minimum artifact bundle should include:

- the tracked commit hash;
- each selected run's `manifest.json`, `config.yaml`, and `backtest_summary.json`;
- config/data hashes reported by the summaries;
- the relevant `Runs/oos_ledger.jsonl` rows or a frozen ledger extract;
- the script used to regenerate figures/tables from those summaries.

Do not evaluate publication claims from an untracked local `Runs/` tree alone.

## Local Verification

Install the repo:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the invariant tests:

```bash
pytest
```

Torch-free or fast checks can be run selectively:

```bash
pytest tests/test_core.py tests/test_reward_terms.py tests/test_block_bootstrap.py -q
pytest -k "holdout or vecnorm or oos_ledger" -q
```

Use `tests/test_oos_ledger_and_significance.py` for the DSR and holdout-ledger
accounting checks.

## Questions Worth Stress-Testing

- Does the feature split ever let validation or OOS see future state?
- Does checkpoint selection use only in-training validation blocks?
- Are best-model weights always paired with the VecNormalize statistics from
the same evaluation step?
- How sensitive are the reported results to seed, exposure-risk scale, and OOS
trial count?
- Are the OOS ledger counts sufficient for the Deflated Sharpe interpretation?
- Which claims remain true if mixed-code exploratory cohorts are excluded?

## Current Caveats

- The published W1-W5 holdouts have been read many times; selection-aware DSR is
below the usual 0.95 bar.
- Two seeds per exposure setting is not enough for a strong stochastic-RL claim.
- Some historical cohorts have mixed-code caveats and should be treated as
exploratory unless explicitly frozen in a publication bundle.
- yfinance daily bars, simple costs, and no capacity model are sufficient for
method research but not for live-trading claims.
