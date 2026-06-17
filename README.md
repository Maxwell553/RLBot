# MarketTrainer (RLBot)

MarketTrainer is a research stack for training a recurrent PPO/LSTM agent on a
daily, long-only, multi-asset portfolio environment. The core objective is
strict walk-forward evaluation: train on historical data, reserve chronological
out-of-sample windows, and select checkpoints without touching the holdout.

This repository is code-first. Large run artifacts live under `Runs/`, which is
gitignored by design. Durable results should be copied into tracked docs or a
small tracked report artifact; do not assume local `Runs/<run_id>/` files exist
for another reader.

## Where To Look

| Topic | Source |
| --- | --- |
| Config fields and defaults | [config/config.yaml](config/config.yaml), [config/README.md](config/README.md) |
| Training, checkpoint selection, backtesting | [docs/TRAINING.md](docs/TRAINING.md) |
| Walk-forward results and research notes | [docs/RESEARCH.md](docs/RESEARCH.md) |
| Modal/cloud runs | [docs/MODAL.md](docs/MODAL.md) |
| Agent/coding invariants | [AGENTS.md](AGENTS.md), [CLAUDE.md](CLAUDE.md) |
| Research automation CLI | [scripts/research.py](scripts/research.py) |
| Library code | [rlbot/](rlbot/) |

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

First run, fetch data and build the cache:

```bash
python scripts/train.py --refresh-data --window 1 --run-id <RUN_ID>
```

Train from an existing cache:

```bash
python scripts/train.py --window 1 --run-id <RUN_ID>
```

Backtest a completed run using its run-local config/cache snapshot:

```bash
python scripts/backtest.py --run-id <RUN_ID> --checkpoint best --detailed
```

Run tests:

```bash
pytest
```

For cloud training, use [docs/MODAL.md](docs/MODAL.md). For research cohorts,
use [scripts/research.py](scripts/research.py) and the workflow in
[docs/RESEARCH.md](docs/RESEARCH.md).

## Current Method At A Glance

- **Universe:** default 10 tradeable assets: `SP500`, `GOLD`, `OIL`,
  `EURUSD`, `USDJPY`, `NIKKEI`, `FTSE`, `BOND10Y`, `COPPER`, `EM`.
  Macro series such as VIX/DXY/rates are observation inputs only.
- **Action space:** `N + 1` weights, where cash competes with risky assets.
  Risky allocations are long-only and capped by `environment.max_single_asset_weight`
  after softmax projection.
- **Observation size:** `10 * N + 28`; derive it from the active universe
  instead of hard-coding dimensions.
- **Data split:** chronological OOS holdout is reserved before train/eval
  splitting. Default `feature_split_mode` is `independent`, with
  `feature_preroll_bars` causal preroll for slow indicators.
- **Execution timing:** observations use data available through the configured
  lag; rebalance fills occur at the next open and are marked to the next close.
- **Reward:** return, feasible benchmark excess, Sortino difference,
  participation/inactivity, churn, turnover, drawdown, concentration, and
  exposure-risk terms. `cash_daily_yield` defaults to `0.0`.
- **Benchmark semantics:** `universe.benchmark` is for reporting only. Reward
  shaping uses `reward.benchmark_cap_weights`; checkpoint selection uses
  `training.best_model_benchmark`.
- **Best checkpoint:** selected after the fee/churn ramp gate by robust
  benchmark-relative eval score, not raw mean NAV. The selected weights and
  matching `VecNormalize` stats are saved together under `models/best/`.
- **OOS evaluation:** use `scripts/backtest.py --run-id <RUN_ID>` so config,
  split metadata, VecNormalize, and cache provenance come from the run snapshot.

## Outputs And Reproducibility

Training writes local artifacts to `Runs/<run_id>/`:

- `manifest.json`
- `config.yaml`
- `data_cache.npz`
- `models/{best,final,checkpoints}/`
- `plots/`
- `logs/`, `tb_logs/`, `eval_logs/`
- `backtest_summary.json` after backtesting

`Runs/` is gitignored because these files are large, machine-local, and often
experimental. The run snapshot is still the source of truth for reproducing that
specific run on the same machine; it is just not a tracked publication artifact.

When a result should be shared in the repo, summarize it in
[docs/RESEARCH.md](docs/RESEARCH.md) or add a small tracked artifact under a
dedicated docs/results location. Avoid README tables that require local
gitignored files to be meaningful.

## Research Workflow

The research loop is intentionally gated:

- W1-W5 are the active walk-forward OOS windows.
- W6 is reserved as embargoed terminal validation.
- Specs under `specs/` may patch method knobs such as reward, curriculum,
  policy, environment, and training settings.
- Universe changes, transaction-cost changes, and split changes are treated as
  changes to the evaluation problem and are restricted by the research tooling.
- OOS reads are logged in `Runs/oos_ledger.jsonl`; repeated holdout probing
  should be treated as budgeted model-selection pressure.

Use small screens and seed repeats before promoting expensive full-cohort runs.
Do not judge a configuration from one lucky window.

Canonical windows used by `--window N`:

| Window | Train through | OOS holdout |
| --- | --- | --- |
| W1 | 2015-12-31 | 2016-01-01 to 2017-12-31 |
| W2 | 2017-12-31 | 2018-01-01 to 2019-12-31 |
| W3 | 2019-12-31 | 2020-01-01 to 2021-12-31 |
| W4 | 2021-12-31 | 2022-01-01 to 2023-12-31 |
| W5 | 2023-12-31 | 2024-01-01 to 2025-12-31 |
| W6 | 2025-12-31 | 2026-01-01 to 2027-12-31 |

## Development Notes

- Python target: `>=3.10`.
- Editable install: `pip install -e ".[dev]"`.
- Main CLIs: `scripts/train.py`, `scripts/backtest.py`,
  `scripts/research.py`, `scripts/infer_weights.py`.
- Installed entry points: `market-trainer-train`,
  `market-trainer-backtest`.
- There is no top-level `train.py`/`backtest.py`; use the scripts above.
- `.cache/`, `data_cache.npz`, `Runs/`, and execution/shadow-trading state are
  ignored.

Before changing data handling, environment execution, checkpoint selection, or
OOS accounting, read [AGENTS.md](AGENTS.md) for invariants that span files.
