# Walk-forward launchers

Thin bash wrappers around `scripts/train.py`. Run from repo root (each script `cd`s to the project root).

| Script | OOS horizon | Macro context (informal) |
|--------|-------------|---------------------------|
| `window1_train.sh` | 2016-01-01 → 2017-12-31 | Post-Yuan devaluation grind |
| `window2_train.sh` | 2018-01-01 → 2019-12-31 | Vol-mageddon, trade-war shocks |
| `window3_train.sh` | 2020-01-01 → 2021-06-30 | COVID crash & recovery |
| `window4_train.sh` | 2021-07-01 → 2022-12-31 | Inflation / rapid tightening |
| `window5_train.sh` | 2023-01-01 → 2024-12-31 | Low-vol large-cap expansion |
| `window6_train.sh` | 2025-01-01 → latest bar | Choppy multi-asset rotation |

Override run id:

```bash
RUN_ID=wf_window1_001 ./scripts/walkforward/window1_train.sh
```

Validate bar counts (no training):

```bash
.venv/bin/python scripts/walkforward/validate_split.py --window 1
```

Full methodology: [RESEARCH.md](../../RESEARCH.md).

**Universe:** All windows use `universe.assets` in `config/config.yaml` (default N=10). After editing that block, run `scripts/train.py --refresh-data` once before launching a window.

**Manifest:** Each run writes `runs/<RUN_ID>/manifest.json` with `universe.tickers`, `n_assets`, and `obs_dim` for OOS backtest alignment.

Multi-seed training: [../run_seed_ensemble.sh](../run_seed_ensemble.sh) (`--window`, `--cohort`, optional `--seeds`).
