# paper_trade — measurement, not deployment

This is **measurement infrastructure**, not a trading system. It observes what a trained
policy *would* allocate on given dates. There is **no broker adapter** and **no
market-impact, capacity, borrow/financing, or spread-crossing model** beyond the static
per-asset costs in `config.yaml`. Do not read paper-trade output as a live-capital claim
(see `docs/claude-review-20260605.md`, P2-E and the "capacity/impact before live framing"
note).

## Usage

```bash
# Emit today's target weights with full provenance (config/data/model hashes):
python scripts/infer_weights.py --run-id <RUN_ID> --checkpoint best --as-of 2022-12-31

# Log intended weights + turnover across several as-of dates (measurement loop):
python scripts/paper_trade.py --run-id <RUN_ID> --dates 2022-01-03,2022-02-01,2022-03-01
```

Outputs are written under `Runs/<run_id>/paper_trade/` (gitignored):
- `target_weights_<as_of>.json` — one audited weight vector per date (cash + per-asset,
  action provenance, config/data/model hashes, VecNormalize path, warmup bars).
- `log.jsonl` — one line per as-of date with cash/gross/turnover-vs-previous.

## Contract

`infer_weights.py` reuses the **same** observation pipeline, recurrent warmup, and frozen
`VecNormalize` as backtest (`rollout_policy_on_slice`), binds the run-local config + data
snapshot by default, and validates that the emitted weights are a long-only simplex with
each risky leg ≤ the configured cap. A broker adapter, if ever added, must consume this
audited payload — it must not re-derive weights.
