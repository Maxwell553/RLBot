"""Paper-trade measurement harness (NOT a deployment / live-capital path).

Calls scripts/infer_weights.py for one or more as-of dates and appends the intended
target weights + turnover-vs-previous to a JSONL log. There is **no broker adapter** and
**no market-impact / capacity model** (see docs/claude-review-20260605.md P2-E): this is
measurement infrastructure to observe what the policy *would* do, not a trading system.

    python scripts/paper_trade.py --run-id <ID> --dates 2022-01-03,2022-02-01

Logs to Runs/<run_id>/paper_trade/log.jsonl (gitignored under Runs/).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlbot.run_artifacts import PROJECT_ROOT, RunPaths  # noqa: E402

REPO = PROJECT_ROOT


def _infer_one(run_id: str, as_of: str, checkpoint: str) -> dict:
    """Run infer_weights for one date and return its payload dict."""
    out_path = RunPaths(run_id).run_meta_dir / "paper_trade" / f"target_weights_{as_of}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "infer_weights.py"),
        "--run-id", run_id,
        "--checkpoint", checkpoint,
        "--as-of", as_of,
        "--out", str(out_path),
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO))
    return json.loads(out_path.read_text(encoding="utf-8"))


def _turnover(prev: dict | None, curr: dict) -> float:
    """L1/2 turnover between two target-weight dicts (0 if no prior)."""
    if prev is None:
        return 0.0
    keys = set(prev["target_weights"]) | set(curr["target_weights"])
    return 0.5 * sum(
        abs(curr["target_weights"].get(k, 0.0) - prev["target_weights"].get(k, 0.0))
        for k in keys
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dates", required=True, help="Comma-separated as-of dates (YYYY-MM-DD).")
    parser.add_argument("--checkpoint", default="best", choices=("best", "final"))
    args = parser.parse_args()

    run_id = args.run_id.strip()
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    log_path = RunPaths(run_id).run_meta_dir / "paper_trade" / "log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    prev: dict | None = None
    with log_path.open("a", encoding="utf-8") as f:
        for d in dates:
            payload = _infer_one(run_id, d, args.checkpoint)
            entry = {
                "as_of": d,
                "run_id": run_id,
                "checkpoint": args.checkpoint,
                "cash_weight": payload["cash_weight"],
                "gross_exposure": payload["gross_exposure"],
                "turnover_vs_prev": _turnover(prev, payload),
                "target_weights": payload["target_weights"],
            }
            f.write(json.dumps(entry) + "\n")
            print(
                f"[paper] {d}: cash={entry['cash_weight']:.2f} "
                f"gross={entry['gross_exposure']:.2f} turnover={entry['turnover_vs_prev']:.3f}"
            )
            prev = payload
    print(f"[paper] appended {len(dates)} entries to {log_path} (measurement only — no broker)")


if __name__ == "__main__":
    main()
