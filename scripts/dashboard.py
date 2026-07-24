#!/usr/bin/env python3
"""Lightweight local training dashboard (HTML) for idle periods while jobs run.

Reads ``Runs/`` manifests, eval logs, and optional Modal status — no server required.
Open the generated HTML in a browser; re-run this script to refresh.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path as _Path

_bootstrap_path = _Path(__file__).resolve().parent / "_bootstrap.py"
_bootstrap_spec = importlib.util.spec_from_file_location("_rlbot_repo_bootstrap", _bootstrap_path)
assert _bootstrap_spec is not None and _bootstrap_spec.loader is not None
_bootstrap_mod = importlib.util.module_from_spec(_bootstrap_spec)
_bootstrap_spec.loader.exec_module(_bootstrap_mod)

from pathlib import Path

from rlbot.curriculum_preflight import build_curriculum_preflight
from rlbot.rl_config import load_config
from rlbot.run_artifacts import PROJECT_ROOT, RUNS_ROOT
from rlbot.run_audit import audit_runs, discover_audit_run_ids


def _html_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_dashboard_html(records, *, generated_at: str) -> str:
    rows = []
    for r in records:
        warn = "; ".join(r.warnings[:3])
        labels = ",".join(r.labels)
        progress = ""
        if r.elapsed_timesteps and r.nominal_timesteps:
            pct = 100.0 * r.elapsed_timesteps / max(r.nominal_timesteps, 1)
            progress = f"{pct:.1f}%"
        rows.append(
            "<tr>"
            f"<td>{_html_escape(r.run_id)}</td>"
            f"<td>{_html_escape(r.training_status or '-')}</td>"
            f"<td>{_html_escape(progress)}</td>"
            f"<td>{r.elapsed_timesteps or '-'}</td>"
            f"<td>{r.best_eval_step or '-'}</td>"
            f"<td>{r.curriculum_stage_at_best or '-'}</td>"
            f"<td>{'' if r.oos_sharpe is None else f'{r.oos_sharpe:.2f}'}</td>"
            f"<td>{'' if r.oos_deflated_sharpe is None else f'{r.oos_deflated_sharpe:.2f}'}</td>"
            f"<td>{_html_escape(labels)}</td>"
            f"<td>{_html_escape(warn)}</td>"
            "</tr>"
        )

    # Active / incomplete runs first for the “job progress” view.
    active = [r for r in records if r.training_status not in ("completed",) or not r.oos_return]
    active_note = f"{len(active)} without completed OOS or still training"

    try:
        pf = build_curriculum_preflight(load_config())
        preflight_warn = "<br/>".join(_html_escape(w) for w in pf.warnings) or "none"
        early = "yes" if pf.early_stop_reachable else "NO — early stop unreachable"
        stationary = pf.stationary_full_dr_steps
    except Exception as exc:  # noqa: BLE001
        preflight_warn = _html_escape(str(exc))
        early = "?"
        stationary = "?"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>RLBot training dashboard</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 1.5rem; color: #111; }}
  h1 {{ font-size: 1.25rem; margin: 0 0 0.5rem; }}
  .meta {{ color: #555; font-size: 0.9rem; margin-bottom: 1rem; }}
  .warn {{ background: #fff3cd; border: 1px solid #e6d8a8; padding: 0.75rem; margin: 1rem 0; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
  th, td {{ border: 1px solid #ddd; padding: 0.35rem 0.5rem; text-align: left; }}
  th {{ background: #f4f4f4; }}
  tr:nth-child(even) {{ background: #fafafa; }}
</style>
</head>
<body>
  <h1>RLBot training dashboard</h1>
  <div class="meta">Generated { _html_escape(generated_at) } · {len(records)} runs · {active_note}</div>
  <div class="warn">
    <strong>Default schedule checks</strong><br/>
    Stationary full-DR steps: {stationary}<br/>
    Early stop reachable: {early}<br/>
    Warnings: {preflight_warn}
  </div>
  <table>
    <thead>
      <tr>
        <th>run_id</th><th>status</th><th>progress</th><th>elapsed</th>
        <th>best@</th><th>stage@best</th><th>OOS Sh</th><th>DSR</th>
        <th>labels</th><th>warnings</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
  <p class="meta">Re-run <code>python scripts/dashboard.py</code> to refresh. Modal spend/FPS live in the training terminal / TB logs.</p>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", default="")
    p.add_argument(
        "--out",
        default=str(RUNS_ROOT / "dashboard.html"),
        help="Output HTML path",
    )
    args = p.parse_args()
    ids = discover_audit_run_ids(root=PROJECT_ROOT, prefix=args.prefix)
    # Prefer recent / incomplete runs at the top of a full dump would be huge —
    # include all but HTML stays usable for dozens–hundreds of rows.
    records = audit_runs(ids, root=PROJECT_ROOT)
    records.sort(
        key=lambda r: (
            0 if r.training_status != "completed" else 1,
            r.run_id,
        )
    )
    html = build_dashboard_html(
        records,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({len(records)} runs)")


if __name__ == "__main__":
    main()
