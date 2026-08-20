"""Experiment spec: a pre-registered hypothesis + a config patch (allow-list-restricted)
expanded into concrete variants. Never touches holdout dates, the universe, or the
walk-forward split — those would change what OOS *is*."""

from __future__ import annotations

import copy
import re
import hashlib
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Config sections an experiment may patch (method knobs only).
_ALLOWED_PREFIXES = (
    "reward.",
    "curriculum.",
    "entropy_schedule.",
    "policy.",
    "hyperparameters.",
    "environment.",
)
_ALLOWED_EXACT = {"data.feature_split_mode"}

# The only base config an experiment may start from. Pointing a spec at a different
# YAML (other universe, costs, split) would bypass the patch firewall entirely.
CANONICAL_BASE_CONFIG = "config/config.yaml"

# Canonical walk-forward windows (docs/RESEARCH.md): window N trains through
# Dec-31 of (2013 + 2N) and holds out the following two calendar years. Specs may
# reference these by name or restate the exact dates — anything else would let an
# experiment place its own favorable holdout, changing what OOS *is*.
CANONICAL_WINDOWS: dict[str, dict[str, str]] = {
    f"W{n}": {
        "train_end": f"{2013 + 2 * n}-12-31",
        "holdout_start": f"{2014 + 2 * n}-01-01",
        "holdout_end": f"{2015 + 2 * n}-12-31",
    }
    for n in range(1, 7)
}

# Reserved terminal validation window: excluded from ALL research specs so at least
# one holdout stays untouched by the iterate-measure loop. Only a final pre-deployment
# validation (human-run, outside research.py) or the tier-5 shadow path may use it.
EMBARGOED_WINDOWS = frozenset({"W6"})

_WINDOW_KEYS = {"name", "train_end", "holdout_start", "holdout_end"}
_WINDOW_DATE_KEYS = ("train_end", "holdout_start", "holdout_end")


def normalize_window(window: dict) -> dict:
    """Validate one spec window against the canonical table; resolve names to dates.

    Rejects unknown keys (a typo'd date key would otherwise be silently dropped and
    the run would fall back to a different holdout than pre-registered). A window
    given by name gets the canonical dates filled in; a window given by dates must
    match a canonical window exactly.
    """
    unknown = set(window) - _WINDOW_KEYS
    if unknown:
        raise ValueError(
            f"window {window!r} has unknown key(s) {sorted(unknown)}; "
            f"allowed: {sorted(_WINDOW_KEYS)}"
        )
    name = str(window.get("name", "")).upper()
    dates = {k: str(window[k]) for k in _WINDOW_DATE_KEYS if window.get(k)}
    canon_by_name = CANONICAL_WINDOWS.get(name)
    if not dates:
        if name and canon_by_name is None:
            raise ValueError(
                f"window name {window.get('name')!r} is not canonical and gives no dates; "
                f"use one of {sorted(set(CANONICAL_WINDOWS) - set(EMBARGOED_WINDOWS))} "
                "or omit windows for the config default."
            )
        if name in EMBARGOED_WINDOWS:
            raise PermissionError(
                f"window {name} is EMBARGOED — it is the reserved terminal validation "
                "window and may not be used by research specs. Use W1–W5."
            )
        return {"name": name, **(canon_by_name or {})} if name else {}
    match = next(
        (wname for wname, c in CANONICAL_WINDOWS.items()
         if all(dates.get(k, c[k]) == c[k] for k in _WINDOW_DATE_KEYS)),
        None,
    )
    if match is None:
        raise PermissionError(
            f"window {window!r} does not match any canonical walk-forward window "
            f"(would change what OOS is). Canonical: {CANONICAL_WINDOWS}"
        )
    if canon_by_name is not None and match != name:
        raise ValueError(f"window {window!r}: name says {name} but dates match {match}")
    resolved = name or match
    if resolved in EMBARGOED_WINDOWS:
        raise PermissionError(
            f"window {resolved} is EMBARGOED — it is the reserved terminal validation "
            "window and may not be used by research specs. Use W1–W5."
        )
    return {"name": resolved, **CANONICAL_WINDOWS[match]}
_ALLOWED_TRAINING = {
    "training.reproducible",
    "training.early_stop_patience",
    "training.timesteps",
    "training.n_envs",
    "training.obs_noise",
    "training.seed",
    "training.viz_freq",
    "training.curriculum_update_freq",
    "training.checkpoint_save_freq_steps",
}


def is_allowed_patch_key(key: str) -> bool:
    """True if a dotted config key may be patched by an experiment.

    Denies (by omission) anything that changes what the OOS test is: universe.*,
    transaction_costs.*, data.* except feature_split_mode, and the split-defining
    training.holdout_days / block_size / eval_stride / eval_n_episodes.
    """
    if key in _ALLOWED_EXACT or key in _ALLOWED_TRAINING:
        return True
    return key.startswith(_ALLOWED_PREFIXES)


def assert_patch_allowed(*patches: dict) -> None:
    bad = sorted({k for p in patches for k in p if not is_allowed_patch_key(k)})
    if bad:
        raise PermissionError(
            "experiment patch targets keys outside the allow-list "
            f"(would change the OOS definition / universe / split): {bad}"
        )


def set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """Set ``d[a][b][c] = value`` for dotted_key 'a.b.c'; intermediate keys must exist."""
    parts = dotted_key.split(".")
    node = d
    for p in parts[:-1]:
        if not isinstance(node, dict) or p not in node:
            raise KeyError(f"patch key {dotted_key!r}: '{p}' not found in base config")
        node = node[p]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise KeyError(f"patch key {dotted_key!r}: '{parts[-1]}' not found in base config")
    node[parts[-1]] = value


@dataclass
class ExperimentSpec:
    id: str
    hypothesis: str = ""
    parent: str | None = None
    base_config: str = "config/config.yaml"
    patch: dict = field(default_factory=dict)  # applied to every variant
    grid: dict = field(default_factory=dict)  # dotted-key -> list, cartesian product
    seeds: list[int] = field(default_factory=lambda: [0])
    windows: list[dict] = field(default_factory=list)  # [{name, train_end, holdout_start,...}]
    timesteps: int | None = None
    checkpoint_rule: str = "best"
    evaluation_tier: int = 1
    success_gates: dict = field(default_factory=dict)
    budget: dict = field(default_factory=dict)
    # sha256 of the spec file text, set by load_spec; cohort.json records it so
    # promote can refuse a spec edited after launch.
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("ExperimentSpec.id is required")
        self.id = str(self.id)
        if re.search(r"__seed\d", self.id):
            raise ValueError(
                f"spec id {self.id!r} must not contain '__seed<digits>' — it would "
                "break the cross-seed report grouping of legacy records."
            )
        if any(int(x) < 0 for x in self.seeds):
            raise ValueError(f"seeds must be non-negative ints, got {self.seeds}")
        assert_patch_allowed(self.patch, self.grid)
        if self.success_gates:
            # Validate pre-registered gate keys NOW — a typo'd gate that first raises
            # at collect (after the compute is spent) defeats pre-registration.
            from rlbot.research.gates import SUPPORTED_SUCCESS_GATES

            unknown = set(self.success_gates) - SUPPORTED_SUCCESS_GATES
            if unknown:
                raise ValueError(
                    f"unknown success_gates key(s) {sorted(unknown)}; "
                    f"supported: {sorted(SUPPORTED_SUCCESS_GATES)}"
                )
        if self.base_config != CANONICAL_BASE_CONFIG:
            raise PermissionError(
                f"base_config must be {CANONICAL_BASE_CONFIG!r} (got {self.base_config!r}); "
                "a different base YAML would bypass the patch firewall."
            )
        self.windows = [normalize_window(dict(w)) for w in self.windows]


def load_spec(path: str | Path) -> ExperimentSpec:
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"spec must be a mapping, got {type(data)}")
    known = ExperimentSpec.__dataclass_fields__.keys()
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown spec keys: {sorted(unknown)}")
    spec = ExperimentSpec(**data)
    # Recorded in cohort.json so promote can refuse a spec edited after launch.
    spec.source_sha256 = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return spec


@dataclass
class Variant:
    variant_id: str
    concrete_patch: dict
    seed: int
    window: dict
    # variant_id minus the seed component — the cross-seed aggregation key. Without
    # it, every report group held exactly one record and "median across seeds" was a
    # median of one.
    group_id: str = ""


def _grid_combos(grid: dict) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    value_lists = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


_COHORT_DIGITS = re.compile(r"(\d{3,})$")


def spec_cohort_tag(spec_id: str) -> str:
    """``w3_dd_coupling_811`` → ``811``; otherwise the spec id as-is."""
    m = _COHORT_DIGITS.search(str(spec_id))
    return m.group(1) if m else str(spec_id)


def _grid_letter(index: int) -> str:
    """0 → a, 25 → z, 26 → aa. Distinguishes grid cells without knob-name essays."""
    if index < 26:
        return chr(ord("a") + index)
    return _grid_letter(index // 26 - 1) + chr(ord("a") + index % 26)


def _window_code(window: dict | None) -> str:
    name = str((window or {}).get("name") or "").upper()
    if name.startswith("W") and name[1:].isdigit():
        return name
    if name.isdigit():
        return f"W{name}"
    return name


def canonical_run_id(
    *,
    spec_id: str,
    window: dict | None,
    grid_index: int,
    n_grid: int,
    seed: int,
) -> tuple[str, str]:
    """Return ``(run_id, group_id)`` in ``W3_811`` / ``W3_811a`` / ``W3_811a_s42`` form.

    Seed 0 is omitted (``W3_811``). Extra seeds use ``_s{seed}``. A one-cell grid
    has no letter; two or more cells get ``a``, ``b``, … The knob values live in
    that run's ``config.yaml`` header, not the directory name.
    """
    cohort = spec_cohort_tag(spec_id)
    letter = _grid_letter(grid_index) if n_grid > 1 else ""
    wcode = _window_code(window)
    if wcode:
        group_id = f"{wcode}_{cohort}{letter}"
    elif letter:
        group_id = f"{cohort}_{letter}"
    else:
        group_id = cohort
    seed_i = int(seed)
    run_id = group_id if seed_i == 0 else f"{group_id}_s{seed_i}"
    return run_id, group_id


def variant_config_header(spec: ExperimentSpec, variant: Variant) -> str:
    """Comment block so a later reader can see the delta without the directory name."""
    lines = [
        f"# {variant.variant_id} — cohort {spec.id}",
        "#",
    ]
    hyp = " ".join(str(spec.hypothesis or "").split())
    if hyp:
        lines.append(f"# Hypothesis: {hyp}")
        lines.append("#")
    if spec.parent:
        lines.append(f"# Parent: {spec.parent}")
    lines.append("# Patch vs config/config.yaml (this file is the source of truth):")
    if variant.concrete_patch:
        for key in sorted(variant.concrete_patch):
            lines.append(f"#   {key}: {variant.concrete_patch[key]!r}")
    else:
        lines.append("#   (none — identical to base)")
    w = variant.window or {}
    if w.get("name") or w.get("train_end"):
        lines.append(
            f"# Window: {w.get('name', '')}  train_end={w.get('train_end', '')}  "
            f"holdout={w.get('holdout_start', '')}..{w.get('holdout_end', '')}"
        )
    lines.append(f"# Seed: {variant.seed}")
    lines.append("#")
    return "\n".join(lines) + "\n"


def resolve_variants(spec: ExperimentSpec) -> list[Variant]:
    """Cartesian product of grid × seeds × windows; ``patch`` applied to all."""
    windows = spec.windows or [{}]
    combos = _grid_combos(spec.grid)
    n_grid = len(combos)
    variants: list[Variant] = []
    for grid_index, combo in enumerate(combos):
        concrete = {**spec.patch, **combo}
        for seed in spec.seeds:
            for window in windows:
                run_id, group_id = canonical_run_id(
                    spec_id=spec.id,
                    window=window,
                    grid_index=grid_index,
                    n_grid=n_grid,
                    seed=int(seed),
                )
                variants.append(
                    Variant(
                        variant_id=run_id,
                        concrete_patch=dict(concrete),
                        seed=int(seed),
                        window=dict(window),
                        group_id=group_id,
                    )
                )
    ids = [v.variant_id for v in variants]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(
            f"Variant id collision in spec {spec.id!r}: {dupes}. Distinct grid cells "
            "would overwrite each other's configs and be skipped at launch — make the "
            "colliding grid values distinguishable."
        )
    return variants


def build_variant_config_dict(base_config_dict: dict, concrete_patch: dict) -> dict:
    """Deep-copy the base config dict and apply a (validated) concrete patch."""
    assert_patch_allowed(concrete_patch)
    out = copy.deepcopy(base_config_dict)
    for key, value in concrete_patch.items():
        set_nested(out, key, value)
    return out
