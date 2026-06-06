"""Experiment spec: a pre-registered hypothesis + a config patch (allow-list-restricted)
expanded into concrete variants. Never touches holdout dates, the universe, or the
walk-forward split — those would change what OOS *is*."""

from __future__ import annotations

import copy
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

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("ExperimentSpec.id is required")
        assert_patch_allowed(self.patch, self.grid)


def load_spec(path: str | Path) -> ExperimentSpec:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"spec must be a mapping, got {type(data)}")
    known = ExperimentSpec.__dataclass_fields__.keys()
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown spec keys: {sorted(unknown)}")
    return ExperimentSpec(**data)


@dataclass
class Variant:
    variant_id: str
    concrete_patch: dict
    seed: int
    window: dict


def _grid_combos(grid: dict) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    value_lists = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


def _short(value: Any) -> str:
    s = str(value).replace(" ", "")
    return s[:16]


def resolve_variants(spec: ExperimentSpec) -> list[Variant]:
    """Cartesian product of grid × seeds × windows; ``patch`` applied to all."""
    windows = spec.windows or [{}]
    variants: list[Variant] = []
    for combo in _grid_combos(spec.grid):
        concrete = {**spec.patch, **combo}
        grid_tag = "_".join(f"{k.split('.')[-1]}={_short(v)}" for k, v in combo.items())
        for seed in spec.seeds:
            for window in windows:
                wname = window.get("name", "") if window else ""
                parts = [spec.id]
                if grid_tag:
                    parts.append(grid_tag)
                parts.append(f"seed{seed}")
                if wname:
                    parts.append(wname)
                variants.append(
                    Variant(
                        variant_id="__".join(parts),
                        concrete_patch=dict(concrete),
                        seed=int(seed),
                        window=dict(window),
                    )
                )
    return variants


def build_variant_config_dict(base_config_dict: dict, concrete_patch: dict) -> dict:
    """Deep-copy the base config dict and apply a (validated) concrete patch."""
    assert_patch_allowed(concrete_patch)
    out = copy.deepcopy(base_config_dict)
    for key, value in concrete_patch.items():
        set_nested(out, key, value)
    return out
