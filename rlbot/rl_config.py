"""
Load and validate ``config.yaml`` — single source of truth for env, reward, and training params.
"""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config" / "config.yaml"

# Supported tradeable universe size (count of ``universe.assets`` keys).
UNIVERSE_MIN_ASSETS = 5
UNIVERSE_MAX_ASSETS = 55


def _req(d: dict[str, Any], key: str, section: str) -> Any:
    if key not in d:
        raise KeyError(f"config.yaml missing '{section}.{key}'")
    return d[key]


def _reward_churn_penalty(rew: dict[str, Any]) -> float:
    if "churn_penalty" in rew:
        return float(rew["churn_penalty"])
    if "churn_lambda" in rew and "churn_penalty_scale" in rew:
        return float(rew["churn_lambda"]) * float(rew["churn_penalty_scale"])
    raise KeyError(
        "config.yaml reward: set 'churn_penalty' "
        "(or legacy churn_lambda + churn_penalty_scale)"
    )


def _float_list(xs: list, name: str, expected_n: int | None = None) -> list[float]:
    if expected_n is not None and len(xs) != expected_n:
        raise ValueError(f"{name} must have {expected_n} entries, got {len(xs)}")
    return [float(x) for x in xs]


def observation_dim_for_universe(n_assets: int, n_macro: int = 4) -> int:
    """Market + live mask + portfolio + meta observation size (macro count fixed at 4 by default)."""
    return 10 * n_assets + 8 + 5 * n_macro


@dataclass(frozen=True)
class EnvironmentConfig:
    initial_cash: float
    max_episode_steps: int
    lookback: int
    obs_lag_default: int
    min_obs_lag: int
    max_obs_lag: int
    max_single_asset_weight: float
    fee_scale_default: float
    stop_loss_fraction: float
    domain_randomize_fee_dr_min: float
    domain_randomize_fee_dr_max: float
    domain_randomize_fee_beta_a: float
    domain_randomize_fee_beta_b: float
    action_smoothing_alpha: float


@dataclass(frozen=True)
class RewardConfig:
    reward_scale: float
    max_step_log_return: float
    max_step_log_return_downside: float
    risk_window: int
    sortino_min_steps: int
    sortino_downside_floor: float
    risk_bonus_scale: float
    benchmark_cap_weights: list[float]
    benchmark_excess_scale: float
    benchmark_excess_clip: float
    benchmark_combined_abs_cap: float
    churn_penalty: float
    turnover_penalty: float
    drawdown_downside_gamma: float
    drawdown_increase_penalty: float
    drawdown_level_penalty: float
    drawdown_level_floor: float
    concentration_penalty: float
    concentration_target_eff_assets: float
    cash_daily_yield: float
    inactivity_penalty_over_50: float
    inactivity_penalty_over_90: float
    eval_inactivity_penalty_scale: float
    participation_bonus: float
    participation_reward_scale: float
    exposure_risk_mode: str
    exposure_risk_penalty_scale: float

    def benchmark_cap_weights_array(self) -> np.ndarray:
        w = np.asarray(self.benchmark_cap_weights, dtype=np.float64)
        s = float(w.sum())
        if s <= 0.0:
            raise ValueError("benchmark_cap_weights must sum to a positive value")
        return w / s


@dataclass(frozen=True)
class TransactionCostsConfig:
    slippage: list[float]
    tx_fee: list[float]
    annual_holding_cost: list[float]
    trading_days_per_year: int

    def slippage_array(self) -> np.ndarray:
        return np.asarray(self.slippage, dtype=np.float64)

    def tx_fee_array(self) -> np.ndarray:
        return np.asarray(self.tx_fee, dtype=np.float64)

    def daily_holding_cost_array(self) -> np.ndarray:
        return np.asarray(self.annual_holding_cost, dtype=np.float64) / float(
            self.trading_days_per_year
        )


@dataclass(frozen=True)
class HyperparametersConfig:
    learning_rate: float
    learning_rate_floor: float
    batch_size: int
    n_steps: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    clip_range_finetune: float
    ent_coef_initial: float
    ent_coef_finetune: float
    vf_coef: float
    max_grad_norm: float
    weight_decay: float


@dataclass(frozen=True)
class PolicyConfig:
    lstm_hidden_size: int
    n_lstm_layers: int
    net_arch_pi: list[int]
    net_arch_vf: list[int]


@dataclass(frozen=True)
class TrainingConfig:
    timesteps: int
    n_envs: int
    obs_noise: float
    seed: int
    block_size: int
    eval_stride: int
    eval_n_episodes: int
    holdout_days: int
    viz_freq: int
    curriculum_update_freq: int
    checkpoint_save_freq_steps: int
    eval_freq_steps: int
    eval_freq_pre_gate_steps: int
    # When True, training envs use deterministic per-env seed streams (seed + env index)
    # instead of fresh OS entropy per episode reset (reseed_on_reset). Default False keeps
    # the diversity behavior; True makes same-seed runs reproducible.
    reproducible: bool = False
    early_stop_patience: int = 0  # >0 enables patience early-stop after curriculum completes
    best_model_score_std_coef: float = 0.75
    best_model_score_dd_coef: float = 2.0
    best_model_score_stitched_blend: float = 0.5
    best_model_benchmark: str = "equal_weight_daily"


@dataclass(frozen=True)
class VecNormalizeConfig:
    norm_obs: bool
    norm_reward_train: bool
    clip_obs: float
    clip_reward: float


@dataclass(frozen=True)
class EntropyScheduleConfig:
    explore_ent: float
    final_ent: float
    early_floor: float
    early_floor_high: float
    warmup_improvements: int
    decay_start_fraction: float
    dr_lock_fraction: float
    early_floor_fraction: float


@dataclass(frozen=True)
class CurriculumConfig:
    budget_short: int
    budget_long: int
    fee_free_fraction: float
    fee_ramp_fraction: float
    churn_ramp_floor: float
    dr_widen_span_fraction: float
    fee_free_long: int
    fee_ramp_end_long: int
    dr_widen_span_long: int
    # Step before which eval NAV is logged but models/best/ is not updated.
    # None → fee_ramp_end for the learn budget; 0 → disable gate.
    best_model_min_step: int | None = None


FEATURE_SPLIT_MODES = ("continuous", "independent")


@dataclass(frozen=True)
class DataConfig:
    since: str
    fracdiff_d: float
    feature_purge_warmup: int
    # "continuous": features precomputed on the contiguous panel then sliced into
    #   train/eval blocks (matches continuous backtest memory; purge NOT applied).
    # "independent" (default): features recomputed per segment over a BOUNDED causal
    #   preroll window of feature_preroll_bars earlier bars (sliced off after
    #   computation) so slow indicators get real warmup with uniform history depth at
    #   every segment head. The preroll bars are the adjacent train blocks, so this is
    #   NOT train/eval feature isolation — vs continuous it differs mainly in
    #   fracdiff's long tail. Panel-head bars with insufficient preroll are
    #   neutralized via feature_purge_warmup.
    feature_split_mode: str = "independent"
    feature_preroll_bars: int = 252


@dataclass(frozen=True)
class UniverseConfig:
    benchmark: str
    assets: dict[str, str]
    tickers: list[str]

    @property
    def n_assets(self) -> int:
        return len(self.tickers)


@dataclass(frozen=True)
class RLConfig:
    path: Path
    universe: UniverseConfig
    environment: EnvironmentConfig
    reward: RewardConfig
    transaction_costs: TransactionCostsConfig
    hyperparameters: HyperparametersConfig
    policy: PolicyConfig
    training: TrainingConfig
    vec_normalize: VecNormalizeConfig
    entropy_schedule: EntropyScheduleConfig
    curriculum: CurriculumConfig
    data: DataConfig
    raw: dict[str, Any] = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.raw)


def validate_universe_asset_count(n_assets: int) -> None:
    """Enforce supported panel width (``universe.assets`` key count)."""
    if n_assets < UNIVERSE_MIN_ASSETS or n_assets > UNIVERSE_MAX_ASSETS:
        raise ValueError(
            f"universe must have between {UNIVERSE_MIN_ASSETS} and {UNIVERSE_MAX_ASSETS} "
            f"tradeable assets, got {n_assets}"
        )


def _parse_universe(uni: dict[str, Any]) -> UniverseConfig:
    benchmark = str(_req(uni, "benchmark", "universe"))
    raw_assets = _req(uni, "assets", "universe")
    if not isinstance(raw_assets, dict) or not raw_assets:
        raise ValueError("universe.assets must be a non-empty mapping of label → yfinance symbol")
    assets = {str(k): str(v) for k, v in raw_assets.items()}
    tickers = list(assets.keys())
    validate_universe_asset_count(len(tickers))
    if benchmark not in assets:
        raise ValueError(
            f"universe.benchmark {benchmark!r} must be a key in universe.assets, got {tickers}"
        )
    return UniverseConfig(benchmark=benchmark, assets=assets, tickers=tickers)


def slice_config_to_n_assets(cfg: RLConfig, n_assets: int) -> RLConfig:
    """Truncate ``universe.assets`` and per-asset lists to the first *n_assets* YAML keys.

    Use for CLI ``--n-assets`` without editing ``config.yaml``. The file must define at
    least *n_assets* keys; ``universe.benchmark`` must remain among the kept keys.
    ``benchmark_cap_weights`` are renormalized after slicing.
    """
    validate_universe_asset_count(n_assets)
    full = cfg.universe.n_assets
    if n_assets == full:
        return cfg
    if n_assets > full:
        raise ValueError(
            f"Requested N={n_assets} assets but config defines only {full} under "
            f"universe.assets; add symbols to config.yaml (supported up to {UNIVERSE_MAX_ASSETS})."
        )
    data = copy.deepcopy(cfg.raw)
    keys = list(data["universe"]["assets"].keys())[:n_assets]
    data["universe"]["assets"] = {k: data["universe"]["assets"][k] for k in keys}
    benchmark = str(data["universe"]["benchmark"])
    if benchmark not in data["universe"]["assets"]:
        raise ValueError(
            f"universe.benchmark {benchmark!r} is not among the first {n_assets} assets "
            f"({keys}); put the benchmark key earlier in universe.assets or lower --n-assets"
        )
    weights = [float(x) for x in data["reward"]["benchmark_cap_weights"][:n_assets]]
    wsum = sum(weights)
    if wsum <= 0.0:
        raise ValueError("benchmark_cap_weights slice must sum to a positive value")
    data["reward"]["benchmark_cap_weights"] = [w / wsum for w in weights]
    tc = data["transaction_costs"]
    for name in ("slippage", "tx_fee", "annual_holding_cost"):
        tc[name] = list(tc[name])[:n_assets]
    return _parse_config(data, cfg.path)


def _benchmark_combined_abs_cap(rew: dict) -> float:
    """Resolve the constant cap on |sortino + benchmark| (reward units).

    Old run snapshots carry the legacy ``benchmark_relative_max_share`` knob (a
    relative cap that proved reward-hackable). For those, translate the share into
    the constant cap the old code produced on a typical 1%-daily-move day:
    ``share/(1-share) * reward_scale * 0.01``. New configs set
    ``benchmark_combined_abs_cap`` directly; 0 disables both terms.
    """
    cap_raw = rew.get("benchmark_combined_abs_cap")
    if cap_raw is not None:
        return float(cap_raw)
    legacy = rew.get("benchmark_relative_max_share")
    if legacy is None:
        raise ValueError(
            "Missing required key 'benchmark_combined_abs_cap' in config section 'reward' "
            "(or legacy 'benchmark_relative_max_share' for old run snapshots)"
        )
    share = float(legacy)
    if not (0.0 <= share < 1.0):
        raise ValueError(
            f"reward.benchmark_relative_max_share (legacy) must be in [0, 1), got {share}"
        )
    if share == 0.0:
        return 0.0
    return (share / (1.0 - share)) * float(_req(rew, "reward_scale", "reward")) * 0.01


def _validate_reward_config(rew: RewardConfig) -> None:
    cap = float(rew.benchmark_combined_abs_cap)
    if not np.isfinite(cap) or cap < 0.0:
        raise ValueError(
            f"reward.benchmark_combined_abs_cap must be a finite value >= 0 "
            f"(0 disables the Sortino + benchmark-excess terms), got {cap}"
        )
    if rew.drawdown_increase_penalty < 0.0:
        raise ValueError(
            f"reward.drawdown_increase_penalty must be >= 0, got {rew.drawdown_increase_penalty}"
        )
    if rew.drawdown_level_penalty < 0.0:
        raise ValueError(
            f"reward.drawdown_level_penalty must be >= 0, got {rew.drawdown_level_penalty}"
        )
    floor = float(rew.drawdown_level_floor)
    if not (0.0 <= floor < 1.0):
        raise ValueError(
            f"reward.drawdown_level_floor must be in [0, 1), got {floor}"
        )
    if rew.concentration_penalty < 0.0:
        raise ValueError(
            f"reward.concentration_penalty must be >= 0, got {rew.concentration_penalty}"
        )
    if rew.concentration_target_eff_assets < 0.0:
        raise ValueError(
            "reward.concentration_target_eff_assets must be >= 0, "
            f"got {rew.concentration_target_eff_assets}"
        )
    if rew.cash_daily_yield < 0.0:
        raise ValueError(
            f"reward.cash_daily_yield must be >= 0, got {rew.cash_daily_yield}"
        )
    from rlbot.eval_selection import EXPOSURE_RISK_MODES

    if rew.exposure_risk_mode not in EXPOSURE_RISK_MODES:
        raise ValueError(
            f"reward.exposure_risk_mode must be one of {sorted(EXPOSURE_RISK_MODES)}, "
            f"got {rew.exposure_risk_mode!r}"
        )
    if rew.exposure_risk_penalty_scale < 0.0:
        raise ValueError(
            "reward.exposure_risk_penalty_scale must be >= 0, "
            f"got {rew.exposure_risk_penalty_scale}"
        )
    if rew.turnover_penalty < 0.0:
        raise ValueError(
            f"reward.turnover_penalty must be >= 0, got {rew.turnover_penalty}"
        )


def validate_config_for_universe(cfg: RLConfig, n_assets: int) -> None:
    """Ensure per-asset config lists match the loaded OHLCV panel width."""
    validate_universe_asset_count(n_assets)
    u = cfg.universe
    if u.n_assets != n_assets:
        raise ValueError(
            f"universe.assets has {u.n_assets} entries but data panel has {n_assets} assets; "
            "align config.yaml universe with --refresh-data"
        )
    if u.benchmark not in u.assets:
        raise ValueError(f"universe.benchmark {u.benchmark!r} missing from universe.assets")

    def _check(xs: list[float], name: str) -> None:
        if len(xs) != n_assets:
            raise ValueError(f"{name} must have {n_assets} entries, got {len(xs)}")

    _check(cfg.reward.benchmark_cap_weights, "reward.benchmark_cap_weights")
    tc = cfg.transaction_costs
    _check(tc.slippage, "transaction_costs.slippage")
    _check(tc.tx_fee, "transaction_costs.tx_fee")
    _check(tc.annual_holding_cost, "transaction_costs.annual_holding_cost")


def _feature_split_mode(value: Any) -> str:
    mode = str(value)
    if mode not in FEATURE_SPLIT_MODES:
        raise ValueError(
            f"data.feature_split_mode must be one of {FEATURE_SPLIT_MODES}, got {mode!r}"
        )
    return mode


def _parse_config(data: dict[str, Any], path: Path) -> RLConfig:
    universe = _parse_universe(_req(data, "universe", "root"))
    env = _req(data, "environment", "root")
    rew = _req(data, "reward", "root")
    tc = _req(data, "transaction_costs", "root")
    hp = _req(data, "hyperparameters", "root")
    pol = _req(data, "policy", "root")
    tr = _req(data, "training", "root")
    vn = _req(data, "vec_normalize", "root")
    ent = _req(data, "entropy_schedule", "root")
    cur = _req(data, "curriculum", "root")
    dat = _req(data, "data", "root")

    cfg = RLConfig(
        path=path.resolve(),
        universe=universe,
        environment=EnvironmentConfig(
            initial_cash=float(_req(env, "initial_cash", "environment")),
            max_episode_steps=int(_req(env, "max_episode_steps", "environment")),
            lookback=int(_req(env, "lookback", "environment")),
            obs_lag_default=int(_req(env, "obs_lag_default", "environment")),
            min_obs_lag=int(_req(env, "min_obs_lag", "environment")),
            max_obs_lag=int(_req(env, "max_obs_lag", "environment")),
            max_single_asset_weight=float(_req(env, "max_single_asset_weight", "environment")),
            fee_scale_default=float(_req(env, "fee_scale_default", "environment")),
            stop_loss_fraction=float(_req(env, "stop_loss_fraction", "environment")),
            domain_randomize_fee_dr_min=float(_req(env, "domain_randomize_fee_dr_min", "environment")),
            domain_randomize_fee_dr_max=float(_req(env, "domain_randomize_fee_dr_max", "environment")),
            domain_randomize_fee_beta_a=float(_req(env, "domain_randomize_fee_beta_a", "environment")),
            domain_randomize_fee_beta_b=float(_req(env, "domain_randomize_fee_beta_b", "environment")),
            action_smoothing_alpha=float(env.get("action_smoothing_alpha", 0.0)),
        ),
        reward=RewardConfig(
            reward_scale=float(_req(rew, "reward_scale", "reward")),
            max_step_log_return=float(_req(rew, "max_step_log_return", "reward")),
            max_step_log_return_downside=float(
                rew.get("max_step_log_return_downside", -0.15)
            ),
            risk_window=int(_req(rew, "risk_window", "reward")),
            sortino_min_steps=int(rew.get("sortino_min_steps", 20)),
            # Default preserves pre-2026-06 run snapshots (old floor 1e-4); the current
            # config.yaml sets an economically meaningful floor (see config comment).
            sortino_downside_floor=float(rew.get("sortino_downside_floor", 1e-4)),
            risk_bonus_scale=float(_req(rew, "risk_bonus_scale", "reward")),
            benchmark_cap_weights=_float_list(
                _req(rew, "benchmark_cap_weights", "reward"),
                "benchmark_cap_weights",
                expected_n=None,
            ),
            benchmark_excess_scale=float(_req(rew, "benchmark_excess_scale", "reward")),
            benchmark_excess_clip=float(_req(rew, "benchmark_excess_clip", "reward")),
            benchmark_combined_abs_cap=_benchmark_combined_abs_cap(rew),
            churn_penalty=_reward_churn_penalty(rew),
            turnover_penalty=float(rew.get("turnover_penalty", 0.0)),
            drawdown_downside_gamma=float(_req(rew, "drawdown_downside_gamma", "reward")),
            drawdown_increase_penalty=float(rew.get("drawdown_increase_penalty", 0.75)),
            drawdown_level_penalty=float(rew.get("drawdown_level_penalty", 3.0)),
            drawdown_level_floor=float(rew.get("drawdown_level_floor", 0.08)),
            concentration_penalty=float(rew.get("concentration_penalty", 0.0)),
            concentration_target_eff_assets=float(
                rew.get("concentration_target_eff_assets", 5.5)
            ),
            cash_daily_yield=float(rew.get("cash_daily_yield", 0.0)),
            inactivity_penalty_over_50=float(_req(rew, "inactivity_penalty_over_50", "reward")),
            inactivity_penalty_over_90=float(_req(rew, "inactivity_penalty_over_90", "reward")),
            eval_inactivity_penalty_scale=float(
                _req(rew, "eval_inactivity_penalty_scale", "reward")
            ),
            participation_bonus=float(_req(rew, "participation_bonus", "reward")),
            participation_reward_scale=float(
                _req(rew, "participation_reward_scale", "reward")
            ),
            exposure_risk_mode=str(rew.get("exposure_risk_mode", "realized_vol")),
            exposure_risk_penalty_scale=float(rew.get("exposure_risk_penalty_scale", 0.0)),
        ),
        transaction_costs=TransactionCostsConfig(
            slippage=_float_list(
                _req(tc, "slippage", "transaction_costs"), "slippage", expected_n=None
            ),
            tx_fee=_float_list(
                _req(tc, "tx_fee", "transaction_costs"), "tx_fee", expected_n=None
            ),
            annual_holding_cost=_float_list(
                _req(tc, "annual_holding_cost", "transaction_costs"),
                "annual_holding_cost",
                expected_n=None,
            ),
            trading_days_per_year=int(_req(tc, "trading_days_per_year", "transaction_costs")),
        ),
        hyperparameters=HyperparametersConfig(
            learning_rate=float(_req(hp, "learning_rate", "hyperparameters")),
            learning_rate_floor=float(_req(hp, "learning_rate_floor", "hyperparameters")),
            batch_size=int(_req(hp, "batch_size", "hyperparameters")),
            n_steps=int(_req(hp, "n_steps", "hyperparameters")),
            n_epochs=int(_req(hp, "n_epochs", "hyperparameters")),
            gamma=float(_req(hp, "gamma", "hyperparameters")),
            gae_lambda=float(_req(hp, "gae_lambda", "hyperparameters")),
            clip_range=float(_req(hp, "clip_range", "hyperparameters")),
            clip_range_finetune=float(_req(hp, "clip_range_finetune", "hyperparameters")),
            ent_coef_initial=float(_req(hp, "ent_coef_initial", "hyperparameters")),
            ent_coef_finetune=float(_req(hp, "ent_coef_finetune", "hyperparameters")),
            vf_coef=float(_req(hp, "vf_coef", "hyperparameters")),
            max_grad_norm=float(_req(hp, "max_grad_norm", "hyperparameters")),
            weight_decay=float(_req(hp, "weight_decay", "hyperparameters")),
        ),
        policy=PolicyConfig(
            lstm_hidden_size=int(_req(pol, "lstm_hidden_size", "policy")),
            n_lstm_layers=int(_req(pol, "n_lstm_layers", "policy")),
            net_arch_pi=[int(x) for x in _req(pol, "net_arch_pi", "policy")],
            net_arch_vf=[int(x) for x in _req(pol, "net_arch_vf", "policy")],
        ),
        training=TrainingConfig(
            timesteps=int(_req(tr, "timesteps", "training")),
            n_envs=int(_req(tr, "n_envs", "training")),
            obs_noise=float(_req(tr, "obs_noise", "training")),
            seed=int(_req(tr, "seed", "training")),
            block_size=int(_req(tr, "block_size", "training")),
            eval_stride=int(_req(tr, "eval_stride", "training")),
            eval_n_episodes=int(_req(tr, "eval_n_episodes", "training")),
            holdout_days=int(_req(tr, "holdout_days", "training")),
            viz_freq=int(_req(tr, "viz_freq", "training")),
            curriculum_update_freq=int(_req(tr, "curriculum_update_freq", "training")),
            checkpoint_save_freq_steps=int(
                _req(tr, "checkpoint_save_freq_steps", "training")
            ),
            eval_freq_steps=int(tr.get("eval_freq_steps", 500_000)),
            eval_freq_pre_gate_steps=int(tr.get("eval_freq_pre_gate_steps", 3_000_000)),
            reproducible=bool(tr.get("reproducible", False)),
            early_stop_patience=int(tr.get("early_stop_patience", 0)),
            best_model_score_std_coef=float(tr.get("best_model_score_std_coef", 0.75)),
            best_model_score_dd_coef=float(tr.get("best_model_score_dd_coef", 2.0)),
            best_model_score_stitched_blend=float(tr.get("best_model_score_stitched_blend", 0.5)),
            best_model_benchmark=str(tr.get("best_model_benchmark", "equal_weight_daily")),
        ),
        vec_normalize=VecNormalizeConfig(
            norm_obs=bool(_req(vn, "norm_obs", "vec_normalize")),
            norm_reward_train=bool(_req(vn, "norm_reward_train", "vec_normalize")),
            clip_obs=float(_req(vn, "clip_obs", "vec_normalize")),
            clip_reward=float(_req(vn, "clip_reward", "vec_normalize")),
        ),
        entropy_schedule=EntropyScheduleConfig(
            explore_ent=float(_req(ent, "explore_ent", "entropy_schedule")),
            final_ent=float(_req(ent, "final_ent", "entropy_schedule")),
            early_floor=float(_req(ent, "early_floor", "entropy_schedule")),
            early_floor_high=float(_req(ent, "early_floor_high", "entropy_schedule")),
            warmup_improvements=int(_req(ent, "warmup_improvements", "entropy_schedule")),
            decay_start_fraction=float(_req(ent, "decay_start_fraction", "entropy_schedule")),
            dr_lock_fraction=float(_req(ent, "dr_lock_fraction", "entropy_schedule")),
            early_floor_fraction=float(_req(ent, "early_floor_fraction", "entropy_schedule")),
        ),
        curriculum=CurriculumConfig(
            budget_short=int(_req(cur, "budget_short", "curriculum")),
            budget_long=int(_req(cur, "budget_long", "curriculum")),
            fee_free_fraction=float(_req(cur, "fee_free_fraction", "curriculum")),
            fee_ramp_fraction=float(_req(cur, "fee_ramp_fraction", "curriculum")),
            churn_ramp_floor=float(_req(cur, "churn_ramp_floor", "curriculum")),
            dr_widen_span_fraction=float(_req(cur, "dr_widen_span_fraction", "curriculum")),
            fee_free_long=int(_req(cur, "fee_free_long", "curriculum")),
            fee_ramp_end_long=int(_req(cur, "fee_ramp_end_long", "curriculum")),
            dr_widen_span_long=int(_req(cur, "dr_widen_span_long", "curriculum")),
            best_model_min_step=(
                None
                if cur.get("best_model_min_step") is None
                else int(cur["best_model_min_step"])
            ),
        ),
        data=DataConfig(
            since=str(_req(dat, "since", "data")),
            fracdiff_d=float(_req(dat, "fracdiff_d", "data")),
            feature_purge_warmup=int(_req(dat, "feature_purge_warmup", "data")),
            feature_split_mode=_feature_split_mode(dat.get("feature_split_mode", "independent")),
            feature_preroll_bars=int(dat.get("feature_preroll_bars", 252)),
        ),
        raw=data,
    )
    _validate_reward_config(cfg.reward)
    from rlbot.eval_selection import EVAL_BENCHMARK_MODES

    bm = cfg.training.best_model_benchmark
    if bm not in EVAL_BENCHMARK_MODES:
        raise ValueError(
            f"training.best_model_benchmark must be one of {sorted(EVAL_BENCHMARK_MODES)}, "
            f"got {bm!r}"
        )
    blend = float(cfg.training.best_model_score_stitched_blend)
    if not 0.0 <= blend <= 1.0:
        raise ValueError(
            f"training.best_model_score_stitched_blend must be in [0, 1], got {blend}"
        )
    return cfg


_CONFIG: RLConfig | None = None


def load_config(path: Path | str | None = None) -> RLConfig:
    """Parse and validate ``config.yaml`` (or the given path)."""
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data)}")
    return _parse_config(data, p)


def get_config() -> RLConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


def set_config(cfg: RLConfig) -> None:
    """Install active config (e.g. after ``--config`` or ``--n-assets``)."""
    global _CONFIG
    _CONFIG = cfg


class WorkerConfigInstaller:
    """Picklable carrier that re-installs the active config inside a subprocess.

    ``SubprocVecEnv`` workers start via spawn/forkserver, so they re-import every
    module with ``_CONFIG = None`` — without this, ``get_config()`` inside a worker
    silently falls back to the default ``config/config.yaml`` and any ``--config`` /
    ``--n-assets`` override never reaches the training envs (reward, costs, cap, DR
    bounds). Construct it from the effective config in the main process and call it
    first thing inside the env factory.
    """

    def __init__(self, cfg: RLConfig) -> None:
        self._raw = cfg.raw
        self._path = str(cfg.path)

    def __call__(self) -> RLConfig:
        cfg = _parse_config(self._raw, Path(self._path))
        set_config(cfg)
        return cfg


def write_config_snapshot(cfg: RLConfig, path: Path | str) -> None:
    """Write the effective config (e.g. after ``--n-assets`` slice) to a run directory."""
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, default_flow_style=None)


def trade_curriculum_milestones(
    learn_budget: int,
    cur: CurriculumConfig | None = None,
) -> tuple[int, int]:
    """Return ``(fee_free_until, fee_ramp_end)`` in environment steps."""
    if cur is None:
        cur = get_config().curriculum
    lb = max(1, int(learn_budget))
    if lb <= cur.budget_short:
        fee_free = max(1, int(cur.fee_free_fraction * lb))
        fee_ramp = max(fee_free + 1, int(cur.fee_ramp_fraction * lb))
        return fee_free, fee_ramp
    ff_short = max(1, int(cur.fee_free_fraction * cur.budget_short))
    fr_short = max(ff_short + 1, int(cur.fee_ramp_fraction * cur.budget_short))
    if lb >= cur.budget_long:
        return cur.fee_free_long, cur.fee_ramp_end_long
    t = (lb - cur.budget_short) / (cur.budget_long - cur.budget_short)
    ff = int(ff_short + t * (cur.fee_free_long - ff_short))
    fr = int(fr_short + t * (cur.fee_ramp_end_long - fr_short))
    fee_free = max(1, ff)
    fee_ramp = max(fee_free + 1, fr)
    return fee_free, fee_ramp


def resolve_best_model_min_step(
    learn_budget: int,
    cur: CurriculumConfig | None = None,
) -> int:
    """Step before which ``models/best/`` is not updated (``0`` disables the gate)."""
    if cur is None:
        cur = get_config().curriculum
    explicit = cur.best_model_min_step
    if explicit is not None:
        return max(0, int(explicit))
    _, fee_ramp_end = trade_curriculum_milestones(learn_budget, cur=cur)
    return fee_ramp_end


def apply_deterministic_seeds(seed: int) -> None:
    """Lock Python, NumPy, and PyTorch RNGs for reproducible training."""
    import torch as th

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    try:
        th.use_deterministic_algorithms(True)
    except Exception:
        th.use_deterministic_algorithms(True, warn_only=True)


