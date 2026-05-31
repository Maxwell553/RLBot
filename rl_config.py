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
import torch as th
import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"

N_ASSETS_EXPECTED = 10


def _req(d: dict[str, Any], key: str, section: str) -> Any:
    if key not in d:
        raise KeyError(f"config.yaml missing '{section}.{key}'")
    return d[key]


def _float_list(xs: list, name: str, n: int = N_ASSETS_EXPECTED) -> list[float]:
    if len(xs) != n:
        raise ValueError(f"{name} must have {n} entries, got {len(xs)}")
    return [float(x) for x in xs]


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


@dataclass(frozen=True)
class RewardConfig:
    reward_scale: float
    max_step_log_return: float
    risk_window: int
    risk_bonus_scale: float
    benchmark_cap_weights: list[float]
    churn_lambda: float
    drawdown_penalty_scale: float
    inactivity_penalty_over_50: float
    inactivity_penalty_over_90: float
    eval_inactivity_penalty_scale: float
    participation_bonus: float

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
    churn_start_fraction: float
    dr_widen_span_fraction: float
    fee_free_long: int
    fee_ramp_end_long: int
    churn_start_long: int
    dr_widen_span_long: int


@dataclass(frozen=True)
class DataConfig:
    since: str
    fracdiff_d: float
    feature_purge_warmup: int


@dataclass(frozen=True)
class RLConfig:
    path: Path
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


def _parse_config(data: dict[str, Any], path: Path) -> RLConfig:
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

    return RLConfig(
        path=path.resolve(),
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
        ),
        reward=RewardConfig(
            reward_scale=float(_req(rew, "reward_scale", "reward")),
            max_step_log_return=float(_req(rew, "max_step_log_return", "reward")),
            risk_window=int(_req(rew, "risk_window", "reward")),
            risk_bonus_scale=float(_req(rew, "risk_bonus_scale", "reward")),
            benchmark_cap_weights=_float_list(
                _req(rew, "benchmark_cap_weights", "reward"),
                "benchmark_cap_weights",
            ),
            churn_lambda=float(_req(rew, "churn_lambda", "reward")),
            drawdown_penalty_scale=float(_req(rew, "drawdown_penalty_scale", "reward")),
            inactivity_penalty_over_50=float(_req(rew, "inactivity_penalty_over_50", "reward")),
            inactivity_penalty_over_90=float(_req(rew, "inactivity_penalty_over_90", "reward")),
            eval_inactivity_penalty_scale=float(
                _req(rew, "eval_inactivity_penalty_scale", "reward")
            ),
            participation_bonus=float(_req(rew, "participation_bonus", "reward")),
        ),
        transaction_costs=TransactionCostsConfig(
            slippage=_float_list(_req(tc, "slippage", "transaction_costs"), "slippage"),
            tx_fee=_float_list(_req(tc, "tx_fee", "transaction_costs"), "tx_fee"),
            annual_holding_cost=_float_list(
                _req(tc, "annual_holding_cost", "transaction_costs"), "annual_holding_cost"
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
            churn_start_fraction=float(_req(cur, "churn_start_fraction", "curriculum")),
            dr_widen_span_fraction=float(_req(cur, "dr_widen_span_fraction", "curriculum")),
            fee_free_long=int(_req(cur, "fee_free_long", "curriculum")),
            fee_ramp_end_long=int(_req(cur, "fee_ramp_end_long", "curriculum")),
            churn_start_long=int(_req(cur, "churn_start_long", "curriculum")),
            dr_widen_span_long=int(_req(cur, "dr_widen_span_long", "curriculum")),
        ),
        data=DataConfig(
            since=str(_req(dat, "since", "data")),
            fracdiff_d=float(_req(dat, "fracdiff_d", "data")),
            feature_purge_warmup=int(_req(dat, "feature_purge_warmup", "data")),
        ),
        raw=data,
    )


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
    """Install active config (e.g. after ``--config`` override)."""
    global _CONFIG
    _CONFIG = cfg
    sync_trading_env_aliases(cfg)


def sync_trading_env_aliases(cfg: RLConfig | None = None) -> None:
    """Keep ``trading_env`` module-level names in sync for legacy imports."""
    import trading_env as te

    c = cfg or get_config()
    e, r = c.environment, c.reward
    te.LOOKBACK = e.lookback
    te.MIN_OBS_LAG = e.min_obs_lag
    te.MAX_OBS_LAG = e.max_obs_lag
    te.MAX_SINGLE_ASSET_WEIGHT = e.max_single_asset_weight
    te.REWARD_SCALE = r.reward_scale
    te.MAX_STEP_LOG_RETURN = r.max_step_log_return
    te.CHURN_LAMBDA = r.churn_lambda
    te.RISK_WINDOW = r.risk_window
    te.RISK_BONUS_SCALE = r.risk_bonus_scale
    te.INACTIVITY_PENALTY_OVER_50 = r.inactivity_penalty_over_50
    te.INACTIVITY_PENALTY_OVER_90 = r.inactivity_penalty_over_90
    te.EVAL_INACTIVITY_PENALTY_SCALE = r.eval_inactivity_penalty_scale
    te.PARTICIPATION_BONUS = r.participation_bonus
    te.STOP_LOSS_FRACTION = e.stop_loss_fraction
    te.ASSET_SLIPPAGE = c.transaction_costs.slippage_array()
    te.ASSET_TX_FEE = c.transaction_costs.tx_fee_array()
    te.ANNUAL_HOLDING_COST = np.asarray(c.transaction_costs.annual_holding_cost, dtype=np.float64)
    te.DAILY_HOLDING_COST = c.transaction_costs.daily_holding_cost_array()


def apply_deterministic_seeds(seed: int) -> None:
    """Lock Python, NumPy, and PyTorch RNGs for reproducible training."""
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


