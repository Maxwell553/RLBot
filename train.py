#!/usr/bin/env python3
"""
Train shared RecurrentPPO (LSTM) policy on synchronized multi-asset daily data.

10 global assets: S&P 500 (SPY), Gold (GLD), Crude Oil WTI (USO),
EUR/USD, USD/JPY, Nikkei 225, FTSE 100, 10-Year Treasury (IEF),
Copper (HG=F), Emerging Markets (EEM).

Artifacts for inference (backtest / paper_trade): ``models/<run_id>/vec_normalize.pkl``
(same stats copied to ``models/<run_id>/best/`` next to ``best_model.zip``).

Anti-overfitting measures:
  - Fractionally differentiated price features (stationary + memory)
  - Observation noise on market features during training
  - Seed shuffling: fresh OS entropy on every episode reset
  - VecNormalize + cosine LR decay with floor
  - Domain randomization: fee_scale and obs_lag vary per training episode (after fee curriculum)
  - Fee curriculum: frictionless → fee ramp → full DR; churn penalty kicks in mid-run (see trade_curriculum_milestones)
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch as th
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from data_utils import (
    fetch_aligned_daily,
    load_cache,
    reserve_chronological_holdout,
    save_cache,
    train_test_split_alternating,
)
from run_artifacts import (
    RunPaths,
    new_run_id,
    snapshot_data_cache,
    write_latest_pointer,
    write_manifest,
)
from trading_env import MultiAssetPortfolioEnv
from visualize import TrainingVizCallback, open_plot_file

ROOT = Path(__file__).resolve().parent
DATA_CACHE = ROOT / "data_cache.npz"


def _persist_trade_artifacts(model: RecurrentPPO, train_env: VecNormalize, paths: RunPaths) -> tuple[Path, Path]:
    """Save VecNormalize stats + final weights so inference matches training input scaling.

    Writes ``models/<id>/vec_normalize.pkl`` and a duplicate
    ``models/<id>/best/vec_normalize.pkl`` (same file as best_model.zip).
    """
    root_vn = paths.models_dir / "vec_normalize.pkl"
    train_env.save(str(root_vn))
    best_vn = paths.best_model_dir / "vec_normalize.pkl"
    shutil.copy2(root_vn, best_vn)
    model.save(str(paths.final_model))
    return root_vn, best_vn


# ── Env factory ──────────────────────────────────────────────────────────

def _make_env_factory(
    ohlcv: np.ndarray,
    rsi: np.ndarray,
    macd: np.ndarray,
    macro: np.ndarray,
    fracdiff: np.ndarray,
    fracdiff_macro: np.ndarray,
    random_start: bool,
    log_dir: Path,
    monitor_stem: str,
    max_episode_steps: int = 252,
    obs_noise_std: float = 0.0,
    reseed_on_reset: bool = False,
    block_boundaries: list | None = None,
    obs_lag_default: int = 1,
    domain_randomize: bool = True,
):
    """Return a callable that creates and wraps a single environment."""

    def _init():
        env = MultiAssetPortfolioEnv(
            ohlcv,
            rsi,
            macd,
            fracdiff=fracdiff,
            fracdiff_macro=fracdiff_macro,
            macro=macro,
            random_start=random_start,
            max_episode_steps=max_episode_steps,
            obs_noise_std=obs_noise_std,
            reseed_on_reset=reseed_on_reset,
            block_boundaries=block_boundaries,
            obs_lag=0,
            obs_lag_default=obs_lag_default,
            fee_scale_default=1.0,
            domain_randomize=domain_randomize,
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        return Monitor(env, filename=str(log_dir / monitor_stem))

    return _init


def _lr_schedule_with_floor(initial_lr: float, floor_lr: float = 1e-6):
    """Cosine-annealing LR that decays to ``floor_lr``.

    With initial_lr=3e-4 and floor_lr=1e-6, the final ~30% of training
    runs at very low LR, letting the model settle into precise weights
    that can beat transaction costs.
    """
    import math

    def schedule(progress_remaining: float) -> float:
        cosine = 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress_remaining)))
        return floor_lr + (initial_lr - floor_lr) * cosine

    return schedule


class AdaptiveEntropyCallback(BaseCallback):
    """High entropy for broad exploration, tapering only after eval improves.

    Phase 1 (explore):  ent_coef = ``explore_ent``.  Held at the explore
        floor until the best eval reward has improved at least
        ``warmup_improvements`` times AND at least ``min_explore_steps``
        total timesteps have elapsed.
    Phase 2 (exploit):  cosine-decay from ``explore_ent`` to ``final_ent``
        over the remaining training budget, but never below
        ``early_floor`` during the first ``early_floor_steps``.
    """

    def __init__(
        self,
        explore_ent: float = 0.05,
        final_ent: float = 0.005,
        early_floor: float = 0.01,
        early_floor_steps: int = 3_000_000,
        warmup_improvements: int = 3,
        eval_log_dir: str = "",
    ):
        super().__init__()
        self.explore_ent = explore_ent
        self.final_ent = final_ent
        self.early_floor = early_floor
        self.early_floor_steps = early_floor_steps
        self.warmup_improvements = warmup_improvements
        self.eval_log_dir = eval_log_dir
        self._exploit_start: float | None = None
        self._last_best: float = -float("inf")
        self._improvements: int = 0

    def _on_step(self) -> bool:
        import math

        if self.eval_log_dir:
            npz = Path(self.eval_log_dir) / "evaluations.npz"
            if npz.is_file():
                data = np.load(str(npz))
                if "results" in data:
                    mean_rewards = data["results"].mean(axis=1)
                    current_best = float(mean_rewards.max())
                    if current_best > self._last_best + 1e-6:
                        self._improvements += 1
                        self._last_best = current_best

        progress_remaining = self.model._current_progress_remaining

        if self._exploit_start is None and self._improvements >= self.warmup_improvements:
            self._exploit_start = progress_remaining
            self.logger.record("config/exploit_phase_started", 1.0 - progress_remaining)

        if self._exploit_start is not None:
            phase_total = self._exploit_start
            phase_elapsed = self._exploit_start - progress_remaining
            frac = min(phase_elapsed / max(phase_total, 1e-12), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * frac))
            ent = self.final_ent + (self.explore_ent - self.final_ent) * cosine
        else:
            ent = self.explore_ent

        # Hard floor during early training to prevent premature convergence
        if self.num_timesteps < self.early_floor_steps:
            ent = max(ent, self.early_floor)

        self.model.ent_coef = ent
        self.logger.record("config/ent_coef", ent)
        self.logger.record("config/eval_improvements", self._improvements)
        return True


# Standard schedules (two columns from training design): 120M long run vs 65M short run.
_CURR_BUDGET_SHORT = 65_000_000
_CURR_BUDGET_LONG = 120_000_000
_CURR_FEE_FREE_SHORT = 2_000_000
_CURR_FEE_RAMP_END_SHORT = 10_000_000
_CURR_CHURN_START_SHORT = 5_000_000
_CURR_FEE_FREE_LONG = 13_300_000
_CURR_FEE_RAMP_END_LONG = 40_000_000
_CURR_CHURN_START_LONG = 60_000_000


def trade_curriculum_milestones(learn_budget: int) -> tuple[int, int, int]:
    """Return ``(fee_free_until, fee_ramp_end, churn_start_step)`` in environment steps.

    - **Frictionless** until ``fee_free_until``; **fee ramp** until ``fee_ramp_end``; then DR.
    - **Churn** penalty scale is 0 before ``churn_start_step``, then 1.

    Anchored at 65M and 120M; budgets in between interpolate; below 65M, scales the short schedule.
    """
    lb = max(1, int(learn_budget))
    if lb <= _CURR_BUDGET_SHORT:
        alpha = lb / _CURR_BUDGET_SHORT
        fee_free = max(1, int(_CURR_FEE_FREE_SHORT * alpha))
        fee_ramp = max(fee_free + 1, int(_CURR_FEE_RAMP_END_SHORT * alpha))
        churn_at = max(1, int(_CURR_CHURN_START_SHORT * alpha))
        return fee_free, fee_ramp, churn_at
    if lb >= _CURR_BUDGET_LONG:
        return _CURR_FEE_FREE_LONG, _CURR_FEE_RAMP_END_LONG, _CURR_CHURN_START_LONG
    t = (lb - _CURR_BUDGET_SHORT) / (_CURR_BUDGET_LONG - _CURR_BUDGET_SHORT)
    ff = int(_CURR_FEE_FREE_SHORT + t * (_CURR_FEE_FREE_LONG - _CURR_FEE_FREE_SHORT))
    fr = int(_CURR_FEE_RAMP_END_SHORT + t * (_CURR_FEE_RAMP_END_LONG - _CURR_FEE_RAMP_END_SHORT))
    ch = int(_CURR_CHURN_START_SHORT + t * (_CURR_CHURN_START_LONG - _CURR_CHURN_START_SHORT))
    fee_free = max(1, ff)
    fee_ramp = max(fee_free + 1, fr)
    churn_at = max(1, ch)
    return fee_free, fee_ramp, churn_at


def fee_curriculum_milestones(learn_budget: int) -> tuple[int, int]:
    """Backward-compatible (fee_free, fee_ramp_end) for logging."""
    ff, fr, _ = trade_curriculum_milestones(learn_budget)
    return ff, fr


def entropy_early_floor_milestones(learn_budget: int) -> int:
    """Entropy floor duration: original ~8M steps for an ~18M run → ``8/18`` of ``learn_budget``."""
    lb = max(1, int(learn_budget))
    return max(1, lb * 8 // 18)


class TradingCurriculumCallback(BaseCallback):
    """Training-only schedule: fee ramp + churn scale (eval envs never see this).

    Milestones from ``trade_curriculum_milestones(learn_budget)``.

    - Steps ``[0, fee_free_until)``: ``fee_scale = 0`` (frictionless).
    - Steps ``[fee_free_until, fee_ramp_end)``: linear ramp to ``fee_scale = 1.0``.
    - Steps ``>= fee_ramp_end``: release (``None``) → domain randomization on reset.
    - Churn: ``churn_scale = 0`` before ``churn_start_step``, then ``1``.
    """

    def __init__(
        self,
        vec_env: VecNormalize,
        learn_budget: int,
        update_freq: int = 50_000,
    ):
        super().__init__()
        self.vec_env = vec_env
        self.learn_budget = int(learn_budget)
        self.fee_free_until, self.fee_ramp_end, self.churn_start_step = trade_curriculum_milestones(
            self.learn_budget
        )
        self.update_freq = max(1, int(update_freq))
        self._last_key: tuple[float | None, float] | None = None

    def _fee_override(self, t: int) -> float | None:
        if t < self.fee_free_until:
            return 0.0
        if t < self.fee_ramp_end:
            span = max(self.fee_ramp_end - self.fee_free_until, 1)
            return float(t - self.fee_free_until) / float(span)
        return None

    def _churn_scale(self, t: int) -> float:
        return 0.0 if t < self.churn_start_step else 1.0

    def _apply(self) -> None:
        t = int(self.num_timesteps)
        fee = self._fee_override(t)
        churn = self._churn_scale(t)
        key = (fee, churn)
        if key != self._last_key:
            self.vec_env.env_method("set_curriculum_state", fee, churn)
            self._last_key = key
            self.logger.record("config/curriculum_fee_override", -1.0 if fee is None else float(fee))
            self.logger.record("config/curriculum_churn_scale", churn)

    def _on_training_start(self) -> None:
        self._last_key = None
        self._apply()

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            self._apply()
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--since", default="2005-01-01",
        help="Fetch start date (UTC). Assets with later listings are backfilled.",
    )
    parser.add_argument("--until", default=None, help="Optional fetch end (UTC)")
    parser.add_argument("--refresh-data", action="store_true", help="Refetch OHLCV from yfinance")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=65_000_000,
        help="Total PPO steps (default 65M; use 120M if you want the long schedule end-state)",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--n-steps", type=int, default=32768)
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel training envs")
    parser.add_argument("--max-ep-steps", type=int, default=63, help="Steps per training episode (~3 months of daily bars)")
    parser.add_argument("--obs-noise", type=float, default=0.02, help="Gaussian noise std added to market features during training (regularization)")
    parser.add_argument("--obs-lag", type=int, default=1, help="Default market-feature lag when not randomizing (eval); training samples 0..2 per episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--test-days", type=int, default=365,
        help="Deprecated alias for --holdout-days (backtest OOS window); prefer --holdout-days",
    )
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=None,
        help=(
            "Reserve the last N calendar days for backtest only; training/eval never see these bars. "
            "Default: same as --test-days (365) for backward compatibility."
        ),
    )
    parser.add_argument("--block-size", type=int, default=126, help="Walk-forward block size in trading bars (~6 months)")
    parser.add_argument("--eval-stride", type=int, default=4, help="Every Nth block goes to eval (4 → 25%% eval)")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--viz-freq", type=int, default=20_000)
    parser.add_argument("--show-viz", action="store_true")
    parser.add_argument("--run-id", default="", metavar="ID")
    parser.add_argument(
        "--resume", default="", metavar="PATH",
        help="Resume from a RecurrentPPO checkpoint .zip (loads weights + VecNormalize stats). Old MLP/PPO checkpoints are incompatible.",
    )
    args = parser.parse_args()
    if args.holdout_days is None:
        args.holdout_days = args.test_days

    run_id = args.run_id.strip() or new_run_id(timesteps=args.timesteps)
    paths = RunPaths(run_id=run_id)
    paths.mkdirs()

    # ── data ─────────────────────────────────────────────────────────────
    if args.refresh_data or not DATA_CACHE.is_file():
        idx, ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro = fetch_aligned_daily(
            since=args.since, until=args.until,
        )
        save_cache(str(DATA_CACHE), idx, ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro)
    else:
        idx, ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro = load_cache(str(DATA_CACHE))

    snapshot_data_cache(DATA_CACHE, paths.data_snapshot)

    (idx_fit, ohlcv_fit, rsi_fit, macd_fit, macro_fit, fd_fit, fdm_fit), (
        idx_hold, ohlcv_hold, rsi_hold, macd_hold, macro_hold, fd_hold, fdm_hold,
    ) = reserve_chronological_holdout(
        idx, ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro,
        holdout_days=args.holdout_days,
    )

    (train_idx, train_ohlcv, train_rsi, train_macd, train_macro, train_fd, train_fdm, train_boundaries), (
        eval_idx, eval_ohlcv, eval_rsi, eval_macd, eval_macro, eval_fd, eval_fdm, eval_boundaries,
    ) = train_test_split_alternating(
        idx_fit, ohlcv_fit, rsi_fit, macd_fit, macro_fit, fd_fit, fdm_fit,
        block_size=args.block_size,
        eval_stride=args.eval_stride,
    )

    if len(train_idx) < 200:
        raise RuntimeError(
            "Not enough training rows after split; widen the date range or reduce --holdout-days."
        )

    write_manifest(
        paths.manifest_path,
        {
            "run_id": run_id,
            "args": vars(args),
            "n_index": int(len(idx)),
            "n_trainable_bars": int(len(idx_fit)),
            "chronological_holdout": {
                "holdout_days": int(args.holdout_days),
                "holdout_bars": int(len(idx_hold)),
                "date_start": str(idx_hold[0]),
                "date_end": str(idx_hold[-1]),
            },
            "n_train_bars": int(len(train_idx)),
            "n_eval_bars": int(len(eval_idx)),
            "data_cache_snapshot": str(paths.data_snapshot),
        },
    )

    print(f"Run id: {run_id}")
    print(f"  plots:   {paths.plots_dir}/")
    print(f"  models:  {paths.models_dir}/")
    print(f"  logs:    {paths.logs_dir}/")
    print(f"  tb_logs: {paths.tb_dir}/")
    print(f"  meta:    {paths.run_meta_dir}/")
    print(f"  network: RecurrentPPO MlpLstmPolicy — LSTM 2×64 + MLP heads [128,128] (pi+vf), AdamW weight_decay=1e-3, gae=0.95, gamma=0.99")
    print(f"  early_stop: off (full {args.timesteps:,} timesteps; eval still saves best_model)")
    print(f"  trade bundle on exit: {paths.models_dir.name}/vec_normalize.pkl + copy → best/ (pair with best_model.zip)")
    print(f"  n_envs={args.n_envs}, n_steps={args.n_steps}, batch={args.batch_size}")
    print(f"  max_ep_steps={args.max_ep_steps} (daily bars), clip=0.20, epochs=3, eval=deterministic_cycle (75 eps)")
    print(f"  obs_noise={args.obs_noise}, reseed_on_reset=True (training)")
    print(f"  obs_lag: train Uniform{{0,1,2}} per episode; eval fixed at {args.obs_lag}")
    print(f"  execution=open[t+1] (realistic: decide after close[t-1], fill at next open)")
    print(f"  reward: return*2000 + Sortino*25 + participation(gross*|w_risky|)*0.1 - inactivity(>50% -5, >90% extra -0.1) - churn - linear_dd*10")
    print(f"  action: softmax(cash+10 assets), long-only risky weights, soft 40% cap per asset")
    print(f"  domain_randomization: fee_scale~U(0.5,1.5), obs_lag~Discrete{{0,1,2}} (training, after fee curriculum)")
    _ff, _fr, _ch = trade_curriculum_milestones(args.timesteps)
    _ef = entropy_early_floor_milestones(args.timesteps)
    print(
        f"  fee curriculum: fee=0 for {_ff:,} steps → ramp to 1.0 by {_fr:,} → release DR; "
        f"churn penalty from step {_ch:,}"
    )
    print(
        f"  entropy: 0.10 (explore, floor=0.01 for {_ef:,} steps, 8/18 of run) → 0.005 (exploit after eval warmup)"
    )
    print(f"  LR={args.learning_rate} (cosine → 1e-6 floor)")
    print(
        f"  OOS holdout: last {args.holdout_days} calendar days → {len(idx_hold)} bars "
        f"({idx_hold[0].date()} .. {idx_hold[-1].date()}) — excluded from training/eval"
    )
    print(f"  split=alternating walk-forward (block={args.block_size}, stride={args.eval_stride}) on trainable-only data")
    print(f"  train={len(train_idx)} bars ({len(train_boundaries)} boundaries), eval={len(eval_idx)} bars ({len(eval_boundaries)} boundaries)")
    if args.resume:
        print(f"  MODE: fine-tune from {args.resume}")

    # ── envs ─────────────────────────────────────────────────────────────
    n_envs = args.n_envs

    train_env = SubprocVecEnv([
        _make_env_factory(
            train_ohlcv, train_rsi, train_macd, train_macro, train_fd, train_fdm,
            random_start=True,
            log_dir=paths.logs_dir,
            monitor_stem=f"train_monitor_{i}",
            max_episode_steps=args.max_ep_steps,
            obs_noise_std=args.obs_noise,
            reseed_on_reset=True,
            block_boundaries=train_boundaries,
            obs_lag_default=args.obs_lag,
            domain_randomize=True,
        )
        for i in range(n_envs)
    ])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    eval_env = SubprocVecEnv([
        _make_env_factory(
            eval_ohlcv, eval_rsi, eval_macd, eval_macro, eval_fd, eval_fdm,
            random_start=False,
            log_dir=paths.logs_dir,
            monitor_stem="eval_monitor",
            max_episode_steps=args.max_ep_steps,
            reseed_on_reset=False,
            block_boundaries=eval_boundaries,
            obs_lag_default=args.obs_lag,
            domain_randomize=False,
        )
    ])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
        training=False,
    )

    # ── model ────────────────────────────────────────────────────────────
    policy_kwargs = dict(
        lstm_hidden_size=64,
        n_lstm_layers=2,
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=th.nn.Tanh,
        ortho_init=True,
        optimizer_class=th.optim.AdamW,
        optimizer_kwargs=dict(weight_decay=1e-3),
    )

    lr_schedule = _lr_schedule_with_floor(args.learning_rate, floor_lr=1e-6)

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        print(f"  Resuming from: {resume_path}")
        model = RecurrentPPO.load(
            str(resume_path),
            env=train_env,
            device="auto",
            tensorboard_log=str(paths.tb_dir),
        )
        model.learning_rate = lr_schedule
        model.n_steps = args.n_steps
        model.batch_size = args.batch_size
        model.ent_coef = 0.001
        model.clip_range = lambda _: 0.1

        stem = resume_path.stem
        parts = stem.split("_", 1)
        vn_path = resume_path.parent / f"{parts[0]}_vecnormalize_{parts[1]}.pkl" if len(parts) == 2 else None
        if vn_path is None or not vn_path.is_file():
            vn_path = resume_path.parent.parent / "vec_normalize.pkl"
        if vn_path and vn_path.is_file():
            loaded_vn = VecNormalize.load(str(vn_path), train_env.venv)
            train_env.obs_rms = loaded_vn.obs_rms
            train_env.ret_rms = loaded_vn.ret_rms
            eval_env.obs_rms = loaded_vn.obs_rms
            print(f"  Restored VecNormalize stats from: {vn_path}")
        else:
            print("  WARNING: No VecNormalize stats found for checkpoint")
            eval_env.obs_rms = train_env.obs_rms

        print(f"  Fine-tune LR={args.learning_rate}, ent_coef=0.001, clip=0.1")
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=lr_schedule,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=3,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.10,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=str(paths.tb_dir),
            seed=args.seed,
            device="auto",
        )
        eval_env.obs_rms = train_env.obs_rms

    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"  total params: {total_params:,}  (trainable: {trainable_params:,})")

    # ── callbacks ────────────────────────────────────────────────────────
    # No StopTrainingOnNoModelImprovement: train for full --timesteps (eval still logs best_model)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(paths.best_model_dir),
        log_path=str(paths.eval_log_dir),
        eval_freq=max(5_000 // n_envs, args.n_steps),
        n_eval_episodes=75,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1_000_000 // n_envs, args.n_steps),
        save_path=str(paths.models_dir / "checkpoints"),
        name_prefix="ppo",
        save_vecnormalize=True,
    )

    callbacks = [eval_callback, checkpoint_callback]
    if not args.resume:
        callbacks.insert(
            0,
            TradingCurriculumCallback(
                train_env,
                learn_budget=args.timesteps,
                update_freq=50_000,
            ),
        )
        callbacks.append(AdaptiveEntropyCallback(
            explore_ent=0.10,
            final_ent=0.005,
            early_floor=0.01,
            early_floor_steps=entropy_early_floor_milestones(args.timesteps),
            warmup_improvements=5,
            eval_log_dir=str(paths.eval_log_dir),
        ))
    if not args.no_viz:
        callbacks.append(
            TrainingVizCallback(
                plot_path=paths.training_plot,
                eval_npz_path=paths.eval_npz,
                plot_freq=args.viz_freq,
            )
        )

    # ── train ────────────────────────────────────────────────────────────
    learn_error: BaseException | None = None
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
            reset_num_timesteps=not bool(args.resume),
        )
    except KeyboardInterrupt:
        print("\n\nCtrl+C detected — saving current weights before exit…")
    except BaseException as e:
        learn_error = e
        print(f"\nWARNING: training stopped with {type(e).__name__}: {e}")
    finally:
        # Always persist VecNormalize + weights so runs are trade-ready even if learn() crashes
        vn_root, vn_best = _persist_trade_artifacts(model, train_env, paths)
        print(f"\nTrade bundle: {paths.final_model.name} + vec_normalize (run + best/)")
        print(f"  VecNormalize: {vn_root}")
        print(f"  Copy next to best: {vn_best}")

    if learn_error is not None:
        raise learn_error

    write_manifest(
        paths.manifest_path,
        {
            "run_id": run_id,
            "args": vars(args),
            "n_index": int(len(idx)),
            "n_train_bars": int(len(train_idx)),
            "n_eval_bars": int(len(eval_idx)),
            "data_cache_snapshot": str(paths.data_snapshot),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "artifacts": {
                "final_model": str(paths.final_model),
                "best_model": str(paths.best_model_dir / "best_model.zip"),
                "best_model_dir": str(paths.best_model_dir),
                "vec_normalize": str(paths.models_dir / "vec_normalize.pkl"),
                "vec_normalize_next_to_best": str(paths.best_model_dir / "vec_normalize.pkl"),
                "training_plot": str(paths.training_plot),
                "tensorboard": str(paths.tb_dir),
                "monitor_logs": str(paths.logs_dir),
                "eval_npz": str(paths.eval_npz),
            },
        },
    )
    write_latest_pointer(run_id)

    print(f"\nSaved final model: {paths.final_model}")
    print(f"VecNormalize stats: {paths.models_dir / 'vec_normalize.pkl'}")
    print(f"Best model + vec (trade): {paths.best_model_dir}/best_model.zip + vec_normalize.pkl")
    print(f"Best checkpoint dir: {paths.models_dir / 'checkpoints'}/")
    if not args.no_viz:
        print(f"Training plot: {paths.training_plot}")
        if args.show_viz:
            open_plot_file(paths.training_plot)


if __name__ == "__main__":
    main()
