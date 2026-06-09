#!/usr/bin/env python3
"""
Train shared RecurrentPPO (LSTM) on synchronized multi-asset daily data.

Universe size and symbols: ``config/config.yaml`` → ``universe.assets`` (5–55);
optional CLI ``--n-assets`` slices the first N keys.

Artifacts for inference (backtest): ``Runs/<run_id>/models/best/best_model.zip`` paired
with ``Runs/<run_id>/models/best/vec_normalize.pkl`` saved at the same eval step (after
``fee_ramp_end`` when the best-model gate is on; end-of-run ``models/vec_normalize.pkl`` is
final-step stats only).

Anti-overfitting measures:
  - Fractionally differentiated price features (stationary + memory)
  - Observation noise on market features during training
  - Seed shuffling: fresh OS entropy on every episode reset
  - VecNormalize + cosine LR decay with floor
  - Domain randomization: Beta-centered fee_scale + obs_lag, bounds widen 10M→40M (65M budget)
  - Fee curriculum (train + eval): frictionless → linear fee/churn ramp → progressive DR on train
"""

from __future__ import annotations

import importlib.util
from pathlib import Path as _Path

_bootstrap_path = _Path(__file__).resolve().parent / "_bootstrap.py"
_bootstrap_spec = importlib.util.spec_from_file_location("_rlbot_repo_bootstrap", _bootstrap_path)
assert _bootstrap_spec is not None and _bootstrap_spec.loader is not None
_bootstrap_mod = importlib.util.module_from_spec(_bootstrap_spec)
_bootstrap_spec.loader.exec_module(_bootstrap_mod)

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _startup_log(msg: str) -> None:
    print(msg, flush=True)


_startup_log("[train] Starting (loading dependencies)...")

import numpy as np

_startup_log(
    "[train] Loading PyTorch and Stable-Baselines3 "
    "(first run in a new shell may take 1–5 minutes)..."
)
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

from rlbot.data_utils import (
    clip_index_until,
    fetch_aligned_daily,
    load_cache,
    reserve_chronological_holdout,
    save_cache,
    select_tradeable_columns,
    WalkforwardEnvPack,
    align_panel_to_timeline,
    train_test_split_alternating,
)
from rlbot.rl_config import (
    UNIVERSE_MAX_ASSETS,
    UNIVERSE_MIN_ASSETS,
    apply_deterministic_seeds,
    get_config,
    load_config,
    observation_dim_for_universe,
    set_config,
    slice_config_to_n_assets,
    validate_config_for_universe,
    write_config_snapshot,
)
from rlbot.vecnorm_utils import sync_vecnormalize_stats
from rlbot.run_artifacts import (
    DEFAULT_DATA_CACHE,
    RunPaths,
    config_sha256,
    git_provenance,
    new_run_id,
    read_run_manifest,
    resolve_data_cache,
    sha256_file,
    merge_manifest,
    write_manifest,
)
from rlbot.trading_env import EpisodeEndNavRecorder, MultiAssetPortfolioEnv
from rlbot.reward_logging import RewardDecompAccumulator
from rlbot.modal_cloud import commit_modal_volumes
from rlbot.visualize import TrainingVizCallback, open_plot_file

_startup_log("[train] Dependencies loaded.")

ROOT = Path(__file__).resolve().parent.parent
DATA_CACHE = DEFAULT_DATA_CACHE


def _persist_trade_artifacts(model: RecurrentPPO, train_env: VecNormalize, paths: RunPaths) -> tuple[Path, Path | None]:
    """Save end-of-run VecNormalize stats + final policy weights.

    Writes ``Runs/<id>/models/vec_normalize.pkl``. ``models/best/vec_normalize.pkl`` is
    written by ``EvalNavBestModelCallback`` at best-save time (stats the checkpoint was
    actually selected under) and is **never** overwritten here — a run that never saved
    a best model returns ``None`` for it rather than pairing best weights with
    end-of-run stats.
    """
    root_vn = paths.models_dir / "vec_normalize.pkl"
    train_env.save(str(root_vn))
    best_vn = paths.best_model_dir / "vec_normalize.pkl"
    best_vn_exists = best_vn.is_file()
    model.save(str(paths.final_model))
    return root_vn, best_vn if best_vn_exists else None


# ── Env factory ──────────────────────────────────────────────────────────

def _make_env_factory(
    pack: WalkforwardEnvPack,
    random_start: bool,
    log_dir: Path,
    monitor_stem: str,
    max_episode_steps: int = 252,
    obs_noise_std: float = 0.0,
    noise_scale: np.ndarray | None = None,
    reseed_on_reset: bool = False,
    env_seed: int | None = None,
    obs_lag_default: int = 1,
    domain_randomize: bool = True,
    inactivity_penalty_scale: float = 1.0,
    record_episode_nav: bool = False,
):
    """Return a callable that creates and wraps a single environment."""

    def _init():
        env = MultiAssetPortfolioEnv(
            **pack.env_kwargs(),
            random_start=random_start,
            max_episode_steps=max_episode_steps,
            obs_noise_std=obs_noise_std,
            noise_scale=noise_scale,
            reseed_on_reset=reseed_on_reset,
            env_seed=env_seed,
            obs_lag=0,
            obs_lag_default=obs_lag_default,
            fee_scale_default=1.0,
            domain_randomize=domain_randomize,
            inactivity_penalty_scale=inactivity_penalty_scale,
        )
        if record_episode_nav:
            return EpisodeEndNavRecorder(env)
        log_dir.mkdir(parents=True, exist_ok=True)
        return Monitor(env, filename=str(log_dir / monitor_stem))

    return _init


class EvalNavBestModelCallback(EvalCallback):
    """Run periodic eval; save ``best_model.zip`` on **max mean ending NAV**, not reward.

    Still logs ``evaluations.npz`` (rewards) for entropy scheduling; deployment model
    is chosen by validation wealth, avoiding passive low-churn reward hacks.

    When ``best_model_min_step`` > 0, eval NAV is logged from step 0 but ``models/best/``
    updates only after that step (default: ``fee_ramp_end`` — full eval fees + churn).
    """

    def __init__(
        self,
        eval_env,
        nav_history_path: Path,
        best_model_save_path: str,
        train_vec_env: VecNormalize | None = None,
        patience: int = 0,
        curriculum_end_step: int = 0,
        best_model_min_step: int = 0,
        **kwargs,
    ):
        self._best_model_dir = Path(best_model_save_path)
        self.nav_history_path = Path(nav_history_path)
        self.best_mean_nav = -np.inf
        self._nav_timesteps: list[int] = []
        self._mean_ending_nav: list[float] = []
        self._train_vec_env = train_vec_env
        self.patience = int(patience)
        self.curriculum_end_step = int(curriculum_end_step)
        self.best_model_min_step = int(best_model_min_step)
        self._post_gate_tracking_started = False
        self._evals_since_best = 0
        self.early_stop_reason: str | None = None
        self._load_nav_history()
        super().__init__(eval_env, best_model_save_path=None, **kwargs)

    def _best_model_gate_open(self) -> bool:
        return self.best_model_min_step <= 0 or self.num_timesteps >= self.best_model_min_step

    def _post_gate_best_nav(self) -> float:
        if self.best_model_min_step <= 0:
            return self.best_mean_nav
        post = [
            n
            for t, n in zip(self._nav_timesteps, self._mean_ending_nav)
            if t >= self.best_model_min_step
        ]
        return float(max(post)) if post else -np.inf

    def _load_nav_history(self) -> None:
        if not self.nav_history_path.is_file():
            return
        try:
            z = np.load(self.nav_history_path, allow_pickle=False)
            self._nav_timesteps = list(np.asarray(z["timesteps"], dtype=np.int64))
            self._mean_ending_nav = list(np.asarray(z["mean_ending_nav"], dtype=np.float64))
            if self._mean_ending_nav:
                if self.best_model_min_step > 0:
                    self.best_mean_nav = self._post_gate_best_nav()
                    self._post_gate_tracking_started = any(
                        t >= self.best_model_min_step for t in self._nav_timesteps
                    )
                else:
                    self.best_mean_nav = float(max(self._mean_ending_nav))
        except (OSError, ValueError, KeyError):
            pass

    def _save_nav_history(self) -> None:
        self.nav_history_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.nav_history_path,
            timesteps=np.asarray(self._nav_timesteps, dtype=np.int64),
            mean_ending_nav=np.asarray(self._mean_ending_nav, dtype=np.float64),
        )

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self._train_vec_env is not None:
                sync_vecnormalize_stats(self._train_vec_env, self.eval_env)
            self.eval_env.env_method("pop_ending_navs")

        continue_training = super()._on_step()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            all_navs: list[float] = []
            for nav_list in self.eval_env.env_method("pop_ending_navs"):
                all_navs.extend(nav_list)
            if all_navs:
                mean_nav = float(np.mean(all_navs))
                self._nav_timesteps.append(int(self.num_timesteps))
                self._mean_ending_nav.append(mean_nav)
                self._save_nav_history()
                self.logger.record("eval/mean_ending_nav", mean_nav)
                gate_open = self._best_model_gate_open()
                self.logger.record("eval/best_model_gate_open", float(gate_open))
                if gate_open:
                    if self.best_model_min_step > 0 and not self._post_gate_tracking_started:
                        self._post_gate_tracking_started = True
                        self.best_mean_nav = -np.inf
                    if mean_nav > self.best_mean_nav:
                        self.best_mean_nav = mean_nav
                        self._evals_since_best = 0
                        if self.verbose >= 1:
                            print(f"New best mean ending NAV: {mean_nav:,.0f}")
                        self._best_model_dir.mkdir(parents=True, exist_ok=True)
                        self.model.save(str(self._best_model_dir / "best_model"))
                        if self._train_vec_env is not None:
                            # Pair the checkpoint with the normalization stats it was
                            # selected under, not whatever exists at end of training.
                            self._train_vec_env.save(
                                str(self._best_model_dir / "vec_normalize.pkl")
                            )
                    elif self.patience > 0 and self.num_timesteps >= self.curriculum_end_step:
                        # Patience early-stop, but only after the curriculum has fully released.
                        self._evals_since_best += 1
                        self.logger.record("eval/evals_since_best", self._evals_since_best)
                        if self._evals_since_best >= self.patience:
                            self.early_stop_reason = (
                                f"no new best mean ending NAV for {self.patience} evals after "
                                f"curriculum (step {self.num_timesteps})"
                            )
                            print(f"[train] early stop: {self.early_stop_reason}")
                            return False

        return continue_training


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
    """High entropy early, then mandatory cosine decay (not eval-gated).

    Phase 1 (explore): ``ent_coef = explore_ent`` until ``decay_start_fraction``
        of the run (default 45%, aligned with fee ramp end). Exploration floors
        apply while ``num_timesteps < early_floor_steps`` (``early_floor_fraction``).
    Phase 2 (decay): cosine schedule from ``explore_ent`` → ``final_ent`` over
        the remaining ``1 - decay_start_fraction`` of training, regardless of
        eval NAV.
    """

    def __init__(
        self,
        explore_ent: float = 0.05,
        final_ent: float = 0.005,
        early_floor: float = 0.01,
        early_floor_steps: int = 3_000_000,
        min_explore_steps: int = 15_000_000,
        decay_start_fraction: float = 0.45,
        warmup_improvements: int = 3,
        eval_log_dir: str = "",
        eval_check_freq: int = 50_000,
        eval_nav_callback: "EvalNavBestModelCallback | None" = None,
    ):
        super().__init__()
        self.explore_ent = explore_ent
        self.final_ent = final_ent
        self.early_floor = early_floor
        self.early_floor_steps = early_floor_steps
        self.min_explore_steps = int(min_explore_steps)
        self.decay_start_fraction = float(np.clip(decay_start_fraction, 0.0, 0.99))
        self.warmup_improvements = warmup_improvements
        self.eval_log_dir = eval_log_dir
        self.eval_check_freq = max(1, int(eval_check_freq))
        self._eval_nav_callback = eval_nav_callback
        self._last_best: float = -float("inf")
        self._improvements: int = 0

    def _sync_eval_improvements(self) -> None:
        """Update improvement count at eval cadence only (no per-step disk I/O)."""
        if self._eval_nav_callback is not None:
            current_best = float(self._eval_nav_callback.best_mean_nav)
        elif self.eval_log_dir:
            npz = Path(self.eval_log_dir) / "evaluations.npz"
            if not npz.is_file():
                return
            data = np.load(str(npz))
            if "results" not in data:
                return
            current_best = float(np.asarray(data["results"]).mean(axis=1).max())
        else:
            return

        if current_best > self._last_best + 1e-6:
            if self.num_timesteps >= self.min_explore_steps:
                self._improvements += 1
            self._last_best = current_best

    def _on_step(self) -> bool:
        import math

        if self.n_calls % self.eval_check_freq == 0:
            self._sync_eval_improvements()

        progress_remaining = self.model._current_progress_remaining
        progress_done = 1.0 - float(progress_remaining)

        if progress_done >= self.decay_start_fraction:
            span = max(1.0 - self.decay_start_fraction, 1e-12)
            frac = min((progress_done - self.decay_start_fraction) / span, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * frac))
            ent = self.final_ent + (self.explore_ent - self.final_ent) * cosine
        else:
            ent = self.explore_ent
            if self.num_timesteps < self.min_explore_steps:
                ent = max(
                    ent,
                    max(self.early_floor, get_config().entropy_schedule.early_floor_high),
                )
            elif self.num_timesteps < self.early_floor_steps:
                ent = max(ent, self.early_floor)

        self.model.ent_coef = ent
        self.logger.record("config/ent_coef", ent)
        self.logger.record("config/eval_improvements", self._improvements)
        self.logger.record("config/entropy_decay_active", float(progress_done >= self.decay_start_fraction))
        return True


def trade_curriculum_milestones(learn_budget: int) -> tuple[int, int]:
    """Return ``(fee_free_until, fee_ramp_end)`` in environment steps.

    - **Frictionless** until ``fee_free_until``; **fee ramp** until ``fee_ramp_end``; then DR.
    - **Churn** penalty scale ramps ``churn_ramp_floor`` → 1.0 over the same fee-ramp window.

    At ≤65M budget: fraction-of-run schedule to avoid a mid-run cliff.
    Between 65M and 120M: interpolate toward long-run anchors; ≥120M uses fixed long milestones.
    """
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


def fee_curriculum_milestones(learn_budget: int) -> tuple[int, int]:
    """Backward-compatible (fee_free, fee_ramp_end) for logging."""
    ff, fr = trade_curriculum_milestones(learn_budget)
    return ff, fr


def resolve_best_model_min_step(learn_budget: int) -> int:
    """Step before which ``models/best/`` is not updated (eval NAV still logged).

    ``curriculum.best_model_min_step``: ``null`` → ``fee_ramp_end``; ``0`` → disable gate.
    """
    explicit = get_config().curriculum.best_model_min_step
    if explicit is not None:
        return max(0, int(explicit))
    _, fee_ramp_end = trade_curriculum_milestones(learn_budget)
    return fee_ramp_end


def entropy_early_floor_milestones(learn_budget: int) -> int:
    """Entropy floor duration as a fraction of ``learn_budget`` (see config ``early_floor_fraction``)."""
    lb = max(1, int(learn_budget))
    frac = get_config().entropy_schedule.early_floor_fraction
    return max(1, int(lb * frac))


def dr_widen_end_milestone(learn_budget: int) -> int:
    """Last step of progressive DR widening (fee/lag bounds); starts at ``fee_ramp_end``."""
    cur = get_config().curriculum
    _, fee_ramp_end = trade_curriculum_milestones(learn_budget)
    lb = max(1, int(learn_budget))
    if lb <= cur.budget_short:
        span = max(1, int(cur.dr_widen_span_fraction * lb))
    elif lb >= cur.budget_long:
        span = cur.dr_widen_span_long
    else:
        span_short = max(1, int(cur.dr_widen_span_fraction * cur.budget_short))
        t = (lb - cur.budget_short) / (cur.budget_long - cur.budget_short)
        span = int(span_short + t * (cur.dr_widen_span_long - span_short))
        span = max(1, span)
    return min(lb, fee_ramp_end + span)


def entropy_dr_lock_milestones(learn_budget: int) -> int:
    """No eval-driven exploit phase until this step (fraction of learn budget)."""
    lb = max(1, int(learn_budget))
    frac = get_config().entropy_schedule.dr_lock_fraction
    return max(1, int(frac * lb))


class TradingCurriculumCallback(BaseCallback):
    """Fee/churn curriculum on train + eval; DR bounds on train only.

    Milestones from ``trade_curriculum_milestones(learn_budget)``.

    - Steps ``[0, fee_free_until)``: ``fee_scale = 0`` (frictionless).
    - Steps ``[fee_free_until, fee_ramp_end)``: linear ramp to ``fee_scale = 1.0``.
    - Steps ``[fee_ramp_end, dr_widen_end)``: progressive widening of DR fee/lag bounds (train).
    - Steps ``>= dr_widen_end``: full DR on train (fee in config DR range, lag in {0, 1, 2}).
    - Churn: ``churn_scale = 0`` before ``fee_free_until``; then ``churn_ramp_floor`` → ``1``
      linearly over the fee-ramp window (train + eval).
    - Eval envs mirror the fee/churn schedule (no domain randomization).
    """

    def __init__(
        self,
        vec_env: VecNormalize,
        learn_budget: int,
        update_freq: int = 50_000,
        eval_vec_env: VecNormalize | None = None,
    ):
        super().__init__()
        self.vec_env = vec_env
        self.eval_vec_env = eval_vec_env
        self.learn_budget = int(learn_budget)
        self.fee_free_until, self.fee_ramp_end = trade_curriculum_milestones(
            self.learn_budget
        )
        self.dr_widen_end = dr_widen_end_milestone(self.learn_budget)
        self.update_freq = max(1, int(update_freq))
        self._last_key: tuple | None = None

    def _fee_override(self, t: int) -> float | None:
        if t < self.fee_free_until:
            return 0.0
        if t < self.fee_ramp_end:
            span = max(self.fee_ramp_end - self.fee_free_until, 1)
            return float(t - self.fee_free_until) / float(span)
        return None

    def _churn_scale(self, t: int) -> float:
        """Churn penalty scale aligned with fee curriculum (0 → floor → 1)."""
        floor = float(get_config().curriculum.churn_ramp_floor)
        if t < self.fee_free_until:
            return 0.0
        if t >= self.fee_ramp_end:
            return 1.0
        span = max(self.fee_ramp_end - self.fee_free_until, 1)
        progress = float(t - self.fee_free_until) / float(span)
        return floor + (1.0 - floor) * progress

    def _dr_bounds(self, t: int) -> tuple[float, float, int, int]:
        """Progressive fee/lag bounds after fee curriculum releases DR."""
        dr_min = get_config().environment.domain_randomize_fee_dr_min
        dr_max = get_config().environment.domain_randomize_fee_dr_max
        env_cfg = get_config().environment
        lag_lo, lag_hi = env_cfg.min_obs_lag, env_cfg.max_obs_lag
        if t < self.fee_ramp_end:
            return dr_min, dr_max, lag_lo, lag_hi
        if t >= self.dr_widen_end:
            return dr_min, dr_max, lag_lo, lag_hi
        progress = (t - self.fee_ramp_end) / max(self.dr_widen_end - self.fee_ramp_end, 1)
        fee_min = 1.0 - (1.0 - dr_min) * progress
        fee_max = 1.0 + (dr_max - 1.0) * progress
        lag_min = int(round(1.0 - progress))
        lag_max = int(round(1.0 + progress))
        lag_min = max(lag_lo, min(lag_min, lag_hi))
        lag_max = max(lag_min, min(lag_max, lag_hi))
        return fee_min, fee_max, lag_min, lag_max

    def _apply(self) -> None:
        t = int(self.num_timesteps)
        fee = self._fee_override(t)
        churn = self._churn_scale(t)
        fee_min, fee_max, lag_min, lag_max = self._dr_bounds(t)
        key = (fee, churn, fee_min, fee_max, lag_min, lag_max)
        if key != self._last_key:
            self.vec_env.env_method("set_curriculum_state", fee, churn)
            self.vec_env.env_method(
                "set_randomization_bounds", fee_min, fee_max, lag_min, lag_max
            )
            if self.eval_vec_env is not None:
                self.eval_vec_env.env_method("set_curriculum_state", fee, churn)
            self._last_key = key
            self.logger.record("config/curriculum_fee_override", -1.0 if fee is None else float(fee))
            self.logger.record("config/curriculum_churn_scale", churn)
            self.logger.record("config/curriculum_fee_dr_min", fee_min)
            self.logger.record("config/curriculum_fee_dr_max", fee_max)
            self.logger.record("config/curriculum_obs_lag_dr_min", float(lag_min))
            self.logger.record("config/curriculum_obs_lag_dr_max", float(lag_max))

    def _on_training_start(self) -> None:
        self._last_key = None
        self._apply()

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            self._apply()
        return True


class RewardDecompCallback(BaseCallback):
    """Aggregate ``info['rew_decomp/*']`` → TensorBoard scalars + a windowed JSON snapshot.

    Makes the reward balance observable (the review's asymmetry finding: inactivity
    can dwarf participation/churn). Logs per-term means and share-of-absolute-reward over
    the window since the last log, then resets, so ``reward_decomp.json`` reflects recent
    (steady-state) behavior rather than a run-wide average. TB scalars give the time series.
    """

    def __init__(self, json_path, log_freq: int = 50_000):
        super().__init__()
        self.json_path = json_path
        self.log_freq = max(int(log_freq), 1)
        self._acc = RewardDecompAccumulator()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos:
            self._acc.update(infos)
        if self.n_calls % self.log_freq == 0 and self._acc.count > 0:
            s = self._acc.summary()
            for term, val in s["mean"].items():
                self.logger.record(f"rew_decomp/mean/{term}", val)
            for term, val in s["abs_share"].items():
                self.logger.record(f"rew_decomp/abs_share/{term}", val)
            write_manifest(self.json_path, {"timesteps": int(self.num_timesteps), **s})
            self._acc.reset()  # windowed: next snapshot reflects the next interval only
        return True


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=str(ROOT / "config" / "config.yaml"))
    pre_args, _ = pre.parse_known_args()
    set_config(load_config(pre_args.config))

    cfg = get_config()
    hp = cfg.hyperparameters
    tr_cfg = cfg.training
    pol = cfg.policy
    vn_cfg = cfg.vec_normalize
    ent_cfg = cfg.entropy_schedule

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(cfg.path),
        help="Path to config.yaml (loaded before other defaults)",
    )
    parser.add_argument(
        "--since", default=cfg.data.since,
        help="Fetch start date (UTC). Assets with later listings are backfilled.",
    )
    parser.add_argument("--until", default=None, help="Optional fetch end (UTC)")
    parser.add_argument("--refresh-data", action="store_true", help="Refetch OHLCV from yfinance")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=tr_cfg.timesteps,
        help="Total PPO steps (default from config.yaml)",
    )
    parser.add_argument("--learning-rate", type=float, default=hp.learning_rate)
    parser.add_argument("--batch-size", type=int, default=hp.batch_size)
    parser.add_argument("--n-steps", type=int, default=hp.n_steps)
    parser.add_argument("--n-envs", type=int, default=tr_cfg.n_envs, help="Parallel training envs")
    parser.add_argument(
        "--max-ep-steps",
        type=int,
        default=cfg.environment.max_episode_steps,
        help="Steps per training episode (~3 months of daily bars)",
    )
    parser.add_argument(
        "--obs-noise",
        type=float,
        default=tr_cfg.obs_noise,
        help="Gaussian noise std added to market features during training (regularization)",
    )
    parser.add_argument(
        "--obs-lag",
        type=int,
        default=cfg.environment.obs_lag_default,
        help="Default market-feature lag when not randomizing (eval); training samples min..max per episode",
    )
    parser.add_argument("--seed", type=int, default=tr_cfg.seed)
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=tr_cfg.holdout_days,
        help=(
            "Reserve the last N calendar days for backtest only; training/eval never see these bars. "
            "Ignored when --train-end and --holdout-start are set."
        ),
    )
    parser.add_argument(
        "--train-end",
        default=None,
        metavar="YYYY-MM-DD",
        help="Last trainable calendar day (inclusive). Requires --holdout-start.",
    )
    parser.add_argument(
        "--holdout-start",
        default=None,
        metavar="YYYY-MM-DD",
        help="First OOS calendar day (inclusive). Requires --train-end.",
    )
    parser.add_argument(
        "--holdout-end",
        default=None,
        metavar="YYYY-MM-DD",
        help="Last OOS calendar day (inclusive). Default: last bar after --until clip.",
    )
    parser.add_argument(
        "--block-size", type=int, default=tr_cfg.block_size, help="Walk-forward block size in trading bars"
    )
    parser.add_argument(
        "--eval-stride", type=int, default=tr_cfg.eval_stride, help="Every Nth block goes to eval"
    )
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--viz-freq", type=int, default=tr_cfg.viz_freq)
    parser.add_argument("--show-viz", action="store_true")
    parser.add_argument("--run-id", default="", metavar="ID")
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Walk-forward window (1–6). When --run-id is omitted, auto id is "
            "W{N}_<month><day> (e.g. W1_604); duplicates get _a, _b, …"
        ),
    )
    parser.add_argument(
        "--n-assets",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"Use the first N keys from universe.assets in config.yaml "
            f"({UNIVERSE_MIN_ASSETS}–{UNIVERSE_MAX_ASSETS}); slices cap weights and costs. "
            "Cannot exceed the number of assets defined in the config file."
        ),
    )
    parser.add_argument(
        "--resume",
        default="",
        metavar="PATH",
        help=(
            "Crash-resume from a checkpoint: restore weights + VecNormalize, continue the "
            "curriculum and entropy schedule from the checkpoint timestep."
        ),
    )
    parser.add_argument(
        "--finetune",
        default="",
        metavar="PATH",
        help=(
            "Fine-tune from a checkpoint: lower LR/entropy/clip, skip curriculum and "
            "adaptive-entropy callbacks (experimental regime, not crash resume)."
        ),
    )
    parser.add_argument(
        "--overwrite-run", action="store_true",
        help="Allow training into an existing Runs/<run-id>/ directory (overwrites its "
             "manifest/models; refused by default — reuse also restores the old run's "
             "best-eval threshold, which can suppress best_model saves).",
    )
    args = parser.parse_args()
    if args.resume.strip() and args.finetune.strip():
        raise SystemExit("Use only one of --resume or --finetune, not both.")
    if Path(args.config).resolve() != cfg.path:
        set_config(load_config(args.config))
        cfg = get_config()

    if args.n_assets is not None:
        cfg = slice_config_to_n_assets(get_config(), args.n_assets)
        set_config(cfg)
        _startup_log(
            f"[train] --n-assets {args.n_assets}: "
            f"{', '.join(cfg.universe.tickers)}"
        )

    apply_deterministic_seeds(args.seed)

    if args.run_id.strip():
        run_id = args.run_id.strip()
    elif args.window is not None:
        run_id = new_run_id(args.window)
    else:
        raise SystemExit(
            "Provide --run-id or --window (auto id: W{window}_<month><day>, e.g. W1_604)."
        )
    paths = RunPaths(run_id=run_id)
    if (paths.run_meta_dir / "manifest.json").is_file() and not (
        args.overwrite_run or args.resume
    ):
        raise SystemExit(
            f"Run directory Runs/{run_id}/ already has a manifest. Refusing to overwrite "
            "an existing run (its artifacts may be referenced by the research registry). "
            "Pick a new --run-id, or pass --overwrite-run to retrain in place."
        )
    paths.mkdirs()
    if args.overwrite_run and not args.resume:
        # A fresh retrain must not inherit the old run's best-eval threshold or its
        # best-time normalization stats; stale ones suppress best_model saves.
        for stale in (
            paths.eval_nav_history,
            paths.best_model_dir / "best_model.zip",
            paths.best_model_dir / "vec_normalize.pkl",
        ):
            stale.unlink(missing_ok=True)
    if args.n_assets is not None:
        write_config_snapshot(cfg, paths.run_meta_dir / "config.yaml")
    else:
        shutil.copy2(cfg.path, paths.run_meta_dir / "config.yaml")

    _startup_log(f"[train] Run id={run_id!r}; loading market data...")

    # ── data ─────────────────────────────────────────────────────────────
    data_cache = resolve_data_cache()
    if args.refresh_data or not data_cache.is_file():
        _startup_log("[train] Fetching OHLCV from yfinance (may take several minutes)...")
        idx, ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro, trend, asset_vol, macro_vol, asset_live = (
            fetch_aligned_daily(
                symbols_dict=cfg.universe.assets,
                since=args.since,
                until=args.until,
                fracdiff_d=cfg.data.fracdiff_d,
            )
        )
        save_cache(
            str(data_cache),
            idx,
            ohlcv,
            rsi,
            macd,
            macro,
            fracdiff,
            fracdiff_macro,
            trend,
            asset_vol,
            macro_vol,
            asset_live=asset_live,
            fracdiff_d=cfg.data.fracdiff_d,
            tickers=cfg.universe.tickers,
        )
        panel_tickers = list(cfg.universe.tickers)
    else:
        (
            idx,
            ohlcv,
            rsi,
            macd,
            macro,
            fracdiff,
            fracdiff_macro,
            trend,
            asset_vol,
            macro_vol,
            asset_live,
            panel_tickers,
        ) = load_cache(str(data_cache), expected_fracdiff_d=cfg.data.fracdiff_d)

    if list(panel_tickers) != cfg.universe.tickers:
        (
            ohlcv,
            rsi,
            macd,
            fracdiff,
            trend,
            panel_tickers,
            asset_live,
            asset_vol,
            macro_vol,
        ) = select_tradeable_columns(
            ohlcv,
            rsi,
            macd,
            fracdiff,
            trend,
            panel_tickers,
            cfg.universe.tickers,
            asset_live=asset_live,
            asset_vol=asset_vol,
            macro_vol=macro_vol,
        )

    validate_config_for_universe(cfg, int(ohlcv.shape[1]))
    n_assets = int(ohlcv.shape[1])
    n_actions = n_assets + 1
    obs_dim = observation_dim_for_universe(n_assets)
    _startup_log(f"[train] Data panel: {len(idx)} bars, N={n_assets} assets.")

    if args.until:
        idx, (ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro, trend, asset_vol, macro_vol, asset_live) = (
            clip_index_until(
                idx,
                ohlcv,
                rsi,
                macd,
                macro,
                fracdiff,
                fracdiff_macro,
                trend,
                asset_vol,
                macro_vol,
                asset_live,
                until=args.until,
            )
        )

    save_cache(
        str(paths.data_snapshot),
        idx,
        ohlcv,
        rsi,
        macd,
        macro,
        fracdiff,
        fracdiff_macro,
        trend,
        asset_vol,
        macro_vol,
        asset_live=asset_live,
        fracdiff_d=cfg.data.fracdiff_d,
        tickers=panel_tickers,
    )
    if args.n_assets is not None:
        print(
            f"  data snapshot: wrote effective N={n_assets} panel to {paths.data_snapshot.name} "
            f"(run-local; global cache may be wider — use --refresh-data when changing --n-assets)"
        )

    (idx_fit, ohlcv_fit, macro_fit, asset_live_fit), (
        idx_hold,
        ohlcv_hold,
        macro_hold,
        asset_live_hold,
    ) = reserve_chronological_holdout(
        idx,
        ohlcv,
        macro,
        asset_live,
        holdout_days=args.holdout_days,
        train_end=args.train_end,
        holdout_start=args.holdout_start,
        holdout_end=args.holdout_end,
    )

    purge = cfg.data.feature_purge_warmup
    split_mode = cfg.data.feature_split_mode
    feat_src = "cache" if data_cache.is_file() and not args.refresh_data else "computed"
    if split_mode == "independent":
        # Per-segment recompute + purge happens inside the split; precomputed continuous
        # features are not used.
        feature_kwargs: dict = {}
        feat_desc = f"independent split: per-segment recompute + purge={purge} applied"
    else:
        rsi_fit, macd_fit, fd_fit, fdm_fit, trend_fit, avol_fit, mvol_fit = align_panel_to_timeline(
            idx,
            idx_fit,
            rsi,
            macd,
            fracdiff,
            fracdiff_macro,
            trend,
            asset_vol,
            macro_vol,
        )
        feature_kwargs = dict(
            rsi=rsi_fit,
            macd=macd_fit,
            fracdiff=fd_fit,
            fracdiff_macro=fdm_fit,
            trend=trend_fit,
            asset_vol=avol_fit,
            macro_vol=mvol_fit,
        )
        feat_desc = (
            f"continuous split: {feat_src} panel → block slice "
            f"(purge={purge} unused; matches continuous backtest memory)"
        )
    _startup_log(
        f"[train] Walk-forward block split ({feat_desc}, "
        f"block={args.block_size}, stride={args.eval_stride})..."
    )
    train_pack, eval_pack = (
        WalkforwardEnvPack.from_tuple(p)
        for p in train_test_split_alternating(
            idx_fit,
            ohlcv_fit,
            macro_fit,
            asset_live_fit,
            block_size=args.block_size,
            eval_stride=args.eval_stride,
            fracdiff_d=cfg.data.fracdiff_d,
            feature_purge_warmup=purge,
            feature_split_mode=split_mode,
            feature_preroll_bars=cfg.data.feature_preroll_bars,
            **feature_kwargs,
        )
    )
    train_idx = train_pack.idx
    eval_idx = eval_pack.idx
    train_boundaries = train_pack.block_boundaries
    eval_boundaries = eval_pack.block_boundaries
    print(f"  features: {feat_desc}")

    if len(train_idx) < 200:
        raise RuntimeError(
            "Not enough training rows after split; widen the date range or reduce --holdout-days."
        )

    universe_meta = {
        "benchmark": cfg.universe.benchmark,
        "tickers": list(panel_tickers),
        "n_assets": n_assets,
        "n_actions": n_actions,
        "obs_dim": obs_dim,
    }

    # Provenance shared by the pre- and post-training manifest writes.
    provenance = {
        "feature_split_mode": cfg.data.feature_split_mode,
        "config_hash": config_sha256(cfg.to_dict()),
        "data_cache_hash": sha256_file(paths.data_snapshot),
        **git_provenance(),
    }

    write_manifest(
        paths.manifest_path,
        {
            "run_id": run_id,
            "config_path": str(cfg.path),
            "args": vars(args),
            "universe": universe_meta,
            "n_index": int(len(idx)),
            "n_trainable_bars": int(len(idx_fit)),
            "chronological_holdout": {
                "holdout_days": int(args.holdout_days),
                "train_end": args.train_end,
                "holdout_start": args.holdout_start,
                "holdout_end": args.holdout_end or (str(idx_hold[-1]) if len(idx_hold) else None),
                "trainable_end": str(idx_fit[-1]) if len(idx_fit) else None,
                "holdout_bars": int(len(idx_hold)),
                "date_start": str(idx_hold[0]) if len(idx_hold) else None,
                "date_end": str(idx_hold[-1]) if len(idx_hold) else None,
            },
            "n_train_bars": int(len(train_idx)),
            "n_eval_bars": int(len(eval_idx)),
            "data_cache_snapshot": str(paths.data_snapshot),
            **provenance,
        },
    )

    print(f"Run id: {run_id}")
    print(
        f"  universe: N={n_assets} tradeable assets "
        f"({', '.join(panel_tickers[:5])}{'...' if n_assets > 5 else ''}) "
        f"[config universe.assets; CLI --n-assets overrides count]"
    )
    print(f"  plots:   {paths.plots_dir}/")
    print(f"  models:  {paths.models_dir}/")
    print(f"  logs:    {paths.logs_dir}/")
    print(f"  tb_logs: {paths.tb_dir}/")
    print(f"  meta:    {paths.run_meta_dir}/")
    print(
        f"  network: RecurrentPPO MlpLstmPolicy — obs_dim={obs_dim}, "
        f"n_actions={n_actions} (cash+{n_assets} assets), LSTM 2×64 + MLP [128,128]"
    )
    _es_patience = int(cfg.training.early_stop_patience)
    if _es_patience > 0:
        print(
            f"  early_stop: patience={_es_patience} evals after curriculum "
            f"(else full {args.timesteps:,} timesteps); best_model by eval NAV"
        )
    else:
        print(f"  early_stop: off (full {args.timesteps:,} timesteps; best_model by eval NAV)")
    print(
        f"  trade bundle: best/ saves model + vec_normalize together on each new best eval NAV; "
        f"exit writes final model + end-of-run vec_normalize.pkl"
    )
    print(f"  n_envs={args.n_envs}, n_steps={args.n_steps}, batch={args.batch_size}")
    rollout_size = int(args.n_steps) * int(args.n_envs)
    if int(args.batch_size) > rollout_size:
        raise ValueError(
            f"batch_size ({args.batch_size}) must be <= n_steps * n_envs "
            f"({args.n_steps} * {args.n_envs} = {rollout_size}) for PPO"
        )
    print(f"  max_ep_steps={args.max_ep_steps} (daily bars, train only; eval spans full segment)")
    print(f"  obs_noise={args.obs_noise}, reseed_on_reset=True (training)")
    print(f"  obs_lag: train Uniform{{0,1,2}} per episode; eval fixed at {args.obs_lag}")
    print(f"  execution=open[t+1] (realistic: decide after close[t-1], fill at next open)")
    print(
        f"  reward: return*{cfg.reward.reward_scale:g} (downside amp gamma={cfg.reward.drawdown_downside_gamma:g}) "
        f"+ bench_excess*{cfg.reward.benchmark_excess_scale:g} "
        f"+ Sortino*{cfg.reward.risk_bonus_scale:g} "
        f"(combined cap {cfg.reward.benchmark_relative_max_share:.0%}) "
        f"+ participation*{cfg.reward.participation_bonus:g}*{cfg.reward.participation_reward_scale:g} "
        f"- inactivity - tx_cost*{cfg.reward.churn_penalty:g}*{cfg.reward.reward_scale:g}"
    )
    print(
        f"  eval inactivity scale: {cfg.reward.eval_inactivity_penalty_scale} "
        f"(train=1.0)"
    )
    print(f"  eval plot: mean ending NAV → Runs/<id>/eval_logs/eval_nav_history.npz")
    print(
        f"  action: softmax(cash+{n_assets} assets), long-only risky weights, "
        f"soft cap per asset (config)"
    )
    print(f"  universe: {', '.join(panel_tickers)}")
    _dre = dr_widen_end_milestone(args.timesteps)
    print(
        f"  domain_randomization: fee_scale~Beta(5,5) on widening bounds, "
        f"obs_lag~Discrete (training, after fee curriculum)"
    )
    _ff, _fr = trade_curriculum_milestones(args.timesteps)
    _ef = entropy_early_floor_milestones(args.timesteps)
    _edl = entropy_dr_lock_milestones(args.timesteps)
    _decay_frac = get_config().entropy_schedule.decay_start_fraction
    _decay_step = int(_decay_frac * args.timesteps)
    print(
        f"  fee curriculum (train + eval): fee=0 for {_ff:,} steps → linear ramp to 1.0 by "
        f"{_fr:,} → progressive DR widen to {_dre:,} → full DR (train only); "
        f"churn scale 0 → {cfg.curriculum.churn_ramp_floor:g} → 1.0 over same fee-ramp window"
    )
    print(f"  feature_split_mode: {cfg.data.feature_split_mode}")
    _bms = resolve_best_model_min_step(args.timesteps)
    if _bms > 0:
        print(
            f"  best_model gate: eval NAV logged always; models/best/ updates from step "
            f"{_bms:,} (fee_ramp_end; full eval fees + churn)"
        )
    else:
        print("  best_model gate: off (eval NAV selects best from step 0)")
    print(
        f"  entropy: explore {ent_cfg.explore_ent} (floor 0.02 until {_edl:,} steps, "
        f"then 0.01 for {_ef:,}) → cosine decay to {ent_cfg.final_ent} from "
        f"{_decay_frac:.0%} of run (step ~{_decay_step:,}), not eval-gated"
    )
    print(f"  LR={args.learning_rate} (cosine → 1e-6 floor)")
    if args.train_end and args.holdout_start:
        print(
            f"  OOS holdout: {args.holdout_start} .. {idx_hold[-1].date()} → {len(idx_hold)} bars "
            f"({idx_hold[0].date()} .. {idx_hold[-1].date()}) — excluded from training/eval"
        )
        print(
            f"  trainable through {args.train_end} → {len(idx_fit)} bars "
            f"({idx_fit[0].date()} .. {idx_fit[-1].date()})"
        )
    else:
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
    _startup_log(
        f"[train] Spawning {n_envs} parallel training envs "
        f"(first launch may take 1–3 minutes)..."
    )

    train_noise_scale = None
    if args.obs_noise > 0.0:
        train_noise_scale = MultiAssetPortfolioEnv.compute_obs_noise_scale(
            train_pack.ohlcv,
            train_pack.rsi,
            train_pack.macd,
            train_pack.fracdiff,
            train_pack.fracdiff_macro,
            train_pack.trend,
            train_pack.macro,
            train_pack.asset_vol,
            train_pack.macro_vol,
            n_assets=n_assets,
            n_noisy_features=MultiAssetPortfolioEnv.noisy_market_feature_count(n_assets),
            lookback=cfg.environment.lookback,
            return_horizons=MultiAssetPortfolioEnv.RETURN_HORIZONS,
            min_t=cfg.environment.lookback + cfg.environment.max_obs_lag,
            max_t=int(train_pack.ohlcv.shape[0]) - 2,
        )

    reproducible = bool(cfg.training.reproducible)
    if reproducible:
        _startup_log(
            "[train] reproducible=True: deterministic per-env seed streams "
            "(seed + env index); same-seed runs reproduce."
        )
    train_env = SubprocVecEnv([
        _make_env_factory(
            train_pack,
            random_start=True,
            noise_scale=train_noise_scale,
            log_dir=paths.logs_dir,
            monitor_stem=f"train_monitor_{i}",
            max_episode_steps=args.max_ep_steps,
            obs_noise_std=args.obs_noise,
            reseed_on_reset=not reproducible,
            env_seed=(int(args.seed) + i) if reproducible else None,
            obs_lag_default=args.obs_lag,
            domain_randomize=True,
            inactivity_penalty_scale=1.0,
        )
        for i in range(n_envs)
    ])
    train_env = VecNormalize(
        train_env,
        norm_obs=vn_cfg.norm_obs,
        norm_reward=vn_cfg.norm_reward_train,
        clip_obs=vn_cfg.clip_obs,
        clip_reward=vn_cfg.clip_reward,
        gamma=hp.gamma,
    )

    eval_env = SubprocVecEnv([
        _make_env_factory(
            eval_pack,
            random_start=False,
            noise_scale=train_noise_scale,
            log_dir=paths.logs_dir,
            monitor_stem="eval_monitor",
            max_episode_steps=args.max_ep_steps,
            reseed_on_reset=False,
            obs_lag_default=args.obs_lag,
            domain_randomize=False,
            inactivity_penalty_scale=cfg.reward.eval_inactivity_penalty_scale,
            record_episode_nav=True,
        )
    ])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=vn_cfg.norm_obs,
        norm_reward=False,
        clip_obs=vn_cfg.clip_obs,
        gamma=hp.gamma,
        training=False,
    )
    _startup_log("[train] Environments ready; building RecurrentPPO policy...")

    # ── model ────────────────────────────────────────────────────────────
    policy_kwargs = dict(
        lstm_hidden_size=pol.lstm_hidden_size,
        n_lstm_layers=pol.n_lstm_layers,
        net_arch=dict(pi=pol.net_arch_pi, vf=pol.net_arch_vf),
        activation_fn=th.nn.Tanh,
        ortho_init=True,
        optimizer_class=th.optim.AdamW,
        optimizer_kwargs=dict(weight_decay=hp.weight_decay),
    )

    lr_schedule = _lr_schedule_with_floor(args.learning_rate, floor_lr=hp.learning_rate_floor)

    finetune_mode = bool(args.finetune.strip())
    checkpoint_arg = args.finetune.strip() or args.resume.strip()
    if checkpoint_arg:
        resume_path = Path(checkpoint_arg)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

        mode_label = "Fine-tuning" if finetune_mode else "Crash-resuming"
        print(f"  {mode_label} from: {resume_path}")
        model = RecurrentPPO.load(
            str(resume_path),
            env=train_env,
            device="auto",
            tensorboard_log=str(paths.tb_dir),
        )
        model.learning_rate = lr_schedule
        model.n_steps = args.n_steps
        model.batch_size = args.batch_size
        if finetune_mode:
            model.ent_coef = hp.ent_coef_finetune
            model.clip_range = lambda _: hp.clip_range_finetune

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

        if finetune_mode:
            print(f"  Fine-tune LR={args.learning_rate}, ent_coef={hp.ent_coef_finetune}, clip={hp.clip_range_finetune}")
        else:
            print(
                f"  Resume at timestep {model.num_timesteps:,}: curriculum + entropy callbacks active"
            )
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=lr_schedule,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=hp.n_epochs,
            gamma=hp.gamma,
            gae_lambda=hp.gae_lambda,
            clip_range=hp.clip_range,
            ent_coef=hp.ent_coef_initial,
            vf_coef=hp.vf_coef,
            max_grad_norm=hp.max_grad_norm,
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
    # Evaluate every 500k global timesteps (500k / n_envs vector steps; decoupled from n_steps).
    eval_freq = max(500_000 // n_envs, 1)
    validation_segments = eval_env.env_method("get_segments")[0]
    n_validation_blocks = (
        len(validation_segments) if validation_segments else tr_cfg.eval_n_episodes
    )
    n_validation_blocks = max(1, int(n_validation_blocks))
    eval_coverage_bars = (
        int(sum(max(0, seg_end - earliest - 1) for earliest, seg_end in validation_segments))
        if validation_segments
        else 0
    )
    print(
        f"  eval: {n_validation_blocks} episode(s) = one full rollout per eval segment "
        f"(config eval_n_episodes={tr_cfg.eval_n_episodes} is fallback only)"
    )
    print(
        f"  eval coverage: {n_validation_blocks} segments / {eval_coverage_bars} scored bars "
        f"(effective sample size of the deterministic eval-selection signal)"
    )
    # Patience early-stop is gated on curriculum completion (dr_widen_end); patience=0 keeps
    # the full --timesteps budget. best_model saves open after fee_ramp_end (full eval fees).
    curriculum_end_step = dr_widen_end_milestone(args.timesteps)
    early_stop_patience = int(tr_cfg.early_stop_patience)
    best_model_min_step = resolve_best_model_min_step(args.timesteps)
    eval_callback = EvalNavBestModelCallback(
        eval_env,
        nav_history_path=paths.eval_nav_history,
        best_model_save_path=str(paths.best_model_dir),
        train_vec_env=train_env,
        patience=early_stop_patience,
        curriculum_end_step=curriculum_end_step,
        best_model_min_step=best_model_min_step,
        log_path=str(paths.eval_log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=n_validation_blocks,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(tr_cfg.checkpoint_save_freq_steps // n_envs, args.n_steps),
        save_path=str(paths.models_dir / "checkpoints"),
        name_prefix="ppo",
        save_vecnormalize=True,
    )

    reward_decomp_callback = RewardDecompCallback(
        json_path=paths.eval_log_dir / "reward_decomp.json",
        log_freq=max(tr_cfg.curriculum_update_freq, 1),
    )
    callbacks = [eval_callback, checkpoint_callback, reward_decomp_callback]
    if not finetune_mode:
        callbacks.insert(
            0,
            TradingCurriculumCallback(
                train_env,
                learn_budget=args.timesteps,
                update_freq=tr_cfg.curriculum_update_freq,
                eval_vec_env=eval_env,
            ),
        )
        callbacks.append(AdaptiveEntropyCallback(
            explore_ent=ent_cfg.explore_ent,
            final_ent=ent_cfg.final_ent,
            early_floor=ent_cfg.early_floor,
            early_floor_steps=entropy_early_floor_milestones(args.timesteps),
            min_explore_steps=entropy_dr_lock_milestones(args.timesteps),
            decay_start_fraction=ent_cfg.decay_start_fraction,
            warmup_improvements=ent_cfg.warmup_improvements,
            eval_log_dir=str(paths.eval_log_dir),
            eval_check_freq=eval_freq,
            eval_nav_callback=eval_callback,
        ))
    if not args.no_viz:
        callbacks.append(
            TrainingVizCallback(
                plot_path=paths.training_plot,
                eval_nav_npz_path=paths.eval_nav_history,
                plot_freq=args.viz_freq,
            )
        )

    # ── train ────────────────────────────────────────────────────────────
    _startup_log(f"[train] Starting PPO learning ({args.timesteps:,} timesteps)...")
    learn_error: BaseException | None = None
    interrupted = False
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
            reset_num_timesteps=not bool(args.resume),
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\n\nCtrl+C detected — saving current weights before exit…")
    except BaseException as e:
        learn_error = e
        print(f"\nWARNING: training stopped with {type(e).__name__}: {e}")
    finally:
        # Always persist VecNormalize + weights so runs are trade-ready even if learn() crashes
        vn_root, vn_best = _persist_trade_artifacts(model, train_env, paths)
        commit_modal_volumes(reason="training exit")
        print(f"\nTrade bundle: {paths.final_model.name} + end-of-run vec_normalize")
        print(f"  VecNormalize (final): {vn_root}")
        if vn_best is not None:
            print(f"  VecNormalize (best, paired with best_model.zip): {vn_best}")
        else:
            print("  WARNING: no best/vec_normalize.pkl — eval never improved NAV")

    if learn_error is not None:
        raise learn_error

    # Best eval-NAV checkpoint provenance for the manifest + training summary.
    best_eval_nav = (
        float(eval_callback.best_mean_nav)
        if np.isfinite(eval_callback.best_mean_nav)
        else None
    )
    best_eval_step = None
    if eval_callback._mean_ending_nav and eval_callback._nav_timesteps:
        i = int(np.argmax(np.asarray(eval_callback._mean_ending_nav)))
        if i < len(eval_callback._nav_timesteps):
            best_eval_step = int(eval_callback._nav_timesteps[i])
    early_stop_reason = getattr(eval_callback, "early_stop_reason", None)

    # Merge (never rebuild) the pre-training manifest: it carries the
    # chronological_holdout block that defines what OOS is for this run; losing it
    # would let a later backtest silently extend the holdout window.
    merge_manifest(
        paths.manifest_path,
        {
            "run_id": run_id,
            "config_path": str(cfg.path),
            "args": vars(args),
            "universe": universe_meta,
            "n_index": int(len(idx)),
            "n_train_bars": int(len(train_idx)),
            "n_eval_bars": int(len(eval_idx)),
            "data_cache_snapshot": str(paths.data_snapshot),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_status": "interrupted" if interrupted else "completed",
            "total_params": total_params,
            "trainable_params": trainable_params,
            "best_eval_nav": best_eval_nav,
            "best_eval_step": best_eval_step,
            "early_stop_reason": early_stop_reason,
            **provenance,
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
                "eval_nav_history": str(paths.eval_nav_history),
            },
        },
    )

    # Machine-readable training summary for the research registry / orchestrator.
    write_manifest(
        paths.run_meta_dir / "training_summary.json",
        {
            "run_id": run_id,
            "timesteps": int(args.timesteps),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "best_eval_nav": best_eval_nav,
            "best_eval_step": best_eval_step,
            "early_stop_reason": early_stop_reason,
            "n_train_bars": int(len(train_idx)),
            "n_eval_bars": int(len(eval_idx)),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            **provenance,
        },
    )

    print(f"\nSaved final model: {paths.final_model}")
    print(f"VecNormalize stats: {paths.models_dir / 'vec_normalize.pkl'}")
    print(
        f"Best model + vec (trade, matched pair): "
        f"{paths.best_model_dir}/best_model.zip + vec_normalize.pkl"
    )
    print(f"Best checkpoint dir: {paths.models_dir / 'checkpoints'}/")
    if not args.no_viz:
        print(f"Training plot: {paths.training_plot}")
        if args.show_viz:
            open_plot_file(paths.training_plot)


if __name__ == "__main__":
    main()
