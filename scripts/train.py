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
  - Domain randomization: Beta-centered fee_scale + obs_lag, bounds widen progressively after fee ramp
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
import subprocess
import sys
import json
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path


def _startup_log(msg: str) -> None:
    print(msg, flush=True)


_startup_log("[train] Starting (loading dependencies)...")

# Fast-fail on contradictory flags before the expensive torch import (also keeps this
# check testable without torch installed).
if any(a == "--resume" or a.startswith("--resume=") for a in sys.argv[1:]) and any(
    a == "--finetune" or a.startswith("--finetune=") for a in sys.argv[1:]
):
    raise SystemExit("Use only one of --resume or --finetune, not both.")

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
    reserve_chronological_validation_tail,
    build_continuous_walkforward_pack,
    build_multi_regime_walkforward_packs,
    save_cache,
    select_tradeable_columns,
    WalkforwardEnvPack,
    align_panel_to_timeline,
    train_test_split_alternating,
)
from rlbot.rl_config import (
    UNIVERSE_MAX_ASSETS,
    UNIVERSE_MIN_ASSETS,
    WorkerConfigInstaller,
    apply_deterministic_seeds,
    get_config,
    load_config,
    observation_dim_for_universe,
    set_config,
    slice_config_to_n_assets,
    validate_config_for_universe,
    write_config_snapshot,
)
from rlbot.training_progress import (
    BudgetProgressBarCallback,
    absolute_progress_done,
    absolute_progress_remaining,
    churn_scale_at_step,
    lr_schedule_with_floor_for_budget,
    resolve_learn_timesteps,
)
from rlbot.vecnorm_utils import sync_vecnormalize_stats
from rlbot.run_artifacts import (
    DEFAULT_DATA_CACHE,
    RunPaths,
    config_sha256,
    git_provenance,
    hardware_profile,
    new_run_id,
    persist_dirty_source_snapshot,
    read_run_manifest,
    resolve_data_cache,
    sha256_file,
    merge_manifest,
    utc_now_iso,
    write_manifest,
)
from rlbot.eval_selection import (
    EvalBenchmarkContext,
    aggregate_eval_portfolio_diagnostics,
    aggregate_multi_regime_scores,
    append_eval_diagnostics_jsonl,
    apply_max_dd_reject_penalty,
    apply_mean_cash_cap_penalty,
    apply_worst_regime_cash_penalty,
    blend_block_and_oos_aligned_scores,
    compute_robust_eval_score,
)
from rlbot.eval_schedule import eval_freq_vector_steps, should_run_scheduled_eval
from rlbot.checkpoint_selection import (
    STRESS_FEE_SCALE,
    STRESS_OBS_LAG,
    ConfirmationState,
    SelectionDiagnostics,
    blend_canonical_stress,
    post_gate_scores,
    trailing_aggregate,
    update_confirmation,
)
from rlbot.research.spec import CANONICAL_WINDOWS
from rlbot.trading_env import EpisodeEndNavRecorder, MultiAssetPortfolioEnv
from rlbot.reward_logging import RewardDecompAccumulator
from rlbot.modal_cloud import commit_modal_volumes
from rlbot.visualize import TrainingVizCallback, open_plot_file
from rlbot.curriculum_preflight import build_curriculum_preflight, format_preflight_text

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
    config_installer: WorkerConfigInstaller | None = None,
):
    """Return a callable that creates and wraps a single environment.

    ``config_installer`` MUST be passed for SubprocVecEnv use: workers are spawned
    with a fresh interpreter where ``get_config()`` would otherwise fall back to the
    default ``config/config.yaml``, silently discarding ``--config``/``--n-assets``
    overrides for everything the env reads at construction (reward, costs, cap, DR).
    """

    def _init():
        if config_installer is not None:
            config_installer()
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
        if domain_randomize:
            # SB3 resets envs in _setup_learn BEFORE the curriculum callback applies
            # its pinned pre-ramp bounds, so without this the very first episode per
            # worker samples fee/lag from the full config DR range. Pin at
            # construction; the curriculum widens from fee_ramp_end onward.
            from rlbot.rl_config import get_config as _gc

            _lag = _gc().environment.obs_lag_default
            env.set_randomization_bounds(1.0, 1.0, _lag, _lag)
        if record_episode_nav:
            return EpisodeEndNavRecorder(env)
        log_dir.mkdir(parents=True, exist_ok=True)
        return Monitor(env, filename=str(log_dir / monitor_stem))

    return _init


class EvalNavBestModelCallback(EvalCallback):
    """Run periodic eval; save ``best_model.zip`` on max **robust eval score**, not reward.

    ``score_mode="excess_nav"`` (legacy):
    score = mean(excess vs passive bench) - std_coef * std(excess) - dd_coef * p75(max_dd_nav)
    ``score_mode="excess_sharpe"``: the return signal is the annualized Sharpe of daily
    excess returns (segment-mean blended with pooled) and the drawdown penalty uses
    the unitless p75(max_dd_frac). See ``compute_robust_eval_score``.
    The return term blends segment-mean and stitched/pooled signals
    (``best_model_score_stitched_blend``).

    Passive benchmark for selection: ``training.best_model_benchmark`` (default equal-weight daily).
    Also logs stitched validation NAV (compounded eval blocks) and drawdown.
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
        *,
        panel_tickers: list[str],
        max_single_asset_weight: float,
        eval_diagnostics_path: Path,
        benchmark_ctx: EvalBenchmarkContext,
        score_std_coef: float = 0.75,
        score_dd_coef: float = 2.0,
        score_stitched_blend: float = 0.5,
        score_mode: str = "excess_nav",
        eval_freq_steps: int = 500_000,
        eval_freq_pre_gate_steps: int = 3_000_000,
        n_envs: int = 16,
        trailing_evals: int = 0,
        trailing_agg: str = "median",
        confirm_evals: int = 0,
        stress_suite: bool = False,
        stress_weight: float = 0.3,
        oos_aligned_env: VecNormalize | None = None,
        oos_aligned_benchmark_ctx: EvalBenchmarkContext | None = None,
        oos_aligned_benchmark_ctxs: list[EvalBenchmarkContext] | None = None,
        oos_aligned_weight: float = 0.0,
        multi_regime_eval_agg: str = "p25",
        eval_score_burn_in_bars: int = 0,
        best_model_max_dd_reject: float = 0.0,
        best_model_max_dd_reject_hard: bool = True,
        best_model_max_dd_reject_coef: float = 50.0,
        best_model_worst_regime_cash_coef: float = 0.0,
        best_model_worst_regime_cash_target: float = 0.0,
        best_model_mean_cash_coef: float = 0.0,
        best_model_mean_cash_cap: float = 1.0,
        reward_decomp_callback: "RewardDecompCallback | None" = None,
        **kwargs,
    ):
        self._best_model_dir = Path(best_model_save_path)
        self.nav_history_path = Path(nav_history_path)
        self.eval_diagnostics_path = Path(eval_diagnostics_path)
        self.benchmark_ctx = benchmark_ctx
        self.panel_tickers = list(panel_tickers)
        self.max_single_asset_weight = float(max_single_asset_weight)
        self._oos_aligned_env = oos_aligned_env
        self._oos_aligned_benchmark_ctx = oos_aligned_benchmark_ctx
        self._oos_aligned_benchmark_ctxs = list(oos_aligned_benchmark_ctxs or [])
        self.oos_aligned_weight = float(oos_aligned_weight)
        self.multi_regime_eval_agg = str(multi_regime_eval_agg).lower()
        self.eval_score_burn_in_bars = int(eval_score_burn_in_bars)
        self.best_model_max_dd_reject = float(best_model_max_dd_reject)
        self.best_model_max_dd_reject_hard = bool(best_model_max_dd_reject_hard)
        self.best_model_max_dd_reject_coef = float(best_model_max_dd_reject_coef)
        self.best_model_worst_regime_cash_coef = float(best_model_worst_regime_cash_coef)
        self.best_model_worst_regime_cash_target = float(best_model_worst_regime_cash_target)
        self.best_model_mean_cash_coef = float(best_model_mean_cash_coef)
        self.best_model_mean_cash_cap = float(best_model_mean_cash_cap)
        self._last_oos_aligned_max_dd = float("nan")
        self.score_std_coef = float(score_std_coef)
        self.score_dd_coef = float(score_dd_coef)
        self.score_stitched_blend = float(score_stitched_blend)
        self.score_mode = str(score_mode)
        self.trailing_evals = max(0, int(trailing_evals))
        self.trailing_agg = str(trailing_agg).lower()
        self.confirm_evals = max(0, int(confirm_evals))
        self.stress_suite = bool(stress_suite)
        self.stress_weight = float(stress_weight)
        self._confirm_state: ConfirmationState | None = None
        self._reward_decomp_callback = reward_decomp_callback
        self.last_selection_diagnostics: SelectionDiagnostics | None = None
        self.best_selection_score = -np.inf
        self.best_mean_nav = -np.inf
        self._nav_timesteps: list[int] = []
        self._mean_ending_nav: list[float] = []
        self._robust_scores: list[float] = []
        self._std_ending_nav: list[float] = []
        self._mean_max_dd_nav: list[float] = []
        self._mean_max_dd_frac: list[float] = []
        self._mean_excess_nav: list[float] = []
        self._stitched_agent_nav: list[float] = []
        self._stitched_excess_nav: list[float] = []
        self._stitched_max_dd_frac: list[float] = []
        self._best_eval_step: int | None = None
        self._eval_freq_steps_post = int(eval_freq_steps)
        self._eval_freq_pre_gate_steps = int(eval_freq_pre_gate_steps)
        self._n_envs = max(int(n_envs), 1)
        self._last_eval_n_calls = 0
        self._post_gate_eval_forced = False
        self._train_vec_env = train_vec_env
        self.patience = int(patience)
        self.curriculum_end_step = int(curriculum_end_step)
        self.best_model_min_step = int(best_model_min_step)
        self._post_gate_tracking_started = False
        self._evals_since_best = 0
        self.early_stop_reason: str | None = None
        self._load_nav_history()
        vec_post = eval_freq_vector_steps(self._eval_freq_steps_post, self._n_envs)
        super().__init__(eval_env, best_model_save_path=None, eval_freq=vec_post, **kwargs)

    def _should_run_eval(self) -> bool:
        return should_run_scheduled_eval(
            n_calls=int(self.n_calls),
            last_eval_n_calls=int(self._last_eval_n_calls),
            num_timesteps=int(self.num_timesteps),
            post_gate_global_freq=self._eval_freq_steps_post,
            pre_gate_global_freq=self._eval_freq_pre_gate_steps,
            best_model_min_step=int(self.best_model_min_step),
            n_envs=self._n_envs,
            post_gate_eval_forced=self._post_gate_eval_forced,
        )

    def _mark_eval_ran(self) -> None:
        self._last_eval_n_calls = int(self.n_calls)
        if (
            self.best_model_min_step > 0
            and self.num_timesteps >= self.best_model_min_step
        ):
            self._post_gate_eval_forced = True

    def _best_model_gate_open(self) -> bool:
        return self.best_model_min_step <= 0 or self.num_timesteps >= self.best_model_min_step

    def _post_gate_best_score(self) -> float:
        if self.best_model_min_step <= 0:
            return self.best_selection_score
        post = [
            s
            for t, s in zip(self._nav_timesteps, self._robust_scores)
            if t >= self.best_model_min_step
        ]
        return float(max(post)) if post else -np.inf

    def _post_gate_score_series(self) -> list[float]:
        """Instantaneous (or stress-blended) scores eligible for trailing selection."""
        return post_gate_scores(
            self._nav_timesteps,
            self._robust_scores,
            gate_step=int(self.best_model_min_step),
        )

    def _trailing_selection_score(self, scores: list[float] | None = None) -> float:
        series = self._post_gate_score_series() if scores is None else scores
        window = max(1, self.trailing_evals) if self.trailing_evals > 0 else 1
        agg = self.trailing_agg if self.trailing_agg in ("median", "mean") else "median"
        return trailing_aggregate(series, window=window, agg=agg)

    def _restore_best_selection_from_history(
        self,
        *,
        saved_best_selection_score: float | None = None,
        saved_best_mean_nav: float | None = None,
    ) -> None:
        """Resume the selection threshold from the checkpoint that was actually saved.

        Prefer the persisted ``best_eval_step`` / ``best_selection_score`` (trailing +
        confirmation) over argmax of instantaneous post-gate scores.
        """
        if not self._robust_scores:
            return
        n = min(len(self._robust_scores), len(self._nav_timesteps))
        if n <= 0:
            return
        scores = np.asarray(self._robust_scores[:n], dtype=np.float64)
        steps = np.asarray(self._nav_timesteps[:n], dtype=np.int64)
        if self.best_model_min_step > 0:
            mask = steps >= self.best_model_min_step
        else:
            mask = np.ones(n, dtype=bool)
        if not np.any(mask):
            self.best_selection_score = -np.inf
            self.best_mean_nav = -np.inf
            return

        idx: int | None = None
        if self._best_eval_step is not None:
            matches = [
                i for i in range(n) if int(steps[i]) == int(self._best_eval_step) and mask[i]
            ]
            if matches:
                idx = int(matches[-1])

        if idx is None:
            # Legacy histories without a selected-step stamp: fall back to argmax.
            masked_scores = scores[mask]
            j_masked = int(np.argmax(masked_scores))
            idx = int(np.nonzero(mask)[0][j_masked])
            self._best_eval_step = int(steps[idx])

        if (
            saved_best_selection_score is not None
            and np.isfinite(saved_best_selection_score)
        ):
            self.best_selection_score = float(saved_best_selection_score)
        else:
            # Reconstruct trailing selection through the selected index (post-gate only).
            through = post_gate_scores(
                steps[: idx + 1],
                scores[: idx + 1],
                gate_step=int(self.best_model_min_step),
            )
            if self.trailing_evals > 1:
                self.best_selection_score = self._trailing_selection_score(through)
            else:
                self.best_selection_score = float(scores[idx])

        if saved_best_mean_nav is not None and np.isfinite(saved_best_mean_nav):
            self.best_mean_nav = float(saved_best_mean_nav)
        elif idx < len(self._mean_ending_nav):
            self.best_mean_nav = float(self._mean_ending_nav[idx])

    def _load_nav_history(self) -> None:
        if not self.nav_history_path.is_file():
            return
        try:
            z = np.load(self.nav_history_path, allow_pickle=False)
            self._nav_timesteps = list(np.asarray(z["timesteps"], dtype=np.int64))
            self._mean_ending_nav = list(np.asarray(z["mean_ending_nav"], dtype=np.float64))
            if "robust_scores" in z:
                self._robust_scores = list(np.asarray(z["robust_scores"], dtype=np.float64))
            elif self._mean_ending_nav:
                self._robust_scores = list(self._mean_ending_nav)
            if "std_ending_nav" in z:
                self._std_ending_nav = list(np.asarray(z["std_ending_nav"], dtype=np.float64))
            if "mean_max_drawdown_nav" in z:
                self._mean_max_dd_nav = list(np.asarray(z["mean_max_drawdown_nav"], dtype=np.float64))
            if "mean_max_drawdown_frac" in z:
                self._mean_max_dd_frac = list(np.asarray(z["mean_max_drawdown_frac"], dtype=np.float64))
            for key, attr in (
                ("mean_excess_nav", "_mean_excess_nav"),
                ("stitched_agent_nav", "_stitched_agent_nav"),
                ("stitched_excess_nav", "_stitched_excess_nav"),
                ("stitched_max_drawdown_frac", "_stitched_max_dd_frac"),
            ):
                if key in z:
                    setattr(self, attr, list(np.asarray(z[key], dtype=np.float64)))
            bes = z.get("best_eval_step")
            if bes is not None:
                self._best_eval_step = int(np.asarray(bes).reshape(-1)[0])
            saved_sel: float | None = None
            if "best_selection_score" in z:
                saved_sel = float(np.asarray(z["best_selection_score"]).reshape(-1)[0])
            saved_nav: float | None = None
            if "best_mean_nav" in z:
                saved_nav = float(np.asarray(z["best_mean_nav"]).reshape(-1)[0])
            if self._nav_timesteps:
                if self.best_model_min_step > 0:
                    self._post_gate_eval_forced = any(
                        int(t) >= self.best_model_min_step for t in self._nav_timesteps
                    )
                self._last_eval_n_calls = int(self._nav_timesteps[-1] // self._n_envs)
            if self._robust_scores:
                if self.best_model_min_step > 0:
                    self._post_gate_tracking_started = any(
                        t >= self.best_model_min_step for t in self._nav_timesteps
                    )
                self._restore_best_selection_from_history(
                    saved_best_selection_score=saved_sel,
                    saved_best_mean_nav=saved_nav,
                )
        except (OSError, ValueError, KeyError):
            pass

    def _save_nav_history(self) -> None:
        self.nav_history_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "timesteps": np.asarray(self._nav_timesteps, dtype=np.int64),
            "mean_ending_nav": np.asarray(self._mean_ending_nav, dtype=np.float64),
            "robust_scores": np.asarray(self._robust_scores, dtype=np.float64),
            "std_ending_nav": np.asarray(self._std_ending_nav, dtype=np.float64),
            "mean_max_drawdown_nav": np.asarray(self._mean_max_dd_nav, dtype=np.float64),
            "mean_max_drawdown_frac": np.asarray(self._mean_max_dd_frac, dtype=np.float64),
            "mean_excess_nav": np.asarray(self._mean_excess_nav, dtype=np.float64),
            "stitched_agent_nav": np.asarray(self._stitched_agent_nav, dtype=np.float64),
            "stitched_excess_nav": np.asarray(self._stitched_excess_nav, dtype=np.float64),
            "stitched_max_drawdown_frac": np.asarray(self._stitched_max_dd_frac, dtype=np.float64),
        }
        if self.best_model_min_step > 0:
            payload["best_model_min_step"] = np.asarray(
                [self.best_model_min_step], dtype=np.int64
            )
        if self._best_eval_step is not None:
            payload["best_eval_step"] = np.asarray([self._best_eval_step], dtype=np.int64)
        if np.isfinite(self.best_selection_score):
            payload["best_selection_score"] = np.asarray(
                [self.best_selection_score], dtype=np.float64
            )
        if np.isfinite(self.best_mean_nav):
            payload["best_mean_nav"] = np.asarray([self.best_mean_nav], dtype=np.float64)
        np.savez_compressed(self.nav_history_path, **payload)

    def _collect_eval_episodes(self) -> list[dict]:
        episodes: list[dict] = []
        for batch in self.eval_env.env_method("pop_eval_episodes"):
            episodes.extend(batch)
        return episodes

    def _log_portfolio_diagnostics(self, episodes: list[dict], metrics: dict) -> None:
        diag = aggregate_eval_portfolio_diagnostics(
            episodes,
            tickers=self.panel_tickers,
            max_single_asset_weight=self.max_single_asset_weight,
            benchmark_ctx=self.benchmark_ctx,
            burn_in_bars=self.eval_score_burn_in_bars,
        )
        panel = diag.get("portfolio") or {}
        if panel:
            self.logger.record("eval/mean_cash_frac", panel.get("mean_cash_frac", 0.0))
            self.logger.record("eval/mean_gross_exposure", panel.get("mean_gross_exposure", 0.0))
            self.logger.record(
                "eval/mean_effective_n_assets", panel.get("mean_effective_n_assets", 0.0)
            )
            self.logger.record(
                "eval/mean_top3_concentration", panel.get("mean_top3_concentration", 0.0)
            )
            self.logger.record("eval/mean_turnover", panel.get("mean_turnover", 0.0))
            self.logger.record("eval/cap_hit_fraction", panel.get("cap_hit_fraction", 0.0))
        append_eval_diagnostics_jsonl(
            self.eval_diagnostics_path,
            {
                "timestep": int(self.num_timesteps),
                "score": metrics,
                "portfolio": panel,
                "segments": diag.get("segments", []),
                "stitched": diag.get("stitched", {}),
            },
        )

    def _on_step(self) -> bool:
        run_eval = self._should_run_eval()
        if run_eval:
            if self._train_vec_env is not None:
                sync_vecnormalize_stats(self._train_vec_env, self.eval_env)
            self.eval_env.env_method("pop_eval_episodes")

        old_freq = self.eval_freq
        self.eval_freq = 1 if run_eval else max(int(self.n_calls) + 1, 2)
        continue_training = super()._on_step()
        self.eval_freq = old_freq

        if run_eval:
            self._mark_eval_ran()
            episodes = self._collect_eval_episodes()
            if episodes:
                metrics = compute_robust_eval_score(
                    episodes,
                    std_coef=self.score_std_coef,
                    dd_coef=self.score_dd_coef,
                    stitched_blend=self.score_stitched_blend,
                    benchmark_ctx=self.benchmark_ctx,
                    score_mode=self.score_mode,
                    burn_in_bars=self.eval_score_burn_in_bars,
                )
                score = float(metrics["score"])
                mean_nav = float(metrics["mean_ending_nav"])
                self._nav_timesteps.append(int(self.num_timesteps))
                self._mean_ending_nav.append(mean_nav)
                self._robust_scores.append(score)
                self._std_ending_nav.append(float(metrics["std_ending_nav"]))
                self._mean_max_dd_nav.append(float(metrics["p75_max_drawdown_nav"]))
                self._mean_max_dd_frac.append(float(metrics["p75_max_drawdown_frac"]))
                self._mean_excess_nav.append(float(metrics["mean_excess_nav"]))
                self._stitched_agent_nav.append(float(metrics.get("stitched_agent_nav", mean_nav)))
                self._stitched_excess_nav.append(float(metrics.get("stitched_excess_nav", 0.0)))
                self._stitched_max_dd_frac.append(
                    float(metrics.get("stitched_max_drawdown_frac", metrics["p75_max_drawdown_frac"]))
                )
                self._save_nav_history()
                self.logger.record("eval/mean_ending_nav", mean_nav)
                self.logger.record("eval/mean_excess_nav", metrics["mean_excess_nav"])
                self.logger.record("eval/std_excess_nav", metrics["std_excess_nav"])
                self.logger.record("eval/robust_score", score)
                self.logger.record(
                    "eval/p75_max_drawdown_nav", metrics["p75_max_drawdown_nav"]
                )
                self.logger.record(
                    "eval/p75_max_drawdown_pct",
                    100.0 * float(metrics["p75_max_drawdown_frac"]),
                )
                if "stitched_agent_nav" in metrics:
                    self.logger.record("eval/stitched_agent_nav", metrics["stitched_agent_nav"])
                    self.logger.record("eval/stitched_excess_nav", metrics["stitched_excess_nav"])
                    self.logger.record(
                        "eval/stitched_max_drawdown_pct",
                        100.0 * float(metrics["stitched_max_drawdown_frac"]),
                    )
                self._log_portfolio_diagnostics(episodes, metrics)
                gate_open = self._best_model_gate_open()
                self.logger.record("eval/best_model_gate_open", float(gate_open))
                raw_score = score
                oos_aligned_score = None
                if self.oos_aligned_weight > 0.0 and self._oos_aligned_env is not None:
                    oos_aligned_score = self._run_oos_aligned_eval()
                    if oos_aligned_score is not None:
                        score = blend_block_and_oos_aligned_scores(
                            raw_score,
                            oos_aligned_score,
                            weight=self.oos_aligned_weight,
                        )
                        self.logger.record("eval/oos_aligned_score", float(oos_aligned_score))
                        self.logger.record("eval/selection_score_oos_blended", float(score))
                stress_score = None
                if gate_open and self.stress_suite:
                    stress_score = self._run_stress_eval()
                    if stress_score is not None:
                        score = blend_canonical_stress(
                            score, stress_score, stress_weight=self.stress_weight
                        )
                        self.logger.record("eval/stress_score", float(stress_score))
                        self.logger.record("eval/selection_score_blended", float(score))
                        # Replace the just-appended raw block score with the final selection score.
                        self._robust_scores[-1] = float(score)
                elif oos_aligned_score is not None:
                    self._robust_scores[-1] = float(score)
                post_scores = self._post_gate_score_series()
                trailing = self._trailing_selection_score(post_scores)
                selection_score = trailing if self.trailing_evals > 1 else score
                self.logger.record("eval/trailing_selection_score", float(selection_score))
                if gate_open:
                    if self.best_model_min_step > 0 and not self._post_gate_tracking_started:
                        self._post_gate_tracking_started = True
                        self.best_selection_score = -np.inf
                        self.best_mean_nav = -np.inf
                        self._confirm_state = None
                    new_state, replace = update_confirmation(
                        self._confirm_state,
                        score=selection_score,
                        best_score=self.best_selection_score,
                        confirms_needed=self.confirm_evals,
                    )
                    self._confirm_state = new_state
                    confirms_seen = 0 if new_state is None else int(new_state.confirms_seen)
                    self.last_selection_diagnostics = SelectionDiagnostics(
                        raw_score=float(raw_score),
                        trailing_score=float(trailing),
                        selection_score=float(selection_score),
                        stress_score=None if stress_score is None else float(stress_score),
                        trailing_window=int(self.trailing_evals),
                        trailing_agg=str(self.trailing_agg),
                        confirms_needed=int(self.confirm_evals),
                        confirms_seen=confirms_seen,
                        gate_open=True,
                        replaced_best=bool(replace),
                    )
                    append_eval_diagnostics_jsonl(
                        self.eval_diagnostics_path.with_name("selection_diagnostics.jsonl"),
                        {
                            "timestep": int(self.num_timesteps),
                            **self.last_selection_diagnostics.to_dict(),
                        },
                    )
                    if replace:
                        self.best_selection_score = selection_score
                        self.best_mean_nav = mean_nav
                        self._best_eval_step = int(self.num_timesteps)
                        self._evals_since_best = 0
                        self._confirm_state = None
                        if self.verbose >= 1:
                            print(
                                f"New best robust eval score: {selection_score:,.3f} "
                                f"(raw {raw_score:,.3f}, "
                                f"mean excess {metrics['mean_excess_nav']:,.0f}, "
                                f"std {metrics['std_excess_nav']:,.0f}, "
                                f"p75 max DD {100.0 * metrics['p75_max_drawdown_frac']:.1f}%)"
                            )
                        self._best_model_dir.mkdir(parents=True, exist_ok=True)
                        self.model.save(str(self._best_model_dir / "best_model"))
                        if self._train_vec_env is not None:
                            self._train_vec_env.save(
                                str(self._best_model_dir / "vec_normalize.pkl")
                            )
                        self._save_nav_history()
                        if self._reward_decomp_callback is not None:
                            self._reward_decomp_callback.snapshot_best(
                                int(self.num_timesteps)
                            )
                    elif self.patience > 0 and self.num_timesteps >= self.curriculum_end_step:
                        self._evals_since_best += 1
                        self.logger.record("eval/evals_since_best", self._evals_since_best)
                        if self._evals_since_best >= self.patience:
                            self.early_stop_reason = (
                                f"no new best robust eval score for {self.patience} evals after "
                                f"curriculum (step {self.num_timesteps})"
                            )
                            print(f"[train] early stop: {self.early_stop_reason}")
                            return False

        return continue_training

    def _run_oos_aligned_eval(self) -> float | None:
        """Continuous validation matching OOS backtest structure.

        Single-pack mode (726): one chronological episode.
        Multi-regime mode (727+): one continuous episode per regime slice, then
        aggregate with ``multi_regime_eval_agg`` (default p25).
        """
        if self._oos_aligned_env is None:
            return None
        multi = len(self._oos_aligned_benchmark_ctxs) > 1
        if not multi and self._oos_aligned_benchmark_ctx is None:
            return None
        try:
            if self._train_vec_env is not None:
                sync_vecnormalize_stats(self._train_vec_env, self._oos_aligned_env)
            self._oos_aligned_env.env_method("pop_eval_episodes")
            from stable_baselines3.common.evaluation import evaluate_policy

            n_eps = max(1, len(self._oos_aligned_benchmark_ctxs)) if multi else 1
            evaluate_policy(
                self.model,
                self._oos_aligned_env,
                n_eval_episodes=n_eps,
                deterministic=self.deterministic,
                render=False,
                callback=None,
                warn=False,
            )
            batches = self._oos_aligned_env.env_method("pop_eval_episodes")
            episodes: list[dict] = []
            for batch in batches:
                episodes.extend(batch)
            if not episodes:
                return None

            if multi:
                ctxs = self._oos_aligned_benchmark_ctxs
                # One episode per env/slice (order matches SubprocVecEnv worker order).
                n = min(len(episodes), len(ctxs))
                per_scores: list[float] = []
                cash_fracs: list[float] = []
                per_max_dds: list[float] = []
                for i in range(n):
                    metrics_i = compute_robust_eval_score(
                        [episodes[i]],
                        std_coef=self.score_std_coef,
                        dd_coef=self.score_dd_coef,
                        stitched_blend=1.0,
                        benchmark_ctx=ctxs[i],
                        score_mode=self.score_mode,
                        burn_in_bars=self.eval_score_burn_in_bars,
                    )
                    per_scores.append(float(metrics_i["score"]))
                    per_max_dds.append(float(metrics_i.get("max_max_drawdown_frac", 0.0)))
                    diag_i = aggregate_eval_portfolio_diagnostics(
                        [episodes[i]],
                        tickers=self.panel_tickers,
                        max_single_asset_weight=self.max_single_asset_weight,
                        benchmark_ctx=ctxs[i],
                        burn_in_bars=self.eval_score_burn_in_bars,
                    )
                    panel_i = diag_i.get("portfolio") or {}
                    if panel_i:
                        cash_fracs.append(float(panel_i.get("mean_cash_frac", 0.0)))
                    append_eval_diagnostics_jsonl(
                        self.eval_diagnostics_path,
                        {
                            "timestep": int(self.num_timesteps),
                            "kind": "oos_aligned",
                            "regime_index": i,
                            "score": metrics_i,
                            "portfolio": panel_i,
                            "segments": diag_i.get("segments", []),
                        },
                    )
                agg_score = aggregate_multi_regime_scores(
                    per_scores, agg=self.multi_regime_eval_agg
                )
                max_dd = float(max(per_max_dds)) if per_max_dds else 0.0
                self._last_oos_aligned_max_dd = max_dd
                agg_score, worst_cash, cash_pen = apply_worst_regime_cash_penalty(
                    agg_score,
                    per_max_dds,
                    cash_fracs,
                    coef=self.best_model_worst_regime_cash_coef,
                    target=self.best_model_worst_regime_cash_target,
                )
                agg_score, mean_cash, mean_cash_pen = apply_mean_cash_cap_penalty(
                    agg_score,
                    cash_fracs,
                    coef=self.best_model_mean_cash_coef,
                    cap=self.best_model_mean_cash_cap,
                )
                agg_score, dd_pen, over_thr = apply_max_dd_reject_penalty(
                    agg_score,
                    max_dd,
                    threshold=self.best_model_max_dd_reject,
                    coef=self.best_model_max_dd_reject_coef,
                    hard=self.best_model_max_dd_reject_hard,
                )
                rejected = bool(over_thr and self.best_model_max_dd_reject_hard)
                self.logger.record("eval/oos_aligned_score", float(agg_score) if np.isfinite(agg_score) else -1e9)
                self.logger.record("eval/oos_aligned_max_dd_frac", max_dd)
                self.logger.record("eval/oos_aligned_dd_rejected", float(over_thr))
                self.logger.record(
                    "eval/oos_aligned_score_min", float(min(per_scores)) if per_scores else 0.0
                )
                self.logger.record(
                    "eval/oos_aligned_score_median",
                    float(np.median(per_scores)) if per_scores else 0.0,
                )
                if cash_fracs:
                    self.logger.record(
                        "eval/oos_aligned_mean_cash_frac", float(np.mean(cash_fracs))
                    )
                if np.isfinite(worst_cash):
                    self.logger.record("eval/oos_aligned_worst_regime_cash", worst_cash)
                if cash_pen > 0.0:
                    self.logger.record("eval/oos_aligned_cash_penalty", cash_pen)
                if mean_cash_pen > 0.0:
                    self.logger.record("eval/oos_aligned_mean_cash_penalty", mean_cash_pen)
                if over_thr and np.isfinite(dd_pen):
                    self.logger.record("eval/oos_aligned_dd_penalty", float(dd_pen))
                append_eval_diagnostics_jsonl(
                    self.eval_diagnostics_path,
                    {
                        "timestep": int(self.num_timesteps),
                        "kind": "oos_aligned_aggregate",
                        "agg": self.multi_regime_eval_agg,
                        "per_regime_scores": per_scores,
                        "per_regime_max_dd": per_max_dds,
                        "per_regime_cash": cash_fracs,
                        "max_dd_frac": max_dd,
                        "worst_regime_cash": (
                            float(worst_cash) if np.isfinite(worst_cash) else None
                        ),
                        "cash_penalty": float(cash_pen),
                        "mean_cash": (
                            float(mean_cash) if np.isfinite(mean_cash) else None
                        ),
                        "mean_cash_penalty": float(mean_cash_pen),
                        "dd_penalty": (
                            float(dd_pen) if np.isfinite(dd_pen) else None
                        ),
                        "dd_rejected": bool(over_thr),
                        "dd_reject_hard": bool(self.best_model_max_dd_reject_hard),
                        "dd_reject_threshold": float(self.best_model_max_dd_reject),
                        "score": float(agg_score) if np.isfinite(agg_score) else None,
                    },
                )
                return float(agg_score)

            metrics = compute_robust_eval_score(
                episodes,
                std_coef=self.score_std_coef,
                dd_coef=self.score_dd_coef,
                stitched_blend=1.0,  # single continuous path
                benchmark_ctx=self._oos_aligned_benchmark_ctx,
                score_mode=self.score_mode,
                burn_in_bars=self.eval_score_burn_in_bars,
            )
            max_dd = float(metrics.get("max_max_drawdown_frac", 0.0))
            self._last_oos_aligned_max_dd = max_dd
            score_out, dd_pen, over_thr = apply_max_dd_reject_penalty(
                float(metrics["score"]),
                max_dd,
                threshold=self.best_model_max_dd_reject,
                coef=self.best_model_max_dd_reject_coef,
                hard=self.best_model_max_dd_reject_hard,
            )
            # Log continuous-path portfolio diagnostics alongside block diagnostics.
            diag = aggregate_eval_portfolio_diagnostics(
                episodes,
                tickers=self.panel_tickers,
                max_single_asset_weight=self.max_single_asset_weight,
                benchmark_ctx=self._oos_aligned_benchmark_ctx,
                burn_in_bars=self.eval_score_burn_in_bars,
            )
            panel = diag.get("portfolio") or {}
            if panel:
                self.logger.record(
                    "eval/oos_aligned_mean_cash_frac", panel.get("mean_cash_frac", 0.0)
                )
                self.logger.record(
                    "eval/oos_aligned_mean_gross_exposure",
                    panel.get("mean_gross_exposure", 0.0),
                )
            self.logger.record("eval/oos_aligned_max_dd_frac", max_dd)
            self.logger.record("eval/oos_aligned_dd_rejected", float(over_thr))
            if over_thr and np.isfinite(dd_pen):
                self.logger.record("eval/oos_aligned_dd_penalty", float(dd_pen))
            append_eval_diagnostics_jsonl(
                self.eval_diagnostics_path,
                {
                    "timestep": int(self.num_timesteps),
                    "kind": "oos_aligned",
                    "score": metrics,
                    "portfolio": panel,
                    "segments": diag.get("segments", []),
                    "dd_rejected": bool(over_thr),
                    "dd_reject_hard": bool(self.best_model_max_dd_reject_hard),
                    "dd_reject_threshold": float(self.best_model_max_dd_reject),
                    "dd_penalty": float(dd_pen) if np.isfinite(dd_pen) else None,
                },
            )
            return float(score_out)
        except Exception as exc:  # noqa: BLE001
            print(f"[train] oos-aligned eval failed: {exc}")
            return None

    def _run_stress_eval(self) -> float | None:
        """Fixed high-fee / high-lag eval; returns robust score or None on failure.

        Agent *and* passive benchmark are priced at ``STRESS_FEE_SCALE`` so the
        excess / Sharpe signal stays like-for-like (not agent-only fee stress).

        When OOS-aligned continuous selection is active (weight ≥ 0.5), stress the
        continuous env so the adverse-fee signal matches the primary selection path.
        Multi-regime: stress every slice and aggregate with the same agg as selection.
        """
        use_oos = (
            self.oos_aligned_weight >= 0.5
            and self._oos_aligned_env is not None
            and (
                self._oos_aligned_benchmark_ctx is not None
                or len(self._oos_aligned_benchmark_ctxs) > 0
            )
        )
        env = self._oos_aligned_env if use_oos else self.eval_env
        multi = use_oos and len(self._oos_aligned_benchmark_ctxs) > 1
        bench_ctx = self._oos_aligned_benchmark_ctx if use_oos else self.benchmark_ctx
        n_eps = (
            max(1, len(self._oos_aligned_benchmark_ctxs))
            if multi
            else (1 if use_oos else self.n_eval_episodes)
        )
        try:
            if self._train_vec_env is not None:
                sync_vecnormalize_stats(self._train_vec_env, env)
            env.env_method(
                "set_eval_controls",
                fee_scale=float(STRESS_FEE_SCALE),
                obs_lag=int(STRESS_OBS_LAG),
            )
            env.env_method("pop_eval_episodes")
            from stable_baselines3.common.evaluation import evaluate_policy

            evaluate_policy(
                self.model,
                env,
                n_eval_episodes=n_eps,
                deterministic=self.deterministic,
                render=False,
                callback=self._log_success_callback if not use_oos else None,
                warn=False,
            )
            batches = env.env_method("pop_eval_episodes")
            episodes: list[dict] = []
            for batch in batches:
                episodes.extend(batch)
            env.env_method("set_eval_controls", fee_scale=None, obs_lag=None)
            if not episodes:
                return None
            if multi:
                ctxs = [
                    EvalBenchmarkContext(
                        ohlcv=c.ohlcv,
                        idx=c.idx,
                        tickers=list(c.tickers),
                        asset_live=c.asset_live,
                        mode=str(c.mode),
                        fee_scale=float(STRESS_FEE_SCALE),
                    )
                    for c in self._oos_aligned_benchmark_ctxs
                ]
                n = min(len(episodes), len(ctxs))
                per_scores = []
                for i in range(n):
                    m = compute_robust_eval_score(
                        [episodes[i]],
                        std_coef=self.score_std_coef,
                        dd_coef=self.score_dd_coef,
                        stitched_blend=1.0,
                        benchmark_ctx=ctxs[i],
                        score_mode=self.score_mode,
                        burn_in_bars=self.eval_score_burn_in_bars,
                    )
                    per_scores.append(float(m["score"]))
                return aggregate_multi_regime_scores(
                    per_scores, agg=self.multi_regime_eval_agg
                )
            stress_ctx = bench_ctx
            if use_oos and bench_ctx is not None:
                stress_ctx = EvalBenchmarkContext(
                    ohlcv=bench_ctx.ohlcv,
                    idx=bench_ctx.idx,
                    tickers=list(bench_ctx.tickers),
                    asset_live=bench_ctx.asset_live,
                    mode=str(bench_ctx.mode),
                    fee_scale=float(STRESS_FEE_SCALE),
                )
            metrics = compute_robust_eval_score(
                episodes,
                std_coef=self.score_std_coef,
                dd_coef=self.score_dd_coef,
                stitched_blend=1.0 if use_oos else self.score_stitched_blend,
                benchmark_ctx=stress_ctx,
                score_mode=self.score_mode,
                burn_in_bars=self.eval_score_burn_in_bars,
            )
            return float(metrics["score"])
        except Exception as exc:  # noqa: BLE001
            print(f"[train] stress eval failed: {exc}")
            try:
                env.env_method("set_eval_controls", fee_scale=None, obs_lag=None)
            except Exception:  # noqa: BLE001
                pass
            return None


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
        decay_start_fraction: float = 0.585,
        learn_budget: int = 50_000_000,
        warmup_improvements: int = 3,
        eval_log_dir: str = "",
        eval_check_freq: int = 50_000,
        eval_nav_callback: "EvalNavBestModelCallback | None" = None,
        lr_schedule=None,
    ):
        super().__init__()
        self.explore_ent = explore_ent
        self.final_ent = final_ent
        self.early_floor = early_floor
        self.early_floor_steps = early_floor_steps
        self.min_explore_steps = int(min_explore_steps)
        self.decay_start_fraction = float(np.clip(decay_start_fraction, 0.0, 0.99))
        self.learn_budget = max(1, int(learn_budget))
        self.warmup_improvements = warmup_improvements
        self.eval_log_dir = eval_log_dir
        self.eval_check_freq = max(1, int(eval_check_freq))
        self._eval_nav_callback = eval_nav_callback
        self._lr_schedule = lr_schedule
        self._last_best: float = -float("inf")
        self._improvements: int = 0

    def _sync_eval_improvements(self) -> None:
        """Update improvement count at eval cadence only (no per-step disk I/O)."""
        if self._eval_nav_callback is not None:
            current_best = float(self._eval_nav_callback.best_selection_score)
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

    def _sync_absolute_progress(self) -> None:
        """Keep LR + entropy on the global budget curve (resume-safe)."""
        t = int(self.num_timesteps)
        remaining = absolute_progress_remaining(t, self.learn_budget)
        self.model._current_progress_remaining = remaining
        if self._lr_schedule is not None and hasattr(self._lr_schedule, "sync_num_timesteps"):
            self._lr_schedule.sync_num_timesteps(t)

    def _on_training_start(self) -> None:
        self._sync_absolute_progress()

    def _on_step(self) -> bool:
        import math

        self._sync_absolute_progress()

        if self.n_calls % self.eval_check_freq == 0:
            self._sync_eval_improvements()

        progress_done = absolute_progress_done(int(self.num_timesteps), self.learn_budget)

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

    At ≤ budget_short (50M default): fraction-of-run schedule to avoid a mid-run cliff.
    Between budget_short and budget_long: interpolate toward long-run anchors; ≥120M uses fixed long milestones.
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
    - Churn + turnover: ``churn_scale = 0`` while frictionless; ramps ``churn_ramp_floor`` → ``1``
      over the fee-ramp window only after fees turn on (train + eval).
    - Eval envs mirror the fee/churn schedule (no domain randomization).
    """

    def __init__(
        self,
        vec_env: VecNormalize,
        learn_budget: int,
        update_freq: int = 50_000,
        eval_vec_env: VecNormalize | None = None,
        oos_aligned_vec_env: VecNormalize | None = None,
    ):
        super().__init__()
        self.vec_env = vec_env
        self.eval_vec_env = eval_vec_env
        self.oos_aligned_vec_env = oos_aligned_vec_env
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
        """Churn penalty scale aligned with fee curriculum (0 while frictionless)."""
        return churn_scale_at_step(
            t,
            fee_free_until=self.fee_free_until,
            fee_ramp_end=self.fee_ramp_end,
            churn_ramp_floor=float(get_config().curriculum.churn_ramp_floor),
        )

    def _dr_bounds(self, t: int) -> tuple[float, float, int, int]:
        """Progressive fee/lag bounds after fee curriculum releases DR."""
        dr_min = get_config().environment.domain_randomize_fee_dr_min
        dr_max = get_config().environment.domain_randomize_fee_dr_max
        env_cfg = get_config().environment
        lag_lo, lag_hi = env_cfg.min_obs_lag, env_cfg.max_obs_lag
        if t < self.fee_ramp_end:
            # No DR before the fee curriculum releases: fee is pinned by the override
            # anyway, and obs_lag stays at the deterministic default. Returning the
            # full-wide bounds here (the old behavior) randomized lag from step 0 and
            # then snapped it to a 1-bar cliff exactly at fee_ramp_end — the same step
            # the best-model gate opens.
            lag_fixed = max(lag_lo, min(env_cfg.obs_lag_default, lag_hi))
            return 1.0, 1.0, lag_fixed, lag_fixed
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
            if self.oos_aligned_vec_env is not None:
                self.oos_aligned_vec_env.env_method("set_curriculum_state", fee, churn)
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
    """Aggregate ``info['rew_decomp/*']`` → TensorBoard + JSONL history + windowed JSON.

    Makes the reward balance observable (the review's asymmetry finding: inactivity
    can dwarf participation/churn). Logs per-term means and share-of-absolute-reward over
    the window since the last log, then resets, so the rolling JSON reflects recent
    (steady-state) behavior. JSONL keeps the full time series; a best-checkpoint
    snapshot is written when ``models/best/`` updates.
    """

    def __init__(
        self,
        json_path,
        log_freq: int = 50_000,
        *,
        clip_reward: float = 10.0,
    ):
        super().__init__()
        self.json_path = Path(json_path)
        self.jsonl_path = self.json_path.with_suffix(".jsonl")
        self.best_json_path = self.json_path.with_name("reward_decomp_best.json")
        self.log_freq = max(int(log_freq), 1)
        self.clip_reward = float(clip_reward)
        self._acc = RewardDecompAccumulator()
        self._n_reward = 0
        self._n_clipped = 0

    def _append_jsonl(self, payload: dict) -> None:
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")

    def snapshot_best(self, timesteps: int) -> None:
        """Persist the current window summary as the best-checkpoint decomposition."""
        if self._acc.count <= 0:
            return
        s = self._acc.summary()
        write_manifest(
            self.best_json_path,
            {"timesteps": int(timesteps), "at_best_checkpoint": True, **s},
        )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos:
            self._acc.update(infos)
        rewards = self.locals.get("rewards")
        if rewards is not None:
            arr = np.asarray(rewards, dtype=np.float64).reshape(-1)
            self._n_reward += int(arr.size)
            self._n_clipped += int(np.sum(np.abs(arr) >= self.clip_reward - 1e-9))
        if self.n_calls % self.log_freq == 0 and self._acc.count > 0:
            s = self._acc.summary()
            clip_rate = (
                float(self._n_clipped) / float(self._n_reward) if self._n_reward else 0.0
            )
            s["vecnormalize_reward_clip_rate"] = clip_rate
            for term, val in s["mean"].items():
                self.logger.record(f"rew_decomp/mean/{term}", val)
            for term, val in s["abs_share"].items():
                self.logger.record(f"rew_decomp/abs_share/{term}", val)
            for k, val in (s.get("extras") or {}).items():
                self.logger.record(f"rew_decomp/extra/{k}", val)
            for k, val in (s.get("reward_quantiles") or {}).items():
                self.logger.record(f"rew_decomp/quantile/{k}", val)
            self.logger.record("rew_decomp/vecnormalize_clip_rate", clip_rate)
            payload = {"timesteps": int(self.num_timesteps), **s}
            write_manifest(self.json_path, payload)
            self._append_jsonl(payload)
            self._acc.reset()
            self._n_reward = 0
            self._n_clipped = 0
        return True


def _load_json_if_exists(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[train] WARNING: could not read {path}: {exc}")
        return None


def _print_reward_decomp(paths: RunPaths) -> dict | None:
    """Print best-checkpoint (preferred) or final reward decomposition."""
    best = _load_json_if_exists(paths.eval_log_dir / "reward_decomp_best.json")
    final = _load_json_if_exists(paths.eval_log_dir / "reward_decomp.json")
    payload = best or final
    label = "best-checkpoint" if best is not None else "final-window"
    if payload is None:
        print("[train] reward decomp: (no reward_decomp_best.json / reward_decomp.json yet)")
        return None
    mean = payload.get("mean") or {}
    share = payload.get("abs_share") or {}
    print(f"\n=== Reward decomp ({label}, timesteps={payload.get('timesteps', '?')}) ===")
    # Stable term order for scanning across runs.
    preferred = [
        "return",
        "benchmark",
        "sortino",
        "participation",
        "inactivity",
        "churn",
        "turnover",
        "drawdown",
        "drawdown_penalty",
        "drawdown_increase",
        "drawdown_level",
        "concentration",
        "volatility",
        "exposure_risk",
    ]
    # Hide terms that are off in the active config (parser-default 0) so a 0.0%
    # row is not mistaken for a live-but-weak signal.
    rwd = get_config().reward
    skip: set[str] = set()
    if float(rwd.participation_bonus) <= 0.0:
        skip.add("participation")
    if float(rwd.inactivity_penalty_over_50) <= 0.0 and float(rwd.inactivity_penalty_over_90) <= 0.0:
        skip.add("inactivity")
    if float(rwd.concentration_penalty) <= 0.0:
        skip.add("concentration")
    if float(rwd.vol_penalty_scale) <= 0.0:
        skip.add("volatility")
    if float(rwd.exposure_risk_penalty_scale) <= 0.0:
        skip.add("exposure_risk")
    keys = [k for k in preferred if k in mean and k not in skip] + [
        k for k in sorted(mean) if k not in preferred and k not in skip
    ]
    print(f"{'term':<22} {'mean':>10} {'abs_share':>10}")
    for k in keys:
        m = float(mean.get(k, 0.0))
        s = float(share.get(k, 0.0))
        print(f"{k:<22} {m:+10.4f} {s:10.1%}")
    return payload


def _print_backtest_cash(summary: dict) -> None:
    pd = summary.get("portfolio_diagnostics") or {}
    weights = pd.get("per_asset_mean_weights") or {}
    print("\n=== OOS backtest allocation (best checkpoint) ===")
    print(
        f"  return={float(summary.get('total_return', float('nan'))):+.2%}  "
        f"sharpe={float(summary.get('sharpe', float('nan'))):.2f}  "
        f"max_dd={float(summary.get('max_drawdown', float('nan'))):+.2%}  "
        f"window={summary.get('oos_window', '?')}"
    )
    print(
        f"  mean_cash={float(pd.get('mean_cash_frac', float('nan'))):.1%}  "
        f"gross={float(pd.get('mean_gross_exposure', float('nan'))):.1%}  "
        f"eff_n={float(pd.get('mean_effective_n_assets', float('nan'))):.2f}  "
        f"turnover={float(pd.get('mean_turnover', float('nan'))):.4f}  "
        f"cap_hit={float(pd.get('cap_hit_fraction', float('nan'))):.1%}"
    )
    if weights:
        ordered = sorted(weights.items(), key=lambda kv: (-float(kv[1]), kv[0]))
        parts = [f"{k}={float(v):.1%}" for k, v in ordered]
        print("  mean weights: " + ", ".join(parts))


def _run_post_train_backtest_and_report(
    *,
    run_id: str,
    paths: RunPaths,
    detailed: bool = False,
) -> None:
    """OOS backtest best checkpoint, then log cash allocation + reward decomp."""
    print("\n" + "=" * 72)
    print(f"[train] Post-train OOS backtest + allocation / reward report ({run_id})")
    print("=" * 72)

    decomp = _print_reward_decomp(paths)

    best_zip = paths.best_model_dir / "best_model.zip"
    checkpoint = "best"
    if not best_zip.is_file():
        final_zip = Path(paths.final_model)
        if not final_zip.is_file():
            print(
                f"[train] Skipping post-backtest: missing {best_zip} "
                "(no best checkpoint was saved)."
            )
            report = {
                "run_id": run_id,
                "backtest": None,
                "reward_decomp": decomp,
                "skipped_reason": "missing_best_model",
            }
            write_manifest(paths.run_meta_dir / "post_train_report.json", report)
            return
        checkpoint = "best"  # resolve via --allow-latest-checkpoint → final.zip
        print(
            f"[train] No best_model.zip (selection never saved a best) — "
            f"falling back to final weights via --allow-latest-checkpoint:\n"
            f"  {final_zip}"
        )

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "backtest.py"),
        "--run-id",
        run_id,
        "--checkpoint",
        checkpoint,
        "--allow-latest-checkpoint",
        # Always write the canonical dashboard path so /ops picks it up even when
        # falling back to final/latest weights (those would otherwise only write
        # backtest_summary_{final,latest}.json).
        "--summary-json",
        str(paths.backtest_summary),
    ]
    if detailed:
        cmd.append("--detailed")
    print(f"[train] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT))
    summary_path = paths.backtest_summary
    summary = _load_json_if_exists(summary_path) if proc.returncode == 0 else None
    if proc.returncode != 0:
        print(f"[train] WARNING: post-backtest exited {proc.returncode}")
    elif summary is None:
        from rlbot.run_artifacts import resolve_backtest_summary_path

        alt = resolve_backtest_summary_path(paths)
        if alt is not None:
            summary = _load_json_if_exists(alt)
            summary_path = alt
        if summary is None:
            print(f"[train] WARNING: post-backtest ok but missing {paths.backtest_summary}")
        else:
            _print_backtest_cash(summary)
    else:
        _print_backtest_cash(summary)

    # Re-print decomp after backtest so it sits next to allocation in the log tail.
    if decomp is not None:
        _print_reward_decomp(paths)

    report = {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "model_path": str(best_zip if best_zip.is_file() else paths.final_model),
        "backtest_summary_path": str(summary_path) if summary_path.is_file() else None,
        "backtest": {
            "total_return": summary.get("total_return"),
            "sharpe": summary.get("sharpe"),
            "max_drawdown": summary.get("max_drawdown"),
            "oos_window": summary.get("oos_window"),
            "portfolio_diagnostics": summary.get("portfolio_diagnostics"),
        }
        if summary
        else None,
        "reward_decomp": decomp,
        "backtest_exit_code": int(proc.returncode),
        "skipped_reason": (
            "fallback_final_no_best" if not best_zip.is_file() else None
        ),
    }
    write_manifest(paths.run_meta_dir / "post_train_report.json", report)
    print(f"[train] Wrote {paths.run_meta_dir / 'post_train_report.json'}")


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
    parser.add_argument(
        "--post-backtest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After a completed run, automatically OOS-backtest --checkpoint best and "
            "print cash allocation + reward decomposition (default: on). "
            "Use --no-post-backtest to skip (e.g. research tiers that must not burn OOS)."
        ),
    )
    parser.add_argument(
        "--post-backtest-detailed",
        action="store_true",
        help="Pass --detailed to the post-train OOS backtest (slower; bootstrap/ensemble).",
    )
    args = parser.parse_args()
    if args.resume.strip() and args.finetune.strip():
        raise SystemExit("Use only one of --resume or --finetune, not both.")
    if Path(args.config).resolve() != cfg.path:
        set_config(load_config(args.config))
        cfg = get_config()

    if args.window is not None:
        window_name = f"W{int(args.window)}"
        canonical = CANONICAL_WINDOWS.get(window_name)
        if canonical is None:
            raise SystemExit(
                f"--window {args.window} is not canonical; choose one of "
                f"{', '.join(sorted(CANONICAL_WINDOWS))}."
            )
        for attr, value in canonical.items():
            current = getattr(args, attr)
            if current is None:
                setattr(args, attr, value)
            elif str(current) != str(value):
                raise SystemExit(
                    f"--window {args.window} implies --{attr.replace('_', '-')} {value}, "
                    f"but got {current}. Omit the explicit date flag or choose the "
                    "matching canonical window."
                )
        _startup_log(
            f"[train] --window {args.window}: using canonical {window_name} "
            f"train_end={args.train_end}, holdout_start={args.holdout_start}, "
            f"holdout_end={args.holdout_end}"
        )

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
    # Full pre-OOS panel (before carving the continuous validation tail).
    idx_trainable = idx_fit
    n_trainable_bars = int(len(idx_fit))

    # Continuous validation for selection:
    # - multi_regime (727+): K overlays spanning the full trainable panel (no carve-out)
    # - chronological tail (726): hold out the last N bars from alternating train/eval
    oos_aligned_pack: WalkforwardEnvPack | None = None
    oos_aligned_packs: list[WalkforwardEnvPack] = []
    chronological_validation: dict | None = None
    if bool(tr_cfg.oos_aligned_eval) and float(tr_cfg.oos_aligned_eval_weight) > 0.0:
        n_val = int(tr_cfg.oos_aligned_eval_bars)
        if bool(getattr(tr_cfg, "multi_regime_eval", False)):
            rsi_fit_full, macd_fit_full, fd_fit_full, fdm_fit_full, trend_fit_full, avol_fit_full, mvol_fit_full = (
                align_panel_to_timeline(
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
            )
            oos_aligned_packs = build_multi_regime_walkforward_packs(
                idx_fit,
                ohlcv_fit,
                macro_fit,
                asset_live_fit,
                rsi=rsi_fit_full,
                macd=macd_fit_full,
                fracdiff=fd_fit_full,
                fracdiff_macro=fdm_fit_full,
                trend=trend_fit_full,
                asset_vol=avol_fit_full,
                macro_vol=mvol_fit_full,
                n_slices=int(tr_cfg.multi_regime_eval_slices),
                slice_bars=n_val,
            )
            chronological_validation = {
                "mode": "multi_regime",
                "n_slices": len(oos_aligned_packs),
                "slice_bars": n_val,
                "agg": str(tr_cfg.multi_regime_eval_agg),
                "oos_aligned_eval_weight": float(tr_cfg.oos_aligned_eval_weight),
                "eval_score_burn_in_bars": int(tr_cfg.eval_score_burn_in_bars),
                "feature_memory": "continuous",
                "slices": [
                    {"start": str(p.idx[0]), "end": str(p.idx[-1]), "n_bars": int(len(p.idx))}
                    for p in oos_aligned_packs
                ],
            }
            print(
                f"  Multi-regime validation: {len(oos_aligned_packs)} × {n_val} bars "
                f"(agg={tr_cfg.multi_regime_eval_agg}, weight={tr_cfg.oos_aligned_eval_weight:g}, "
                f"burn_in={tr_cfg.eval_score_burn_in_bars})"
            )
            for i, p in enumerate(oos_aligned_packs):
                print(f"    slice[{i}]: {p.idx[0].date()} .. {p.idx[-1].date()}")
        else:
            (idx_fit, ohlcv_fit, macro_fit, asset_live_fit), (
                idx_val,
                ohlcv_val,
                macro_val,
                asset_live_val,
            ) = reserve_chronological_validation_tail(
                idx_fit,
                ohlcv_fit,
                macro_fit,
                asset_live_fit,
                n_bars=n_val,
                min_remaining=max(500, 4 * int(tr_cfg.block_size)),
            )
            rsi_v, macd_v, fd_v, fdm_v, trend_v, avol_v, mvol_v = align_panel_to_timeline(
                idx,
                idx_val,
                rsi,
                macd,
                fracdiff,
                fracdiff_macro,
                trend,
                asset_vol,
                macro_vol,
            )
            oos_aligned_pack = build_continuous_walkforward_pack(
                idx_val,
                ohlcv_val,
                macro_val,
                asset_live_val,
                rsi=rsi_v,
                macd=macd_v,
                fracdiff=fd_v,
                fracdiff_macro=fdm_v,
                trend=trend_v,
                asset_vol=avol_v,
                macro_vol=mvol_v,
            )
            chronological_validation = {
                "mode": "chronological_tail",
                "n_bars": int(len(idx_val)),
                "start": str(idx_val[0]),
                "end": str(idx_val[-1]),
                "oos_aligned_eval_weight": float(tr_cfg.oos_aligned_eval_weight),
                "eval_score_burn_in_bars": int(tr_cfg.eval_score_burn_in_bars),
                "feature_memory": "continuous",
            }
            print(
                f"  OOS-aligned validation: {len(idx_val)} bars "
                f"{idx_val[0].date()} .. {idx_val[-1].date()} "
                f"(weight={tr_cfg.oos_aligned_eval_weight:g}, burn_in={tr_cfg.eval_score_burn_in_bars})"
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
    started_at = utc_now_iso()
    dirty_frag = persist_dirty_source_snapshot(
        paths.run_meta_dir / "provenance",
        root=ROOT,
    )
    resume_path = str(getattr(args, "resume", "") or "").strip()
    resume_parent_step = None
    if resume_path:
        m_ckpt = re.search(r"ppo_(\d+)_steps", resume_path)
        if m_ckpt:
            resume_parent_step = int(m_ckpt.group(1))
    provenance = {
        "feature_split_mode": cfg.data.feature_split_mode,
        "config_hash": config_sha256(cfg.to_dict()),
        "data_cache_hash": sha256_file(paths.data_snapshot),
        "started_at_utc": started_at,
        "hardware": hardware_profile(),
        "provenance": dirty_frag,
        "resume_parent": resume_path or None,
        "resume_parent_step": resume_parent_step,
        "nominal_timesteps": int(args.timesteps),
        **{k: dirty_frag.get(k) for k in ("git_commit", "git_dirty")},
    }

    write_manifest(
        paths.manifest_path,
        {
            "run_id": run_id,
            "config_path": str(cfg.path),
            "args": vars(args),
            "universe": universe_meta,
            "n_index": int(len(idx)),
            "n_trainable_bars": n_trainable_bars,
            "chronological_holdout": {
                "holdout_days": int(args.holdout_days),
                "train_end": args.train_end,
                "holdout_start": args.holdout_start,
                "holdout_end": args.holdout_end or (str(idx_hold[-1]) if len(idx_hold) else None),
                "trainable_end": str(idx_trainable[-1]) if len(idx_trainable) else None,
                "holdout_bars": int(len(idx_hold)),
                "date_start": str(idx_hold[0]) if len(idx_hold) else None,
                "date_end": str(idx_hold[-1]) if len(idx_hold) else None,
            },
            "chronological_validation": chronological_validation,
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
            f"(else full {args.timesteps:,} timesteps); best_model by robust eval score"
        )
    else:
        print(f"  early_stop: off (full {args.timesteps:,} timesteps; best_model by robust eval score)")
    try:
        print(format_preflight_text(build_curriculum_preflight(cfg, budget=int(args.timesteps))))
    except Exception as exc:  # noqa: BLE001
        print(f"  [preflight] skipped: {exc}")
    print(
        f"  trade bundle: best/ saves model + vec_normalize together on each new best robust score; "
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
        f"(combined abs cap {cfg.reward.benchmark_combined_abs_cap:g}) "
        f"- drawdown_penalty(inc={cfg.reward.drawdown_increase_penalty:g}, "
        f"lvl={cfg.reward.drawdown_level_penalty:g}@{cfg.reward.drawdown_level_floor:.0%}) "
        f"- exposure_risk({cfg.reward.exposure_risk_mode}, scale={cfg.reward.exposure_risk_penalty_scale:g}) "
        f"- vol_penalty({cfg.reward.vol_penalty_mode}, scale={cfg.reward.vol_penalty_scale:g}) "
        f"- tx_cost*{cfg.reward.churn_penalty:g}*{cfg.reward.reward_scale:g} "
        f"- turnover*{cfg.reward.turnover_penalty:g}*{cfg.reward.reward_scale:g} "
        f"(both × curriculum_churn_scale×VIX) "
        f"| cash_yield={cfg.reward.cash_daily_yield:g}/day"
    )
    print(
        f"  eval selection ({cfg.training.best_model_benchmark}): "
        f"{1.0 - cfg.training.best_model_score_stitched_blend:g}*mean(excess) + "
        f"{cfg.training.best_model_score_stitched_blend:g}*stitched_excess "
        f"- {cfg.training.best_model_score_std_coef:g}*std(excess) "
        f"- {cfg.training.best_model_score_dd_coef:g}*p75(max_dd_nav)"
    )
    print(f"  eval plot: mean ending NAV + robust score → Runs/<id>/eval_logs/eval_nav_history.npz")
    print(f"  eval portfolio diagnostics → Runs/<id>/eval_logs/eval_portfolio_diagnostics.jsonl")
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
            f"  best_model gate: eval logged always; models/best/ updates from step "
            f"{_bms:,} (fee_ramp_end; full eval fees + churn)"
        )
    else:
        print("  best_model gate: off (robust eval score selects best from step 0)")
    print(
        f"  entropy: explore {ent_cfg.explore_ent} (floor 0.02 until {_edl:,} steps, "
        f"then 0.01 for {_ef:,}) → cosine decay to {ent_cfg.final_ent} from "
        f"{_decay_frac:.0%} of run (step ~{_decay_step:,}), not eval-gated"
    )
    if str(getattr(hp, "lr_schedule", "cosine")).lower() == "phase_aware":
        _lr_hold = dr_widen_end_milestone(int(args.timesteps))
        print(
            f"  LR={args.learning_rate} (phase-aware: hold until DR widen end "
            f"~{_lr_hold:,}, then cosine → {hp.learning_rate_floor} floor)"
        )
    else:
        print(f"  LR={args.learning_rate} (cosine → {hp.learning_rate_floor} floor)")
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
        print(f"  MODE: crash-resume from {args.resume}")

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
    worker_config = WorkerConfigInstaller(cfg)
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
            config_installer=worker_config,
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
            config_installer=worker_config,
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

    oos_aligned_env: VecNormalize | None = None
    oos_aligned_benchmark_ctx: EvalBenchmarkContext | None = None
    oos_aligned_benchmark_ctxs: list[EvalBenchmarkContext] = []
    packs_for_oos = (
        oos_aligned_packs
        if oos_aligned_packs
        else ([oos_aligned_pack] if oos_aligned_pack is not None else [])
    )
    if packs_for_oos:
        factories = []
        for i, pack in enumerate(packs_for_oos):
            oos_ep_steps = max(1, int(pack.ohlcv.shape[0]) - 2)
            factories.append(
                _make_env_factory(
                    pack,
                    random_start=False,
                    noise_scale=train_noise_scale,
                    log_dir=paths.logs_dir,
                    monitor_stem=f"oos_aligned_monitor_{i}",
                    max_episode_steps=oos_ep_steps,
                    reseed_on_reset=False,
                    obs_lag_default=args.obs_lag,
                    domain_randomize=False,
                    inactivity_penalty_scale=cfg.reward.eval_inactivity_penalty_scale,
                    record_episode_nav=True,
                    config_installer=worker_config,
                )
            )
            oos_aligned_benchmark_ctxs.append(
                EvalBenchmarkContext(
                    ohlcv=pack.ohlcv,
                    idx=pack.idx,
                    tickers=list(panel_tickers),
                    asset_live=pack.asset_live,
                    mode=str(tr_cfg.best_model_benchmark),
                )
            )
        oos_aligned_env = SubprocVecEnv(factories)
        oos_aligned_env = VecNormalize(
            oos_aligned_env,
            norm_obs=vn_cfg.norm_obs,
            norm_reward=False,
            clip_obs=vn_cfg.clip_obs,
            gamma=hp.gamma,
            training=False,
        )
        oos_aligned_benchmark_ctx = oos_aligned_benchmark_ctxs[0]

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

    # Phase-aware LR: hold the initial LR through DR widening (fee/lag conditions are
    # still shifting; a decayed LR freezes adaptation), then cosine-decay to the floor
    # over the settled full-DR remainder. "cosine" (legacy) keeps the global curve.
    lr_hold_until = (
        dr_widen_end_milestone(int(args.timesteps))
        if str(getattr(hp, "lr_schedule", "cosine")).lower() == "phase_aware"
        else 0
    )
    lr_schedule = lr_schedule_with_floor_for_budget(
        args.learning_rate,
        hp.learning_rate_floor,
        int(args.timesteps),
        hold_until_step=lr_hold_until,
    )

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
            # Prefer the stats saved NEXT TO the checkpoint (e.g. best/vec_normalize.pkl
            # for best_model.zip) before falling back to the run-level end-of-run stats —
            # the grandparent fallback alone silently mispaired best weights with
            # end-of-run normalization.
            sibling = resume_path.parent / "vec_normalize.pkl"
            vn_path = sibling if sibling.is_file() else resume_path.parent.parent / "vec_normalize.pkl"
        if vn_path and vn_path.is_file():
            loaded_vn = VecNormalize.load(str(vn_path), train_env.venv)
            train_env.obs_rms = loaded_vn.obs_rms
            train_env.ret_rms = loaded_vn.ret_rms
            eval_env.obs_rms = loaded_vn.obs_rms
            if oos_aligned_env is not None:
                oos_aligned_env.obs_rms = loaded_vn.obs_rms
            print(f"  Restored VecNormalize stats from: {vn_path}")
        else:
            print("  WARNING: No VecNormalize stats found for checkpoint")
            eval_env.obs_rms = train_env.obs_rms
            if oos_aligned_env is not None:
                oos_aligned_env.obs_rms = train_env.obs_rms

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
        if oos_aligned_env is not None:
            oos_aligned_env.obs_rms = train_env.obs_rms

    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"  total params: {total_params:,}  (trainable: {trainable_params:,})")

    # ── callbacks ────────────────────────────────────────────────────────
    eval_freq_steps = int(tr_cfg.eval_freq_steps)
    eval_freq_pre_gate = int(tr_cfg.eval_freq_pre_gate_steps)
    eval_freq = eval_freq_vector_steps(eval_freq_steps, n_envs)
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
    if oos_aligned_env is not None:
        if oos_aligned_packs:
            print(
                f"  Multi-regime selection: {len(oos_aligned_packs)} × "
                f"{tr_cfg.oos_aligned_eval_bars} bars, agg={tr_cfg.multi_regime_eval_agg}, "
                f"weight={tr_cfg.oos_aligned_eval_weight:g}, "
                f"burn_in={tr_cfg.eval_score_burn_in_bars}"
            )
        else:
            print(
                f"  OOS-aligned selection: continuous {tr_cfg.oos_aligned_eval_bars} bars, "
                f"weight={tr_cfg.oos_aligned_eval_weight:g}, "
                f"burn_in={tr_cfg.eval_score_burn_in_bars} "
                f"(primary checkpoint signal matches backtest structure)"
            )
    if bool(getattr(cfg.environment, "two_head_actions", False)):
        print("  Policy action map: two-head (exposure logit + risky allocation)")
    # Patience early-stop is gated on curriculum completion (dr_widen_end); patience=0 keeps
    # the full --timesteps budget. best_model saves open after fee_ramp_end (full eval fees).
    curriculum_end_step = dr_widen_end_milestone(args.timesteps)
    early_stop_patience = int(tr_cfg.early_stop_patience)
    best_model_min_step = resolve_best_model_min_step(args.timesteps)
    print(
        f"  eval cadence: {eval_freq_pre_gate:,} global steps pre fee-ramp, "
        f"{eval_freq_steps:,} after (gate {best_model_min_step:,})"
    )
    eval_benchmark_ctx = EvalBenchmarkContext(
        ohlcv=eval_pack.ohlcv,
        idx=eval_pack.idx,
        tickers=list(panel_tickers),
        asset_live=eval_pack.asset_live,
        mode=str(tr_cfg.best_model_benchmark),
        fee_scale=1.0,
    )
    eval_callback = EvalNavBestModelCallback(
        eval_env,
        nav_history_path=paths.eval_nav_history,
        best_model_save_path=str(paths.best_model_dir),
        train_vec_env=train_env,
        patience=early_stop_patience,
        curriculum_end_step=curriculum_end_step,
        best_model_min_step=best_model_min_step,
        panel_tickers=panel_tickers,
        max_single_asset_weight=cfg.environment.max_single_asset_weight,
        eval_diagnostics_path=paths.eval_portfolio_diagnostics_jsonl,
        benchmark_ctx=eval_benchmark_ctx,
        score_std_coef=tr_cfg.best_model_score_std_coef,
        score_dd_coef=tr_cfg.best_model_score_dd_coef,
        score_stitched_blend=tr_cfg.best_model_score_stitched_blend,
        score_mode=tr_cfg.best_model_score_mode,
        eval_freq_steps=eval_freq_steps,
        eval_freq_pre_gate_steps=eval_freq_pre_gate,
        n_envs=n_envs,
        trailing_evals=int(getattr(tr_cfg, "best_model_trailing_evals", 0)),
        trailing_agg=str(getattr(tr_cfg, "best_model_trailing_agg", "median")),
        confirm_evals=int(getattr(tr_cfg, "best_model_confirm_evals", 0)),
        stress_suite=bool(getattr(tr_cfg, "best_model_stress_suite", False)),
        stress_weight=float(getattr(tr_cfg, "best_model_stress_weight", 0.3)),
        oos_aligned_env=oos_aligned_env,
        oos_aligned_benchmark_ctx=oos_aligned_benchmark_ctx,
        oos_aligned_benchmark_ctxs=oos_aligned_benchmark_ctxs,
        oos_aligned_weight=(
            float(tr_cfg.oos_aligned_eval_weight) if oos_aligned_env is not None else 0.0
        ),
        multi_regime_eval_agg=str(getattr(tr_cfg, "multi_regime_eval_agg", "p25")),
        eval_score_burn_in_bars=int(tr_cfg.eval_score_burn_in_bars),
        best_model_max_dd_reject=float(getattr(tr_cfg, "best_model_max_dd_reject", 0.0)),
        best_model_max_dd_reject_hard=bool(
            getattr(tr_cfg, "best_model_max_dd_reject_hard", True)
        ),
        best_model_max_dd_reject_coef=float(
            getattr(tr_cfg, "best_model_max_dd_reject_coef", 50.0)
        ),
        best_model_worst_regime_cash_coef=float(
            getattr(tr_cfg, "best_model_worst_regime_cash_coef", 0.0)
        ),
        best_model_worst_regime_cash_target=float(
            getattr(tr_cfg, "best_model_worst_regime_cash_target", 0.0)
        ),
        best_model_mean_cash_coef=float(
            getattr(tr_cfg, "best_model_mean_cash_coef", 0.0)
        ),
        best_model_mean_cash_cap=float(
            getattr(tr_cfg, "best_model_mean_cash_cap", 1.0)
        ),
        log_path=str(paths.eval_log_dir),
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

    # curriculum_update_freq is configured in GLOBAL timesteps; callbacks count
    # vector steps (n_calls), so divide by n_envs — otherwise the ramp granularity
    # (and thus training behavior) silently varies with the GPU profile's env count.
    callback_update_freq = max(tr_cfg.curriculum_update_freq // n_envs, 1)
    reward_decomp_callback = RewardDecompCallback(
        json_path=paths.eval_log_dir / "reward_decomp.json",
        log_freq=callback_update_freq,
        clip_reward=float(cfg.vec_normalize.clip_reward),
    )
    eval_callback._reward_decomp_callback = reward_decomp_callback
    callbacks = [eval_callback, checkpoint_callback, reward_decomp_callback]
    resume_mode = bool(args.resume.strip())
    learn_timesteps, reset_num_timesteps = resolve_learn_timesteps(
        budget=int(args.timesteps),
        start=int(model.num_timesteps),
        resume=resume_mode,
    )
    if learn_timesteps > 0:
        callbacks.append(BudgetProgressBarCallback(budget_timesteps=int(args.timesteps)))
    if not finetune_mode:
        callbacks.insert(
            0,
            TradingCurriculumCallback(
                train_env,
                learn_budget=args.timesteps,
                update_freq=callback_update_freq,
                eval_vec_env=eval_env,
                oos_aligned_vec_env=oos_aligned_env,
            ),
        )
        callbacks.append(AdaptiveEntropyCallback(
            explore_ent=ent_cfg.explore_ent,
            final_ent=ent_cfg.final_ent,
            early_floor=ent_cfg.early_floor,
            early_floor_steps=entropy_early_floor_milestones(args.timesteps),
            min_explore_steps=entropy_dr_lock_milestones(args.timesteps),
            decay_start_fraction=ent_cfg.decay_start_fraction,
            learn_budget=int(args.timesteps),
            warmup_improvements=ent_cfg.warmup_improvements,
            eval_log_dir=str(paths.eval_log_dir),
            eval_check_freq=eval_freq,
            eval_nav_callback=eval_callback,
            lr_schedule=lr_schedule,
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
    if resume_mode and learn_timesteps == 0:
        _startup_log(
            f"[train] Already at budget ({args.timesteps:,} steps at "
            f"{model.num_timesteps:,}); skipping learn()."
        )
    elif resume_mode:
        _startup_log(
            f"[train] Resuming PPO: {model.num_timesteps:,} / {args.timesteps:,} "
            f"({learn_timesteps:,} steps remaining)..."
        )
    else:
        _startup_log(f"[train] Starting PPO learning ({args.timesteps:,} timesteps)...")
    learn_error: BaseException | None = None
    interrupted = False
    try:
        if learn_timesteps > 0:
            model.learn(
                total_timesteps=learn_timesteps,
                callback=CallbackList(callbacks),
                progress_bar=False,
                reset_num_timesteps=reset_num_timesteps,
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
    best_eval_score = (
        float(eval_callback.best_selection_score)
        if np.isfinite(eval_callback.best_selection_score)
        else None
    )
    # Prefer the step that actually wrote best_model.zip (trailing/confirmation),
    # not argmax of instantaneous post-gate scores.
    best_eval_step = getattr(eval_callback, "_best_eval_step", None)
    if best_eval_step is not None:
        best_eval_step = int(best_eval_step)
    elif eval_callback._robust_scores and eval_callback._nav_timesteps:
        navs_arr = np.asarray(eval_callback._robust_scores, dtype=np.float64)
        steps_arr = np.asarray(eval_callback._nav_timesteps, dtype=np.int64)
        n = min(len(navs_arr), len(steps_arr))
        navs_arr, steps_arr = navs_arr[:n], steps_arr[:n]
        gate = int(eval_callback.best_model_min_step)
        post_gate = steps_arr >= gate if gate > 0 else np.ones(n, dtype=bool)
        if post_gate.any():
            j = int(np.argmax(navs_arr[post_gate]))
            best_eval_step = int(steps_arr[post_gate][j])
    early_stop_reason = getattr(eval_callback, "early_stop_reason", None)
    elapsed_timesteps = int(getattr(model, "num_timesteps", 0) or 0)
    cumulative_timesteps = elapsed_timesteps
    if resume_parent_step is not None and elapsed_timesteps < resume_parent_step:
        # Session counter was reset somehow; prefer parent+session if available.
        cumulative_timesteps = resume_parent_step + max(0, elapsed_timesteps)
    elif resume_parent_step is not None:
        cumulative_timesteps = elapsed_timesteps  # SB3 keeps absolute counter on resume

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
            "n_trainable_bars": n_trainable_bars,
            "n_train_bars": int(len(train_idx)),
            "n_eval_bars": int(len(eval_idx)),
            "chronological_validation": chronological_validation,
            "data_cache_snapshot": str(paths.data_snapshot),
            "started_at_utc": provenance.get("started_at_utc"),
            "finished_at_utc": utc_now_iso(),
            "training_status": "interrupted" if interrupted else "completed",
            "total_params": total_params,
            "trainable_params": trainable_params,
            "best_eval_nav": best_eval_nav,
            "best_eval_score": best_eval_score,
            "best_eval_step": best_eval_step,
            "early_stop_reason": early_stop_reason,
            "elapsed_timesteps": elapsed_timesteps,
            "cumulative_timesteps": cumulative_timesteps,
            "nominal_timesteps": int(args.timesteps),
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
                "eval_portfolio_diagnostics": str(paths.eval_portfolio_diagnostics_jsonl),
                "reward_decomp_jsonl": str(paths.eval_log_dir / "reward_decomp.jsonl"),
                "reward_decomp_best": str(paths.eval_log_dir / "reward_decomp_best.json"),
                "source_provenance_dir": str(paths.run_meta_dir / "provenance"),
            },
        },
    )

    # Machine-readable training summary for the research registry / orchestrator.
    write_manifest(
        paths.run_meta_dir / "training_summary.json",
        {
            "run_id": run_id,
            "timesteps": int(args.timesteps),
            "elapsed_timesteps": elapsed_timesteps,
            "cumulative_timesteps": cumulative_timesteps,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "best_eval_nav": best_eval_nav,
            "best_eval_score": best_eval_score,
            "best_eval_step": best_eval_step,
            "early_stop_reason": early_stop_reason,
            "n_train_bars": int(len(train_idx)),
            "n_eval_bars": int(len(eval_idx)),
            "started_at_utc": provenance.get("started_at_utc"),
            "finished_at_utc": utc_now_iso(),
            **{k: provenance[k] for k in ("git_commit", "git_dirty", "config_hash", "data_cache_hash", "feature_split_mode") if k in provenance},
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

    if interrupted:
        print("[train] Interrupted — skipping post-train backtest.")
    elif bool(getattr(args, "post_backtest", True)):
        _run_post_train_backtest_and_report(
            run_id=run_id,
            paths=paths,
            detailed=bool(getattr(args, "post_backtest_detailed", False)),
        )
    else:
        print("[train] Post-train backtest disabled (--no-post-backtest).")


if __name__ == "__main__":
    main()
