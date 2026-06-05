"""Fast RecurrentPPO load for inference (no real AdamW / optimizer state)."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Iterator

import torch as th
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from rlbot.vecnorm_utils import freeze_vec_normalize_for_inference


class _DummyOptimizer:
    """Placeholder so SB3 policy __init__ does not import torch._dynamo/sympy."""

    def __init__(self, params, lr=0.0, **kwargs) -> None:
        self.param_groups = [{"params": list(params)}]

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        return

    def zero_grad(self, set_to_none: bool = False) -> None:
        return

    def step(self, closure=None) -> None:
        return


@contextlib.contextmanager
def skip_training_optimizer() -> Iterator[None]:
    """Route ``torch.optim.Adam*`` to a no-op optimizer during policy construction."""
    original_adam = th.optim.Adam
    original_adamw = th.optim.AdamW
    th.optim.Adam = _DummyOptimizer  # type: ignore[misc, assignment]
    th.optim.AdamW = _DummyOptimizer  # type: ignore[misc, assignment]
    try:
        yield
    finally:
        th.optim.Adam = original_adam  # type: ignore[misc]
        th.optim.AdamW = original_adamw  # type: ignore[misc]


def _patch_policy_kwargs_for_inference(data: dict[str, Any]) -> None:
    """Saved checkpoints embed ``torch.optim.adamw.AdamW`` in nested ``policy_kwargs``."""
    pk = data.get("policy_kwargs")
    if isinstance(pk, dict):
        pk["optimizer_class"] = _DummyOptimizer


def load_recurrent_ppo_inference(
    path: str | Path,
    *,
    device: str = "cpu",
) -> Any:
    """
    Load ``RecurrentPPO`` policy weights for ``predict()`` only.

    Does **not** load VecNormalize statistics. For scaled observations use
    ``load_vec_normalize_for_inference`` or ``load_recurrent_ppo_with_vecnorm``.
    """
    from sb3_contrib import RecurrentPPO

    path = str(Path(path).resolve())
    with skip_training_optimizer():
        data, params, pytorch_variables = load_from_zip_file(path, device=device, load_data=True)
        assert data is not None and params is not None
        _patch_policy_kwargs_for_inference(data)

        model = RecurrentPPO(
            policy=data["policy_class"],
            env=None,  # type: ignore[arg-type]
            device=device,
            _init_setup_model=False,
        )
        model.__dict__.update(data)
        model._setup_model()

        if pytorch_variables is not None:
            for name, tensor in pytorch_variables.items():
                attr = getattr(model, name, None)
                if attr is not None:
                    setattr(model, name, tensor.to(model.device))

        model.policy.load_state_dict(params["policy"], strict=True)
        model.policy.set_training_mode(False)

    return model


def load_vec_normalize_for_inference(
    stats_path: str | Path,
    venv: VecEnv,
) -> VecNormalize:
    """Load frozen training observation statistics (``vec_normalize.pkl``)."""
    vec_env = VecNormalize.load(str(Path(stats_path).resolve()), venv)
    return freeze_vec_normalize_for_inference(vec_env)


def load_recurrent_ppo_with_vecnorm(
    model_path: str | Path,
    stats_path: str | Path,
    venv: VecEnv,
    *,
    device: str = "cpu",
) -> tuple[Any, VecNormalize]:
    """
    Load policy weights and hydrate VecNormalize running stats for deployment.

    Returns ``(model, vec_env)`` with ``model.set_env(vec_env)`` applied.
    """
    model = load_recurrent_ppo_inference(model_path, device=device)
    vec_env = load_vec_normalize_for_inference(stats_path, venv)
    model.set_env(vec_env)
    return model, vec_env


def swap_recurrent_ppo_weights(model: Any, path: str | Path, *, device: str = "cpu") -> None:
    """Reload ``policy.pth`` only (for batch backtests across checkpoints)."""
    _, params, _ = load_from_zip_file(str(Path(path).resolve()), device=device, load_data=False)
    assert params is not None
    model.policy.load_state_dict(params["policy"], strict=True)
