"""Fast RecurrentPPO load for inference (no real AdamW / optimizer state)."""

from __future__ import annotations

import contextlib
from functools import partial
from pathlib import Path
from typing import Any, Iterator

import torch as th
import torch.nn as nn
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.type_aliases import Schedule


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


def _build_policy_without_optimizer(self: ActorCriticPolicy, lr_schedule: Schedule) -> None:
    """Mirror ``ActorCriticPolicy._build`` but skip ``optimizer_class(...)``."""
    import numpy as np

    self._build_mlp_extractor()

    latent_dim_pi = self.mlp_extractor.latent_dim_pi

    if isinstance(self.action_dist, DiagGaussianDistribution):
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            latent_dim=latent_dim_pi, log_std_init=self.log_std_init
        )
    elif isinstance(self.action_dist, StateDependentNoiseDistribution):
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
        )
    elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
        self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
    else:
        raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

    self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
    if self.ortho_init:
        module_gains = {
            self.features_extractor: np.sqrt(2),
            self.mlp_extractor: np.sqrt(2),
            self.action_net: 0.01,
            self.value_net: 1,
        }
        if not self.share_features_extractor:
            del module_gains[self.features_extractor]
            module_gains[self.pi_features_extractor] = np.sqrt(2)
            module_gains[self.vf_features_extractor] = np.sqrt(2)

        for module, gain in module_gains.items():
            module.apply(partial(self.init_weights, gain=gain))

    self.optimizer = None  # type: ignore[assignment]


@contextlib.contextmanager
def skip_training_optimizer() -> Iterator[None]:
    """Patch SB3 policy construction so no real ``torch.optim.AdamW`` is created."""
    original_build = ActorCriticPolicy._build
    original_adam = th.optim.Adam
    original_adamw = th.optim.AdamW
    ActorCriticPolicy._build = _build_policy_without_optimizer  # type: ignore[method-assign]
    th.optim.Adam = _DummyOptimizer  # type: ignore[misc, assignment]
    th.optim.AdamW = _DummyOptimizer  # type: ignore[misc, assignment]
    try:
        yield
    finally:
        ActorCriticPolicy._build = original_build  # type: ignore[method-assign]
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
    Load ``RecurrentPPO`` for ``predict()`` only.

    Uses ``policy.pth`` weights and never instantiates a real AdamW optimizer
    (SB3's default ``load()`` does, which can stall for minutes on PyTorch 2.x).
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


def swap_recurrent_ppo_weights(model: Any, path: str | Path, *, device: str = "cpu") -> None:
    """Reload ``policy.pth`` only (for batch backtests across checkpoints)."""
    _, params, _ = load_from_zip_file(str(Path(path).resolve()), device=device, load_data=False)
    assert params is not None
    model.policy.load_state_dict(params["policy"], strict=True)
