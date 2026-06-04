"""Ensure repo root is on ``sys.path`` when running ``python scripts/<cli>.py``."""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# SB3 load() builds AdamW; on PyTorch 2.x that can pull torch._dynamo → sympy (minutes).
# Inference/backtest only need policy weights — disable dynamo before any torch import.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# PyTorch 2.x + Python 3.14: noisy JIT overload source warnings (harmless).
warnings.filterwarnings(
    "ignore",
    message=r"Unable to retrieve source for @torch\.jit\._overload",
    category=UserWarning,
    module=r"torch\._jit_internal",
)

ROOT = Path(__file__).resolve().parent.parent
_root = str(ROOT)
if _root not in sys.path:
    sys.path.insert(0, _root)
