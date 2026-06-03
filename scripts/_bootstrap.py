"""Ensure repo root is on ``sys.path`` when running ``python scripts/<cli>.py``."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

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
