"""Auto-research harness: experiment specs, run registry, reports, and OOS gates.

Torch-free by design — the orchestrator (scripts/research.py) shells out to the
canonical train/backtest CLIs, so these modules import only stdlib + rlbot.rl_config /
rlbot.run_artifacts and can be unit-tested without the training stack.
"""
