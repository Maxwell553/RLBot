Isolated bundle for run 120M_4_19_26 — checkpoint at 48M training steps
=======================================================================

Selection (from runs/120M_4_19_26/eval_logs/evaluations.npz):
  Best mean eval reward in timesteps [40_000_000, 60_000_000] occurs at
  ~47_972_352 steps (per-step reward mean ≈ 1.59).
  Nearest 1M checkpoint: 48_000_000 steps.

Model (copied from models/120M_4_19_26/checkpoints/ — originals unchanged):
  model/ppo_48000000_steps.zip   RecurrentPPO weights at 48M steps
  model/vec_normalize.pkl        Matching VecNormalize stats (ppo_vecnormalize_48000000_steps.pkl)
  model/ppo_portfolio_final.zip  Duplicate of ppo_48000000_steps.zip for drop-in use with paper_trade defaults

Also copied (from the same run — originals unchanged):
  plots_120M_4_19_26/   training plot, training_episodes.npz, etc.
  logs_120M_4_19_26/    Monitor CSVs
  tb_logs_120M_4_19_26/ TensorBoard event files
  runs_120M_4_19_26/    manifest.json, eval_logs/evaluations.npz, data_cache snapshot
  checkpoints_1M/       Same 48M .zip + .pkl pair for provenance

Paper trade from repo root (example):
  cd paper_trade && python paper_trade.py \\
    --model ../paper_trade_2/model/ppo_portfolio_final.zip \\
    --vec-normalize ../paper_trade_2/model/vec_normalize.pkl

Or use ../paper_trade_2/model/ppo_48000000_steps.zip explicitly.
