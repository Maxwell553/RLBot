"""feature_split_mode: continuous slices the global panel; independent recomputes
per segment and applies the warmup purge."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rlbot.data_utils import (
    N_MACRO,
    WalkforwardEnvPack,
    compute_feature_panel,
    train_test_split_alternating,
)

BLOCK = 30
STRIDE = 4
PURGE = 25


def _panel(n_bars: int = 300, n_a: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n_bars, freq="B")
    rets = rng.normal(0.0005, 0.01, size=(n_bars, n_a))
    price = 100.0 * np.exp(np.cumsum(rets, axis=0))
    ohlcv = np.zeros((n_bars, n_a, 5), dtype=np.float64)
    for c in range(4):
        ohlcv[:, :, c] = price
    ohlcv[:, :, 4] = 1e6
    macro = 10.0 * np.exp(np.cumsum(rng.normal(0.0, 0.005, size=(n_bars, N_MACRO)), axis=0))
    return idx, ohlcv, macro


def _split(mode: str, *, precomputed: bool = False):
    idx, ohlcv, macro = _panel()
    kwargs: dict = {}
    if precomputed:
        rsi, macd, fd, fdm, trend, avol, mvol = compute_feature_panel(ohlcv, macro)
        kwargs = dict(
            rsi=rsi, macd=macd, fracdiff=fd, fracdiff_macro=fdm,
            trend=trend, asset_vol=avol, macro_vol=mvol,
        )
    tr, ev = train_test_split_alternating(
        idx, ohlcv, macro,
        block_size=BLOCK, eval_stride=STRIDE, feature_purge_warmup=PURGE,
        feature_split_mode=mode, **kwargs,
    )
    return idx, WalkforwardEnvPack.from_tuple(tr), WalkforwardEnvPack.from_tuple(ev)


def test_continuous_slices_global_panel() -> None:
    """Continuous mode returns rows of the panel computed on the full timeline."""
    idx, ohlcv, macro = _panel()
    rsi_g, macd_g, fd_g, fdm_g, trend_g, avol_g, mvol_g = compute_feature_panel(ohlcv, macro)
    _, tr, ev = _split("continuous")
    for pack in (tr, ev):
        loc = idx.get_indexer(pack.idx)
        assert np.all(loc >= 0)
        assert np.allclose(pack.rsi, rsi_g[loc])
        assert np.allclose(pack.macd, macd_g[loc])
        assert np.allclose(pack.fracdiff, fd_g[loc])
        assert np.allclose(pack.trend, trend_g[loc])


def test_continuous_precomputed_equals_recomputed() -> None:
    """Passing cache features vs recomputing once gives identical continuous output."""
    _, tr_p, ev_p = _split("continuous", precomputed=True)
    _, tr_r, ev_r = _split("continuous", precomputed=False)
    for a, b in ((tr_p, tr_r), (ev_p, ev_r)):
        assert np.allclose(a.rsi, b.rsi)
        assert np.allclose(a.fracdiff, b.fracdiff)
        assert np.allclose(a.asset_vol, b.asset_vol)


def _segment_starts(pack: WalkforwardEnvPack) -> list[int]:
    return [0] + list(pack.block_boundaries)


def test_independent_purges_each_segment_head() -> None:
    """Independent mode neutralizes the first PURGE bars of every segment."""
    _, tr, ev = _split("independent")
    for pack in (tr, ev):
        starts = _segment_starts(pack)
        ends = starts[1:] + [pack.rsi.shape[0]]
        for s, e in zip(starts, ends):
            n = min(PURGE, e - s)
            assert np.allclose(pack.rsi[s : s + n], 50.0)
            assert np.allclose(pack.macd[s : s + n], 0.0)
            assert np.allclose(pack.fracdiff[s : s + n], 0.0)
            assert np.allclose(pack.fracdiff_macro[s : s + n], 0.0)
            assert np.allclose(pack.trend[s : s + n], 0.0)


def test_continuous_does_not_purge() -> None:
    """Continuous mode leaves segment heads non-neutral (purge is not applied)."""
    _, tr, _ = _split("continuous")
    # at least one of the indicator panels is non-neutral somewhere in the first PURGE bars
    head = tr.fracdiff[:PURGE]
    assert not np.allclose(head, 0.0)


def test_boundaries_and_shape_match_across_modes() -> None:
    """Both modes segment identically and return the same row count."""
    _, tr_c, ev_c = _split("continuous")
    _, tr_i, ev_i = _split("independent")
    assert list(tr_c.block_boundaries) == list(tr_i.block_boundaries)
    assert list(ev_c.block_boundaries) == list(ev_i.block_boundaries)
    assert tr_c.rsi.shape == tr_i.rsi.shape
    assert ev_c.rsi.shape == ev_i.rsi.shape
    # boundaries stay within range
    for b in list(tr_c.block_boundaries) + list(ev_c.block_boundaries):
        assert 0 < b < tr_c.rsi.shape[0] + ev_c.rsi.shape[0]


def test_independent_ignores_precomputed_features() -> None:
    """Precomputed (continuous) features passed in independent mode are ignored."""
    _, tr_none, _ = _split("independent", precomputed=False)
    _, tr_pre, _ = _split("independent", precomputed=True)
    assert np.allclose(tr_none.rsi, tr_pre.rsi)
    assert np.allclose(tr_none.fracdiff, tr_pre.fracdiff)
