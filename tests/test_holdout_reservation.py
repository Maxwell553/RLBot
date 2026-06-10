"""Chronological OOS holdout reservation (rlbot.data_utils.reserve_chronological_holdout):
the trainable segment ends strictly before the holdout begins, boundary bars land on the
right side, and the two slices never overlap. Torch-free."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlbot.data_utils import reserve_chronological_holdout


def _panel(n_bars: int = 120, n_a: int = 3, seed: int = 0):
    """Business-day index + tagged arrays (row i carries value i) like test_feature_split_modes."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-02", periods=n_bars, freq="B")
    rets = rng.normal(0.0005, 0.01, size=(n_bars, n_a))
    price = 100.0 * np.exp(np.cumsum(rets, axis=0))
    ohlcv = np.zeros((n_bars, n_a, 5), dtype=np.float64)
    for c in range(4):
        ohlcv[:, :, c] = price
    ohlcv[:, :, 4] = 1e6
    row_tag = np.arange(n_bars, dtype=np.float64)  # row identity survives slicing
    return idx, ohlcv, row_tag


# ── days-based reservation ───────────────────────────────────────────────
def test_days_based_trainable_ends_strictly_before_holdout() -> None:
    idx, ohlcv, tag = _panel()
    (tr_idx, tr_ohlcv, tr_tag), (ho_idx, ho_ohlcv, ho_tag) = reserve_chronological_holdout(
        idx, ohlcv, tag, holdout_days=30
    )
    assert tr_idx.max() < ho_idx.min()
    # cutoff semantics: trainable is date <= last - holdout_days, holdout is the rest
    cutoff = idx[-1] - pd.Timedelta(days=30)
    assert tr_idx.max() <= cutoff
    assert ho_idx.min() > cutoff
    # boundary bars on the right side: every bar <= cutoff trains, every later bar holds out
    assert list(tr_idx) == list(idx[idx <= cutoff])
    assert list(ho_idx) == list(idx[idx > cutoff])


def test_days_based_partition_no_overlap_no_gap() -> None:
    idx, ohlcv, tag = _panel()
    (tr_idx, _, tr_tag), (ho_idx, _, ho_tag) = reserve_chronological_holdout(
        idx, ohlcv, tag, holdout_days=45
    )
    # exact partition of the panel: no overlap, no dropped bars, nothing invented
    assert len(tr_idx) + len(ho_idx) == len(idx)
    assert len(tr_idx.intersection(ho_idx)) == 0
    assert tr_idx.append(ho_idx).equals(idx)
    # arrays are sliced consistently with the index
    np.testing.assert_array_equal(np.concatenate([tr_tag, ho_tag]), np.arange(len(idx)))


def test_days_based_arrays_match_index_rows() -> None:
    idx, ohlcv, tag = _panel()
    (tr_idx, tr_ohlcv, tr_tag), (ho_idx, ho_ohlcv, ho_tag) = reserve_chronological_holdout(
        idx, ohlcv, tag, holdout_days=20
    )
    assert tr_ohlcv.shape[0] == len(tr_idx) == len(tr_tag)
    assert ho_ohlcv.shape[0] == len(ho_idx) == len(ho_tag)
    # holdout is the chronological tail
    np.testing.assert_array_equal(ho_tag, np.arange(len(idx) - len(ho_idx), len(idx)))


# ── date-based reservation ───────────────────────────────────────────────
def test_date_based_boundary_bars() -> None:
    idx, ohlcv, tag = _panel()
    train_end = idx[79]  # exact bar date
    holdout_start = idx[80]
    (tr_idx, _, _), (ho_idx, _, _) = reserve_chronological_holdout(
        idx, ohlcv, tag, train_end=train_end, holdout_start=holdout_start
    )
    # a bar exactly on train_end is trainable; a bar exactly on holdout_start holds out
    assert idx[79] in tr_idx
    assert idx[80] in ho_idx
    assert tr_idx.max() == idx[79]
    assert ho_idx.min() == idx[80]
    assert tr_idx.max() < ho_idx.min()
    # adjacent dates → exact partition
    assert len(tr_idx) + len(ho_idx) == len(idx)


def test_date_based_gap_excluded_from_both() -> None:
    idx, ohlcv, tag = _panel()
    (tr_idx, _, tr_tag), (ho_idx, _, ho_tag) = reserve_chronological_holdout(
        idx, ohlcv, tag, train_end=idx[59], holdout_start=idx[70]
    )
    assert tr_idx.max() == idx[59]
    assert ho_idx.min() == idx[70]
    # bars in (train_end, holdout_start) belong to neither slice
    gap = idx[(idx > idx[59]) & (idx < idx[70])]
    assert len(gap) == 10
    assert len(tr_idx.intersection(gap)) == 0
    assert len(ho_idx.intersection(gap)) == 0
    assert len(tr_idx) + len(ho_idx) == len(idx) - len(gap)
    np.testing.assert_array_equal(tr_tag, np.arange(0, 60))
    np.testing.assert_array_equal(ho_tag, np.arange(70, len(idx)))


def test_date_based_holdout_end_clips_tail() -> None:
    idx, ohlcv, tag = _panel()
    (_, _, _), (ho_idx, _, ho_tag) = reserve_chronological_holdout(
        idx, ohlcv, tag,
        train_end=idx[79], holdout_start=idx[80], holdout_end=idx[99],
    )
    assert ho_idx.min() == idx[80]
    assert ho_idx.max() == idx[99]
    np.testing.assert_array_equal(ho_tag, np.arange(80, 100))
    # without holdout_end the holdout runs to the last bar
    (_, _, _), (ho_full, _, _) = reserve_chronological_holdout(
        idx, ohlcv, tag, train_end=idx[79], holdout_start=idx[80]
    )
    assert ho_full.max() == idx[-1]


# ── error behavior ───────────────────────────────────────────────────────
def test_empty_dataset_raises() -> None:
    empty = pd.DatetimeIndex([])
    with pytest.raises(ValueError, match="Empty dataset"):
        reserve_chronological_holdout(empty, np.zeros((0, 3)), holdout_days=30)


def test_nonpositive_holdout_days_raises() -> None:
    idx, ohlcv, tag = _panel()
    with pytest.raises(ValueError, match="holdout_days must be positive"):
        reserve_chronological_holdout(idx, ohlcv, tag, holdout_days=0)


def test_train_end_not_before_holdout_start_raises() -> None:
    idx, ohlcv, tag = _panel()
    with pytest.raises(ValueError, match="strictly before"):
        reserve_chronological_holdout(
            idx, ohlcv, tag, train_end=idx[80], holdout_start=idx[80]
        )


def test_date_based_requires_both_dates() -> None:
    idx, ohlcv, tag = _panel()
    with pytest.raises(ValueError, match="both train_end and holdout_start"):
        reserve_chronological_holdout(idx, ohlcv, tag, train_end=idx[80])
    with pytest.raises(ValueError, match="both train_end and holdout_start"):
        reserve_chronological_holdout(idx, ohlcv, tag, holdout_start=idx[80])


def test_empty_holdout_window_raises() -> None:
    idx, ohlcv, tag = _panel()
    # holdout entirely after the panel ends → no holdout rows
    with pytest.raises(ValueError, match="No rows in holdout"):
        reserve_chronological_holdout(
            idx, ohlcv, tag,
            train_end=idx[-1] + pd.Timedelta(days=10),
            holdout_start=idx[-1] + pd.Timedelta(days=20),
        )


def test_no_trainable_rows_raises() -> None:
    idx, ohlcv, tag = _panel()
    with pytest.raises(ValueError, match="No trainable rows"):
        reserve_chronological_holdout(
            idx, ohlcv, tag,
            train_end=idx[0] - pd.Timedelta(days=10),
            holdout_start=idx[0],
        )
