"""Headless forward collector: ticks write execution/ caches without the UI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rlbot.forward_loop import (
    LoopLock,
    paper_day_due,
    rl_shadow_due,
    tick,
    write_status,
)


def test_paper_day_due_skips_same_session() -> None:
    assert paper_day_due(None, session="2026-08-17")
    assert paper_day_due("2026-08-16", session="2026-08-17")
    assert not paper_day_due("2026-08-17", session="2026-08-17")


def test_rl_shadow_due_weekdays_after_1800_et() -> None:
    monday_open = pd.Timestamp("2026-08-17 10:00", tz="America/New_York")
    monday_eve = pd.Timestamp("2026-08-17 18:15", tz="America/New_York")
    saturday = pd.Timestamp("2026-08-15 19:00", tz="America/New_York")
    assert not rl_shadow_due(monday_open, None)
    assert rl_shadow_due(monday_eve, None)
    assert not rl_shadow_due(monday_eve, "2026-08-17")
    assert not rl_shadow_due(saturday, None)
    assert rl_shadow_due(monday_open, "2026-08-17", force=True)
    assert not rl_shadow_due(saturday, None, force=True)


def test_shadow_ledger_needs_reopen_cash_reset(tmp_path: Path) -> None:
    from rlbot.forward_loop import shadow_ledger_needs_reopen

    path = tmp_path / "shadow_ledger_RLModel.jsonl"
    assert shadow_ledger_needs_reopen(path) is True
    path.write_text(
        '{"run_id": "RLModel", "target_weights": {"CASH": 1.0}, '
        '"note": "Reset to 100k cash (flat paper book)."}\n',
        encoding="utf-8",
    )
    assert shadow_ledger_needs_reopen(path) is True
    path.write_text(
        '{"target_weights": {"CASH": 0.08, "SP500": 0.18, "GOLD": 0.12}, "note": null}\n',
        encoding="utf-8",
    )
    assert shadow_ledger_needs_reopen(path) is False


def test_soft_rl_shadow_reopens_flat_ledger_before_close(tmp_path: Path, monkeypatch) -> None:
    recorded: list[str] = []

    def _run(*args, **kwargs):  # noqa: ANN001
        recorded.append("record")

        class _P:
            returncode = 0
            stderr = b""

        return _P()

    ledger = tmp_path / "execution" / "shadow_ledger_RLModel.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        '{"target_weights": {"CASH": 1.0}, "note": "Reset to 100k cash (flat paper book)."}\n',
        encoding="utf-8",
    )
    (tmp_path / "Runs" / "RLModel" / "models" / "best").mkdir(parents=True)
    (tmp_path / "Runs" / "RLModel" / "models" / "best" / "best_model.zip").write_bytes(b"x")

    monkeypatch.setattr("rlbot.forward_loop.subprocess.run", _run)
    from rlbot.forward_loop import _soft_rl_shadow

    out = _soft_rl_shadow(
        run_id="RLModel",
        now=pd.Timestamp("2026-08-18 15:30", tz="America/New_York"),
        last_shadow_date="2026-08-18",
        root=tmp_path,
    )
    assert out.get("recorded") is True
    assert recorded == ["record", "record"]  # record + reconcile



def test_tick_writes_status_and_skips_network(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_refresh(run_id, **kwargs):  # noqa: ANN001
        captured["run_id"] = run_id
        captured["root"] = kwargs.get("root")
        return {
            "run_id": run_id,
            "n_bars": 3,
            "live": {"last_price_bar": "2026-08-17 09:35", "prices_stale": False},
        }

    writes: list[str] = []

    def _fake_write(payload):  # noqa: ANN001
        writes.append(str(payload.get("run_id")))
        return tmp_path / f"forward_mark_{payload.get('run_id')}.json"

    monkeypatch.setattr("rlbot.forward_loop.refresh_forward_mark_live", _fake_refresh)
    monkeypatch.setattr("rlbot.forward_loop.write_forward_mark", _fake_write)
    monkeypatch.setattr("rlbot.forward_loop.publish_public_forward", lambda *a, **k: None)
    monkeypatch.setattr("rlbot.forward_loop.resolve_active_forward_run_id", lambda root=None: "GENERAL_EQUITY1")
    monkeypatch.setattr(
        "rlbot.forward_loop._soft_paper_ge1",
        lambda **kwargs: {"ok": True, "as_of": "2026-08-17", "actions": ["hold"]},
    )
    monkeypatch.setattr(
        "rlbot.forward_loop._soft_paper_crest",
        lambda **kwargs: {"ok": True, "skipped": "pack_missing"},
    )
    monkeypatch.setattr(
        "rlbot.forward_loop._soft_rl_shadow",
        lambda **kwargs: {"ok": True, "skipped": "not_due"},
    )

    status = tick(
        run_id="GENERAL_EQUITY1",
        root=tmp_path,
        now=pd.Timestamp("2026-08-17 10:32", tz="America/New_York"),
        run_paper=True,
        run_rl_shadow=True,
    )
    assert status["run_id"] == "GENERAL_EQUITY1"
    assert status["n_bars"] == 3
    assert status["paper_ge1_date"] == "2026-08-17"
    assert (tmp_path / "execution" / "forward_loop_status.json").is_file()
    assert captured["run_id"] == "GENERAL_EQUITY1"
    assert captured["root"] == tmp_path
    assert "GENERAL_EQUITY1" in writes
    assert (status.get("sleeves") or {}).get("GENERAL_EQUITY1", {}).get("ok") is True


def test_second_tick_skips_ge1_paper_same_session(tmp_path: Path, monkeypatch) -> None:
    calls = {"ge1": 0}

    def _ge1(**kwargs):  # noqa: ANN001
        calls["ge1"] += 1
        return {"ok": True, "as_of": "2026-08-17", "actions": ["rebalance"]}

    monkeypatch.setattr("rlbot.forward_loop.refresh_forward_mark_live", lambda *a, **k: {"run_id": "GENERAL_EQUITY1", "n_bars": 1, "live": {}})
    monkeypatch.setattr("rlbot.forward_loop.write_forward_mark", lambda payload: None)
    monkeypatch.setattr("rlbot.forward_loop.publish_public_forward", lambda *a, **k: None)
    monkeypatch.setattr("rlbot.forward_loop._soft_paper_ge1", _ge1)
    monkeypatch.setattr("rlbot.forward_loop._soft_paper_crest", lambda **kwargs: {"ok": True, "skipped": "pack_missing"})
    monkeypatch.setattr("rlbot.forward_loop._soft_rl_shadow", lambda **kwargs: {"ok": True, "skipped": "not_due"})

    now = pd.Timestamp("2026-08-17 11:00", tz="America/New_York")
    tick(run_id="GENERAL_EQUITY1", root=tmp_path, now=now)
    # Simulate the real skip path: status already recorded today's paper date.
    from rlbot.forward_loop import paper_day_due, read_status

    st = read_status(tmp_path)
    assert not paper_day_due(st.get("paper_ge1_date"), session="2026-08-17")

    def _ge1_skip(**kwargs):  # noqa: ANN001
        calls["ge1"] += 1
        from rlbot.forward_loop import paper_day_due as due

        if not due(kwargs.get("last_date"), session=kwargs.get("session")):
            return {"ok": True, "skipped": "already_today", "as_of": kwargs.get("last_date")}
        return {"ok": True, "as_of": "2026-08-17"}

    monkeypatch.setattr("rlbot.forward_loop._soft_paper_ge1", _ge1_skip)
    tick(run_id="GENERAL_EQUITY1", root=tmp_path, now=now)
    assert calls["ge1"] == 2  # second call still enters helper…
    # …but helper reports skip rather than a new as_of trade.
    st2 = read_status(tmp_path)
    assert st2["sleeves"]["GENERAL_EQUITY1"]["skipped"] == "already_today"


def test_loop_lock_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "forward_loop.lock"
    a = LoopLock(path)
    b = LoopLock(path)
    assert a.acquire(blocking=False)
    assert not b.acquire(blocking=False)
    a.release()
    assert b.acquire(blocking=False)
    b.release()


def test_write_status_roundtrip(tmp_path: Path) -> None:
    path = write_status({"schema": "test", "n_bars": 2}, tmp_path)
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "n_bars" in text
