from pathlib import Path

import pytest

from rlbot.workflow_store import (
    WorkflowActor,
    WorkflowConflict,
    WorkflowNotFound,
    WorkflowStore,
)


@pytest.fixture
def store(tmp_path: Path) -> WorkflowStore:
    return WorkflowStore(tmp_path / "workflow.sqlite3")


@pytest.fixture
def instruments() -> list[dict[str, str]]:
    return [
        {"name": f"Asset {index}", "ticker": f"T{index}", "group": "Equity"}
        for index in range(5)
    ]


def test_submission_is_server_owned_and_idempotent(
    store: WorkflowStore,
    instruments: list[dict[str, str]],
) -> None:
    investor = WorkflowActor("user_1", "org_1", "investor")
    first = store.submit(
        investor,
        idempotency_key="request-123",
        name="Northstar",
        instruments=instruments,
        configuration={"maxWeight": 20, "riskPreference": "balanced"},
    )
    repeated = store.submit(
        investor,
        idempotency_key="request-123",
        name="Ignored duplicate",
        instruments=instruments,
        configuration={"maxWeight": 40, "riskPreference": "growth"},
    )

    assert repeated["id"] == first["id"]
    assert first["state"] == "draft"
    assert first["quoteAmount"] is None
    assert first["paymentState"] == "unpaid"
    assert first["version"] == 1
    assert not first["immutable"]


def test_tenant_isolation_hides_other_organizations(
    store: WorkflowStore,
    instruments: list[dict[str, str]],
) -> None:
    owner = WorkflowActor("owner", "org_1", "investor")
    outsider = WorkflowActor("other", "org_2", "investor")
    request = store.submit(
        owner,
        idempotency_key="tenant-test",
        name="Private mandate",
        instruments=instruments,
        configuration={"maxWeight": 20, "riskPreference": "balanced"},
    )

    assert store.list(outsider) == []
    with pytest.raises(WorkflowNotFound):
        store.get(outsider, request["id"])


def test_controlled_lifecycle_and_payment_immutability(
    store: WorkflowStore,
    instruments: list[dict[str, str]],
) -> None:
    investor = WorkflowActor("owner", "org_1", "investor")
    operator = WorkflowActor("operator", "ops", "operator")
    request = store.submit(
        investor,
        idempotency_key="lifecycle-test",
        name="Northstar",
        instruments=instruments,
        configuration={"maxWeight": 20, "riskPreference": "defensive"},
    )
    mandate_id = request["id"]

    report = [{"ticker": item["ticker"], "eligible": True} for item in instruments]
    request = store.record_preflight(operator, mandate_id, report)
    assert request["state"] == "preflight_passed"

    request = store.transition(operator, mandate_id, "issue_quote")
    assert request["state"] == "quote_issued"
    assert request["quoteAmount"] == 16_750

    request = store.transition(operator, mandate_id, "create_checkout")
    assert request["state"] == "checkout"
    request = store.verify_payment(mandate_id, "evt_123")
    assert request["state"] == "payment_verified"
    assert request["paymentState"] == "verified"
    assert request["immutable"]

    request = store.transition(operator, mandate_id, "queue_training")
    request = store.transition(operator, mandate_id, "start_training")
    request = store.transition(operator, mandate_id, "start_validation")
    request = store.transition(operator, mandate_id, "authorize_oos_evaluation")
    request = store.transition(operator, mandate_id, "release_report")
    assert request["state"] == "released"
    assert [event["eventType"] for event in request["auditLog"]][-1] == "release_report"


def test_investor_can_cancel_before_payment(
    store: WorkflowStore,
    instruments: list[dict[str, str]],
) -> None:
    investor = WorkflowActor("owner", "org_1", "investor")
    outsider = WorkflowActor("other", "org_2", "investor")
    request = store.submit(
        investor,
        idempotency_key="cancel-test",
        name="Northstar",
        instruments=instruments,
        configuration={"maxWeight": 20, "riskPreference": "balanced"},
    )
    assert "cancel" in request["allowedActions"]

    with pytest.raises(WorkflowNotFound):
        store.cancel(outsider, request["id"])

    cancelled = store.cancel(investor, request["id"], {"reason": "changed mind"})
    assert cancelled["state"] == "cancelled"
    assert cancelled["allowedActions"] == []
    assert cancelled["auditLog"][-1]["eventType"] == "cancel"

    # Idempotent once cancelled.
    assert store.cancel(investor, request["id"])["state"] == "cancelled"


def test_cancel_rejected_after_payment(
    store: WorkflowStore,
    instruments: list[dict[str, str]],
) -> None:
    investor = WorkflowActor("owner", "org_1", "investor")
    operator = WorkflowActor("operator", "ops", "operator")
    request = store.submit(
        investor,
        idempotency_key="cancel-paid",
        name="Northstar",
        instruments=instruments,
        configuration={"maxWeight": 20, "riskPreference": "balanced"},
    )
    mandate_id = request["id"]
    report = [{"ticker": item["ticker"], "eligible": True} for item in instruments]
    store.record_preflight(operator, mandate_id, report)
    store.transition(operator, mandate_id, "issue_quote")
    store.transition(operator, mandate_id, "create_checkout")
    store.verify_payment(mandate_id, "evt_paid")
    with pytest.raises(WorkflowConflict):
        store.cancel(investor, mandate_id)


def test_transition_rejects_out_of_order_action(
    store: WorkflowStore,
    instruments: list[dict[str, str]],
) -> None:
    investor = WorkflowActor("owner", "org_1", "investor")
    operator = WorkflowActor("operator", "ops", "operator")
    request = store.submit(
        investor,
        idempotency_key="bad-order",
        name="Northstar",
        instruments=instruments,
        configuration={"maxWeight": 20, "riskPreference": "balanced"},
    )
    with pytest.raises(WorkflowConflict):
        store.transition(operator, request["id"], "queue_training")
