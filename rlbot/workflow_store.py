"""Transactional mandate workflow store for the product-facing API.

This module deliberately has no dependency on ``Runs/`` or training artifacts.
It models commercial workflow state; research execution remains controlled by
the existing CLI and OOS ledger.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKFLOW_STATES = (
    "draft",
    "preflight_passed",
    "quote_issued",
    "checkout",
    "payment_verified",
    "queued",
    "training",
    "validation",
    "governed_oos_evaluation",
    "released",
    "cancelled",
)

# Investor (or operator) may cancel before payment locks the mandate.
_CANCELLABLE_STATES = frozenset({
    "draft",
    "preflight_passed",
    "quote_issued",
    "checkout",
})

_TRANSITIONS = {
    "issue_quote": ("preflight_passed", "quote_issued"),
    "create_checkout": ("quote_issued", "checkout"),
    "queue_training": ("payment_verified", "queued"),
    "start_training": ("queued", "training"),
    "start_validation": ("training", "validation"),
    "authorize_oos_evaluation": ("validation", "governed_oos_evaluation"),
    "release_report": ("governed_oos_evaluation", "released"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class WorkflowActor:
    user_id: str
    org_id: str
    role: str

    @property
    def is_operator(self) -> bool:
        return self.role in {"operator", "admin"}


class WorkflowConflict(ValueError):
    pass


class WorkflowNotFound(LookupError):
    pass


class WorkflowForbidden(PermissionError):
    pass


class WorkflowStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS mandates (
                    id TEXT PRIMARY KEY,
                    org_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    state TEXT NOT NULL,
                    current_version_id TEXT NOT NULL,
                    assigned_operator TEXT,
                    quote_amount INTEGER,
                    payment_state TEXT NOT NULL DEFAULT 'unpaid',
                    eligibility_json TEXT,
                    run_plan_json TEXT,
                    release_json TEXT,
                    idempotency_key TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(org_id, idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS mandate_versions (
                    id TEXT PRIMARY KEY,
                    mandate_id TEXT NOT NULL REFERENCES mandates(id),
                    version INTEGER NOT NULL,
                    instruments_json TEXT NOT NULL,
                    configuration_json TEXT NOT NULL,
                    immutable INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    UNIQUE(mandate_id, version)
                );
                CREATE TABLE IF NOT EXISTS workflow_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mandate_id TEXT NOT NULL REFERENCES mandates(id),
                    actor_id TEXT NOT NULL,
                    actor_role TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    from_state TEXT,
                    to_state TEXT,
                    detail_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_mandates_org_updated
                    ON mandates(org_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_events_mandate
                    ON workflow_events(mandate_id, id);
                """
            )
            mandate_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(mandates)").fetchall()
            }
            if "run_plan_json" not in mandate_columns:
                connection.execute("ALTER TABLE mandates ADD COLUMN run_plan_json TEXT")
            if "release_json" not in mandate_columns:
                connection.execute("ALTER TABLE mandates ADD COLUMN release_json TEXT")

    def submit(
        self,
        actor: WorkflowActor,
        *,
        idempotency_key: str,
        name: str,
        instruments: list[dict[str, Any]],
        configuration: dict[str, Any],
    ) -> dict[str, Any]:
        if actor.is_operator:
            raise WorkflowForbidden("Operator accounts cannot submit investor mandates")
        now = _utc_now()
        mandate_id = f"mand_{uuid.uuid4().hex}"
        version_id = f"{mandate_id}_v1"
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id FROM mandates WHERE org_id = ? AND idempotency_key = ?",
                (actor.org_id, idempotency_key),
            ).fetchone()
            if existing:
                return self.get(actor, str(existing["id"]))
            connection.execute(
                """
                INSERT INTO mandates (
                    id, org_id, owner_id, name, state, current_version_id,
                    idempotency_key, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'draft', ?, ?, ?, ?)
                """,
                (mandate_id, actor.org_id, actor.user_id, name, version_id, idempotency_key, now, now),
            )
            connection.execute(
                """
                INSERT INTO mandate_versions (
                    id, mandate_id, version, instruments_json,
                    configuration_json, created_at
                ) VALUES (?, ?, 1, ?, ?, ?)
                """,
                (
                    version_id,
                    mandate_id,
                    json.dumps(instruments, separators=(",", ":")),
                    json.dumps(configuration, separators=(",", ":")),
                    now,
                ),
            )
            self._event(
                connection,
                mandate_id,
                actor,
                "mandate_submitted",
                None,
                "draft",
                {"version": 1},
                now,
            )
        return self.get(actor, mandate_id)

    def list(self, actor: WorkflowActor) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if actor.is_operator:
                rows = connection.execute("SELECT id FROM mandates ORDER BY updated_at DESC").fetchall()
            else:
                rows = connection.execute(
                    "SELECT id FROM mandates WHERE org_id = ? ORDER BY updated_at DESC",
                    (actor.org_id,),
                ).fetchall()
        return [self.get(actor, str(row["id"])) for row in rows]

    def get(self, actor: WorkflowActor, mandate_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT m.*, v.version, v.instruments_json, v.configuration_json, v.immutable
                FROM mandates m
                JOIN mandate_versions v ON v.id = m.current_version_id
                WHERE m.id = ?
                """,
                (mandate_id,),
            ).fetchone()
            if row is None:
                raise WorkflowNotFound("Unknown mandate")
            if not actor.is_operator and row["org_id"] != actor.org_id:
                raise WorkflowNotFound("Unknown mandate")
            events = connection.execute(
                """
                SELECT actor_id, actor_role, event_type, from_state, to_state, detail_json, created_at
                FROM workflow_events WHERE mandate_id = ? ORDER BY id
                """,
                (mandate_id,),
            ).fetchall()
        return self._serialize(row, events)

    def record_preflight(
        self,
        actor: WorkflowActor,
        mandate_id: str,
        report: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self._require_operator(actor)
        passed = bool(report) and all(bool(item.get("eligible")) for item in report)
        with self._connect() as connection:
            row = self._locked_row(connection, actor, mandate_id)
            if row["state"] not in {"draft", "preflight_passed"}:
                raise WorkflowConflict(f"Preflight is not allowed from {row['state']}")
            next_state = "preflight_passed" if passed else "draft"
            now = _utc_now()
            connection.execute(
                """
                UPDATE mandates
                SET state = ?, eligibility_json = ?,
                    assigned_operator = COALESCE(assigned_operator, ?), updated_at = ?
                WHERE id = ?
                """,
                (
                    next_state,
                    json.dumps(report, separators=(",", ":")),
                    actor.user_id,
                    now,
                    mandate_id,
                ),
            )
            self._event(
                connection,
                mandate_id,
                actor,
                "preflight_passed" if passed else "preflight_failed",
                str(row["state"]),
                next_state,
                {"eligible": passed, "checks": len(report)},
                now,
            )
        return self.get(actor, mandate_id)

    def transition(
        self,
        actor: WorkflowActor,
        mandate_id: str,
        action: str,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require_operator(actor)
        if action not in _TRANSITIONS:
            raise WorkflowConflict(f"Unknown or controlled action: {action}")
        expected, next_state = _TRANSITIONS[action]
        with self._connect() as connection:
            row = self._locked_row(connection, actor, mandate_id)
            if row["state"] != expected:
                raise WorkflowConflict(f"{action} requires {expected}; current state is {row['state']}")
            now = _utc_now()
            quote_amount = row["quote_amount"]
            run_plan_json = row["run_plan_json"]
            release_json = row["release_json"]
            if action == "issue_quote":
                version = connection.execute(
                    "SELECT instruments_json, configuration_json FROM mandate_versions WHERE id = ?",
                    (row["current_version_id"],),
                ).fetchone()
                assert version is not None
                instruments = json.loads(version["instruments_json"])
                configuration = json.loads(version["configuration_json"])
                quote_amount = self._quote(len(instruments), configuration)
            elif action == "queue_training":
                cohort_id = f"{mandate_id}_v{self._version_number(connection, row['current_version_id'])}"
                run_plan_json = json.dumps(
                    {
                        "cohortId": cohort_id,
                        "windows": [1, 2, 3, 4, 5],
                        "seeds": [42, 101, 777],
                        "totalJobs": 15,
                        "status": "materialized",
                    },
                    separators=(",", ":"),
                )
            elif action == "release_report":
                release_json = json.dumps(detail or {}, separators=(",", ":"))
            connection.execute(
                """
                UPDATE mandates
                SET state = ?, quote_amount = ?, assigned_operator = COALESCE(assigned_operator, ?),
                    run_plan_json = ?, release_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    next_state,
                    quote_amount,
                    actor.user_id,
                    run_plan_json,
                    release_json,
                    now,
                    mandate_id,
                ),
            )
            self._event(
                connection,
                mandate_id,
                actor,
                action,
                expected,
                next_state,
                detail or {},
                now,
            )
        return self.get(actor, mandate_id)

    def cancel(
        self,
        actor: WorkflowActor,
        mandate_id: str,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Cancel a pre-payment mandate. Investor (own org) or operator."""
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM mandates WHERE id = ?", (mandate_id,)).fetchone()
            if row is None:
                raise WorkflowNotFound("Unknown mandate")
            if not actor.is_operator and row["org_id"] != actor.org_id:
                raise WorkflowNotFound("Unknown mandate")
            state = str(row["state"])
            if state == "cancelled":
                return self.get(actor, mandate_id)
            if state not in _CANCELLABLE_STATES:
                raise WorkflowConflict(
                    f"Cancel is only allowed before payment verification; current state is {state}"
                )
            now = _utc_now()
            connection.execute(
                "UPDATE mandates SET state = 'cancelled', updated_at = ? WHERE id = ?",
                (now, mandate_id),
            )
            self._event(
                connection,
                mandate_id,
                actor,
                "cancel",
                state,
                "cancelled",
                detail or {},
                now,
            )
        return self.get(actor, mandate_id)

    def verify_payment(self, mandate_id: str, provider_event_id: str) -> dict[str, Any]:
        webhook_actor = WorkflowActor("payment_webhook", "system", "system")
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM mandates WHERE id = ?", (mandate_id,)).fetchone()
            if row is None:
                raise WorkflowNotFound("Unknown mandate")
            if row["state"] == "payment_verified":
                pass
            elif row["state"] != "checkout":
                raise WorkflowConflict(f"Payment verification requires checkout; current state is {row['state']}")
            else:
                now = _utc_now()
                connection.execute(
                    """
                    UPDATE mandates SET state = 'payment_verified', payment_state = 'verified',
                        updated_at = ? WHERE id = ?
                    """,
                    (now, mandate_id),
                )
                connection.execute(
                    "UPDATE mandate_versions SET immutable = 1 WHERE id = ?",
                    (row["current_version_id"],),
                )
                self._event(
                    connection,
                    mandate_id,
                    webhook_actor,
                    "payment_verified",
                    "checkout",
                    "payment_verified",
                    {"provider_event_id": provider_event_id},
                    now,
                )
        return self.get(WorkflowActor("system", str(row["org_id"]), "admin"), mandate_id)

    @staticmethod
    def _quote(instrument_count: int, configuration: dict[str, Any]) -> int:
        risk = str(configuration.get("riskPreference") or "balanced")
        risk_addon = {"defensive": 0, "balanced": 1500, "growth": 2500}.get(risk, 1500)
        return 12_500 + instrument_count * 850 + risk_addon

    @staticmethod
    def _version_number(connection: sqlite3.Connection, version_id: str) -> int:
        row = connection.execute(
            "SELECT version FROM mandate_versions WHERE id = ?",
            (version_id,),
        ).fetchone()
        return int(row["version"]) if row is not None else 1

    @staticmethod
    def _require_operator(actor: WorkflowActor) -> None:
        if not actor.is_operator:
            raise WorkflowForbidden("Research Operations role required")

    @staticmethod
    def _locked_row(
        connection: sqlite3.Connection,
        actor: WorkflowActor,
        mandate_id: str,
    ) -> sqlite3.Row:
        WorkflowStore._require_operator(actor)
        row = connection.execute("SELECT * FROM mandates WHERE id = ?", (mandate_id,)).fetchone()
        if row is None:
            raise WorkflowNotFound("Unknown mandate")
        return row

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        mandate_id: str,
        actor: WorkflowActor,
        event_type: str,
        from_state: str | None,
        to_state: str | None,
        detail: dict[str, Any],
        created_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO workflow_events (
                mandate_id, actor_id, actor_role, event_type,
                from_state, to_state, detail_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mandate_id,
                actor.user_id,
                actor.role,
                event_type,
                from_state,
                to_state,
                json.dumps(detail, separators=(",", ":")),
                created_at,
            ),
        )

    @staticmethod
    def _serialize(row: sqlite3.Row, events: list[sqlite3.Row]) -> dict[str, Any]:
        state = str(row["state"])
        allowed = [
            action
            for action, (expected, _) in _TRANSITIONS.items()
            if expected == state
        ]
        if state == "draft":
            allowed = ["run_preflight"]
        if state in _CANCELLABLE_STATES:
            allowed = [*allowed, "cancel"]
        return {
            "id": row["id"],
            "organizationId": row["org_id"],
            "ownerId": row["owner_id"],
            "name": row["name"],
            "state": state,
            "version": row["version"],
            "versionId": row["current_version_id"],
            "immutable": bool(row["immutable"]),
            "assignedOperator": row["assigned_operator"],
            "quoteAmount": row["quote_amount"],
            "paymentState": row["payment_state"],
            "instruments": json.loads(row["instruments_json"]),
            "configuration": json.loads(row["configuration_json"]),
            "eligibility": json.loads(row["eligibility_json"]) if row["eligibility_json"] else [],
            "runPlan": json.loads(row["run_plan_json"]) if row["run_plan_json"] else None,
            "release": json.loads(row["release_json"]) if row["release_json"] else None,
            "allowedActions": allowed,
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
            "auditLog": [
                {
                    "actorId": event["actor_id"],
                    "actorRole": event["actor_role"],
                    "eventType": event["event_type"],
                    "fromState": event["from_state"],
                    "toState": event["to_state"],
                    "detail": json.loads(event["detail_json"]),
                    "createdAt": event["created_at"],
                }
                for event in events
            ],
        }
