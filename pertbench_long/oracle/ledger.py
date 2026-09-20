"""SQLite ledger with atomic credit accounting and durable purchase phases."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL,
    budget INTEGER NOT NULL,
    charged INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS requests (
    run_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    price INTEGER NOT NULL,
    status TEXT NOT NULL,
    evidence_version INTEGER NOT NULL,
    artifact_json TEXT NOT NULL,
    purchase_phase TEXT NOT NULL DEFAULT 'AUTHORIZED',
    source_sha256 TEXT,
    staging_name TEXT,
    PRIMARY KEY (run_id, request_id)
);
CREATE TABLE IF NOT EXISTS purchases (
    run_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    artifact_json TEXT NOT NULL,
    price INTEGER NOT NULL,
    PRIMARY KEY (run_id, experiment_id)
);
"""

PHASE_AUTHORIZED = "AUTHORIZED"
PHASE_MATERIALIZED = "MATERIALIZED"
PHASE_DELIVERED = "DELIVERED"
PHASE_FAILED = "FAILED"


@dataclass
class LedgerSnapshot:
    run_id: str
    episode_id: str
    budget: int
    charged: int
    remaining: int
    status: str
    unique_purchases: int


class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(requests)").fetchall()}
        if "purchase_phase" not in cols:
            conn.execute("ALTER TABLE requests ADD COLUMN purchase_phase TEXT NOT NULL DEFAULT 'DELIVERED'")
        if "source_sha256" not in cols:
            conn.execute("ALTER TABLE requests ADD COLUMN source_sha256 TEXT")
        if "staging_name" not in cols:
            conn.execute("ALTER TABLE requests ADD COLUMN staging_name TEXT")

    def init_run(self, run_id: str, episode_id: str, budget: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT run_id, episode_id FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if existing:
                if existing["episode_id"] != episode_id:
                    conn.execute("ROLLBACK")
                    return
                conn.execute("COMMIT")
                return
            conn.execute(
                "INSERT INTO runs(run_id, episode_id, budget, charged, status) VALUES (?,?,?,?,?)",
                (run_id, episode_id, int(budget), 0, "RUNNING"),
            )
            conn.execute("COMMIT")

    def snapshot(self, run_id: str) -> LedgerSnapshot:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            n = conn.execute("SELECT COUNT(*) AS n FROM purchases WHERE run_id=?", (run_id,)).fetchone()["n"]
            return LedgerSnapshot(
                run_id=run_id,
                episode_id=row["episode_id"],
                budget=int(row["budget"]),
                charged=int(row["charged"]),
                remaining=int(row["budget"]) - int(row["charged"]),
                status=row["status"],
                unique_purchases=int(n),
            )

    def bound_episode_id(self, run_id: str) -> str:
        return self.snapshot(run_id).episode_id

    def set_status(self, run_id: str, status: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("UPDATE runs SET status=? WHERE run_id=?", (status, run_id))
            conn.execute("COMMIT")

    def get_request(self, run_id: str, request_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM requests WHERE run_id=? AND request_id=?",
                (run_id, request_id),
            ).fetchone()
            return dict(row) if row else None

    def get_purchase(self, run_id: str, experiment_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM purchases WHERE run_id=? AND experiment_id=?",
                (run_id, experiment_id),
            ).fetchone()
            return dict(row) if row else None

    def list_purchases(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM purchases WHERE run_id=?", (run_id,)).fetchall()
            return [dict(r) for r in rows]

    def authorize(
        self,
        *,
        run_id: str,
        episode_id: str,
        request_id: str,
        experiment_id: str,
        payload_hash: str,
        price: int,
        source_sha256: str,
        staging_name: str,
    ) -> dict[str, Any]:
        """Charge and record AUTHORIZED. No Agent-visible files are created here."""
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if run is None:
                conn.execute("ROLLBACK")
                return {"error": "INVALID_STATE", "message": "unknown run"}
            if run["episode_id"] != episode_id:
                conn.execute("ROLLBACK")
                return {"error": "INVALID_STATE", "message": "run is bound to a different episode"}
            if run["status"] != "RUNNING":
                conn.execute("ROLLBACK")
                return {"error": "INVALID_STATE", "status": run["status"]}
            existing_req = conn.execute(
                "SELECT * FROM requests WHERE run_id=? AND request_id=?",
                (run_id, request_id),
            ).fetchone()
            if existing_req:
                if existing_req["payload_hash"] != payload_hash:
                    conn.execute("ROLLBACK")
                    return {"error": "IDEMPOTENCY_CONFLICT"}
                phase = existing_req["purchase_phase"]
                if phase == PHASE_FAILED:
                    conn.execute("ROLLBACK")
                    return {
                        "error": "FAILED_REQUEST",
                        "message": "request_id previously failed and was refunded; retry with a new request_id",
                    }
                conn.execute("COMMIT")
                artifact = json.loads(existing_req["artifact_json"]) if existing_req["artifact_json"] else {}
                return {
                    "ok": True,
                    "replay": True,
                    "charged": 0,
                    "remaining": int(run["budget"]) - int(run["charged"]),
                    "evidence_version": int(existing_req["evidence_version"]),
                    "artifact": artifact,
                    "purchase_phase": phase,
                    "source_sha256": existing_req["source_sha256"],
                    "staging_name": existing_req["staging_name"],
                    "experiment_id": existing_req["experiment_id"],
                }
            owned = conn.execute(
                "SELECT * FROM purchases WHERE run_id=? AND experiment_id=?",
                (run_id, experiment_id),
            ).fetchone()
            charged_now = 0
            new_charged = int(run["charged"])
            artifact: dict[str, Any] = {}
            if owned is None:
                if int(run["charged"]) + int(price) > int(run["budget"]):
                    conn.execute("ROLLBACK")
                    return {"error": "BUDGET_EXCEEDED", "remaining": int(run["budget"]) - int(run["charged"])}
                new_charged = int(run["charged"]) + int(price)
                charged_now = int(price)
                conn.execute("UPDATE runs SET charged=? WHERE run_id=?", (new_charged, run_id))
                conn.execute(
                    "INSERT INTO purchases(run_id, experiment_id, artifact_json, price) VALUES (?,?,?,?)",
                    (run_id, experiment_id, json.dumps(artifact, sort_keys=True), int(price)),
                )
            else:
                artifact = json.loads(owned["artifact_json"]) if owned["artifact_json"] else {}
            evidence_version = conn.execute(
                "SELECT COUNT(*) AS n FROM purchases WHERE run_id=?",
                (run_id,),
            ).fetchone()["n"]
            conn.execute(
                "INSERT INTO requests(run_id, request_id, experiment_id, payload_hash, price, status, evidence_version, "
                "artifact_json, purchase_phase, source_sha256, staging_name) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    request_id,
                    experiment_id,
                    payload_hash,
                    int(price) if owned is None else 0,
                    "ok",
                    int(evidence_version),
                    json.dumps(artifact, sort_keys=True),
                    PHASE_AUTHORIZED,
                    source_sha256,
                    staging_name,
                ),
            )
            remaining = int(run["budget"]) - new_charged
            conn.execute("COMMIT")
            return {
                "ok": True,
                "replay": owned is not None,
                "charged": charged_now,
                "remaining": remaining,
                "evidence_version": int(evidence_version),
                "artifact": artifact,
                "purchase_phase": PHASE_AUTHORIZED,
                "source_sha256": source_sha256,
                "staging_name": staging_name,
                "experiment_id": experiment_id,
            }

    def mark_materialized(self, run_id: str, request_id: str, artifact: dict[str, Any]) -> None:
        artifact_json = json.dumps(artifact, sort_keys=True)
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            req = conn.execute(
                "SELECT * FROM requests WHERE run_id=? AND request_id=?",
                (run_id, request_id),
            ).fetchone()
            if req is None:
                conn.execute("ROLLBACK")
                raise KeyError(request_id)
            if req["purchase_phase"] == PHASE_FAILED:
                conn.execute("ROLLBACK")
                raise KeyError("cannot materialize a FAILED request")
            if req["purchase_phase"] not in {PHASE_AUTHORIZED, PHASE_MATERIALIZED}:
                conn.execute("ROLLBACK")
                raise KeyError(f"illegal phase {req['purchase_phase']} for materialize")
            owned = conn.execute(
                "SELECT * FROM purchases WHERE run_id=? AND experiment_id=?",
                (run_id, req["experiment_id"]),
            ).fetchone()
            if owned is None:
                conn.execute("ROLLBACK")
                raise KeyError("purchase row missing; refuse to reveal")
            conn.execute(
                "UPDATE requests SET purchase_phase=?, artifact_json=? WHERE run_id=? AND request_id=?",
                (PHASE_MATERIALIZED, artifact_json, run_id, request_id),
            )
            conn.execute(
                "UPDATE purchases SET artifact_json=? WHERE run_id=? AND experiment_id=?",
                (artifact_json, run_id, req["experiment_id"]),
            )
            conn.execute("COMMIT")

    def mark_delivered(self, run_id: str, request_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            req = conn.execute(
                "SELECT * FROM requests WHERE run_id=? AND request_id=?",
                (run_id, request_id),
            ).fetchone()
            if req is None:
                conn.execute("ROLLBACK")
                raise KeyError(request_id)
            if req["purchase_phase"] == PHASE_FAILED:
                conn.execute("ROLLBACK")
                raise KeyError("cannot deliver a FAILED request")
            if req["purchase_phase"] not in {PHASE_MATERIALIZED, PHASE_DELIVERED}:
                conn.execute("ROLLBACK")
                raise KeyError(f"illegal phase {req['purchase_phase']} for deliver")
            owned = conn.execute(
                "SELECT * FROM purchases WHERE run_id=? AND experiment_id=?",
                (run_id, req["experiment_id"]),
            ).fetchone()
            if owned is None:
                conn.execute("ROLLBACK")
                raise KeyError("purchase row missing; refuse to reveal")
            conn.execute(
                "UPDATE requests SET purchase_phase=? WHERE run_id=? AND request_id=?",
                (PHASE_DELIVERED, run_id, request_id),
            )
            run = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            conn.execute("COMMIT")
            return {
                "ok": True,
                "charged": 0,
                "remaining": int(run["budget"]) - int(run["charged"]),
                "evidence_version": int(req["evidence_version"]),
                "artifact": json.loads(req["artifact_json"]),
                "purchase_phase": PHASE_DELIVERED,
            }

    def fail_and_refund(self, run_id: str, request_id: str) -> None:
        """Refund at most once, and only if the request never reached MATERIALIZED/DELIVERED."""
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            req = conn.execute(
                "SELECT * FROM requests WHERE run_id=? AND request_id=?",
                (run_id, request_id),
            ).fetchone()
            if req is None:
                conn.execute("COMMIT")
                return
            phase = req["purchase_phase"]
            if phase == PHASE_FAILED:
                conn.execute("COMMIT")
                return
            if phase in {PHASE_MATERIALIZED, PHASE_DELIVERED}:
                conn.execute("COMMIT")
                return
            price = int(req["price"])
            if price > 0:
                run = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
                conn.execute("UPDATE runs SET charged=? WHERE run_id=?", (max(0, int(run["charged"]) - price), run_id))
                conn.execute(
                    "DELETE FROM purchases WHERE run_id=? AND experiment_id=?",
                    (run_id, req["experiment_id"]),
                )
            conn.execute(
                "UPDATE requests SET purchase_phase=?, status=? WHERE run_id=? AND request_id=?",
                (PHASE_FAILED, "failed", run_id, request_id),
            )
            conn.execute("COMMIT")

    def commit_purchase(
        self,
        *,
        run_id: str,
        request_id: str,
        experiment_id: str,
        payload_hash: str,
        price: int,
        artifact: dict[str, Any],
        already_owned: bool,
        episode_id: str | None = None,
        source_sha256: str = "",
        staging_name: str = "",
    ) -> dict[str, Any]:
        """Compatibility wrapper used by older tests: authorize + mark materialized + delivered."""
        snap = self.snapshot(run_id)
        auth = self.authorize(
            run_id=run_id,
            episode_id=episode_id or snap.episode_id,
            request_id=request_id,
            experiment_id=experiment_id,
            payload_hash=payload_hash,
            price=price,
            source_sha256=source_sha256 or artifact.get("sha256", ""),
            staging_name=staging_name or request_id,
        )
        if auth.get("error"):
            return auth
        if not auth.get("replay"):
            self.mark_materialized(run_id, request_id, artifact)
        else:
            stored = self.get_request(run_id, request_id)
            if stored and not stored.get("artifact_json"):
                self.mark_materialized(run_id, request_id, artifact)
        delivered = self.mark_delivered(run_id, request_id)
        delivered["charged"] = auth["charged"]
        delivered["replay"] = auth.get("replay", False) or already_owned
        delivered["remaining"] = auth["remaining"]
        if artifact:
            delivered["artifact"] = artifact
        return delivered
