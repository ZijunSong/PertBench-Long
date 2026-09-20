"""SQLite ledger with atomic credit accounting."""

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

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init_run(self, run_id: str, episode_id: str, budget: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT run_id FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if existing:
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
    ) -> dict[str, Any]:
        artifact_json = json.dumps(artifact, sort_keys=True)
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if run is None:
                conn.execute("ROLLBACK")
                raise KeyError(run_id)
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
                conn.execute("COMMIT")
                return {
                    "ok": True,
                    "charged": 0,
                    "remaining": int(run["budget"]) - int(run["charged"]),
                    "evidence_version": int(existing_req["evidence_version"]),
                    "artifact": json.loads(existing_req["artifact_json"]),
                    "replay": True,
                }
            owned = conn.execute(
                "SELECT * FROM purchases WHERE run_id=? AND experiment_id=?",
                (run_id, experiment_id),
            ).fetchone()
            charged_now = 0
            new_charged = int(run["charged"])
            if owned is None:
                if int(run["charged"]) + int(price) > int(run["budget"]):
                    conn.execute("ROLLBACK")
                    return {"error": "BUDGET_EXCEEDED", "remaining": int(run["budget"]) - int(run["charged"])}
                new_charged = int(run["charged"]) + int(price)
                charged_now = int(price)
                conn.execute(
                    "UPDATE runs SET charged=? WHERE run_id=?",
                    (new_charged, run_id),
                )
                conn.execute(
                    "INSERT INTO purchases(run_id, experiment_id, artifact_json, price) VALUES (?,?,?,?)",
                    (run_id, experiment_id, artifact_json, int(price)),
                )
            evidence_version = conn.execute(
                "SELECT COUNT(*) AS n FROM purchases WHERE run_id=?",
                (run_id,),
            ).fetchone()["n"]
            if owned is not None:
                artifact = json.loads(owned["artifact_json"])
                artifact_json = owned["artifact_json"]
                evidence_version = conn.execute(
                    "SELECT COUNT(*) AS n FROM purchases WHERE run_id=?",
                    (run_id,),
                ).fetchone()["n"]
            conn.execute(
                "INSERT INTO requests(run_id, request_id, experiment_id, payload_hash, price, status, evidence_version, artifact_json) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    request_id,
                    experiment_id,
                    payload_hash,
                    int(price) if owned is None else 0,
                    "ok",
                    int(evidence_version),
                    artifact_json,
                ),
            )
            remaining = int(run["budget"]) - new_charged
            conn.execute("COMMIT")
            return {
                "ok": True,
                "charged": charged_now,
                "remaining": remaining,
                "evidence_version": int(evidence_version),
                "artifact": json.loads(artifact_json),
                "replay": already_owned or owned is not None,
            }
