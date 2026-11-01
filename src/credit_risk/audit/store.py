"""Append-only audit trail backed by SQLite (standard library, no extra deps).

Each decision is stored with its key fields indexed and the full decision record
as JSON, so any past decision can be reconstructed exactly: model version, PD,
decision, SHAP factors, counterfactuals, risk metrics, retrieved sources and the
explanation.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from credit_risk.config import Config, get_config


def _connect(cfg: Config) -> sqlite3.Connection:
    db_path = cfg.path("paths.audit_db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(cfg: Config | None = None) -> None:
    cfg = cfg or get_config()
    with _connect(cfg) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                audit_id      TEXT PRIMARY KEY,
                created_at    TEXT NOT NULL,
                model_version TEXT,
                decision      TEXT,
                pd            REAL,
                record_json   TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save(record: dict[str, Any], cfg: Config | None = None) -> str:
    cfg = cfg or get_config()
    init_db(cfg)
    audit_id = record.get("audit_id") or uuid.uuid4().hex[:12]
    record["audit_id"] = audit_id
    record.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    with _connect(cfg) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO decisions "
            "(audit_id, created_at, model_version, decision, pd, record_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                audit_id,
                record["created_at"],
                record.get("model_version"),
                record.get("decision"),
                record.get("pd"),
                json.dumps(record),
            ),
        )
        conn.commit()
    return audit_id


def get(audit_id: str, cfg: Config | None = None) -> dict[str, Any] | None:
    cfg = cfg or get_config()
    init_db(cfg)
    with _connect(cfg) as conn:
        row = conn.execute(
            "SELECT record_json FROM decisions WHERE audit_id = ?", (audit_id,)
        ).fetchone()
    return json.loads(row["record_json"]) if row else None


def list_recent(limit: int = 20, cfg: Config | None = None) -> list[dict[str, Any]]:
    cfg = cfg or get_config()
    init_db(cfg)
    with _connect(cfg) as conn:
        rows = conn.execute(
            "SELECT audit_id, created_at, decision, pd, model_version, record_json "
            "FROM decisions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        rec = json.loads(r["record_json"])
        applicant = rec.get("applicant", {})
        out.append({
            "audit_id": r["audit_id"],
            "created_at": r["created_at"],
            "decision": r["decision"],
            "pd": r["pd"],
            "model_version": r["model_version"],
            "amt_credit": applicant.get("AMT_CREDIT"),
            "amt_income": applicant.get("AMT_INCOME_TOTAL"),
            "expected_loss": rec.get("risk", {}).get("expected_loss"),
        })
    return out


def stats(cfg: Config | None = None) -> dict[str, Any]:
    """Decision counts across the whole audit trail."""
    cfg = cfg or get_config()
    init_db(cfg)
    with _connect(cfg) as conn:
        rows = conn.execute(
            "SELECT decision, COUNT(*) AS n, AVG(pd) AS mean_pd "
            "FROM decisions GROUP BY decision"
        ).fetchall()
    by_decision = {r["decision"]: {"n": r["n"], "mean_pd": r["mean_pd"]} for r in rows}
    return {"total": sum(v["n"] for v in by_decision.values()),
            "by_decision": by_decision}
