"""SQLite persistence for analysis jobs and editor feedback."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, created_at REAL NOT NULL, title TEXT, source TEXT, result TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, promo_id TEXT NOT NULL,
  vote INTEGER NOT NULL, note TEXT, created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_job ON feedback(job_id);
"""


class Store:
    def __init__(self, path: str, keep_jobs: int = 300):
        self.path, self.keep_jobs = path, keep_jobs
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def save_job(self, title: str, source: str, result: dict) -> str:
        jid = uuid.uuid4().hex[:12]
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO jobs VALUES (?,?,?,?,?)",
                      (jid, time.time(), title, source, json.dumps(result, ensure_ascii=False)))
            c.execute("DELETE FROM jobs WHERE id NOT IN (SELECT id FROM jobs ORDER BY created_at DESC LIMIT ?)",
                      (self.keep_jobs,))
        return jid

    def get_job(self, jid: str):
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        if not row:
            return None
        return {"id": row["id"], "created_at": row["created_at"], "title": row["title"], "source": row["source"],
                "result": json.loads(row["result"])}

    def add_feedback(self, job_id: str, promo_id: str, vote: int, note: str | None) -> int:
        with self._lock, self._conn() as c:
            cur = c.execute("INSERT INTO feedback (job_id, promo_id, vote, note, created_at) VALUES (?,?,?,?,?)",
                            (job_id, promo_id, vote, (note or "")[:500], time.time()))
            return cur.lastrowid

    def feedback_summary(self, job_id: str | None = None) -> dict:
        q = "SELECT promo_id, SUM(CASE WHEN vote>0 THEN 1 ELSE 0 END) up, SUM(CASE WHEN vote<0 THEN 1 ELSE 0 END) down " \
            "FROM feedback {} GROUP BY promo_id"
        with self._conn() as c:
            if job_id:
                rows = c.execute(q.format("WHERE job_id=?"), (job_id,)).fetchall()
            else:
                rows = c.execute(q.format("")).fetchall()
            totals = c.execute("SELECT COUNT(*) n, SUM(vote>0) up, SUM(vote<0) down FROM feedback").fetchone()
        return {"by_promo": {r["promo_id"]: {"up": r["up"], "down": r["down"]} for r in rows},
                "total": {"votes": totals["n"] or 0, "up": totals["up"] or 0, "down": totals["down"] or 0}}
