"""SQLite WAL cache for LLM strategy results.

Key: ``(question_hash, strategy, model_id, prompt_set_id)``.
Payload: the entire ``Attempt`` JSON for that question+strategy.

Crash-safety: WAL mode flushes one row per write; if the process is
killed, at most the in-flight call is lost.

This module is deliberately thread-safe-for-readers but assumes a single
writer process (i.e. one running ``annotate.py``).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / ".llm_cache.sqlite"


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS attempts (
    question_hash   TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    prompt_set_id   TEXT NOT NULL,
    payload         TEXT NOT NULL,
    created_at      REAL NOT NULL,
    PRIMARY KEY (question_hash, strategy, model_id, prompt_set_id)
);
"""


def question_hash(question: str) -> str:
    """Stable per-question hash. Matches data_plan.md §2 spec."""
    return hashlib.sha1(question.strip().lower().encode("utf-8")).hexdigest()


class AttemptCache:
    """Tiny single-process SQLite cache over LLM attempts."""

    def __init__(self, path: Path | str = DEFAULT_CACHE_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # SQLite connections are NOT safe to share across threads by default;
        # we use a single connection guarded by self._lock. The annotator is
        # asyncio + a single event loop, so a single conn is fine.
        self._conn = sqlite3.connect(
            str(self.path),
            isolation_level=None,  # autocommit
            check_same_thread=False,
        )
        self._conn.executescript(_SCHEMA_SQL)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA busy_timeout=5000;")
        except sqlite3.OperationalError as e:
            log.warning("Could not set WAL mode on %s: %s", self.path, e)
        # Counters for the QC report.
        self.n_hits = 0
        self.n_misses = 0

    # --- read ---

    def get(
        self,
        q_hash: str,
        strategy: str,
        model_id: str,
        prompt_set_id: str,
    ) -> dict | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT payload FROM attempts "
                "WHERE question_hash=? AND strategy=? AND model_id=? AND prompt_set_id=?",
                (q_hash, strategy, model_id, prompt_set_id),
            )
            row = cur.fetchone()
        if row is None:
            self.n_misses += 1
            return None
        self.n_hits += 1
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            log.warning("Corrupt cache payload for %s/%s — re-running.", q_hash, strategy)
            return None

    # --- write ---

    def put(
        self,
        q_hash: str,
        strategy: str,
        model_id: str,
        prompt_set_id: str,
        payload: dict[str, Any],
        *,
        created_at: float,
    ) -> None:
        blob = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO attempts "
                "(question_hash, strategy, model_id, prompt_set_id, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (q_hash, strategy, model_id, prompt_set_id, blob, created_at),
            )

    # --- stats ---

    @property
    def hit_rate(self) -> float:
        total = self.n_hits + self.n_misses
        return self.n_hits / total if total else 0.0

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._conn.close()

    def __enter__(self) -> "AttemptCache":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
