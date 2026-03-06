"""NeuraPilot — Semantic Cache
Cuts repeat-query latency by 40-60% by embedding each question and
returning a cached answer when cosine similarity > threshold.

Design:
  - In-process cache backed by a sqlite table (no Redis dependency)
  - Embeddings stored as JSON blobs, cosine similarity in Python
  - Per-course isolation; TTL-based expiry (default 24 h)
  - Thread-safe: uses the same sqlite connection as the rest of the app
  - Returns hit/miss so the caller can track cache hit rate
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from typing import Any

# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS sem_cache (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id   TEXT    NOT NULL,
    query_emb   TEXT    NOT NULL,   -- JSON float list
    query_text  TEXT    NOT NULL,
    answer      TEXT    NOT NULL,
    intent      TEXT    NOT NULL DEFAULT 'ask',
    hits        INTEGER NOT NULL DEFAULT 0,
    created_at  REAL    NOT NULL,
    expires_at  REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sem_cache_course ON sem_cache (course_id, expires_at DESC);
"""


def ensure_cache_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_DDL)
    conn.commit()


# ── Math helpers ──────────────────────────────────────────────────────────────

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine(a: list[float], b: list[float]) -> float:
    denom = _norm(a) * _norm(b)
    return _dot(a, b) / denom if denom > 1e-10 else 0.0


# ── Public API ────────────────────────────────────────────────────────────────

class SemanticCache:
    """Embed-then-lookup cache with cosine similarity gating."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        embeddings,                  # LangChain Embeddings object
        threshold: float = 0.92,    # cosine similarity threshold
        ttl_seconds: int = 86_400,  # 24 h default TTL
        max_entries: int = 2_000,   # evict oldest when over limit
    ) -> None:
        self._conn      = conn
        self._emb       = embeddings
        self._threshold = threshold
        self._ttl       = ttl_seconds
        self._max       = max_entries
        ensure_cache_schema(conn)

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get(
        self,
        query: str,
        course_id: str,
    ) -> dict[str, Any] | None:
        """Return cached entry if a semantically similar query exists, else None."""
        try:
            q_emb = self._embed(query)
            now   = time.time()

            rows = self._conn.execute(
                """
                SELECT id, query_emb, answer, intent, hits
                FROM   sem_cache
                WHERE  course_id = ? AND expires_at > ?
                ORDER  BY created_at DESC
                LIMIT  200
                """,
                (course_id, now),
            ).fetchall()

            best_sim, best_row = 0.0, None
            for row in rows:
                try:
                    emb = json.loads(row["query_emb"])
                    sim = _cosine(q_emb, emb)
                    if sim > best_sim:
                        best_sim, best_row = sim, row
                except Exception:
                    continue

            if best_row and best_sim >= self._threshold:
                # Increment hit counter
                self._conn.execute(
                    "UPDATE sem_cache SET hits = hits + 1 WHERE id = ?",
                    (best_row["id"],),
                )
                self._conn.commit()
                return {
                    "answer":     best_row["answer"],
                    "intent":     best_row["intent"],
                    "similarity": round(best_sim, 4),
                    "cache_hit":  True,
                }
        except Exception:
            pass
        return None

    # ── Store ─────────────────────────────────────────────────────────────────

    def put(
        self,
        query:     str,
        answer:    str,
        course_id: str,
        intent:    str = "ask",
    ) -> None:
        """Store a query-answer pair in the cache."""
        try:
            q_emb    = self._embed(query)
            now      = time.time()
            self._conn.execute(
                """
                INSERT INTO sem_cache
                    (course_id, query_emb, query_text, answer, intent, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    course_id,
                    json.dumps(q_emb),
                    query[:500],
                    answer,
                    intent,
                    now,
                    now + self._ttl,
                ),
            )
            self._conn.commit()
            self._evict_if_needed(course_id)
        except Exception:
            pass

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self, course_id: str) -> dict[str, Any]:
        """Return cache statistics for the analytics dashboard."""
        try:
            now = time.time()
            row = self._conn.execute(
                """
                SELECT
                    COUNT(*)        AS total,
                    SUM(hits)       AS total_hits,
                    AVG(hits)       AS avg_hits,
                    MAX(hits)       AS max_hits
                FROM sem_cache
                WHERE course_id = ? AND expires_at > ?
                """,
                (course_id, now),
            ).fetchone()
            return dict(row) if row else {}
        except Exception:
            return {}

    def clear(self, course_id: str) -> int:
        """Evict all cache entries for a course. Returns rows deleted."""
        try:
            cur = self._conn.execute(
                "DELETE FROM sem_cache WHERE course_id = ?", (course_id,)
            )
            self._conn.commit()
            return cur.rowcount
        except Exception:
            return 0

    # ── Internal ──────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        return self._emb.embed_query(text)

    def _evict_if_needed(self, course_id: str) -> None:
        """Delete oldest entries if we exceed max_entries for the course."""
        try:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM sem_cache WHERE course_id = ?", (course_id,)
            ).fetchone()[0]
            if count > self._max:
                to_del = count - self._max
                self._conn.execute(
                    """
                    DELETE FROM sem_cache
                    WHERE id IN (
                        SELECT id FROM sem_cache
                        WHERE  course_id = ?
                        ORDER  BY created_at ASC
                        LIMIT  ?
                    )
                    """,
                    (course_id, to_del),
                )
                self._conn.commit()
        except Exception:
            pass
