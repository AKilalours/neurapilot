"""NeuraPilot SQLite persistence layer.

Design decisions:
- WAL journal mode for concurrent reads without blocking writes.
- PRAGMA synchronous=NORMAL — durable enough for local data, fast enough for UX.
- All mutations go through typed helpers (no raw SQL outside this module).
- Schema versioning table for safe future migrations.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from neurapilot.config import Settings


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS courses (
    course_id   TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS interactions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           INTEGER NOT NULL,
    course_id    TEXT NOT NULL,
    strict       INTEGER NOT NULL DEFAULT 1,
    user_text    TEXT NOT NULL,
    intent       TEXT NOT NULL DEFAULT '',
    topic        TEXT NOT NULL DEFAULT '',
    output       TEXT NOT NULL DEFAULT '',
    sources_json TEXT NOT NULL DEFAULT '[]',
    latency_ms   INTEGER NOT NULL DEFAULT 0,
    -- Evaluation scores (RAGAS-style)
    faithfulness       REAL,
    answer_relevance   REAL,
    context_precision  REAL
);

CREATE TABLE IF NOT EXISTS mastery (
    course_id  TEXT NOT NULL,
    topic      TEXT NOT NULL,
    alpha      REAL NOT NULL DEFAULT 1.0,
    beta       REAL NOT NULL DEFAULT 1.0,
    attempts   INTEGER NOT NULL DEFAULT 0,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (course_id, topic)
);

CREATE TABLE IF NOT EXISTS flashcards (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id       TEXT NOT NULL,
    topic           TEXT NOT NULL DEFAULT '',
    question        TEXT NOT NULL,
    answer          TEXT NOT NULL,
    citations_json  TEXT NOT NULL DEFAULT '[]',
    difficulty      TEXT NOT NULL DEFAULT 'medium',
    bloom_level     TEXT NOT NULL DEFAULT 'remember',
    -- SM-2 scheduling fields
    ease            REAL NOT NULL DEFAULT 2.5,
    interval_days   INTEGER NOT NULL DEFAULT 0,
    reps            INTEGER NOT NULL DEFAULT 0,
    due_ts          INTEGER NOT NULL,
    created_at      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL,
    metric       TEXT NOT NULL,
    score        REAL NOT NULL,
    ts           INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS threads (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id   TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT 'New Chat',
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   INTEGER NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL DEFAULT '',
    meta_json   TEXT NOT NULL DEFAULT '{}',
    ts          INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS bookmarks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id   TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    content     TEXT NOT NULL DEFAULT '',
    tag         TEXT NOT NULL DEFAULT '',
    created_at  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_interactions_course ON interactions (course_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_flashcards_due      ON flashcards (course_id, due_ts ASC);
CREATE INDEX IF NOT EXISTS idx_mastery_course      ON mastery (course_id);
CREATE INDEX IF NOT EXISTS idx_messages_thread     ON messages (thread_id, ts ASC);
CREATE INDEX IF NOT EXISTS idx_threads_course      ON threads (course_id, updated_at DESC);
"""


# ── Connection factory ────────────────────────────────────────────────────────


def connect(settings: Settings) -> sqlite3.Connection:
    """Open (or create) the SQLite database with optimal pragmas."""
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.executescript(_SCHEMA)
    _ensure_schema_version(conn)
    return conn


def _ensure_schema_version(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version(version) VALUES(?)", (_SCHEMA_VERSION,))
        conn.commit()


# ── Courses ───────────────────────────────────────────────────────────────────


def upsert_course(
    conn: sqlite3.Connection,
    course_id: str,
    title: str,
    description: str = "",
) -> None:
    now = int(time.time())
    conn.execute(
        """
        INSERT INTO courses(course_id, title, description, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(course_id) DO UPDATE SET
            title       = excluded.title,
            description = excluded.description,
            updated_at  = excluded.updated_at
        """,
        (course_id, title, description, now, now),
    )
    conn.commit()


def list_courses(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT course_id, title, description, created_at FROM courses ORDER BY created_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


# ── Interactions ──────────────────────────────────────────────────────────────


def log_interaction(
    conn: sqlite3.Connection,
    *,
    course_id: str,
    strict: bool,
    user_text: str,
    intent: str,
    topic: str,
    output: str,
    sources: list[dict[str, Any]],
    latency_ms: int,
    faithfulness: float | None = None,
    answer_relevance: float | None = None,
    context_precision: float | None = None,
) -> int:
    """Insert an interaction record and return its row id."""
    now = int(time.time())
    cur = conn.execute(
        """
        INSERT INTO interactions(
            ts, course_id, strict, user_text, intent, topic, output,
            sources_json, latency_ms, faithfulness, answer_relevance, context_precision
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now,
            course_id,
            1 if strict else 0,
            user_text,
            intent,
            topic,
            output,
            json.dumps(sources, ensure_ascii=False),
            int(latency_ms),
            faithfulness,
            answer_relevance,
            context_precision,
        ),
    )
    conn.commit()
    return cur.lastrowid or 0


def get_recent_interactions(
    conn: sqlite3.Connection,
    course_id: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, ts, user_text, intent, topic, latency_ms,
               faithfulness, answer_relevance, context_precision
        FROM interactions
        WHERE course_id = ?
        ORDER BY ts DESC
        LIMIT ?
        """,
        (course_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


# ── Mastery (Bayesian Beta model) ─────────────────────────────────────────────


def update_mastery(
    conn: sqlite3.Connection,
    course_id: str,
    topic: str,
    correct: bool,
) -> None:
    """Update Beta(alpha, beta) mastery model for a topic.

    We use a simple conjugate prior:
    - P(mastery) starts at Beta(1,1) = Uniform
    - Each correct answer: alpha += 1
    - Each incorrect answer: beta += 1
    - P_mastery = alpha / (alpha + beta)
    """
    now = int(time.time())
    row = conn.execute(
        "SELECT alpha, beta, attempts FROM mastery WHERE course_id=? AND topic=?",
        (course_id, topic),
    ).fetchone()
    if row is None:
        alpha, beta, attempts = 1.0, 1.0, 0
    else:
        alpha, beta, attempts = float(row["alpha"]), float(row["beta"]), int(row["attempts"])

    if correct:
        alpha += 1.0
    else:
        beta += 1.0
    attempts += 1

    conn.execute(
        """
        INSERT INTO mastery(course_id, topic, alpha, beta, attempts, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(course_id, topic) DO UPDATE SET
            alpha      = excluded.alpha,
            beta       = excluded.beta,
            attempts   = excluded.attempts,
            updated_at = excluded.updated_at
        """,
        (course_id, topic, alpha, beta, attempts, now),
    )
    conn.commit()


def get_mastery(conn: sqlite3.Connection, course_id: str) -> list[dict[str, Any]]:
    """Return mastery data with computed P_mastery for each topic."""
    rows = conn.execute(
        "SELECT topic, alpha, beta, attempts, updated_at FROM mastery WHERE course_id=? ORDER BY updated_at DESC",
        (course_id,),
    ).fetchall()
    result = []
    for r in rows:
        alpha, beta = float(r["alpha"]), float(r["beta"])
        p = alpha / (alpha + beta)
        # 95% credible interval half-width using Normal approximation
        n = alpha + beta - 2  # effective samples
        ci_half = 1.96 * (p * (1 - p) / max(n, 1)) ** 0.5 if n > 0 else 0.5
        result.append({
            "topic": r["topic"],
            "p_mastery": round(p, 3),
            "ci_lower": round(max(0.0, p - ci_half), 3),
            "ci_upper": round(min(1.0, p + ci_half), 3),
            "attempts": r["attempts"],
            "updated_at": r["updated_at"],
        })
    return result


# ── Flashcards ────────────────────────────────────────────────────────────────


def add_flashcards(
    conn: sqlite3.Connection,
    course_id: str,
    topic: str,
    cards: list[dict[str, Any]],
) -> int:
    """Insert flashcards; returns count of cards successfully stored."""
    now = int(time.time())
    count = 0
    for card in cards:
        q = str(card.get("q", "")).strip()
        a = str(card.get("a", "")).strip()
        if not q or not a:
            continue
        conn.execute(
            """
            INSERT INTO flashcards(
                course_id, topic, question, answer, citations_json,
                difficulty, bloom_level, ease, interval_days, reps, due_ts, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                course_id,
                topic,
                q,
                a,
                json.dumps(card.get("citations", []), ensure_ascii=False),
                str(card.get("difficulty", "medium")),
                str(card.get("bloom_level", "remember")),
                2.5,   # initial ease factor (SM-2 standard)
                0,     # first interval
                0,     # reps
                now,   # due immediately for new cards
                now,
            ),
        )
        count += 1
    conn.commit()
    return count


def get_due_flashcards(
    conn: sqlite3.Connection,
    course_id: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return flashcards due for review (SM-2 scheduling)."""
    now = int(time.time())
    rows = conn.execute(
        """
        SELECT id, topic, question, answer, citations_json,
               difficulty, bloom_level, ease, interval_days, reps, due_ts
        FROM flashcards
        WHERE course_id = ? AND due_ts <= ?
        ORDER BY due_ts ASC
        LIMIT ?
        """,
        (course_id, now, limit),
    ).fetchall()
    return [
        {
            "id": r["id"],
            "topic": r["topic"],
            "question": r["question"],
            "answer": r["answer"],
            "citations": json.loads(r["citations_json"] or "[]"),
            "difficulty": r["difficulty"],
            "bloom_level": r["bloom_level"],
            "ease": float(r["ease"]),
            "interval_days": int(r["interval_days"]),
            "reps": int(r["reps"]),
            "due_ts": int(r["due_ts"]),
        }
        for r in rows
    ]


def sm2_review(conn: sqlite3.Connection, card_id: int, quality: int) -> dict[str, Any]:
    """Apply SM-2 algorithm and update the flashcard's scheduling.

    quality: 0-5 (SuperMemo convention)
      0-2: blackout / incorrect → reset
      3-5: correct with varying confidence

    Returns updated scheduling info.
    """
    quality = max(0, min(5, int(quality)))
    row = conn.execute(
        "SELECT ease, interval_days, reps FROM flashcards WHERE id=?",
        (card_id,),
    ).fetchone()
    if row is None:
        return {}

    ease = float(row["ease"])
    interval_days = int(row["interval_days"])
    reps = int(row["reps"])

    if quality < 3:
        # Failed: reset repetition count, schedule for tomorrow
        reps = 0
        interval_days = 1
    else:
        # Passed: apply SM-2 interval schedule
        if reps == 0:
            interval_days = 1
        elif reps == 1:
            interval_days = 6
        else:
            interval_days = max(1, round(interval_days * ease))
        reps += 1

    # Update ease factor: EF' = EF + (0.1 - (5-q)(0.08 + (5-q)*0.02))
    ease += 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
    ease = max(1.3, ease)   # SM-2 minimum ease factor

    due_ts = int(time.time()) + interval_days * 86_400
    conn.execute(
        "UPDATE flashcards SET ease=?, interval_days=?, reps=?, due_ts=? WHERE id=?",
        (ease, interval_days, reps, due_ts, card_id),
    )
    conn.commit()
    return {"ease": round(ease, 3), "interval_days": interval_days, "reps": reps, "due_ts": due_ts}


def count_flashcards(conn: sqlite3.Connection, course_id: str) -> dict[str, int]:
    """Return total and due flashcard counts for a course."""
    total = conn.execute(
        "SELECT COUNT(*) FROM flashcards WHERE course_id=?", (course_id,)
    ).fetchone()[0]
    now = int(time.time())
    due = conn.execute(
        "SELECT COUNT(*) FROM flashcards WHERE course_id=? AND due_ts<=?", (course_id, now)
    ).fetchone()[0]
    return {"total": total, "due": due}


# ── Conversation threads ──────────────────────────────────────────────────────

def create_thread(conn: sqlite3.Connection, course_id: str, title: str = "") -> int:
    now = int(time.time())
    cur = conn.execute(
        "INSERT INTO threads(course_id, title, created_at, updated_at) VALUES (?,?,?,?)",
        (course_id, title or f"Chat {now}", now, now),
    )
    conn.commit()
    return cur.lastrowid or 0


def list_threads(conn: sqlite3.Connection, course_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, title, created_at, updated_at FROM threads WHERE course_id=? ORDER BY updated_at DESC LIMIT 50",
        (course_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def rename_thread(conn: sqlite3.Connection, thread_id: int, title: str) -> None:
    conn.execute("UPDATE threads SET title=?, updated_at=? WHERE id=?", (title, int(time.time()), thread_id))
    conn.commit()


def delete_thread(conn: sqlite3.Connection, thread_id: int) -> None:
    conn.execute("DELETE FROM messages WHERE thread_id=?", (thread_id,))
    conn.execute("DELETE FROM threads WHERE id=?", (thread_id,))
    conn.commit()


def add_message(conn: sqlite3.Connection, thread_id: int, role: str, content: str, meta_json: str = "{}") -> int:
    now = int(time.time())
    cur = conn.execute(
        "INSERT INTO messages(thread_id, role, content, meta_json, ts) VALUES (?,?,?,?,?)",
        (thread_id, role, content, meta_json, now),
    )
    conn.execute("UPDATE threads SET updated_at=? WHERE id=?", (now, thread_id))
    conn.commit()
    return cur.lastrowid or 0


def get_messages(conn: sqlite3.Connection, thread_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, role, content, meta_json, ts FROM messages WHERE thread_id=? ORDER BY ts ASC",
        (thread_id,),
    ).fetchall()
    return [{"id": r["id"], "role": r["role"], "content": r["content"],
             "meta": json.loads(r["meta_json"] or "{}"), "ts": r["ts"]} for r in rows]


# ── Bookmarks / pinned answers ────────────────────────────────────────────────

def add_bookmark(conn: sqlite3.Connection, course_id: str, title: str, content: str, tag: str = "") -> int:
    now = int(time.time())
    cur = conn.execute(
        "INSERT INTO bookmarks(course_id, title, content, tag, created_at) VALUES (?,?,?,?,?)",
        (course_id, title[:120], content[:4000], tag, now),
    )
    conn.commit()
    return cur.lastrowid or 0


def list_bookmarks(conn: sqlite3.Connection, course_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, title, content, tag, created_at FROM bookmarks WHERE course_id=? ORDER BY created_at DESC",
        (course_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def delete_bookmark(conn: sqlite3.Connection, bookmark_id: int) -> None:
    conn.execute("DELETE FROM bookmarks WHERE id=?", (bookmark_id,))
    conn.commit()
