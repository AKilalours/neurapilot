"""Unit tests for NeuraPilot core modules.

Run with: pytest tests/ -v
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from neurapilot.config import Settings


# ── Test fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """Settings pointing to temporary directories for test isolation."""
    return Settings(
        DATA_DIR=str(tmp_path / "data"),
        UPLOAD_DIR=str(tmp_path / "uploads"),
        DB_PATH=str(tmp_path / "data" / "test.db"),
        CHROMA_DIR=str(tmp_path / "chroma"),
    )


@pytest.fixture
def db_conn(tmp_settings: Settings) -> sqlite3.Connection:
    from neurapilot.core.db import connect
    return connect(tmp_settings)


# ── Config tests ──────────────────────────────────────────────────────────────


def test_settings_defaults():
    s = Settings()
    assert s.llm_provider in ("ollama", "openai")
    assert s.top_k > 0
    assert s.candidate_k >= s.top_k


def test_settings_invalid_provider():
    with pytest.raises(Exception):
        Settings(LLM_PROVIDER="invalid_provider")


def test_settings_candidate_k_validation():
    """candidate_k must be >= top_k."""
    with pytest.raises(Exception):
        Settings(TOP_K=10, CANDIDATE_K=5)


# ── DB tests ──────────────────────────────────────────────────────────────────


def test_db_connect_creates_schema(db_conn):
    tables = {
        row[0]
        for row in db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    for expected in ("courses", "interactions", "mastery", "flashcards", "eval_log"):
        assert expected in tables, f"Missing table: {expected}"


def test_upsert_course(db_conn, tmp_settings: Settings):
    from neurapilot.core.db import upsert_course, list_courses
    upsert_course(db_conn, "cs101", "Intro to CS", "A beginner course")
    courses = list_courses(db_conn)
    assert any(c["course_id"] == "cs101" for c in courses)


def test_log_and_retrieve_interaction(db_conn):
    from neurapilot.core.db import log_interaction, get_recent_interactions
    row_id = log_interaction(
        db_conn,
        course_id="test",
        strict=True,
        user_text="What is gradient descent?",
        intent="ask",
        topic="gradient descent",
        output="It's an optimization algorithm. [S1]",
        sources=[{"key": "S1", "source": "lec1.pdf", "page": 3, "chunk_id": "abc123"}],
        latency_ms=420,
        faithfulness=0.9,
        answer_relevance=0.85,
        context_precision=0.75,
    )
    assert row_id > 0

    interactions = get_recent_interactions(db_conn, "test", limit=10)
    assert len(interactions) == 1
    assert interactions[0]["intent"] == "ask"
    assert interactions[0]["faithfulness"] == pytest.approx(0.9)


# ── Mastery tests ─────────────────────────────────────────────────────────────


def test_mastery_bayesian_update(db_conn):
    from neurapilot.core.db import update_mastery, get_mastery

    # Initial state: uniform Beta(1,1) → P=0.5
    update_mastery(db_conn, "c1", "backprop", correct=True)
    update_mastery(db_conn, "c1", "backprop", correct=True)
    update_mastery(db_conn, "c1", "backprop", correct=False)

    rows = get_mastery(db_conn, "c1")
    assert len(rows) == 1
    row = rows[0]
    # alpha=3, beta=2 → P = 3/5 = 0.6
    assert abs(row["p_mastery"] - 0.6) < 0.01
    assert row["attempts"] == 3
    assert 0.0 <= row["ci_lower"] <= row["p_mastery"] <= row["ci_upper"] <= 1.0


# ── Flashcard tests ───────────────────────────────────────────────────────────


def test_add_and_retrieve_flashcards(db_conn):
    from neurapilot.core.db import add_flashcards, get_due_flashcards, count_flashcards

    cards = [
        {"q": "What is backprop?", "a": "Gradient computation algorithm.", "difficulty": "medium"},
        {"q": "Define softmax.", "a": "Normalizes logits to probabilities.", "difficulty": "easy"},
        {"q": "", "a": "empty q — should be skipped"},
    ]
    n = add_flashcards(db_conn, "c1", "neural nets", cards)
    assert n == 2  # skips the empty-q card

    counts = count_flashcards(db_conn, "c1")
    assert counts["total"] == 2
    assert counts["due"] == 2  # due immediately (new cards)

    due = get_due_flashcards(db_conn, "c1", limit=10)
    assert len(due) == 2


def test_sm2_correct_answer(db_conn):
    from neurapilot.core.db import add_flashcards, get_due_flashcards, sm2_review

    add_flashcards(db_conn, "c1", "t", [{"q": "Q", "a": "A"}])
    due = get_due_flashcards(db_conn, "c1")
    assert due

    card_id = due[0]["id"]

    # Quality 5 (perfect recall)
    result = sm2_review(db_conn, card_id, quality=5)
    assert result["interval_days"] == 1  # first rep: 1 day
    assert result["ease"] > 2.5  # ease increases for quality=5

    # Second review: interval should jump to 6
    result2 = sm2_review(db_conn, card_id, quality=4)
    assert result2["interval_days"] == 6


def test_sm2_failed_answer_resets(db_conn):
    from neurapilot.core.db import add_flashcards, get_due_flashcards, sm2_review

    add_flashcards(db_conn, "c1", "t", [{"q": "Q", "a": "A"}])
    due = get_due_flashcards(db_conn, "c1")
    card_id = due[0]["id"]

    # Two successful reviews
    sm2_review(db_conn, card_id, quality=4)
    sm2_review(db_conn, card_id, quality=4)

    # Failed review: resets reps and sets interval to 1 day
    result = sm2_review(db_conn, card_id, quality=1)
    assert result["reps"] == 0
    assert result["interval_days"] == 1


# ── Storage tests ─────────────────────────────────────────────────────────────


def test_storage_atomic_write(tmp_settings: Settings):
    from neurapilot.storage import load_courses, save_courses

    courses = {"ml101": {"title": "ML 101", "description": ""}}
    save_courses(tmp_settings, courses)

    loaded = load_courses(tmp_settings)
    assert "ml101" in loaded
    assert loaded["ml101"]["title"] == "ML 101"


def test_storage_missing_file_returns_empty(tmp_settings: Settings):
    from neurapilot.storage import load_courses
    loaded = load_courses(tmp_settings)
    assert loaded == {}


# ── Rewrite tests ─────────────────────────────────────────────────────────────


def test_rewrite_fallback():
    """rewrite_query fallback should handle LLM returning non-JSON gracefully."""
    from unittest.mock import MagicMock
    from neurapilot.rag.rewrite import rewrite_query

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="NOT VALID JSON AT ALL")

    result = rewrite_query(mock_llm, "What is gradient descent?")
    assert result.query == "What is gradient descent?"
    assert isinstance(result.must_terms, list)
    assert isinstance(result.hyde_passage, str)


def test_rewrite_parses_valid_json():
    from unittest.mock import MagicMock
    from neurapilot.rag.rewrite import rewrite_query

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"query": "gradient descent optimization", "hyde": "Gradient descent updates weights.", "must_terms": ["gradient", "descent", "learning rate"]}'
    )

    result = rewrite_query(mock_llm, "How does gradient descent work?")
    assert result.query == "gradient descent optimization"
    assert "gradient" in result.must_terms
    assert len(result.must_terms) <= 8


# ── Rerank tests ──────────────────────────────────────────────────────────────


def test_rerank_by_terms():
    from unittest.mock import MagicMock
    from neurapilot.rag.rewrite import rerank_by_terms

    doc_a = MagicMock()
    doc_a.page_content = "gradient descent is an optimization method for neural networks"

    doc_b = MagicMock()
    doc_b.page_content = "cooking recipes for pasta carbonara"

    doc_c = MagicMock()
    doc_c.page_content = "gradient descent and learning rate interaction"

    ranked = rerank_by_terms([doc_b, doc_c, doc_a], ["gradient", "descent"], top_k=2)
    assert len(ranked) == 2
    # doc_b (irrelevant) should not be in top 2
    assert doc_b not in ranked


# ── Telemetry tests ───────────────────────────────────────────────────────────


def test_pipeline_trace_timing():
    from neurapilot.core.telemetry import PipelineTrace, timed_node

    trace = PipelineTrace()
    with timed_node(trace, "classify") as t:
        time.sleep(0.01)
    with timed_node(trace, "retrieve") as t:
        time.sleep(0.01)

    summary = trace.summary()
    assert "classify" in summary
    assert "retrieve" in summary
    assert summary["classify"] > 0
    assert trace.total_ms > 15
