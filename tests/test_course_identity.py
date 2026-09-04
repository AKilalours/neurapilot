"""Regression tests for course identity: upload directories and collections.

These cover the bug fixed alongside them. course_slug's predecessor, _safe_id,
sanitized a course_id by dropping every character outside [a-z0-9_-]. That is
lossy, so distinct course ids collapsed onto one identity:

    "CS 101" -> "cs101"
    "cs101"  -> "cs101"

Both then shared one upload directory and one Chroma collection, so querying
one course retrieved the other course's documents with no visible cause.
collection_name() additionally truncated to 63 characters, so two long ids
sharing a prefix collided the same way.

Run with: pytest tests/test_course_identity.py -v
"""
from __future__ import annotations

import pytest

from neurapilot.config import Settings
from neurapilot.storage import collection_name, course_slug, course_upload_dir


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        DATA_DIR=str(tmp_path / "data"),
        UPLOAD_DIR=str(tmp_path / "uploads"),
        DB_PATH=str(tmp_path / "data" / "test.db"),
        CHROMA_DIR=str(tmp_path / "chroma"),
    )


# -- Pairs that collided under the old sanitizer -----------------------------

COLLIDING_PAIRS = [
    ("CS 101", "cs101"),
    ("Math-201", "math 201"),
    ("intro to ai", "Intro To AI"),
    ("ml/advanced", "ml advanced"),
    ("bio!!!", "bio"),
    ("a" * 100 + "-one", "a" * 100 + "-two"),   # collided after truncation
    ("course.1", "course1"),
]


@pytest.mark.parametrize("first,second", COLLIDING_PAIRS)
def test_distinct_course_ids_get_distinct_slugs(first, second):
    assert course_slug(first) != course_slug(second)


@pytest.mark.parametrize("first,second", COLLIDING_PAIRS)
def test_distinct_course_ids_get_distinct_collections(first, second, settings):
    assert collection_name(settings, first) != collection_name(settings, second)


@pytest.mark.parametrize("first,second", COLLIDING_PAIRS)
def test_distinct_course_ids_get_distinct_upload_dirs(first, second, settings):
    assert course_upload_dir(settings, first) != course_upload_dir(settings, second)


def test_old_sanitizer_would_have_collided():
    """Proves these tests fail against the previous implementation."""
    def old_safe_id(course_id: str) -> str:
        return "".join(
            c for c in course_id.lower() if c.isalnum() or c in ("-", "_")
        ).strip("-_") or "default"

    assert old_safe_id("CS 101") == old_safe_id("cs101")


# -- Invariants --------------------------------------------------------------

def test_slug_is_deterministic():
    assert course_slug("CS 101") == course_slug("CS 101")


def test_slug_is_never_empty():
    for weird in ["", "!!!", "...", "---", "   "]:
        assert course_slug(weird)


def test_slug_is_a_single_path_component():
    for weird in ["../../etc", "a/b/c", "x\\y"]:
        slug = course_slug(weird)
        assert "/" not in slug and "\\" not in slug


@pytest.mark.parametrize(
    "course_id",
    ["cs101", "a", "x" * 500, "Intro to Machine Learning (Fall 2026)", "日本語"],
)
def test_collection_name_satisfies_chroma_constraints(course_id, settings):
    name = collection_name(settings, course_id)

    assert 3 <= len(name) <= 63
    assert name[0].isalnum() and name[-1].isalnum()
    assert all(c.isalnum() or c in ("-", "_", ".") for c in name)


def test_upload_dir_and_collection_share_one_identity(settings):
    """Directory and collection must be derived from the same slug."""
    course_id = "Intro to AI"
    slug = course_slug(course_id)

    assert course_upload_dir(settings, course_id).name == slug
    assert slug[:20] in collection_name(settings, course_id) or slug in collection_name(settings, course_id)


def test_store_reexports_the_shared_implementation():
    """rag.store must not carry a second copy of the naming logic."""
    pytest.importorskip("langchain_chroma", reason="vector store deps not installed")
    from neurapilot.rag import store  # noqa: PLC0415

    assert store.collection_name is collection_name
