"""File system helpers for NeuraPilot.

Keeps all path logic in one place so the rest of the code
never constructs paths manually.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neurapilot.config import Settings


# ── Directory management ──────────────────────────────────────────────────────


def ensure_dirs(settings: Settings) -> None:
    """Create all required directories if they don't exist."""
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)


def course_upload_dir(settings: Settings, course_id: str) -> Path:
    """Return (and create) the upload directory for a given course."""
    ensure_dirs(settings)
    d = Path(settings.upload_dir) / _safe_id(course_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Courses manifest ─────────────────────────────────────────────────────────


def _courses_path(settings: Settings) -> Path:
    return Path(settings.data_dir) / "courses.json"


def load_courses(settings: Settings) -> dict[str, dict[str, Any]]:
    """Load the courses manifest from disk (returns empty dict if absent)."""
    ensure_dirs(settings)
    p = _courses_path(settings)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_courses(settings: Settings, courses: dict[str, dict[str, Any]]) -> None:
    """Atomically write the courses manifest."""
    ensure_dirs(settings)
    tmp = _courses_path(settings).with_suffix(".tmp")
    tmp.write_text(json.dumps(courses, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(_courses_path(settings))  # atomic on POSIX


# ── Helpers ───────────────────────────────────────────────────────────────────


def _safe_id(course_id: str) -> str:
    """Sanitize a course_id for use as a directory/collection name."""
    return "".join(c for c in course_id.lower() if c.isalnum() or c in ("-", "_")).strip("-_") or "default"
