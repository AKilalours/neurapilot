"""Regression tests for untrusted upload path handling.

These cover the bug fixed alongside them: POST /courses/{course_id}/upload
built its destination as ``course_upload_dir(...) / file.filename`` using the
client-supplied filename verbatim. Joining an untrusted name onto a directory
with ``/`` does not keep the result inside that directory, so a caller could
write a .pdf, .txt or .md file anywhere the server process had permission to
write.

Run with: pytest tests/test_upload_paths.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

from neurapilot.storage import resolve_within, safe_filename


# -- The payloads that exploited the original code ---------------------------

TRAVERSAL_PAYLOADS = [
    "../../../../etc/cron.d/evil.md",
    "../../../.ssh/authorized_keys.txt",
    "....//....//etc/shadow.txt",
    "/etc/cron.d/evil.md",              # absolute: replaced the base entirely
    "/tmp/owned.txt",
    "C:\\Windows\\System32\\evil.md",   # Windows separators
    "..\\..\\..\\evil.txt",
    "..",
    ".",
    "",
    None,
]


@pytest.mark.parametrize("payload", TRAVERSAL_PAYLOADS)
def test_safe_filename_yields_a_single_harmless_component(payload):
    """No payload may survive as anything other than one plain filename."""
    name = safe_filename(payload)

    assert name, "must never return an empty name"
    assert "/" not in name
    assert "\\" not in name
    assert not name.startswith("."), "must not produce '..', '.' or a dotfile"
    assert Path(name).name == name, "must be a single path component"


@pytest.mark.parametrize("payload", TRAVERSAL_PAYLOADS)
def test_destination_stays_inside_the_upload_directory(payload, tmp_path):
    """The end-to-end property the endpoint depends on."""
    upload_dir = tmp_path / "uploads" / "cs101"
    upload_dir.mkdir(parents=True)

    dest = resolve_within(upload_dir, safe_filename(payload))

    assert upload_dir.resolve() in dest.parents


def test_original_vulnerable_join_would_have_escaped(tmp_path):
    """Proves these tests would have failed against the old code."""
    upload_dir = tmp_path / "uploads" / "cs101"
    upload_dir.mkdir(parents=True)

    # What the endpoint used to do.
    naive_relative = upload_dir / "../../../../etc/cron.d/evil.md"
    naive_absolute = upload_dir / "/etc/cron.d/evil.md"

    assert upload_dir.resolve() not in naive_relative.resolve().parents
    assert naive_absolute == Path("/etc/cron.d/evil.md")


def test_resolve_within_rejects_an_escaping_name(tmp_path):
    base = tmp_path / "uploads"
    base.mkdir()

    with pytest.raises(ValueError):
        resolve_within(base, "../escaped.md")


def test_resolve_within_rejects_a_symlink_out_of_the_directory(tmp_path):
    """Defence in depth: a symlink inside the directory must not redirect a write."""
    base = tmp_path / "uploads"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (base / "link.md").symlink_to(outside / "target.md")

    with pytest.raises(ValueError):
        resolve_within(base, "link.md")


# -- Ordinary filenames must still work --------------------------------------

@pytest.mark.parametrize(
    "given,expected",
    [
        ("lecture_01.pdf", "lecture_01.pdf"),
        ("Week 3 Notes.md", "Week_3_Notes.md"),
        ("data-set.v2.txt", "data-set.v2.txt"),
        ("syllabus.PDF", "syllabus.PDF"),
        ("/home/akila/Documents/thesis.pdf", "thesis.pdf"),
    ],
)
def test_legitimate_filenames_survive(given, expected):
    assert safe_filename(given) == expected


def test_extension_check_runs_on_the_sanitized_name():
    """The endpoint checks the extension after sanitizing, not before.

    Checking first would let 'evil.md/../../shell.py' pass an .md check while
    landing somewhere else entirely.
    """
    assert Path(safe_filename("evil.md/../../shell.py")).suffix == ".py"
