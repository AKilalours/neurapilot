"""Regression tests for the evaluation score parser and the hallucination guard.

Both cover bugs found while benchmarking the `cv` course, where all 16 questions
returned faithfulness=1.000 including an out-of-scope control question that the
pipeline answered from parametric knowledge.

Run with: pytest tests/test_eval_parser_and_guard.py -v
"""
from __future__ import annotations

import pytest

pytest.importorskip("langchain_core", reason="LLM stack not installed")

from neurapilot.evaluation.metrics import _is_refusal, _parse_score  # noqa: E402
from neurapilot.rag.agent_graph import apply_hallucination_guard  # noqa: E402
from neurapilot.rag.prompts import NOT_FOUND_MARKER  # noqa: E402


# -- _parse_score ------------------------------------------------------------
#
# The judge is asked for a bare decimal but small models narrate first. Taking
# the FIRST number picked up a sentence count, which clamped to 1.0 and made the
# metric report a perfect score for every answer.

@pytest.mark.parametrize(
    "reply,expected",
    [
        # Narrated replies: the score is the LAST number, not the first.
        ("There are 5 sentences, 4 are supported by the context, so 0.8", 0.8),
        ("Count: 3 sentences. Supported: 2. Score: 0.67", 0.67),
        ("The answer contains 6 claims; 3 are grounded. 0.5", 0.5),
        ("1 of 4 sentences is supported -> 0.25", 0.25),
        # Bare replies still work.
        ("0.75", 0.75),
        ("Score: 0.4", 0.4),
        ("1.0", 1.0),
        ("0", 0.0),
        # No in-range value: report nothing rather than invent a score.
        ("4 of 5 sentences are supported", None),
        ("unable to determine", None),
        ("", None),
    ],
)
def test_parse_score_takes_the_last_in_range_value(reply, expected):
    assert _parse_score(reply) == expected


def test_old_first_number_parser_would_have_reported_a_perfect_score():
    """Proves this suite fails against the previous implementation."""
    import re

    def old_parse(raw: str):
        m = re.findall(r"\d+\.?\d*", raw.strip())
        if not m:
            return None
        return round(max(0.0, min(1.0, float(m[0]))), 3)

    narrated = "There are 5 sentences, 4 are supported by the context, so 0.8"
    assert old_parse(narrated) == 1.0
    assert _parse_score(narrated) == 0.8


def test_parse_score_never_exceeds_one():
    for reply in ["0.9", "1.0", "0.0", "the score is 1"]:
        score = _parse_score(reply)
        assert score is None or 0.0 <= score <= 1.0


# -- refusal detection -------------------------------------------------------
#
# The old check compared for equality against "Not found in documents.", a
# string the prompt never emits, so refusals were scored as if they were answers.

def test_refusal_detection_matches_the_string_the_prompt_actually_emits():
    assert _is_refusal("⚠️ Not found in documents for this part.")
    assert _is_refusal(NOT_FOUND_MARKER)
    assert not _is_refusal("**TL;DR:** Density estimation is ill-posed [S1].")


# -- hallucination guard -----------------------------------------------------

class _Doc:
    def __init__(self, text: str = "x") -> None:
        self.page_content = text
        self.metadata: dict = {}


def test_guard_refuses_when_nothing_was_retrieved():
    assert apply_hallucination_guard("Paris.", [], strict=True, enabled=True) == NOT_FOUND_MARKER


def test_guard_blocks_the_observed_common_knowledge_fallback():
    """The exact failure seen on 'What is the capital of France?'."""
    answer = (
        "⚠️ Not found in documents for this part.\n\n"
        "However, I can provide a general answer based on common knowledge. "
        "The capital of France is Paris."
    )
    assert apply_hallucination_guard(answer, [_Doc()], strict=True, enabled=True) == NOT_FOUND_MARKER


def test_guard_leaves_a_grounded_answer_untouched():
    grounded = "**TL;DR:** Density estimation is ill-posed because model selection is central [S1]."
    assert apply_hallucination_guard(grounded, [_Doc()], strict=True, enabled=True) == grounded


def test_guard_does_not_rewrite_a_partial_not_found_without_a_hedge():
    """A multi-part answer may legitimately decline one part and answer others."""
    partial = (
        "**Key Points:**\n"
        "- Bernoulli distributions model binary variables [S1].\n"
        "- ⚠️ Not found in documents for this part: the exam date."
    )
    assert apply_hallucination_guard(partial, [_Doc()], strict=True, enabled=True) == partial


def test_guard_is_a_no_op_when_disabled():
    answer = "Not found in documents. However, based on common knowledge, Paris."
    assert apply_hallucination_guard(answer, [_Doc()], strict=True, enabled=False) == answer


def test_guard_does_not_apply_hedge_check_in_tutor_mode():
    """Tutor mode is explicitly allowed to teach beyond the documents."""
    answer = "Not found in documents, but based on general knowledge, here is context."
    assert apply_hallucination_guard(answer, [_Doc()], strict=False, enabled=True) == answer
