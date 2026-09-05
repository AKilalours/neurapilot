"""NeuraPilot Evaluation Metrics — RAGAS-style offline scoring.

Implements three core RAG quality metrics:

1. Faithfulness: Are all claims in the answer traceable to the context?
   - Computed by asking the LLM to check each sentence against context
   - Score: fraction of sentences that are supported

2. Answer Relevance: Does the answer actually address the question?
   - Computed by reverse-generating questions from the answer and measuring
     similarity to the original question (simplified proxy)

3. Context Precision: Are the retrieved chunks actually relevant to the question?
   - Computed by asking the LLM which retrieved chunks contributed to the answer
   - Score: fraction of chunks that were useful

References:
  - RAGAS: https://arxiv.org/abs/2309.15217

Design: All metrics are optional (return None on failure) so evaluation
never blocks the main pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate


@dataclass
class EvalScores:
    """Container for evaluation metric scores (0.0–1.0 each)."""
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_precision": self.context_precision,
        }

    def mean_score(self) -> float | None:
        scores = [s for s in [self.faithfulness, self.answer_relevance, self.context_precision]
                  if s is not None]
        return round(sum(scores) / len(scores), 3) if scores else None


# ── Prompts ───────────────────────────────────────────────────────────────────

_FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Given a CONTEXT and an ANSWER, determine what fraction of the answer's
claims are directly supported by the context.

Respond with ONLY a decimal between 0.0 and 1.0 and nothing else.
No explanation, no counts, no words. Example: 0.75""",
    ),
    ("human", "CONTEXT:\n{context}\n\nANSWER:\n{answer}"),
])

_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Given a QUESTION and an ANSWER, rate how directly and completely
the answer addresses the question.

Return ONLY a decimal between 0.0 and 1.0.
1.0 = fully addresses the question
0.0 = completely off-topic""",
    ),
    ("human", "QUESTION: {question}\n\nANSWER:\n{answer}"),
])

_PRECISION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Given a QUESTION and a list of retrieved CHUNKS, determine what fraction
of the chunks contain information useful for answering the question.

Return ONLY a decimal between 0.0 and 1.0. Example: 0.6""",
    ),
    ("human", "QUESTION: {question}\n\nCHUNKS:\n{chunks}"),
])


# ── Metric functions ──────────────────────────────────────────────────────────


def _is_refusal(answer: str) -> bool:
    """True when the pipeline declined to answer.

    Scoring a refusal is meaningless, so these are excluded. The previous
    equality check against "Not found in documents." never matched, because the
    string the prompt actually emits is "⚠️ Not found in documents for this
    part."; substring matching covers both.
    """
    return "not found in documents" in answer.strip().lower()


def _parse_score(raw: str) -> float | None:
    """Extract a float score from LLM output.

    The judge is asked for a bare decimal, but small models frequently narrate
    first ("There are 5 sentences, 4 are supported, so 0.8"). Taking the FIRST
    number picks up a sentence count and, once clamped to [0, 1], silently
    reports a perfect 1.0. Scores are stated last, so scan backwards and return
    the last value that is actually inside [0, 1].

    Returns None when no in-range value exists, rather than inventing one.
    """
    matches = re.findall(r"\d+\.?\d*", raw.strip())
    for token in reversed(matches):
        try:
            value = float(token)
        except ValueError:
            continue
        if 0.0 <= value <= 1.0:
            return round(value, 3)
    return None


def compute_faithfulness(
    llm: BaseChatModel,
    answer: str,
    docs: list[Document],
) -> float | None:
    """Compute faithfulness score: how well the answer is grounded in context."""
    if not docs or not answer or _is_refusal(answer):
        return None
    context = "\n\n".join(d.page_content[:500] for d in docs[:6])
    try:
        out = llm.invoke(_FAITHFULNESS_PROMPT.format_messages(context=context, answer=answer[:2000]))
        return _parse_score(getattr(out, "content", str(out)))
    except Exception:
        return None


def compute_answer_relevance(
    llm: BaseChatModel,
    question: str,
    answer: str,
) -> float | None:
    """Compute answer relevance score: how well the answer addresses the question."""
    if not answer or _is_refusal(answer):
        return None
    try:
        out = llm.invoke(_RELEVANCE_PROMPT.format_messages(question=question, answer=answer[:2000]))
        return _parse_score(getattr(out, "content", str(out)))
    except Exception:
        return None


def compute_context_precision(
    llm: BaseChatModel,
    question: str,
    docs: list[Document],
) -> float | None:
    """Compute context precision: what fraction of retrieved chunks were useful."""
    if not docs:
        return None
    chunks = "\n\n---\n\n".join(
        f"[Chunk {i}]: {d.page_content[:300]}"
        for i, d in enumerate(docs[:8], start=1)
    )
    try:
        out = llm.invoke(_PRECISION_PROMPT.format_messages(question=question, chunks=chunks))
        return _parse_score(getattr(out, "content", str(out)))
    except Exception:
        return None


def evaluate_response(
    llm: BaseChatModel,
    question: str,
    answer: str,
    docs: list[Document],
) -> EvalScores:
    """Run all three evaluation metrics and return combined scores.

    Designed to be called asynchronously after the main pipeline returns,
    so it never adds latency to the user-facing response.
    """
    return EvalScores(
        faithfulness=compute_faithfulness(llm, answer, docs),
        answer_relevance=compute_answer_relevance(llm, question, answer),
        context_precision=compute_context_precision(llm, question, docs),
    )
