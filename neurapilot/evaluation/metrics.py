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

Count the number of sentences in ANSWER, then count how many are supported by CONTEXT.
Return ONLY a decimal between 0.0 and 1.0. Example: 0.75""",
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


def _parse_score(raw: str) -> float | None:
    """Extract a float score from LLM output."""
    matches = re.findall(r"\d+\.?\d*", raw.strip())
    if not matches:
        return None
    score = float(matches[0])
    return round(max(0.0, min(1.0, score)), 3)


def compute_faithfulness(
    llm: BaseChatModel,
    answer: str,
    docs: list[Document],
) -> float | None:
    """Compute faithfulness score: how well the answer is grounded in context."""
    if not docs or not answer or answer == "Not found in documents.":
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
    if not answer or answer == "Not found in documents.":
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
