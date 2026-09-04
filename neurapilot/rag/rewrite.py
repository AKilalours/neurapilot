"""Query rewriting and term extraction for NeuraPilot.

Implements Hypothetical Document Embeddings (HyDE) + term-boosted retrieval:
1. Rewrite the user's raw question into an optimized search query
2. Extract "must terms" — key concepts that relevant chunks should mention
3. HyDE: generate a short hypothetical answer passage for embedding
   (the embedding of a hypothetical answer is often closer to relevant docs
    than the embedding of the question itself)

References:
  - HyDE: https://arxiv.org/abs/2212.10496
  - Query2Doc: https://arxiv.org/abs/2303.07678
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel


@dataclass(frozen=True)
class RewriteResult:
    """Output from the query rewriting node."""
    query: str              # Optimized retrieval query
    hyde_passage: str       # Hypothetical answer passage (for embedding)
    must_terms: list[str]   # Key terms for reranking boost
    original: str           # Original user question (for logging)


# ── Prompts ───────────────────────────────────────────────────────────────────

from neurapilot.rag.prompts import REWRITE_PROMPT as _REWRITE_PROMPT


# ── Rerank helpers ────────────────────────────────────────────────────────────


def rerank_by_terms(
    docs: list,
    must_terms: list[str],
    top_k: int,
) -> list:
    """Re-rank retrieved docs by term overlap with must_terms.

    Docs containing more key terms score higher and bubble to the top.
    This complements semantic similarity with exact-match signal.
    """
    if not must_terms or not docs:
        return docs[:top_k]

    scored: list[tuple[int, object]] = []
    for doc in docs:
        content = (doc.page_content or "").lower()
        score = sum(1 for term in must_terms if term in content)
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


# ── Main rewrite function ─────────────────────────────────────────────────────


def rewrite_query(llm: BaseChatModel, question: str) -> RewriteResult:
    """Rewrite a student question into retrieval-optimized form.

    Falls back gracefully if the LLM returns malformed JSON.
    """
    try:
        out = llm.invoke(_REWRITE_PROMPT.format_messages(question=question))
        raw = getattr(out, "content", str(out)).strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
        obj = json.loads(raw)

        query = str(obj.get("query", "")).strip() or question
        hyde = str(obj.get("hyde", "")).strip()
        terms_raw = obj.get("must_terms", [])
        if not isinstance(terms_raw, list):
            terms_raw = []
        terms = list(dict.fromkeys(
            str(t).strip().lower()
            for t in terms_raw
            if str(t).strip()
        ))[:8]
        return RewriteResult(query=query, hyde_passage=hyde, must_terms=terms, original=question)

    except Exception:
        # Graceful fallback: use the original question, extract naive terms
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", question.lower())
        # Remove common stopwords
        stopwords = {"the", "and", "for", "that", "this", "with", "from", "what", "how", "why"}
        terms = list(dict.fromkeys(w for w in words if w not in stopwords))[:6]
        return RewriteResult(
            query=question,
            hyde_passage="",
            must_terms=terms,
            original=question,
        )
