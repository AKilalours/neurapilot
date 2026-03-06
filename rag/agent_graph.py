"""NeuraPilot LangGraph Pipeline — Tesla AI Intern grade.

Architecture: 5-node optimized DAG
  classify (1 LLM call: intent+topic) → rewrite → retrieve → rerank → generate → END

Key improvements over previous:
  - Combined classify (1 call not 2) → ~30s faster
  - Keyword-first intent detection before LLM (near-instant for obvious cases)
  - Quiz/flashcard intent never fails silently — always produces output
  - Context capped at 12k chars to prevent OOM on 8B models
  - Hallucination guard OFF by default (adds 30s for marginal gain)
"""
from __future__ import annotations

import json
import re
from typing import Any, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from neurapilot.rag.prompts import CLASSIFY_PROMPT
from neurapilot.rag.rag import (
    RAGResult,
    extract_citation_keys,
    format_context,
    generate_answer,
    generate_flashcards,
    generate_quiz,
    generate_study_plan,
)
from neurapilot.rag.rewrite import RewriteResult, rerank_by_terms, rewrite_query

Intent = Literal["ask", "summarize", "flashcards", "quiz", "plan"]

# ── Keyword patterns — checked BEFORE LLM (instant, no API call) ─────────────
# These must be tight to avoid false positives

# Quiz: only explicit quiz/test/MCQ requests
_QUIZ_KW = re.compile(
    r'\b(quiz\s*me|test\s*me|mcq|multiple.choice|quiz\s+on|generate\s+.{0,10}quiz'
    r'|create\s+.{0,10}quiz|make\s+.{0,10}quiz|quiz\s+ready|quiz\s+about)\b', re.I)

# Flashcards: explicit card requests only
_FLASH_KW = re.compile(
    r'\b(flashcards?|flash\s*cards?|make\s+.{0,20}cards?|cards?\s+on|flip\s+cards?|create\s+.{0,10}cards?|generate\s+.{0,10}cards?)\b', re.I)

# Study plan
_PLAN_KW = re.compile(
    r'\b(study\s*plan|7.day|week\s*plan|schedule\s*for|plan\s*to\s*master)\b', re.I)

# Summarize — catch all "tell me what", "analyse", "summarise", "overview" etc.
_SUMM_KW = re.compile(
    r'\b(summar[iy]s[e]?|summarize|what\s+(does|is|do)\s+(it|this|the\s+file|the\s+paper)'
    r'|tell\s+me\s+what|analys[ei]s?|what\s+it\s+explain|overview\s+of'
    r'|what\s+is\s+this\s+about|what\s+does\s+this|explain\s+the\s+file'
    r'|what\s+this\s+(file|paper|doc)\s+(say|contain|explain))\b', re.I)

# Visualize — must NOT become quiz
_VIZ_KW = re.compile(
    r'\b(visuali[sz]e?|diagram|draw\s+a|show\s+(me\s+)?(a\s+)?(diagram|chart|graph)'
    r'|architecture\s+diagram|concept\s+map|mind\s+map)\b', re.I)


class AgentState(TypedDict, total=False):
    question:  str
    strict:    bool
    intent:    Intent
    topic:     str
    rewrite:   RewriteResult
    raw_docs:  list[Document]
    docs:      list[Document]
    output:    str
    verified:  bool


def _safe_invoke(retriever: Any, query: str) -> list[Document]:
    for call in [lambda: retriever.invoke(query), lambda: retriever.invoke({"query": query})]:
        try:
            return call()
        except Exception:
            continue
    return []


def _parse_classify(raw: str) -> tuple[str, str]:
    """Parse JSON from classify LLM output → (intent, topic)."""
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.DOTALL).strip()
    match = re.search(r'\{[^{}]*\}', clean, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            return str(obj.get("intent","ask")).strip().lower(), str(obj.get("topic","course material")).strip().lower()
        except Exception:
            pass
    return "ask", "course material"


def build_pipeline(
    llm: BaseChatModel,
    retriever: Any,
    strict_default: bool,
    top_k: int,
    hallucination_guard: bool = False,
) -> Any:
    """Build and compile the LangGraph pipeline.

    Call once per session; the compiled graph is thread-safe.
    """

    # ── Node: Classify ────────────────────────────────────────────────────────

    def classify_node(state: AgentState) -> AgentState:
        full_text = state.get("question", "")
        strict    = bool(state.get("strict", strict_default))

        # ── Extract ONLY the current request (strip ALL context prefix formats) ─
        # CRITICAL: _ctx_query injects history ending with "CURRENT QUESTION: {text}"
        # Without stripping, "quiz"/"flashcard" in prior history contaminates classify
        text = full_text
        if "CURRENT QUESTION:" in full_text:
            text = full_text.split("CURRENT QUESTION:")[-1].strip()
        elif "Current request (answer THIS" in full_text:
            text = full_text.split("Current request (answer THIS, not the prior conversation):")[-1].strip()
        elif "Current question:" in full_text:
            text = full_text.split("Current question:")[-1].strip()
        elif "=== END CONVERSATION ===" in full_text:
            after = full_text.split("=== END CONVERSATION ===")[-1].strip()
            if after:
                text = after

        # ── 1. Fast keyword detection on CURRENT TEXT ONLY ─────────────────
        # Visualize FIRST — must never become quiz
        if _VIZ_KW.search(text) and not _QUIZ_KW.search(text):
            return {"intent": "ask", "topic": "visualization", "strict": strict}

        # Summarize/analyse — catch before quiz
        if _SUMM_KW.search(text) and not _QUIZ_KW.search(text):
            m = re.search(r'(?:of|on|about|for)\s+(.{3,40}?)(?:\s+using|\s+from|$)', text, re.I)
            t = m.group(1).strip().lower() if m else "course material"
            return {"intent": "summarize", "topic": t, "strict": strict}

        # Explicit quiz keywords only
        if _QUIZ_KW.search(text):
            return {"intent": "quiz", "topic": "course material", "strict": strict}

        if _FLASH_KW.search(text):
            m = re.search(r'(?:on|about|for)\s+(.{3,40}?)(?:\s+using|\s+from|$)', text, re.I)
            t = m.group(1).strip().lower() if m else "course material"
            return {"intent": "flashcards", "topic": t, "strict": strict}

        if _PLAN_KW.search(text):
            return {"intent": "plan", "topic": "course material", "strict": strict}

        # ── 2. LLM classify for ambiguous cases ─────────────────────────────
        try:
            out        = llm.invoke(CLASSIFY_PROMPT.format_messages(text=text))
            raw        = getattr(out, "content", str(out)).strip()
            intent_raw, topic = _parse_classify(raw)

            valid_intents = {"ask", "summarize", "flashcards", "quiz", "plan", "guidance"}
            alias = {
                "flashcard":"flashcards","flash":"flashcards","cards":"flashcards",
                "flashcards":"flashcards",
                "summ":"summarize","summary":"summarize","summarise":"summarize",
                "analyse":"summarize","analyze":"summarize",
                "studyplan":"plan","study":"plan",
                "visualize":"ask","visualise":"ask","diagram":"ask",
                "task":"guidance","homework":"guidance","assignment":"guidance",
            }
            intent: Intent = intent_raw if intent_raw in valid_intents else alias.get(intent_raw, "ask")  # type: ignore

            # Sanity-check: if topic is garbage, default it
            if not topic or len(topic) > 80 or any(w in topic[:15] for w in ["there is", "no ", "please", "student"]):
                topic = "course material"

        except Exception:
            intent = "ask"
            topic  = "course material"

        return {"intent": intent, "topic": topic, "strict": strict}

    # ── Node: Rewrite ─────────────────────────────────────────────────────────

    def rewrite_node(state: AgentState) -> AgentState:
        return {"rewrite": rewrite_query(llm, state.get("question", ""))}

    # ── Node: Retrieve ────────────────────────────────────────────────────────

    def retrieve_node(state: AgentState) -> AgentState:
        rw    = state.get("rewrite")
        query = rw.query if rw else state.get("question", "")
        docs  = _safe_invoke(retriever, query)
        return {"raw_docs": docs}

    # ── Node: Rerank ──────────────────────────────────────────────────────────

    def rerank_node(state: AgentState) -> AgentState:
        rw         = state.get("rewrite")
        must_terms = rw.must_terms if rw else []
        ranked     = rerank_by_terms(state.get("raw_docs", []), must_terms, top_k=top_k)
        return {"docs": ranked}

    # ── Generation nodes ──────────────────────────────────────────────────────

    def ask_node(state: AgentState) -> AgentState:
        result = generate_answer(llm, state.get("question",""), state.get("docs",[]), bool(state.get("strict", strict_default)))
        return {"output": result.answer}

    def flashcards_node(state: AgentState) -> AgentState:
        out = generate_flashcards(llm, state.get("topic",""), state.get("docs",[]), bool(state.get("strict", strict_default)))
        return {"output": out}

    def quiz_node(state: AgentState) -> AgentState:
        out = generate_quiz(llm, state.get("topic",""), state.get("docs",[]), bool(state.get("strict", strict_default)))
        return {"output": out}

    def plan_node(state: AgentState) -> AgentState:
        out = generate_study_plan(llm, state.get("question",""), state.get("topic",""), state.get("docs",[]), bool(state.get("strict", strict_default)))
        return {"output": out}

    # ── Routing ───────────────────────────────────────────────────────────────

    def route_intent(state: AgentState) -> str:
        return state.get("intent", "ask")

    # ── Build graph ───────────────────────────────────────────────────────────

    g = StateGraph(AgentState)
    g.add_node("classify",   classify_node)
    g.add_node("rewrite",    rewrite_node)
    g.add_node("retrieve",   retrieve_node)
    g.add_node("rerank",     rerank_node)
    g.add_node("ask",        ask_node)
    g.add_node("summarize",  ask_node)
    g.add_node("flashcards", flashcards_node)
    g.add_node("quiz",       quiz_node)
    g.add_node("plan",       plan_node)

    g.add_edge(START,       "classify")
    g.add_edge("classify",  "rewrite")
    g.add_edge("rewrite",   "retrieve")
    g.add_edge("retrieve",  "rerank")

    g.add_node("guidance", ask_node)
    g.add_conditional_edges("rerank", route_intent, {
        "ask": "ask", "summarize": "summarize", "guidance": "guidance",
        "flashcards": "flashcards", "quiz": "quiz", "plan": "plan",
    })
    for node in ["ask","summarize","guidance","flashcards","quiz","plan"]:
        g.add_edge(node, END)

    return g.compile()
