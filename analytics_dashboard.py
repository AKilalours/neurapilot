"""NeuraPilot — Analytics Dashboard
Senior-level observability: p95 latency, cost/request, hit@10, cache hit rate,
intent distribution, eval scores over time, mastery heatmap.

Run standalone:  streamlit run analytics_dashboard.py
Or imported as a tab in ui_streamlit.py via render_analytics_tab()
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Cost model (tokens → $)
# Adjust these when you switch providers / models
# ─────────────────────────────────────────────────────────────────────────────
COST_PER_1K_INPUT  = 0.00015   # $ — gpt-4o-mini input
COST_PER_1K_OUTPUT = 0.00060   # $ — gpt-4o-mini output
AVG_INPUT_TOKENS   = 1_200     # typical prompt size
AVG_OUTPUT_TOKENS  = 400       # typical completion size
COST_PER_REQUEST   = (
    AVG_INPUT_TOKENS  / 1_000 * COST_PER_1K_INPUT +
    AVG_OUTPUT_TOKENS / 1_000 * COST_PER_1K_OUTPUT
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _p(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, math.ceil(len(s) * pct / 100) - 1)
    return s[idx]


def _sparkline(values: list[float], width: int = 20) -> str:
    """ASCII sparkline."""
    bars = "▁▂▃▄▅▆▇█"
    if not values or max(values) == min(values):
        return bars[0] * width
    mn, mx = min(values), max(values)
    return "".join(bars[int((v - mn) / (mx - mn) * (len(bars) - 1))] for v in values[-width:])


def _color_latency(ms: float) -> str:
    if ms < 2_000:  return "🟢"
    if ms < 5_000:  return "🟡"
    return "🔴"


def _color_score(s: float) -> str:
    if s >= 0.80: return "🟢"
    if s >= 0.60: return "🟡"
    return "🔴"


# ─────────────────────────────────────────────────────────────────────────────
# Data fetchers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_interactions(conn: sqlite3.Connection, course_id: str, days: int = 30) -> list[dict]:
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        """
        SELECT id, ts, intent, topic, latency_ms,
               faithfulness, answer_relevance, context_precision
        FROM   interactions
        WHERE  course_id = ? AND ts >= ?
        ORDER  BY ts ASC
        """,
        (course_id, cutoff),
    ).fetchall()
    return [dict(r) for r in rows]


def _fetch_cache_stats(conn: sqlite3.Connection, course_id: str) -> dict:
    try:
        now = time.time()
        r = conn.execute(
            """
            SELECT COUNT(*) AS total, SUM(hits) AS total_hits, AVG(hits) AS avg_hits
            FROM   sem_cache
            WHERE  course_id = ? AND expires_at > ?
            """,
            (course_id, now),
        ).fetchone()
        return dict(r) if r else {}
    except Exception:
        return {}


def _fetch_mastery(conn: sqlite3.Connection, course_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT topic, alpha, beta, review_count, last_reviewed
        FROM   mastery
        WHERE  course_id = ?
        ORDER  BY alpha / (alpha + beta) DESC
        LIMIT  20
        """,
        (course_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Main render function — call this from the Analytics tab in ui_streamlit.py
# ─────────────────────────────────────────────────────────────────────────────

def render_analytics_tab(conn: sqlite3.Connection, course_id: str) -> None:
    """Render the full analytics dashboard inside a Streamlit tab."""

    # ── Controls ─────────────────────────────────────────────────────────────
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(
            '<div style="font-size:22px;font-weight:800;color:#f1f5f9;margin-bottom:4px">'
            '📊 Analytics Dashboard</div>'
            '<div style="font-size:13px;color:#64748b">Real-time observability · '
            'p95 latency · cost · quality · cache</div>',
            unsafe_allow_html=True,
        )
    with c2:
        days = st.selectbox("Period", [7, 14, 30, 90], index=2, key="ana_days")

    data    = _fetch_interactions(conn, course_id, days=days)
    c_stats = _fetch_cache_stats(conn, course_id)
    mastery = _fetch_mastery(conn, course_id)

    if not data:
        st.info("No interactions yet for this course. Ask NeuraPilot some questions first!")
        return

    latencies   = [r["latency_ms"] for r in data if r["latency_ms"]]
    faith_vals  = [r["faithfulness"] for r in data if r["faithfulness"] is not None]
    relev_vals  = [r["answer_relevance"] for r in data if r["answer_relevance"] is not None]
    prec_vals   = [r["context_precision"] for r in data if r["context_precision"] is not None]

    total_reqs  = len(data)
    total_cost  = total_reqs * COST_PER_REQUEST
    cache_hits  = int(c_stats.get("total_hits") or 0)
    cache_total = int(c_stats.get("total") or 0)
    cache_hr    = cache_hits / max(cache_hits + total_reqs, 1)

    # ── KPI Row ───────────────────────────────────────────────────────────────
    st.markdown("---")
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    def _kpi(col, label: str, value: str, sub: str = "", color: str = "#f1f5f9"):
        col.markdown(
            f'<div style="background:#1e1e2e;border:1px solid #2a2a3e;border-radius:12px;'
            f'padding:16px 14px;text-align:center">'
            f'<div style="font-size:11px;font-weight:700;letter-spacing:1px;'
            f'text-transform:uppercase;color:#64748b;margin-bottom:6px">{label}</div>'
            f'<div style="font-size:26px;font-weight:800;color:{color};'
            f'font-family:JetBrains Mono,monospace">{value}</div>'
            f'<div style="font-size:11px;color:#475569;margin-top:4px">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    p50  = _p(latencies, 50)
    p95  = _p(latencies, 95)
    p99  = _p(latencies, 99)
    avg_faith = sum(faith_vals) / len(faith_vals) if faith_vals else 0
    avg_rel   = sum(relev_vals) / len(relev_vals) if relev_vals else 0

    lat_color  = "#4ade80" if p95 < 2000 else "#fbbf24" if p95 < 5000 else "#f87171"
    cost_color = "#4ade80" if total_cost < 1.0 else "#fbbf24"

    _kpi(k1, "p95 Latency",   f"{p95/1000:.1f}s",   f"p50={p50/1000:.1f}s · p99={p99/1000:.1f}s", lat_color)
    _kpi(k2, "Cost / Req",    f"${COST_PER_REQUEST:.4f}", f"Total ${total_cost:.3f}", cost_color)
    _kpi(k3, "Total Requests",f"{total_reqs:,}",     f"last {days} days")
    _kpi(k4, "Faithfulness",  f"{avg_faith:.0%}",    f"{_color_score(avg_faith)} avg eval")
    _kpi(k5, "Relevance",     f"{avg_rel:.0%}",      f"{_color_score(avg_rel)} avg eval")
    _kpi(k6, "Cache Hit Rate",f"{cache_hr:.0%}",     f"{cache_hits} hits · {cache_total} entries")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Latency over time + Intent breakdown ──────────────────────────────────
    col_lat, col_int = st.columns([3, 2])

    with col_lat:
        st.markdown("#### ⏱ Latency over time")
        # Group by day
        from collections import defaultdict
        day_lat: dict[str, list[float]] = defaultdict(list)
        for r in data:
            day = r["ts"][:10]
            if r["latency_ms"]:
                day_lat[day].append(r["latency_ms"])

        days_sorted = sorted(day_lat.keys())
        if days_sorted:
            p50s = [_p(day_lat[d], 50) / 1000 for d in days_sorted]
            p95s = [_p(day_lat[d], 95) / 1000 for d in days_sorted]

            # Simple ASCII chart fallback + real chart via st.line_chart
            import pandas as pd
            df_lat = pd.DataFrame({
                "p50 (s)": p50s,
                "p95 (s)": p95s,
            }, index=days_sorted)
            st.line_chart(df_lat, height=200)

            sla_breaches = sum(1 for v in [_p(day_lat[d], 95) for d in days_sorted] if v > 5000)
            if sla_breaches:
                st.warning(f"⚠️ {sla_breaches} day(s) with p95 > 5 s SLA breach")
            else:
                st.success("✅ All days within p95 < 5 s SLA")

    with col_int:
        st.markdown("#### 🎯 Intent distribution")
        from collections import Counter
        intent_counts = Counter(r["intent"] for r in data)
        INTENT_COLORS = {
            "ask":        "#3b82f6",
            "summarize":  "#06b6d4",
            "quiz":       "#f87171",
            "flashcards": "#4ade80",
            "plan":       "#a78bfa",
        }
        total_i = sum(intent_counts.values())
        for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
            pct  = count / total_i
            col  = INTENT_COLORS.get(intent, "#94a3b8")
            st.markdown(
                f'<div style="margin:5px 0">'
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:13px;color:#cbd5e1;margin-bottom:3px">'
                f'<span>{intent.upper()}</span><span>{count} ({pct:.0%})</span></div>'
                f'<div style="height:8px;border-radius:4px;background:#1e293b">'
                f'<div style="height:8px;border-radius:4px;width:{pct*100:.1f}%;'
                f'background:{col}"></div></div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Eval scores over time ─────────────────────────────────────────────────
    st.markdown("#### 📈 Eval quality scores over time")
    if faith_vals:
        import pandas as pd
        eval_rows = [
            {
                "date":          r["ts"][:10],
                "Faithfulness":  r["faithfulness"]  or 0,
                "Relevance":     r["answer_relevance"] or 0,
                "Precision":     r["context_precision"] or 0,
            }
            for r in data
            if r["faithfulness"] is not None
        ]
        if eval_rows:
            df_eval = pd.DataFrame(eval_rows)
            df_daily = df_eval.groupby("date").mean().reset_index()
            df_daily = df_daily.set_index("date")
            st.line_chart(df_daily[["Faithfulness", "Relevance", "Precision"]], height=200)
    else:
        st.info("Eval scores will appear after a few interactions.")

    # ── Cache analytics ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### ⚡ Semantic cache")
    ca, cb, cc = st.columns(3)

    savings_cost = cache_hits * COST_PER_REQUEST

    # Real cache latency from session state (populated by _run_agent)
    import streamlit as _st2
    cache_lats  = _st2.session_state.get("_cache_latencies", [])
    full_lats   = _st2.session_state.get("_full_latencies", [])
    avg_cache_ms = sum(cache_lats) / len(cache_lats) if cache_lats else None
    avg_full_ms  = sum(full_lats)  / len(full_lats)  if full_lats  else (p50 if latencies else None)

    # Latency reduction % = how much faster cache is vs full pipeline
    if avg_cache_ms and avg_full_ms and avg_full_ms > 0:
        reduction_pct = (1 - avg_cache_ms / avg_full_ms) * 100
    else:
        reduction_pct = None

    savings_ms = cache_hits * (avg_full_ms or p50) if (avg_full_ms or p50) else 0

    ca.metric("Cache entries (live)", f"{cache_total:,}")
    cb.metric("Cache hit latency",
              f"{avg_cache_ms:.0f} ms" if avg_cache_ms else "Run a cached query",
              delta=f"vs {avg_full_ms:.0f} ms full pipeline" if avg_full_ms else None)
    cc.metric("Latency reduction",
              f"{reduction_pct:.0f}%" if reduction_pct else "—",
              delta=f"~${savings_cost:.4f} cost saved" if savings_cost else None)

    # ── hit@10 proxy ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🎯 Retrieval quality — hit@10 proxy")
    st.caption(
        "hit@10 is estimated from context_precision scores. "
        "A score ≥ 0.7 counts as a retrieval hit."
    )
    if prec_vals:
        hits_at_10 = sum(1 for v in prec_vals if v >= 0.70)
        h10 = hits_at_10 / len(prec_vals)
        hcol = "#4ade80" if h10 >= 0.80 else "#fbbf24" if h10 >= 0.60 else "#f87171"
        st.markdown(
            f'<div style="font-size:48px;font-weight:900;color:{hcol};'
            f'font-family:JetBrains Mono,monospace;text-align:center;padding:16px 0">'
            f'hit@10 = {h10:.2f}</div>',
            unsafe_allow_html=True,
        )
        bar_w = int(h10 * 100)
        st.markdown(
            f'<div style="height:12px;border-radius:6px;background:#1e293b;margin:0 auto;max-width:600px">'
            f'<div style="height:12px;border-radius:6px;width:{bar_w}%;background:{hcol}"></div></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"{hits_at_10} / {len(prec_vals)} queries returned high-precision results")
    else:
        st.info("Context precision scores will appear after interactions.")

    # ── Mastery heatmap ───────────────────────────────────────────────────────
    if mastery:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🧠 Topic mastery heatmap")
        for m in mastery[:12]:
            alpha, beta = m.get("alpha", 1), m.get("beta", 1)
            p_know = alpha / (alpha + beta)
            confidence = 1 - 2 * math.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)))
            col   = "#4ade80" if p_know >= 0.75 else "#fbbf24" if p_know >= 0.50 else "#f87171"
            rcount = m.get("review_count", 0)
            topic  = (m.get("topic") or "unknown")[:40]
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:12px;margin:6px 0">'
                f'<div style="width:160px;font-size:13px;color:#cbd5e1;white-space:nowrap;'
                f'overflow:hidden;text-overflow:ellipsis">{topic}</div>'
                f'<div style="flex:1;height:10px;border-radius:5px;background:#1e293b">'
                f'<div style="height:10px;border-radius:5px;width:{p_know*100:.0f}%;'
                f'background:{col}"></div></div>'
                f'<div style="width:50px;font-size:12px;color:{col};text-align:right;'
                f'font-family:JetBrains Mono,monospace">{p_know:.0%}</div>'
                f'<div style="width:60px;font-size:11px;color:#475569">×{rcount}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── System SLO table ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📋 SLO scorecard")

    slos = [
        ("p50 latency",    f"{p50/1000:.2f}s",    "< 2.0s",  p50  < 2_000),
        ("p95 latency",    f"{p95/1000:.2f}s",    "< 5.0s",  p95  < 5_000),
        ("p99 latency",    f"{p99/1000:.2f}s",    "< 10.0s", p99  < 10_000),
        ("Faithfulness",   f"{avg_faith:.2f}",     "> 0.70",  avg_faith > 0.70),
        ("Relevance",      f"{avg_rel:.2f}",       "> 0.70",  avg_rel   > 0.70),
        ("hit@10 proxy",   f"{_p(prec_vals,50) if prec_vals else 0:.2f}", "> 0.70",
         (_p(prec_vals,50) if prec_vals else 0) > 0.70),
        ("Cost / req",     f"${COST_PER_REQUEST:.4f}", "< $0.01", COST_PER_REQUEST < 0.01),
        ("Cache hit rate", f"{cache_hr:.0%}",      "> 20%",   cache_hr > 0.20),
    ]

    import pandas as pd
    df_slo = pd.DataFrame(slos, columns=["Metric", "Current", "Target", "Pass"])
    df_slo["Status"] = df_slo["Pass"].map({True: "✅ Pass", False: "❌ Fail"})
    st.dataframe(
        df_slo[["Metric", "Current", "Target", "Status"]],
        use_container_width=True,
        hide_index=True,
    )

    passing = sum(1 for _, _, _, p in slos if p)
    total_s = len(slos)
    pct_ok  = passing / total_s
    color   = "#4ade80" if pct_ok >= 0.85 else "#fbbf24" if pct_ok >= 0.60 else "#f87171"
    st.markdown(
        f'<div style="text-align:center;font-size:18px;font-weight:700;color:{color};'
        f'margin-top:12px">{passing}/{total_s} SLOs passing ({pct_ok:.0%})</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from neurapilot.config import get_settings
    from neurapilot.core import db as dbmod

    st.set_page_config(page_title="NeuraPilot Analytics", layout="wide")
    S  = get_settings()
    DB = dbmod.connect(S)

    courses_path = Path(S.data_dir) / "courses.json"
    if courses_path.exists():
        import json as _json
        courses = _json.loads(courses_path.read_text())
    else:
        courses = {}

    if not courses:
        st.warning("No courses found. Run NeuraPilot first and ask some questions.")
        st.stop()

    cid = st.selectbox("Course", list(courses.keys()),
                       format_func=lambda k: courses[k].get("title", k))
    render_analytics_tab(DB, cid)
