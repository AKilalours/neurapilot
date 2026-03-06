"""NeuraPilot UI — Production Grade.

FIXES IN THIS VERSION:
  ✅ FIX 1: First message response always renders — moved rerun() AFTER save
  ✅ FIX 2: Multiple images queued properly — pending_imgs is a LIST not scalar
  ✅ FIX 3: Voice TTS fixed — uses st.markdown injection into parent page (not <script> tag)
  ✅ FIX 4: All uploaded images shown in chat, each with its own AI analysis
  ✅ FIX 5: Messages never lost — save to DB before rerun

FEATURES:
  • Persistent threads (sidebar like Claude.ai)
  • Auto-titles from first message
  • Conversation memory (last 8 turns as context)
  • Multiple image queue — upload 2 images, each gets analyzed separately
  • Voice TTS on every answer (Read Aloud button — works in Chrome/Edge/Safari)
  • Code blocks with ▶ Run Python button + inline matplotlib plots
  • Flashcards, Quiz, Study Plan, Summarize
  • Eval score bars (Faithfulness / Relevance / Precision)
  • Regenerate last response
  • Bookmarks, Chat export
  • All file types: PDF CSV XLSX PPTX DOCX JSON YAML XML PNG JPG...
"""
from __future__ import annotations
import base64, io, json, re, time, traceback
from pathlib import Path
from typing import Any
import streamlit as st
from neurapilot.config import get_settings
from neurapilot.core import db as dbmod
from neurapilot.evaluation.metrics import evaluate_response
from neurapilot.rag.agent_graph import build_pipeline
from neurapilot.rag.ingest import ingest_course
from neurapilot.rag.llm import build_llm_bundle
from neurapilot.rag.store import get_retriever
from neurapilot.storage import course_upload_dir, load_courses, save_courses

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuraPilot · Akila Lourdes Miriyala Francis", page_icon="🧠",
    layout="wide", initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ─────────────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"]  { background:#212121; font-family:'Inter',sans-serif; }
[data-testid="stSidebar"]           { background:#171717; border-right:1px solid #2a2a2a; }
[data-testid="stSidebar"] *         { font-family:'Inter',sans-serif !important; }

/* ── Chat input — NeuraPilot style ──────────────────────────────────────── */
[data-testid="stChatInput"] {
    background:#2f2f2f !important;
    border-radius:16px !important;
    border:1px solid #404040 !important;
    box-shadow:0 2px 12px rgba(0,0,0,0.4) !important;
}
[data-testid="stChatInput"] textarea {
    background:transparent !important;
    border:none !important;
    border-radius:16px !important;
    color:#ececec !important;
    font-family:'Inter',sans-serif !important;
    font-size:16px !important;
    padding:14px 18px !important;
    line-height:1.6 !important;
}
[data-testid="stChatInput"] textarea:focus { outline:none !important; }

/* ── Chat messages — NeuraPilot clean style ────────────────────────────── */
[data-testid="stChatMessage"]       { background:transparent !important; border:none !important; padding:0 !important; }
[data-testid="stChatMessage"] p,
[data-testid="stMarkdownContainer"] > div > p { font-size:17px !important; line-height:1.85 !important; color:#ececec !important; margin-bottom:12px !important; }
[data-testid="stChatMessage"] li    { font-size:17px !important; line-height:1.85 !important; color:#ececec !important; }
[data-testid="stChatMessage"] h1    { color:#ffffff !important; font-weight:700 !important; font-size:24px !important; margin-top:24px !important; }
[data-testid="stChatMessage"] h2    { color:#ffffff !important; font-weight:700 !important; font-size:21px !important; margin-top:20px !important; }
[data-testid="stChatMessage"] h3    { color:#ffffff !important; font-weight:600 !important; font-size:18px !important; margin-top:16px !important; }
[data-testid="stChatMessage"] strong { color:#ffffff !important; font-size:17px !important; }
[data-testid="stChatMessage"] code  { font-size:15px !important; background:#2a2a2a !important; border-radius:5px !important; padding:2px 7px !important; }
[data-testid="stChatMessage"] pre   { background:#1a1a1a !important; border-radius:10px !important; border:1px solid #333 !important; }

/* ── User bubble — NeuraPilot style ───────────────────────────────────── */
[data-testid="stChatMessage"][data-testid*="user"] .stChatMessage,
div[class*="stChatMessage"] > div:has([data-testid*="user"]) {
    background:#2f2f2f !important;
    border-radius:18px !important;
    max-width:75% !important;
    margin-left:auto !important;
}

/* ── Buttons ──────────────────────────────────────────────────────────── */
.stButton>button {
    border-radius:8px !important;
    font-family:'Inter',sans-serif !important;
    font-weight:600 !important;
    font-size:14px !important;
    transition:all 0.15s !important;
    border:1px solid #404040 !important;
    background:#2f2f2f !important;
    color:#ececec !important;
}
.stButton>button:hover { background:#404040 !important; border-color:#555 !important; }
.stButton>button[kind="primary"] { background:#10a37f !important; border-color:#10a37f !important; color:white !important; }
.stButton>button[kind="primary"]:hover { background:#0d8a6a !important; }

/* ── Source chips ─────────────────────────────────────────────────────── */
.chip {
    background:#2a2a2a;
    border:1px solid #383838;
    border-radius:6px;
    padding:3px 10px;
    font-size:12px;
    color:#666;
    font-family:'JetBrains Mono',monospace;
    margin:2px 3px;
    display:inline-block;
}

/* ── Eval bars ────────────────────────────────────────────────────────── */
.eval-row   { display:flex; gap:16px; margin:10px 0 4px; flex-wrap:wrap; }
.eval-block { flex:1; min-width:110px; }
.eval-lbl   { font-size:11px; font-weight:700; letter-spacing:.8px; text-transform:uppercase; color:#555; margin-bottom:4px; }
.eval-bar   { height:4px; border-radius:2px; background:#333; overflow:hidden; }
.eval-fill  { height:100%; border-radius:2px; }
.eval-val   { font-size:14px; font-weight:800; margin-top:4px; font-family:'JetBrains Mono',monospace; }

/* ── Intent tags ──────────────────────────────────────────────────────── */
.itag { display:inline-block; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:700; letter-spacing:.5px; }
.i-ask       { background:#10a37f18; color:#10a37f; }
.i-summarize { background:#06b6d418; color:#06b6d4; }
.i-quiz      { background:#f8717118; color:#f87171; }
.i-flashcards{ background:#4ade8018; color:#4ade80; }
.i-plan      { background:#a78bfa18; color:#a78bfa; }
.i-vision    { background:#ec489918; color:#ec4899; }
.i-guidance  { background:#fbbf2418; color:#fbbf24; }

/* ── Flashcards ───────────────────────────────────────────────────────── */
.fc-q  { background:#1e1e1e; border:1px solid #383838; border-left:3px solid #10a37f; border-radius:12px; padding:18px 22px; margin:8px 0 3px; }
.fc-a  { background:#1a2620; border:1px solid #1e3a2a; border-left:3px solid #4ade80; border-radius:12px; padding:16px 20px; margin-bottom:18px; color:#86efac; }
.fc-q p, .fc-a p { font-size:16px !important; }
.fc-badge { display:inline-block; padding:2px 10px; border-radius:10px; font-size:11px; font-weight:800; margin:0 4px 8px 0; letter-spacing:.5px; text-transform:uppercase; }
.fc-easy   { background:#4ade8015; color:#4ade80; border:1px solid #4ade8030; }
.fc-medium { background:#fbbf2415; color:#fbbf24; border:1px solid #fbbf2430; }
.fc-hard   { background:#f8717115; color:#f87171; border:1px solid #f8717130; }
.fc-bloom  { background:#a78bfa15; color:#a78bfa; border:1px solid #a78bfa30; }

/* ── Quiz banner ──────────────────────────────────────────────────────── */
.quiz-banner { background:#1a1f2e; border:1px solid #2a3650; border-radius:14px; padding:22px 26px; margin:6px 0; }

/* ── Thinking box ─────────────────────────────────────────────────────── */
.think-box   { background:#1a1a1a; border:1px solid #2a2a2a; border-radius:12px; padding:14px 18px; margin:8px 0; }
.think-title { font-size:12px; font-weight:700; letter-spacing:.8px; text-transform:uppercase; color:#444; margin-bottom:10px; }
.think-step  { font-size:13px; color:#444; padding:4px 0 4px 14px; border-left:2px solid #2a2a2a; margin:4px 0; font-family:'JetBrains Mono',monospace; }
.think-step.live { color:#10a37f99; border-left-color:#10a37f55; }
.think-step.done { color:#555; }

/* ── Voice button ─────────────────────────────────────────────────────── */
.vbtn { background:none; color:#555; border:1px solid #333; border-radius:6px; padding:4px 12px; cursor:pointer; font-size:12px; font-family:'Inter',sans-serif; font-weight:600; transition:all .12s; margin:4px 2px 0 0; }
.vbtn:hover { background:#2a2a2a; color:#888; }

/* ── Meta row ─────────────────────────────────────────────────────────── */
.meta-row { display:flex; align-items:center; gap:8px; margin-top:12px; padding-top:10px; border-top:1px solid #2a2a2a; flex-wrap:wrap; }
.meta-txt { font-size:11px; color:#444; font-family:'JetBrains Mono',monospace; }

/* ── Viz panel ────────────────────────────────────────────────────────── */
.viz-panel { background:#1a1a1a; border:1px solid #2a2a2a; border-radius:14px; padding:20px; margin-top:12px; }
.viz-title { font-size:11px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:#10a37f; margin-bottom:12px; display:flex; align-items:center; gap:8px; }

/* ── Scrollbar ────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:#333; border-radius:3px; }

/* ── Image queue badge ────────────────────────────────────────────────── */
.img-queue-badge { display:inline-block; background:#10a37f20; color:#10a37f; border:1px solid #10a37f40; border-radius:12px; padding:2px 10px; font-size:12px; font-weight:700; margin:2px 4px; }

/* ── Selectbox / inputs ───────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div { background:#2a2a2a !important; border:1px solid #404040 !important; border-radius:8px !important; color:#ececec !important; }

/* ── Progress ─────────────────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div { background:#10a37f !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Resources
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _sett(): return get_settings()
@st.cache_resource
def _bndl(): return build_llm_bundle(_sett())
@st.cache_resource
def _db():   return dbmod.connect(_sett())

S  = _sett()
B  = _bndl()
DB = _db()

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
_DEF: dict[str, Any] = {
    "tid":           None,
    "msgs":          [],
    "quiz_obj":      None,
    "quiz_idx":      0,
    "quiz_score":    0,
    "img_queue":     [],
    "strict":        False,   # OFF by default — strict=ON blocks quiz/flashcard generation
    "show_think":    True,
    "_active_course":  None,   # tracks which course is active to clear stale state
    "_show_nc":        False,  # toggles new-course form in sidebar
    "_pending_course": None,   # newly created course id — auto-select on next rerun
}
for k, v in _DEF.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _pipe(cid: str, strict_d: bool):
    return build_pipeline(
        llm=B.llm, retriever=get_retriever(S, B.embeddings, cid),
        strict_default=strict_d, top_k=S.top_k, hallucination_guard=False,
    )

# ─────────────────────────────────────────────────────────────────────────────
# FIX 3: Voice — injected into parent page via st.markdown
#         DO NOT use <script> tags — Streamlit strips them silently
#         DO NOT use components.html() — runs in sandboxed iframe, no speechSynthesis
#         CORRECT: onclick attribute on a <button> rendered by st.markdown(unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────
def _voice(text: str, key: str) -> None:
    """TTS via Web Speech API using components.html.

    components.html() renders with allow-scripts + allow-same-origin so
    speechSynthesis IS accessible. This is the only reliable method in Streamlit.
    Height=44px so it looks inline like a small button.
    Text is base64-encoded to handle apostrophes and special chars.
    """
    import streamlit.components.v1 as components
    c = re.sub(r'\[S\d+\]', '', text)
    c = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', c)
    c = re.sub(r'[#`|>\[\]\\]', '', c)
    c = re.sub(r'\s+', ' ', c.replace('\n', ' ')).strip()[:3000]
    b64_text = base64.b64encode(c.encode('utf-8')).decode('ascii')
    k = re.sub(r'[^a-zA-Z0-9]', '_', key)[:30]
    components.html(
        f"""<!DOCTYPE html><html><head><style>
body{{margin:0;padding:2px 0;background:transparent;}}
button{{background:transparent;color:#5a5a72;border:1px solid #2a2a38;
border-radius:6px;padding:5px 14px;cursor:pointer;font-size:12px;
font-family:system-ui,sans-serif;font-weight:600;transition:all .12s;}}
button:hover{{background:#1a1a28;color:#9494ae;}}
</style></head><body>
<button id="b{k}" onclick="(function(b){{
  var s=window.speechSynthesis;
  if(!s){{b.textContent='Not supported';return;}}
  if(s.speaking){{s.cancel();b.textContent='🔊 Read Aloud';b.style.color='';return;}}
  try{{
    var u=new SpeechSynthesisUtterance(decodeURIComponent(escape(atob('{b64_text}'))));
    u.lang='en-US';u.rate=0.93;u.pitch=1.0;u.volume=1.0;
    u.onend=function(){{b.textContent='🔊 Read Aloud';b.style.color='';}};
    u.onerror=function(e){{b.textContent='Error: '+e.error;b.style.color='#f87171';}};
    b.textContent='⏹ Stop';b.style.color='#e8a045';
    s.speak(u);
  }}catch(err){{b.textContent='Error';}}
}})(this)">🔊 Read Aloud</button>
</body></html>""",
        height=44, scrolling=False,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _chips(sources: list) -> str:
    if not sources: return ""
    parts = []
    for s in sources:
        pg = f" p.{s['page']}" if s.get("page") is not None else ""
        parts.append(f'<span class="chip">{s["key"]}: {Path(s["source"]).name}{pg}</span>')
    return " ".join(parts)

def _eval_html(sc: dict) -> str:
    vals = {k: v for k, v in sc.items() if v is not None}
    if not vals: return ""
    cfg = {"faithfulness": ("#e8a045","Faithfulness"), "answer_relevance": ("#22d3ee","Relevance"), "context_precision": ("#4ade80","Precision")}
    blocks = []
    for k, v in vals.items():
        color, lbl = cfg.get(k, ("#6464805", k))
        pct = int(v * 100)
        tc  = "#4ade80" if pct >= 80 else "#fbbf24" if pct >= 60 else "#f87171"
        blocks.append(f'<div class="eval-block"><div class="eval-lbl">{lbl}</div>'
                       f'<div class="eval-bar"><div class="eval-fill" style="width:{pct}%;background:{color}80"></div></div>'
                       f'<div class="eval-val" style="color:{tc}">{pct}%</div></div>')
    return f'<div class="eval-row">{"".join(blocks)}</div>'

def _itag(i: str) -> str:
    return f'<span class="itag i-{i}">{i.upper()}</span>'

# ── Module-level persistent namespace store ──────────────────────────────────
# Keyed by session_id — survives Streamlit reruns unlike function-scoped dicts
_PY_NS_STORE: dict = {}

_PY_BOOT = (
    "import pandas as pd, numpy as np, matplotlib\n"
    "matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt, warnings, math, json\n"
    "warnings.filterwarnings('ignore')\n"
    "from sklearn.model_selection import train_test_split\n"
    "from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler\n"
    "from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso\n"
    "from sklearn.tree import DecisionTreeClassifier\n"
    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n"
    "from sklearn.metrics import mean_squared_error, r2_score, accuracy_score\n"
    "from sklearn.datasets import fetch_california_housing, load_iris, load_digits\n"
    "try:\n"
    "    import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim\n"
    "except ImportError: pass\n"
    "try: import seaborn as sns\n"
    "except ImportError: pass\n"
)


def _get_session_id() -> str:
    """Get a stable session ID that survives Streamlit reruns."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        if ctx:
            return ctx.session_id
    except Exception:
        pass
    if "_sid" not in st.session_state:
        import uuid
        st.session_state._sid = str(uuid.uuid4())
    return st.session_state._sid


def _get_py_ns() -> dict:
    """Return the persistent Python namespace for this session.
    Stored at MODULE level — survives Streamlit reruns.
    Variables from Step 1 stay available for Step 2, 3, etc.
    """
    sid = _get_session_id()
    if sid not in _PY_NS_STORE:
        ns: dict = {}
        try:
            exec(_PY_BOOT, ns)  # noqa: S102
        except Exception:
            pass
        _PY_NS_STORE[sid] = ns
    st.session_state._py_ns = _PY_NS_STORE[sid]
    return _PY_NS_STORE[sid]


def _auto_fix_code(code: str) -> tuple:
    """
    Universal auto-fix engine for ALL AI-generated Python code.
    Covers: Machine Learning, Deep Learning, NLP, Computer Vision, Reinforcement Learning,
            Data Science, Statistics, Mathematics, Web APIs, Databases, Algorithms,
            Data Structures, Visualization, Finance, Biology/Bioinformatics, Physics.
    Runs BEFORE exec() — student never sees avoidable errors.
    """
    import re as _r
    fixes = []

    # ═══════════════════════════════════════════════════════════════════════
    # UNIVERSAL FIXES — apply to all code
    # ═══════════════════════════════════════════════════════════════════════

    # U1: plt.show() blocks Streamlit rendering
    if 'plt.show()' in code:
        code = code.replace('plt.show()', 'plt.tight_layout()')
        fixes.append('🔧 plt.show() removed (Streamlit renders figures automatically)')

    # U2: Small default figure size — unreadable plots
    if 'plt.subplots()' in code and 'figsize' not in code:
        code = code.replace('plt.subplots()', 'plt.subplots(figsize=(10, 5))')
        fixes.append('🔧 figsize=(10,5) added for readable plots')

    if 'plt.figure()' in code and 'figsize' not in code:
        code = code.replace('plt.figure()', 'plt.figure(figsize=(10, 5))')

    # U3: Suppress noisy warnings for clean student output
    if 'import warnings' not in code and (
        'sklearn' in code or 'torch' in code or 'tensorflow' in code
        or 'keras' in code or 'scipy' in code
    ):
        code = 'import warnings; warnings.filterwarnings("ignore")\n' + code

    # U4: print() on tensors — need .item() or .numpy()
    if 'torch' in code:
        code = _r.sub(r'print\(loss\)', 'print(loss.item())', code)
        code = _r.sub(r'print\(loss\.item\(\)\)', 'print(loss.item())', code)  # idempotent

    # U5: Missing random seed — non-reproducible results
    if ('random.seed' not in code and 'np.random.seed' not in code
            and 'random_state' not in code and 'torch.manual_seed' not in code):
        if 'import numpy' in code or 'import random' in code:
            code = 'import random; random.seed(42)\nimport numpy as np; np.random.seed(42)\n' + code
            fixes.append('🔧 random seeds added (reproducible results)')

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 1 — SKLEARN / CLASSICAL ML
    # ═══════════════════════════════════════════════════════════════════════

    # 1a: load_boston removed sklearn 1.2
    if 'load_boston' in code:
        code = code.replace('from sklearn.datasets import load_boston',
                            'from sklearn.datasets import fetch_california_housing')
        code = _r.sub(r'load_boston\s*\(\s*\)', 'fetch_california_housing()', code)
        code = _r.sub(r'\bboston\b', 'housing', code)
        fixes.append('🔧 load_boston → fetch_california_housing (removed in sklearn 1.2)')

    # 1b: LogisticRegression convergence
    if 'LogisticRegression()' in code:
        code = code.replace('LogisticRegression()', 'LogisticRegression(max_iter=1000)')
        fixes.append('🔧 LogisticRegression(max_iter=1000) — prevents ConvergenceWarning')

    if _r.search(r'LogisticRegression\(max_iter\s*=\s*[1-9]\d?\)', code):
        code = _r.sub(r'LogisticRegression\(max_iter\s*=\s*\d+\)',
                      'LogisticRegression(max_iter=1000)', code)
        fixes.append('🔧 LogisticRegression max_iter increased to 1000')

    # 1c: SVC needs probability=True for predict_proba
    if 'SVC()' in code and 'predict_proba' in code:
        code = code.replace('SVC()', 'SVC(probability=True)')
        fixes.append('🔧 SVC(probability=True) — required to call predict_proba()')

    # 1d: KMeans n_init FutureWarning
    if 'KMeans(' in code and 'n_init' not in code:
        code = _r.sub(r'KMeans\(n_clusters\s*=\s*(\w+)\)',
                      r'KMeans(n_clusters=\1, n_init=10)', code)
        fixes.append('🔧 KMeans(n_init=10) — prevents FutureWarning in sklearn 1.4+')

    # 1e: GaussianMixture has no .pdf() — use scipy
    if 'GaussianMixture' in code and '.pdf(' in code:
        if 'from scipy.stats import norm' not in code:
            code = 'from scipy.stats import norm\n' + code
        fixes.append('🔧 Use scipy norm.pdf() — sklearn GaussianMixture has no .pdf()')

    # 1f: GMM on categorical column
    if 'GaussianMixture' in code and '[:, 5]' in code:
        code = code.replace('[:, 5]', '[:, 0]')
        fixes.append('🔧 column 5 → column 0 (GMM needs continuous variable, not categorical)')

    # 1g: GaussianMixture — completely rewrite bad GMM code with correct version
    if 'GaussianMixture' in code:
        # Detect: full dataset X (8 features) fed to GMM — must rewrite
        # GMM needs ONE continuous column, not the full feature matrix
        has_full_X    = 'X = data.data' in code or 'data.data' in code
        has_bad_fit   = 'gmm.fit(X' in code  # catches gmm.fit(X) and gmm.fit(X.reshape...)
        has_bad_plot  = 'gmm.predict(X)' in code or 'gmm.pdf(' in code
        # Only rewrite if it's California Housing / generic sklearn data context
        # Do NOT rewrite if it's NLP, Vision, Transformers, etc.
        is_calhousing_ctx = (
            'fetch_california_housing' in code or 'load_boston' in code
            or ('data.data' in code and 'transform' not in code.lower()
                and 'tokenize' not in code.lower() and 'bert' not in code.lower()
                and 'torch' not in code.lower() and 'embedding' not in code.lower())
        )
        needs_rewrite = has_full_X and (has_bad_fit or has_bad_plot) and is_calhousing_ctx

        if needs_rewrite:
            # Replace the entire GMM block with a correct, self-contained version
            code = """import warnings; warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

# Load dataset — use MedInc (column 0), a continuous variable ideal for GMM
data = fetch_california_housing()
medinc = data.data[:, 0]  # Median Income — continuous, good for GMM

# Fit GMM with 2 components on a single 1D feature reshaped to 2D
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(medinc.reshape(-1, 1))

# Plot histogram + GMM component curves using scipy norm.pdf (correct API)
x = np.linspace(medinc.min(), medinc.max(), 300).reshape(-1, 1)
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(medinc, bins=50, density=True, alpha=0.4, color='steelblue', label='Data')

for i in range(gmm.n_components):
    mean = gmm.means_[i, 0]
    std  = np.sqrt(gmm.covariances_[i, 0, 0])
    ax.plot(x, gmm.weights_[i] * norm.pdf(x, mean, std),
            linewidth=2.5, label=f'Component {i+1}  μ={mean:.2f}  σ={std:.2f}')

ax.set_title('Gaussian Mixture Model — Median Income Distribution')
ax.set_xlabel('Median Income (tens of thousands $)')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
print(f"Component weights: {gmm.weights_.round(3)}")
print(f"Component means:   {gmm.means_.flatten().round(3)}")
"""
            fixes.append('🔧 GMM code fully rewritten — original had wrong input shape. '
                         'Using MedInc column with correct scipy norm.pdf() visualization')
        else:
            # Partial fixes only — don't replace topic-specific code
            if has_bad_fit and 'reshape' not in code:
                code = _r.sub(r'gmm\.fit\((\w+)\)',
                              lambda m: f'gmm.fit({m.group(1)}.reshape(-1, 1))',
                              code, count=1)
                fixes.append('🔧 .reshape(-1,1) added — GaussianMixture expects 2D array')
            if '.pdf(' in code and 'gmm.pdf(' in code:
                if 'from scipy.stats import norm' not in code:
                    code = 'from scipy.stats import norm\n' + code
                fixes.append('🔧 GaussianMixture has no .pdf() — use scipy.stats.norm.pdf() instead')

    # 1h: train_test_split reproducibility
    if 'train_test_split(' in code and 'random_state' not in code:
        code = _r.sub(
            r'train_test_split\(([^)]+)\)',
            lambda m: f'train_test_split({m.group(1).rstrip()}, random_state=42)',
            code
        )
        fixes.append('🔧 random_state=42 added to train_test_split')

    # 1i: Data leakage — scaler fit on test set
    if _r.search(r'scaler\.fit\(X_test\)', code):
        code = _r.sub(r'scaler\.fit\(X_test\)',
                      'scaler.transform(X_test)  # fixed: use transform only on test set', code)
        fixes.append('🔧 scaler.fit(X_test) → scaler.transform(X_test) — prevents data leakage')

    # 1j: Ridge alpha=0 singular matrix
    if _r.search(r'Ridge\(alpha\s*=\s*0[^.]', code):
        code = _r.sub(r'Ridge\(alpha\s*=\s*0\b', 'Ridge(alpha=0.01', code)
        fixes.append('🔧 Ridge(alpha=0) → alpha=0.01 — alpha=0 causes singular matrix')

    # 1k: GridSearchCV — add cv explicitly
    if 'GridSearchCV(' in code and 'cv=' not in code:
        code = _r.sub(r'GridSearchCV\(([^)]+)\)',
                      lambda m: f'GridSearchCV({m.group(1)}, cv=5)', code)
        fixes.append('🔧 cv=5 added to GridSearchCV')

    # 1l: MLP needs max_iter
    for cls in ['MLPClassifier', 'MLPRegressor']:
        if f'{cls}()' in code:
            code = code.replace(f'{cls}()', f'{cls}(max_iter=500)')
            fixes.append(f'🔧 {cls}(max_iter=500) — prevents ConvergenceWarning')

    # 1m: Fake test input arrays → real sample
    for arr_m in list(_r.finditer(r'(\w+)\s*=\s*np\.array\(\[\[([^\]]+)\]\]\)', code)):
        vals = [v.strip() for v in arr_m.group(2).split(',') if v.strip()]
        n = len(vals)
        vname = arr_m.group(1)
        all_num = all(_r.match(r'^-?\d+(\.\d+)?$', v) for v in vals)
        is_seq  = all_num and n >= 3 and vals == [str(i) for i in range(1, n+1)]
        is_fake = all_num and n <= 20 and ('fit(' in code or 'predict' in code)
        if is_seq or is_fake:
            rep = 'X_test[0:1]' if 'X_test' in code else 'X[0:1]'
            code = code.replace(arr_m.group(0),
                f'{vname} = {rep}  # real sample — correct shape')
            fixes.append(
                f'🔧 np.array([[...{n} values]]) → {rep} '
                f'(real data sample guarantees correct feature count)')
            break

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 2 — PYTORCH / DEEP LEARNING
    # ═══════════════════════════════════════════════════════════════════════

    # 2a: Broaden .T fix — catches ALL tensor transpose issues not just matmul
    if '.T)' in code or ('.T,' in code and 'torch' in code):
        import re as _rt
        if 'torch' in code:
            f2 = _rt.sub(r'(\w+)\.T', lambda m: m.group(1)+'.transpose(-2,-1)', code)
            if f2 != code:
                fixes.append('🔧 .T → .transpose(-2,-1) — .T fails on batched 3D tensors')
                code = f2

    # 2a2: Fix nn.Linear input dim mismatching TransformerEncoderLayer d_model
    # Error: mat1 (NxM) and mat2 (PxQ) cannot be multiplied
    if 'TransformerEncoderLayer' in code and 'nn.Linear(' in code:
        import re as _rt2
        dm = _rt2.search(r'TransformerEncoderLayer\s*\(\s*d_model\s*=\s*(\d+)', code)
        if dm:
            d_model = dm.group(1)
            def _fix_linear(m):
                in_f = m.group(1).strip()
                out_f = m.group(2).strip()
                if in_f.isdigit() and in_f != d_model:
                    return f'nn.Linear({d_model}, {out_f})'
                return m.group(0)
            fixed_code = _rt2.sub(r'nn\.Linear\((\d+)\s*,\s*(\d+)\)', _fix_linear, code)
            if fixed_code != code:
                fixes.append(f'🔧 nn.Linear input dim corrected to d_model={d_model} (prevents mat shape mismatch)')
                code = fixed_code

    # 2a3: Fix wrong input tensor shape for Transformer
    if 'TransformerEncoderLayer' in code and ('torch.rand' in code or 'torch.zeros' in code or 'torch.randn' in code):
        import re as _rt3
        dm = _rt3.search(r'TransformerEncoderLayer\s*\(\s*d_model\s*=\s*(\d+)', code)
        if dm:
            d_model = int(dm.group(1))
            def _fix_rand(m):
                fn = m.group(1)
                dims = [x.strip() for x in m.group(2).split(',')]
                if len(dims) == 3 and dims[2].isdigit() and int(dims[2]) != d_model:
                    return f'torch.{fn}({dims[0]}, {dims[1]}, {d_model})'
                return m.group(0)
            fixed_code = _rt3.sub(r'torch\.(rand|zeros|randn|ones)\(([^)]+)\)', _fix_rand, code)
            if fixed_code != code:
                fixes.append(f'🔧 Tensor shape fixed — last dim must equal d_model={d_model}')
                code = fixed_code

    # 2b: .cuda() when no GPU — switch to device-agnostic
    if '.cuda()' in code and 'torch' in code:
        if 'device' not in code:
            code = ("device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
                    + code.replace('.cuda()', '.to(device)'))
            fixes.append('🔧 .cuda() → .to(device) — works on both GPU and CPU')

    # 2c: Missing optimizer.zero_grad() in training loop
    if 'loss.backward()' in code and 'zero_grad()' not in code:
        code = code.replace('loss.backward()', 'optimizer.zero_grad()\n    loss.backward()')

    # 2d: model.train() missing
    if 'torch' in code and 'nn.Module' in code:
        if 'model.train()' not in code and 'for epoch' in code:
            code = code.replace('for epoch', 'model.train()\nfor epoch', 1)
            fixes.append('🔧 model.train() added before training loop')

    # 2e: CrossEntropyLoss + Softmax = double softmax
    if 'CrossEntropyLoss' in code and 'Softmax' in code:
        fixes.append('💡 CrossEntropyLoss already applies softmax — remove nn.Softmax() from model output')

    # 2f: DataLoader shuffle missing for training
    if 'DataLoader(' in code and 'shuffle' not in code and 'train' in code.lower():
        import re as _rt4
        code = _rt4.sub(r'DataLoader\((\w+_train[^)]*)\)',
                        r'DataLoader(, shuffle=True)', code)
        fixes.append('🔧 shuffle=True added to training DataLoader')

    # 2g: .numpy() without .detach()
    if '.numpy()' in code and 'torch' in code and '.detach()' not in code:
        code = code.replace('.numpy()', '.detach().numpy()')
        fixes.append('🔧 .detach().numpy() — must detach from computation graph first')

    # 2h: torch.Tensor deprecated constructor
    if 'torch.Tensor(' in code:
        code = code.replace('torch.Tensor(', 'torch.tensor(')
        fixes.append('🔧 torch.Tensor() → torch.tensor() (lowercase preferred)')

        # ═══════════════════════════════════════════════════════════════════════
    # GROUP 3 — TENSORFLOW / KERAS
    # ═══════════════════════════════════════════════════════════════════════

    # 3a: Old keras import style (TF 1.x)
    if 'from keras' in code and 'tensorflow' not in code:
        code = code.replace('from keras', 'from tensorflow.keras')
        fixes.append('🔧 from keras → from tensorflow.keras (TF 2.x style)')

    # 3b: model.fit missing validation_split
    if 'model.fit(' in code and 'validation_split' not in code and 'validation_data' not in code:
        code = _r.sub(
            r'(model\.fit\([^)]+)(epochs\s*=\s*\d+)([^)]*\))',
            lambda m: m.group(1) + m.group(2) + ', validation_split=0.2' + m.group(3),
            code
        )
        fixes.append('🔧 validation_split=0.2 added to model.fit()')

    # 3c: Missing model.compile() before model.fit()
    if 'model.fit(' in code and 'model.compile(' not in code:
        fixes.append('💡 model.compile(optimizer, loss, metrics) must be called before model.fit()')

    # 3d: Deprecated tf.placeholder (TF 1.x)
    if 'tf.placeholder' in code:
        fixes.append('💡 tf.placeholder removed in TF 2.x — use tf.keras Input layers or eager tensors')

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 4 — NLP / TRANSFORMERS / HUGGINGFACE
    # ═══════════════════════════════════════════════════════════════════════

    # 4a: tokenizer padding/truncation missing
    if 'tokenizer(' in code and 'truncation' not in code and 'padding' not in code:
        code = _r.sub(
            r'tokenizer\(([^)]+)\)',
            lambda m: f'tokenizer({m.group(1)}, truncation=True, padding=True)',
            code, count=1
        )
        fixes.append('🔧 truncation=True, padding=True added to tokenizer call')

    # 4b: NLTK resources not downloaded
    if 'nltk' in code:
        needed = []
        if 'word_tokenize' in code or 'sent_tokenize' in code:
            needed.append("'punkt'")
        if 'stopwords' in code:
            needed.append("'stopwords'")
        if 'WordNetLemmatizer' in code:
            needed.append("'wordnet'")
        if 'pos_tag' in code:
            needed.append("'averaged_perceptron_tagger'")
        if needed and 'nltk.download' not in code:
            dl = '\n'.join([f"nltk.download({r}, quiet=True)" for r in needed])
            code = 'import nltk\n' + dl + '\n' + code
            fixes.append(f'🔧 nltk.download() added for: {", ".join(needed)}')

    # 4c: spaCy model not loaded
    if 'spacy.load(' in code and 'en_core_web' not in code:
        code = code.replace("spacy.load('en')", "spacy.load('en_core_web_sm')")
        code = code.replace('spacy.load("en")', "spacy.load('en_core_web_sm')")
        fixes.append("🔧 spacy.load('en_core_web_sm') — 'en' model name deprecated")

    # 4d: CountVectorizer/TfidfVectorizer fit on test
    if _r.search(r'vectorizer\.fit\(.*test', code):
        fixes.append('💡 Do not fit vectorizer on test data — use fit_transform on train, transform on test')

    # 4e: cosine_similarity with 1D arrays — sklearn expects 2D (n_samples, n_features)
    # Error: "Expected 2D array, got 1D array instead"
    if 'cosine_similarity' in code and ('sklearn' in code or 'from sklearn' in code):
        # Fix: cosine_similarity(vec[0], vec[1]) → cosine_similarity(vec[0].reshape(1,-1), vec[1].reshape(1,-1))
        # Pattern: cosine_similarity(X[N], Y[M]) or cosine_similarity(arr[N], arr[M])
        code = _r.sub(
            r'cosine_similarity\(\s*(\w+)\[(\d+)\]\s*,\s*(\w+)\[(\d+)\]\s*\)',
            r'cosine_similarity(\1[\2].reshape(1, -1), \3[\4].reshape(1, -1))',
            code
        )
        # Pattern: cosine_similarity(vec, vec2) where vecs are likely 1D — wrap both
        # Only if no .reshape already present and not already 2D slices like [:, :]
        if '.reshape' not in code and 'cosine_similarity' in code:
            code = _r.sub(
                r'cosine_similarity\(\s*(\w+)\s*,\s*(\w+)\s*\)',
                lambda m: (
                    f'cosine_similarity({m.group(1)}.reshape(1, -1) if {m.group(1)}.ndim == 1 '
                    f'else {m.group(1)}, '
                    f'{m.group(2)}.reshape(1, -1) if {m.group(2)}.ndim == 1 '
                    f'else {m.group(2)})'
                ),
                code
            )
        fixes.append('🔧 cosine_similarity inputs reshaped to 2D — sklearn requires (n_samples, n_features) arrays')

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 5 — COMPUTER VISION / IMAGE PROCESSING
    # ═══════════════════════════════════════════════════════════════════════

    # 5a: cv2 imshow() doesn't work in Streamlit
    if 'cv2.imshow(' in code:
        code = _r.sub(
            r'cv2\.imshow\([^)]+\)\s*\n?\s*cv2\.waitKey\([^)]*\)\s*\n?\s*cv2\.destroyAllWindows\(\)',
            '# cv2.imshow replaced — using matplotlib for Streamlit\nplt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))',
            code
        )
        if 'cv2.imshow(' in code:
            code = _r.sub(r'cv2\.imshow\(([^,]+),\s*(\w+)\)',
                          r'plt.imshow(cv2.cvtColor(\2, cv2.COLOR_BGR2RGB)); plt.axis("off")',
                          code)
        fixes.append('🔧 cv2.imshow() → plt.imshow() (cv2.imshow does not work in Streamlit)')

    # 5b: cv2 reads BGR — matplotlib needs RGB
    if 'cv2.imread(' in code and 'COLOR_BGR2RGB' not in code and 'plt.' in code:
        code = _r.sub(
            r'(\w+)\s*=\s*cv2\.imread\(([^)]+)\)',
            lambda m: (f'{m.group(1)} = cv2.imread({m.group(2)})\n'
                       f'{m.group(1)} = cv2.cvtColor({m.group(1)}, cv2.COLOR_BGR2RGB)'),
            code
        )
        fixes.append('🔧 cv2.COLOR_BGR2RGB added — OpenCV reads BGR, matplotlib needs RGB')

    # 5c: PIL Image show() doesn't work in Streamlit
    if '.show()' in code and ('PIL' in code or 'Image' in code):
        code = _r.sub(r'(\w+)\.show\(\)',
                      r'plt.imshow(\1); plt.axis("off")', code)
        fixes.append('🔧 image.show() → plt.imshow() (PIL .show() not supported in Streamlit)')

    # 5d: torchvision transforms — ToTensor deprecated in newer versions
    if 'transforms.ToTensor()' in code and 'torchvision' in code:
        fixes.append('💡 In torchvision 0.18+, use transforms.v2.ToImage() + ToDtype() instead of ToTensor()')

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 6 — REINFORCEMENT LEARNING
    # ═══════════════════════════════════════════════════════════════════════

    # 6a: OpenAI Gym deprecated — use Gymnasium
    if 'import gym' in code and 'gymnasium' not in code:
        code = code.replace('import gym', 'import gymnasium as gym')
        fixes.append('🔧 import gym → import gymnasium as gym (gym deprecated, use gymnasium)')

    # 6b: env.reset() returns tuple in Gymnasium (obs, info)
    if 'gymnasium' in code or ('gym' in code and 'reset()' in code):
        if _r.search(r'(\w+)\s*=\s*env\.reset\(\)', code):
            code = _r.sub(r'(\w+)\s*=\s*env\.reset\(\)',
                          r'\1, _ = env.reset()', code)
            fixes.append('🔧 obs, _ = env.reset() — Gymnasium reset() returns (obs, info)')

    # 6c: env.step() returns 5 values in Gymnasium
    if ('gymnasium' in code or 'gym' in code) and 'env.step(' in code:
        if _r.search(r'(\w+),\s*(\w+),\s*(\w+),\s*(\w+)\s*=\s*env\.step', code):
            code = _r.sub(
                r'(\w+),\s*(\w+),\s*(\w+),\s*(\w+)\s*=\s*env\.step\(([^)]+)\)',
                r'\1, \2, \3, _, \4 = env.step(\5)', code
            )
            fixes.append('🔧 env.step() returns 5 values in Gymnasium: obs,reward,terminated,truncated,info')

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 7 — PANDAS / DATA SCIENCE
    # ═══════════════════════════════════════════════════════════════════════

    # 7a: df.append() removed in pandas 2.0
    if '.append(' in code and ('DataFrame' in code or 'df' in code.lower()):
        code = _r.sub(
            r'(\w+)\s*=\s*(\w+)\.append\(([^)]+),\s*ignore_index\s*=\s*True\)',
            r'\1 = pd.concat([\2, pd.DataFrame([\3])], ignore_index=True)',
            code
        )
        if '.append(' in code:
            code = _r.sub(
                r'(\w+)\.append\(([^)]+)\)',
                r'pd.concat([\1, pd.DataFrame([\2])], ignore_index=True)',
                code
            )
            fixes.append('🔧 df.append() → pd.concat() (removed in pandas 2.0)')

    # 7b: inplace=True in chained operations (warning in pandas 2.0)
    if 'inplace=True' in code:
        fixes.append('💡 inplace=True may cause warnings in pandas 2.0 — prefer df = df.method()')

    # 7c: Deprecated pd.to_datetime format
    if 'pd.to_datetime(' in code and 'format=' not in code and 'infer_datetime_format' in code:
        code = code.replace('infer_datetime_format=True', "format='mixed'")
        fixes.append('🔧 infer_datetime_format → format="mixed" (deprecated in pandas 2.0)')

    # 7d: fillna with method= deprecated
    if '.fillna(' in code and "method='" in code:
        code = code.replace(".fillna(method='ffill')", '.ffill()')
        code = code.replace(".fillna(method='bfill')", '.bfill()')
        fixes.append('🔧 fillna(method=) → .ffill()/.bfill() (method= deprecated in pandas 2.0)')

    # 7e: Missing axis in apply
    if '.apply(lambda' in code and 'axis=' not in code and 'DataFrame' in code:
        fixes.append('💡 df.apply(lambda, axis=1) for row-wise operations, axis=0 for column-wise')

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 8 — NUMPY / MATH
    # ═══════════════════════════════════════════════════════════════════════

    # 8a: Deprecated numpy type aliases
    for old_t, new_t in [
        ('np.bool,', 'bool,'), ('np.bool)', 'bool)'), ('np.bool\n', 'bool\n'),
        ('np.int,',  'int,'),  ('np.int)',  'int)'),  ('np.int\n',  'int\n'),
        ('np.float,','float,'),('np.float)','float)'),('np.float\n','float\n'),
        ('np.complex,','complex,'),('np.complex)','complex)'),
        ('np.object,','object,'),('np.str,','str,'),
    ]:
        if old_t in code:
            code = code.replace(old_t, new_t)
            fixes.append(f'🔧 {old_t.strip()} → {new_t.strip()} (deprecated numpy alias)')

    # 8b: Division returning int in Python 3
    if _r.search(r'\b(\d+)\s*/\s*(\d+)\b', code) and 'import numpy' not in code:
        fixes.append('💡 In Python 3, use // for integer division and / for float division')

    # 8c: np.matrix deprecated — use np.array
    if 'np.matrix(' in code:
        code = code.replace('np.matrix(', 'np.array(')
        fixes.append('🔧 np.matrix() → np.array() (np.matrix deprecated)')

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 9 — MATPLOTLIB / SEABORN / VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════

    # 9a: seaborn heatmap fmt warning
    if 'sns.heatmap' in code and 'annot=True' in code and 'fmt=' not in code:
        code = code.replace('annot=True', "annot=True, fmt='.2f'")
        fixes.append('🔧 fmt=".2f" added to sns.heatmap (prevents dtype warning)')

    # 9b: seaborn deprecated distplot → histplot
    if 'sns.distplot(' in code:
        code = code.replace('sns.distplot(', 'sns.histplot(')
        fixes.append('🔧 sns.distplot() → sns.histplot() (distplot removed in seaborn 0.12)')

    # 9c: seaborn deprecated kdeplot shade → fill
    if 'shade=True' in code and 'sns' in code:
        code = code.replace('shade=True', 'fill=True')
        fixes.append('🔧 shade=True → fill=True (shade deprecated in seaborn 0.12)')

    # 9d: plt.xticks rotation — add ha for readability
    if 'plt.xticks(' in code and 'rotation=' in code and 'ha=' not in code:
        code = _r.sub(r'plt\.xticks\(rotation\s*=\s*(\d+)',
                      r"plt.xticks(rotation=\1, ha='right'", code)
        fixes.append("🔧 ha='right' added to plt.xticks for readable labels")

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 10 — STATISTICS / SCIPY
    # ═══════════════════════════════════════════════════════════════════════

    # 10a: scipy.stats.pearsonr returns (r, p) — AI often forgets p-value
    if 'pearsonr(' in code:
        if not _r.search(r'\w+,\s*\w+\s*=\s*.*pearsonr', code):
            code = _r.sub(r'(\w+)\s*=\s*.*pearsonr\(([^)]+)\)',
                          r'\1, p_value = stats.pearsonr(\2)', code)
            fixes.append('🔧 pearsonr returns (r, p_value) — both unpacked')

    # 10b: scipy.stats deprecated in newer versions
    if 'from scipy.stats import' not in code and 'scipy.stats.' in code:
        code = 'from scipy import stats\n' + code
        fixes.append('🔧 from scipy import stats added')

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 11 — WEB / APIs / REQUESTS
    # ═══════════════════════════════════════════════════════════════════════

    # 11a: requests without timeout — hangs forever
    if 'requests.get(' in code and 'timeout' not in code:
        code = _r.sub(r'requests\.get\(([^)]+)\)',
                      lambda m: f'requests.get({m.group(1)}, timeout=10)', code)
        fixes.append('🔧 timeout=10 added to requests.get() — prevents hanging')

    if 'requests.post(' in code and 'timeout' not in code:
        code = _r.sub(r'requests\.post\(([^)]+)\)',
                      lambda m: f'requests.post({m.group(1)}, timeout=10)', code)
        fixes.append('🔧 timeout=10 added to requests.post()')

    # 11b: Missing response.raise_for_status()
    if 'requests.' in code and 'raise_for_status' not in code and 'response' in code:
        code = _r.sub(r'(response\s*=\s*requests\.\w+\([^)]+\))',
                      r'\1\nresponse.raise_for_status()  # raises error for 4xx/5xx', code)
        fixes.append('🔧 response.raise_for_status() added — catches HTTP errors')

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 12 — FILE HANDLING
    # ═══════════════════════════════════════════════════════════════════════

    # 12a: open() without encoding
    if "open('" in code and 'encoding' not in code and ('read()' in code or 'write(' in code):
        code = _r.sub(r"open\('([^']+)',\s*'([rwa]b?)'\)",
                      r"open('\1', '\2', encoding='utf-8')", code)
        code = _r.sub(r'open\("([^"]+)",\s*"([rwa]b?)"\)',
                      r'open("\1", "\2", encoding="utf-8")', code)
        fixes.append("🔧 encoding='utf-8' added to open() — prevents encoding errors")

    # 12b: File not closed — prefer with statement
    if 'open(' in code and 'with open' not in code and '.close()' not in code:
        fixes.append("💡 Use 'with open(file) as f:' to ensure file is always closed properly")

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 13 — DATABASES / SQL
    # ═══════════════════════════════════════════════════════════════════════

    # 13a: SQL injection risk — f-string in SQL (safe string check, no regex)
    if 'execute(f"' in code or "execute(f'" in code:
        fixes.append('⚠️ SQL injection risk: use parameterized queries — cursor.execute(sql, (value,)) instead of f-strings')

    # 13b: Missing connection close
    if ('sqlite3.connect(' in code or 'psycopg2.connect(' in code) and '.close()' not in code:
        fixes.append("💡 Always close DB connections: conn.close() or use 'with' context manager")

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 14 — ALGORITHMS / DATA STRUCTURES
    # ═══════════════════════════════════════════════════════════════════════

    # 14a: Mutable default argument (classic Python gotcha)
    if _r.search(r'def \w+\([^)]*=\s*\[\]', code):
        fixes.append('⚠️ Mutable default argument: def f(x=[]) is a bug — use def f(x=None) and x = x or []')

    if _r.search(r'def \w+\([^)]*=\s*\{\}', code):
        fixes.append('⚠️ Mutable default argument: def f(x={}) is a bug — use def f(x=None) and x = x or {}')

    # 14b: Recursion limit — safe check without backreference
    if 'def ' in code and code.count('return ') > 0:
        fn_names = _r.findall(r'def (\w+)\s*\(', code)
        is_recursive = any(
            code.count(fn + '(') > 1
            for fn in fn_names if fn not in ('__init__', 'forward', 'call')
        )
        if is_recursive and 'sys.setrecursionlimit' not in code:
            fixes.append('💡 Recursive function detected — add sys.setrecursionlimit(10000) if hitting RecursionError')

    # 14c: List comprehension inside loop (inefficient pattern — inform)
    if code.count('for ') > 2 and '[' in code:
        pass  # too many false positives, skip

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 15 — FINANCE / QUANTITATIVE
    # ═══════════════════════════════════════════════════════════════════════

    # 15a: yfinance API change
    if 'yfinance' in code or 'yf.download(' in code:
        if 'auto_adjust' not in code and 'yf.download(' in code:
            code = _r.sub(r'yf\.download\(([^)]+)\)',
                          lambda m: f'yf.download({m.group(1)}, auto_adjust=True)', code)
            fixes.append('🔧 auto_adjust=True added to yf.download() — adjusts for splits/dividends')

    # 15b: pandas_datareader deprecated source
    if 'web.DataReader(' in code and "'yahoo'" in code:
        fixes.append('💡 pandas_datareader Yahoo is unreliable — use yfinance instead: yf.download(ticker, start, end)')

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 16 — BIOINFORMATICS / SCIENTIFIC
    # ═══════════════════════════════════════════════════════════════════════

    # 16a: Biopython deprecated parsers
    if 'Bio.SeqIO' in code and 'parse(' in code:
        if 'with open' not in code and 'open(' not in code:
            fixes.append('💡 Biopython SeqIO.parse() needs an open file handle: SeqIO.parse(open(file), format)')

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 17 — GENERAL PYTHON BEST PRACTICES
    # ═══════════════════════════════════════════════════════════════════════

    # 17a: except: bare (catches everything including KeyboardInterrupt)
    if _r.search(r'except\s*:', code):
        code = _r.sub(r'except\s*:', 'except Exception:', code)
        fixes.append('🔧 except: → except Exception: (bare except catches KeyboardInterrupt too)')

    # 17b: == None instead of is None
    if '== None' in code:
        code = code.replace('== None', 'is None')
        fixes.append('🔧 == None → is None (use identity check for None)')

    if '!= None' in code:
        code = code.replace('!= None', 'is not None')
        fixes.append('🔧 != None → is not None')

    # 17c: print statement (Python 2 style)
    if _r.search(r'^print\s+[^(]', code, _r.MULTILINE):
        code = _r.sub(r'^print\s+(.+)$', r'print(\1)', code, flags=_r.MULTILINE)
        fixes.append('🔧 print statement → print() function (Python 3 style)')

    # 17d: xrange (Python 2) → range (Python 3)
    if 'xrange(' in code:
        code = code.replace('xrange(', 'range(')
        fixes.append('🔧 xrange() → range() (xrange removed in Python 3)')

    # 17e: unicode() (Python 2) → str() (Python 3)
    if 'unicode(' in code:
        code = code.replace('unicode(', 'str(')
        fixes.append('🔧 unicode() → str() (unicode removed in Python 3)')

    # 17f: has_key() (Python 2) → in operator
    if '.has_key(' in code:
        code = _r.sub(r'(\w+)\.has_key\(([^)]+)\)', r'\2 in \1', code)
        fixes.append('🔧 dict.has_key() → key in dict (has_key removed in Python 3)')

    return code, fixes


def _run_py(code: str) -> None:
    import contextlib

    # ── Auto-fix known AI code bugs before running ────────────────────────────
    code, auto_fixes = _auto_fix_code(code)
    for fix_msg in auto_fixes:
        st.info(fix_msg)

    # ── shared namespace (persists across all code blocks in this session) ──
    ns = _get_py_ns()

    ob = io.StringIO()
    eb = io.StringIO()
    with contextlib.redirect_stdout(ob), contextlib.redirect_stderr(eb):
        try:
            exec(code, ns)  # noqa: S102
        except Exception:
            eb.write(traceback.format_exc())

    if ob.getvalue():
        st.code(ob.getvalue(), language="text")
    if eb.getvalue():
        err = eb.getvalue()
        hint = ""

        # ── Smart error hints ─────────────────────────────────────────────
        if "NameError" in err:
            var = re.search(r"name '(\w+)' is not defined", err)
            if var:
                hint = (f"\n\n💡 **Tip:** `{var.group(1)}` was not found. "
                        "Run the earlier code block(s) above first by clicking "
                        "**▶ Run Python** on each step in order.")

        elif "mat1 and mat2 shapes cannot be multiplied" in err:
            # Extract shape info for a specific fix hint
            shapes = re.search(r"\((\d+x\d+) and (\d+x\d+)\)", err)
            shape_info = f" ({shapes.group(1)} vs {shapes.group(2)})" if shapes else ""
            hint = (
                f"\n\n💡 **Matrix shape mismatch{shape_info}** — this is a PyTorch dimension bug "
                "in the generated code.\n\n"
                "**For attention mechanisms**, the fix is to transpose correctly:\n"
                "```python\n"
                "# ❌ Wrong — treats last two dims incorrectly\n"
                "scores = torch.matmul(query, key.T)\n\n"
                "# ✅ Correct — transpose only the last two dimensions\n"
                "scores = torch.matmul(query, key.transpose(-2, -1))\n"
                "```\n"
                "Replace `key.T` with `key.transpose(-2, -1)` in the attention code."
            )

        elif "RuntimeError" in err and "size mismatch" in err:
            hint = ("\n\n💡 **Size mismatch** — check that your input tensor dimensions match "
                    "the model's expected input size (e.g. `nn.Linear(in, out)` — `in` must "
                    "match your data's last dimension).")

        elif "ModuleNotFoundError" in err or "ImportError" in err:
            mod = re.search(r"No module named '([\w.]+)'", err)
            if mod:
                pkg = mod.group(1).split(".")[0]
                hint = (f"\n\n💡 **Missing package:** Run this in your terminal first:\n"
                        f"```bash\npip install {pkg}\n```")

        elif "CUDA" in err or "cuda" in err:
            hint = ("\n\n💡 **CUDA error** — your code is trying to use GPU but it may not "
                    "be available. Add `.cpu()` or remove `.cuda()` / `.to(device)` calls "
                    "to run on CPU instead.")

        st.error(f"```\n{err}```{hint}")

    # Show any matplotlib figures
    try:
        import matplotlib.pyplot as plt
        figs = [plt.figure(i) for i in plt.get_fignums()]
        for fig in figs:
            st.pyplot(fig, use_container_width=True)
        plt.close("all")
    except Exception:
        pass

    # Show any DataFrames produced
    try:
        import pandas as pd
        for vn, vv in ns.items():
            if (not vn.startswith("_")
                    and isinstance(vv, pd.DataFrame)
                    and len(vv) > 0):
                st.caption(f"📊 `{vn}`: {vv.shape[0]} rows × {vv.shape[1]} cols")
                st.dataframe(vv.head(25), use_container_width=True)
    except Exception:
        pass

    # Show a small "variables available" summary so the student knows what carried over
    skip = {"pd", "np", "plt", "matplotlib", "warnings", "train_test_split",
            "StandardScaler", "LinearRegression", "LogisticRegression",
            "mean_squared_error", "r2_score", "accuracy_score"}
    try:
        import pandas as pd, numpy as np
        live_vars = {
            k: type(v).__name__
            for k, v in ns.items()
            if not k.startswith("_") and k not in skip
        }
        if live_vars:
            summary = "  ·  ".join(f"`{k}` ({t})" for k, t in list(live_vars.items())[:8])
            st.caption(f"🔗 **Session variables:** {summary}")
    except Exception:
        pass

def _build_smart_diagram(topic: str, content: str, diagram_type: str = "auto") -> None:
    """
    100% Python-native beautiful visualization engine.
    Zero LLM involvement in drawing code — guaranteed quality diagrams every time.
    Parses the AI text, extracts real concepts/steps, renders via matplotlib.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patheffects as pe
    import numpy as np
    import re as _re

    # ── Palette ────────────────────────────────────────────────────────────
    BG       = '#0a0a14'
    BOX_FILL = '#0f1520'
    ACCENT   = '#3b82f6'
    AMBER    = '#f59e0b'
    GREEN    = '#10b981'
    PURPLE   = '#8b5cf6'
    PINK     = '#ec4899'
    CYAN     = '#06b6d4'
    ORANGE   = '#f97316'
    TEXT_HI  = '#f1f5f9'
    TEXT_MD  = '#94a3b8'
    TEXT_DIM = '#475569'
    PALETTE  = [ACCENT, AMBER, GREEN, PURPLE, PINK, CYAN, ORANGE, '#a78bfa']

    # ── Helpers ─────────────────────────────────────────────────────────────
    def clean(t: str) -> str:
        t = _re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', t)
        t = _re.sub(r'\[S\d+\]|\(page=\d+\)', '', t)
        return t.strip(' :•-–—\t')

    def word_wrap(text: str, width: int = 28) -> str:
        words = text.split(); lines, cur = [], []
        for w in words:
            cur.append(w)
            if len(' '.join(cur)) > width:
                lines.append(' '.join(cur[:-1])); cur = [w]
        if cur: lines.append(' '.join(cur))
        return '\n'.join(lines[:3])

    def parse_concepts(txt: str):
        """Extract (title, description) pairs from AI markdown text."""
        pairs = []
        seen = set()
        # 1) **Bold Term**: description
        for m in _re.finditer(r'\*\*([^*]{3,55})\*\*\s*[:\-–]\s*([^*\n]{10,200})', txt):
            t = clean(m.group(1)); d = clean(m.group(2))
            key = t.lower()[:20]
            if len(t) > 2 and key not in seen:
                pairs.append((t, d)); seen.add(key)
        # 2) Bullet lines  • Term: description
        if len(pairs) < 3:
            for line in txt.splitlines():
                line = line.strip()
                if not line or not line[0] in '•*-':
                    continue
                c = clean(_re.sub(r'^[•*\-–]+\s*', '', line))
                parts = c.split(':', 1)
                if len(parts) == 2 and 3 < len(parts[0]) < 48:
                    t, d = parts[0].strip(), parts[1].strip()
                    key = t.lower()[:20]
                    if key not in seen and len(t) > 2 and len(d) > 5:
                        pairs.append((t, d)); seen.add(key)
        # 3) Section headers (bold only, no colon)
        if len(pairs) < 3:
            for m in _re.finditer(r'\*\*([^*]{4,50})\*\*', txt):
                t = clean(m.group(1))
                key = t.lower()[:20]
                if key not in seen and len(t) > 3:
                    pairs.append((t, '')); seen.add(key)
        return pairs[:8] or [('Concept A', ''), ('Concept B', ''), ('Concept C', '')]

    # ── Shared: draw a beautiful card/box ───────────────────────────────────
    def draw_card(ax, x, y, W, H, title, desc, color, num=None, alpha=1.0):
        # Drop shadow
        ax.add_patch(FancyBboxPatch((x+0.06, y-0.06), W, H,
            boxstyle="round,pad=0.10", linewidth=0,
            facecolor='#000010', alpha=0.50, zorder=1))
        # Glow ring
        ax.add_patch(FancyBboxPatch((x-0.04, y-0.04), W+0.08, H+0.08,
            boxstyle="round,pad=0.10", linewidth=0,
            facecolor=color, alpha=0.10, zorder=2))
        # Main card
        ax.add_patch(FancyBboxPatch((x, y), W, H,
            boxstyle="round,pad=0.09", linewidth=2.0,
            edgecolor=color, facecolor=BOX_FILL, alpha=alpha, zorder=3))
        # Top accent stripe
        ax.add_patch(FancyBboxPatch((x+0.18, y+H-0.115), W-0.36, 0.10,
            boxstyle="round,pad=0.02", linewidth=0,
            facecolor=color, alpha=0.80, zorder=4))
        # Left accent bar
        ax.add_patch(FancyBboxPatch((x+0.04, y+H*0.14), 0.10, H*0.72,
            boxstyle="round,pad=0.01", linewidth=0,
            facecolor=color, alpha=0.85, zorder=4))
        # Number circle badge
        cx_off = 0
        if num is not None:
            bx, by = x + W*0.14, y + H - 0.42
            ax.add_patch(plt.Circle((bx, by), 0.27,
                color=color, zorder=5, clip_on=False))
            ax.text(bx, by, str(num), ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white', zorder=6)
            cx_off = 0.12
        # Title
        short = title if len(title) <= 24 else title[:22]+'…'
        ax.text(x + W*(0.55+cx_off*0.3), y + H*0.67, short,
                ha='center', va='center',
                fontsize=10, fontweight='bold', color=TEXT_HI, zorder=5)
        # Description
        if desc:
            ax.text(x + W*0.56, y + H*0.27, word_wrap(desc[:90], 30),
                    ha='center', va='center', fontsize=7.8,
                    color=TEXT_MD, zorder=5, linespacing=1.45)

    def arrow(ax, x1, y1, x2, y2, color=AMBER, rad=0.0, lbl=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle='->', color=color, lw=2.2,
                mutation_scale=22, connectionstyle=f'arc3,rad={rad}'), zorder=7)
        if lbl:
            ax.text((x1+x2)/2, (y1+y2)/2+0.12, lbl, ha='center', va='bottom',
                    fontsize=7.5, color=AMBER, fontweight='600', zorder=8)

    def page_title(ax, title, subtitle, fig_h):
        ax.text(7, fig_h - 0.30, title, ha='center', va='top',
                fontsize=18, fontweight='bold', color=TEXT_HI,
                path_effects=[pe.withStroke(linewidth=5, foreground=BG)], zorder=10)
        ax.text(7, fig_h - 0.80, subtitle, ha='center', va='top',
                fontsize=10, color=TEXT_DIM, zorder=10)
        ax.plot([1.0, 13.0], [fig_h-1.05, fig_h-1.05], color=ACCENT, lw=0.4, alpha=0.25)

    # ── Choose diagram type ────────────────────────────────────────────────
    low = content.lower()
    dt = diagram_type.lower()
    if 'pipeline' in dt or 'flow' in dt:
        dtype = 'pipeline'
    elif 'architecture' in dt:
        dtype = 'architecture'
    elif 'comparison' in dt or 'side' in dt:
        dtype = 'comparison'
    elif 'timeline' in dt:
        dtype = 'timeline'
    elif 'concept' in dt or 'map' in dt or 'mind' in dt:
        dtype = 'mindmap'
    else:  # auto
        has_process = any(w in low for w in ['step', 'first', 'then ', 'next', 'pipeline', 'process', 'phase', 'stage'])
        has_compare = any(w in low for w in ['compare', 'versus', 'vs.', 'differ', 'unlike', 'while', 'whereas'])
        has_arch    = any(w in low for w in ['architecture', 'module', 'layer', 'component', 'system', 'consist'])
        if has_compare:   dtype = 'comparison'
        elif has_arch:    dtype = 'architecture'
        elif has_process: dtype = 'pipeline'
        else:             dtype = 'mindmap'

    concepts = parse_concepts(content)
    n = len(concepts)

    # ══════════════════════════════════════════════════════════════════
    # PIPELINE — numbered step-by-step flow (3 per row max)
    # ══════════════════════════════════════════════════════════════════
    if dtype == 'pipeline':
        COLS = 3
        rows = (n + COLS - 1) // COLS
        CW, CH, GX, GY = 3.8, 2.4, 0.7, 1.2
        total_w = COLS * CW + (COLS-1) * GX
        sx = (14 - total_w) / 2
        fig_h = 1.6 + rows * (CH + GY) + 0.3
        fig, ax = plt.subplots(figsize=(14, fig_h))
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        ax.set_xlim(0, 14); ax.set_ylim(0, fig_h); ax.axis('off')
        page_title(ax, topic, 'Step-by-Step Process Flow', fig_h)
        pos = []
        for i, (t, d) in enumerate(concepts):
            r, c = divmod(i, COLS)
            x = sx + c * (CW + GX)
            y = (fig_h - 1.5) - r * (CH + GY) - CH
            draw_card(ax, x, y, CW, CH, t, d, PALETTE[i % len(PALETTE)], num=i+1)
            pos.append((x, y, CW, CH))
        # Arrows
        for i in range(len(pos)-1):
            x1,y1,w1,h1 = pos[i]; x2,y2,w2,h2 = pos[i+1]
            r1, r2 = i//COLS, (i+1)//COLS
            if r1 == r2:
                arrow(ax, x1+w1, y1+h1/2, x2, y2+h2/2)
            else:
                ax.annotate('', xy=(x2+w2/2, y2+h2),
                    xytext=(x1+w1/2, y1),
                    arrowprops=dict(arrowstyle='->', color=AMBER, lw=2.0,
                        mutation_scale=18, connectionstyle='arc3,rad=0.0'), zorder=7)
        plt.tight_layout(pad=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════
    # MIND MAP — central concept with radiating nodes
    # ══════════════════════════════════════════════════════════════════
    elif dtype == 'mindmap':
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        ax.set_xlim(-7, 7); ax.set_ylim(-5.5, 5.5); ax.axis('off')
        # Central node
        ax.add_patch(FancyBboxPatch((-3.2, -1.1), 6.4, 2.2,
            boxstyle="round,pad=0.15", linewidth=3,
            edgecolor=AMBER, facecolor='#130e00', alpha=0.97, zorder=6))
        ax.add_patch(FancyBboxPatch((-3.2, -1.1), 6.4, 2.2,
            boxstyle="round,pad=0.15", linewidth=0,
            facecolor=AMBER, alpha=0.08, zorder=5))
        short_t = topic if len(topic) <= 32 else topic[:30]+'…'
        ax.text(0, 0.30, short_t, ha='center', va='center',
                fontsize=14, fontweight='bold', color=AMBER, zorder=7)
        ax.text(0, -0.50, 'Core Concept', ha='center', va='center',
                fontsize=9, color='#a07820', zorder=7)
        # Satellite nodes
        angles = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, n, endpoint=False)
        R_base = [3.8 if i%2==0 else 4.1 for i in range(n)]
        for i, ((t, d), angle, R) in enumerate(zip(concepts, angles, R_base)):
            nx = R * np.cos(angle)
            ny = R * np.sin(angle)
            color = PALETTE[i % len(PALETTE)]
            # Connector from center
            start_r = 1.6
            sx0, sy0 = start_r*np.cos(angle), start_r*np.sin(angle)
            end_r = R - 1.6
            ex, ey = end_r*np.cos(angle), end_r*np.sin(angle)
            ax.plot([sx0, ex], [sy0, ey], color=color, lw=1.8, alpha=0.35, zorder=2)
            ax.annotate('', xy=(nx - 1.5*np.cos(angle), ny - 1.5*np.sin(angle)),
                        xytext=(ex, ey),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8, mutation_scale=14), zorder=3)
            # Node box
            NW, NH = 3.2, 1.2
            bx, by = nx - NW/2, ny - NH/2
            ax.add_patch(FancyBboxPatch((bx-0.05, by-0.05), NW+0.1, NH+0.1,
                boxstyle="round,pad=0.09", linewidth=0,
                facecolor=color, alpha=0.12, zorder=4))
            ax.add_patch(FancyBboxPatch((bx, by), NW, NH,
                boxstyle="round,pad=0.09", linewidth=2,
                edgecolor=color, facecolor=BOX_FILL, zorder=5))
            ax.add_patch(FancyBboxPatch((bx, by+NH-0.08), NW, 0.08,
                boxstyle="round,pad=0.01", linewidth=0,
                facecolor=color, alpha=0.75, zorder=6))
            disp = t if len(t) <= 26 else t[:24]+'…'
            ax.text(nx, by+NH*0.65, disp, ha='center', va='center',
                    fontsize=9, fontweight='bold', color=TEXT_HI, zorder=7)
            if d:
                ax.text(nx, by+NH*0.25, d[:36]+'…' if len(d)>36 else d,
                        ha='center', va='center', fontsize=7.2, color=TEXT_MD, zorder=7)
        ax.set_title(topic, pad=16, fontsize=18, fontweight='bold', color=TEXT_HI,
                     path_effects=[pe.withStroke(linewidth=5, foreground=BG)])
        plt.tight_layout(pad=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════
    # ARCHITECTURE — hub-and-spoke with vertical hierarchy
    # ══════════════════════════════════════════════════════════════════
    elif dtype == 'architecture':
        fig_h = max(10.0, 3.0 + n * 1.7)
        fig, ax = plt.subplots(figsize=(14, fig_h))
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        ax.set_xlim(0, 14); ax.set_ylim(0, fig_h); ax.axis('off')
        page_title(ax, topic, 'System Architecture & Components', fig_h)
        # Top system box
        draw_card(ax, 3.5, fig_h-3.8, 7.0, 1.9,
                  topic[:22]+'…' if len(topic)>22 else topic,
                  'Core System', AMBER)
        # Vertical connector
        ax.plot([7, 7], [fig_h-3.8, fig_h-4.5], color=AMBER, lw=2.2, alpha=0.6, zorder=5)
        # Horizontal bus
        left_x = 1.0; right_x = 13.0; bus_y = fig_h - 4.8
        ax.plot([left_x, right_x], [bus_y, bus_y], color=ACCENT, lw=1.8, alpha=0.35, zorder=4)
        # Component boxes fanned out
        CW, CH = 3.0, 2.0
        total_w = n * CW + (n-1) * 0.5
        csx = (14 - total_w) / 2
        comp_y = bus_y - CH - 0.5
        for i, (name, desc) in enumerate(concepts):
            cx = csx + i * (CW + 0.5)
            cx_mid = cx + CW/2
            color = PALETTE[i % len(PALETTE)]
            draw_card(ax, cx, comp_y, CW, CH, name, desc, color)
            # Vertical drop from bus
            ax.plot([cx_mid, cx_mid], [bus_y, comp_y+CH], color=color, lw=1.5, alpha=0.5, zorder=5)
            ax.scatter([cx_mid], [bus_y], s=60, color=color, zorder=6, edgecolors=BG, linewidths=1.5)
        plt.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════
    # COMPARISON — two sides, feature-by-feature
    # ══════════════════════════════════════════════════════════════════
    elif dtype == 'comparison':
        half = max(n // 2, 2)
        left_concepts  = concepts[:half]
        right_concepts = concepts[half:half*2] if len(concepts) >= half*2 else concepts[:half]
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 8))
        fig.patch.set_facecolor(BG)
        short_topic = topic[:38]+'…' if len(topic)>38 else topic
        fig.suptitle(short_topic, fontsize=17, fontweight='bold', color=TEXT_HI, y=0.97)
        for ax2, side_c, side_col, side_icon, side_lbl in [
            (ax_l, left_concepts,  ACCENT, '🔵', left_concepts[0][0]  if left_concepts  else 'Method A'),
            (ax_r, right_concepts, AMBER,  '🟡', right_concepts[0][0] if right_concepts else 'Method B'),
        ]:
            ax2.set_facecolor(BG); ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis('off')
            # Header
            ax2.add_patch(FancyBboxPatch((0.2, 8.2), 9.6, 1.6,
                boxstyle="round,pad=0.1", linewidth=2.5,
                edgecolor=side_col, facecolor=side_col+'20', zorder=2))
            ax2.add_patch(FancyBboxPatch((0.2, 8.2), 9.6, 0.1,
                boxstyle="round,pad=0.01", linewidth=0,
                facecolor=side_col, alpha=0.8, zorder=3))
            head_lbl = side_lbl[:22]+'…' if len(side_lbl)>22 else side_lbl
            ax2.text(5, 9.07, head_lbl, ha='center', va='center',
                     fontsize=13, fontweight='bold', color=TEXT_HI, zorder=4)
            # Feature rows
            items = [(t,d) for t,d in side_c[1:]] if len(side_c)>1 else [(t,d) for t,d in side_c]
            items = items[:5]
            for ji, (feat_t, feat_d) in enumerate(items):
                yp = 7.3 - ji * 1.45
                ax2.add_patch(FancyBboxPatch((0.2, yp-0.62), 9.6, 1.22,
                    boxstyle="round,pad=0.08", linewidth=1.2,
                    edgecolor=side_col+'55', facecolor=BOX_FILL, zorder=2))
                ax2.add_patch(FancyBboxPatch((0.2, yp+0.55), 9.6, 0.05,
                    boxstyle="round,pad=0.01", linewidth=0,
                    facecolor=side_col, alpha=0.4, zorder=3))
                ax2.text(0.75, yp+0.08, '▸', ha='left', va='center',
                         fontsize=12, color=side_col, zorder=4)
                ft = feat_t[:26]+'…' if len(feat_t)>26 else feat_t
                ax2.text(1.2, yp+0.14, ft,
                         ha='left', va='center', fontsize=9, fontweight='bold',
                         color=TEXT_HI, zorder=4)
                if feat_d:
                    fd = feat_d[:55]+'…' if len(feat_d)>55 else feat_d
                    ax2.text(1.2, yp-0.22, fd,
                             ha='left', va='center', fontsize=7.8, color=TEXT_MD, zorder=4)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════
    # TIMELINE — horizontal progression
    # ══════════════════════════════════════════════════════════════════
    elif dtype == 'timeline':
        fig, ax = plt.subplots(figsize=(14, 6.5))
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        ax.set_xlim(0, 14); ax.set_ylim(0, 6.5); ax.axis('off')
        page_title(ax, topic, 'Progression / Timeline', 6.5)
        # Central axis line
        ax.plot([0.8, 13.2], [3.2, 3.2], color=AMBER, lw=2.5, alpha=0.30, zorder=2)
        arrow(ax, 12.8, 3.2, 13.5, 3.2, color=AMBER)
        step = 12.4 / max(n-1, 1)
        for i, (ms, desc) in enumerate(concepts):
            x = 0.8 + i * step
            color = PALETTE[i % len(PALETTE)]
            # Dot on axis
            ax.scatter([x], [3.2], s=200, color=color, zorder=5, edgecolors=BG, linewidths=2.5)
            ax.text(x, 3.2, str(i+1), ha='center', va='center',
                    fontsize=8, fontweight='bold', color=BG, zorder=6)
            above = (i % 2 == 0)
            by = 4.35 if above else 1.35
            BW = min(2.8, step * 0.9)
            # Box
            ax.add_patch(FancyBboxPatch((x-BW/2-0.04, by-0.64), BW+0.08, 1.28,
                boxstyle="round,pad=0.08", linewidth=0,
                facecolor=color, alpha=0.12, zorder=3))
            ax.add_patch(FancyBboxPatch((x-BW/2, by-0.6), BW, 1.2,
                boxstyle="round,pad=0.08", linewidth=1.8,
                edgecolor=color, facecolor=BOX_FILL, zorder=4))
            ax.add_patch(FancyBboxPatch((x-BW/2, by+0.56), BW, 0.07,
                boxstyle="round,pad=0.01", linewidth=0,
                facecolor=color, alpha=0.8, zorder=5))
            dn = ms[:20]+'…' if len(ms)>20 else ms
            ax.text(x, by+0.12, dn, ha='center', va='center',
                    fontsize=9, fontweight='bold', color=TEXT_HI, zorder=6)
            if desc:
                ax.text(x, by-0.28, desc[:28], ha='center', va='center',
                        fontsize=7, color=TEXT_MD, zorder=6)
            # Connector
            conn_y1 = 3.35 if above else 3.05
            conn_y2 = by - 0.6 if above else by + 1.2
            ax.plot([x, x], [conn_y1, conn_y2], color=color, lw=1.5, alpha=0.45, zorder=3)
        plt.tight_layout(pad=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def _generate_visualization(topic: str, content: str, diagram_type: str = "auto") -> None:
    """Public entry point — calls the native Python diagram engine directly."""
    _build_smart_diagram(topic, content, diagram_type)


def _vision_ans(question: str, img_b64: str, img_name: str) -> str:
    """Ask LLM about an image. Falls back with helpful error if model isn't multimodal."""
    prompt = (
        f"The user uploaded an image called '{img_name}' and asks:\n\n{question}\n\n"
        "Please analyze the image thoroughly:\n"
        "- If it's a chart/plot: describe axes, data ranges, trends, outliers, key insights\n"
        "- If it's data visualization: interpret statistical patterns and findings\n"
        "- If it's a diagram: explain each component clearly\n"
        "Be specific, educational, and helpful."
    )
    try:
        from langchain_core.messages import HumanMessage
        out = B.llm.invoke([HumanMessage(content=[
            {"type": "text",      "text": prompt},
            {"type": "image_url", "image_url": {"url": img_b64}},
        ])])
        return getattr(out, "content", str(out)).strip()
    except Exception:
        return (
            f"📷 **Image received: {img_name}**\n\n"
            "Your current model doesn't support vision.\n\n"
            "**Enable vision with:**\n"
            "```bash\n# Ollama — LLaVA model\nollama pull llava\n```\n"
            "Then in `.env`:\n```\nOLLAMA_CHAT_MODEL=llava\n```\n\n"
            "**Or use OpenAI GPT-4o:**\n```\nLLM_PROVIDER=openai\nOPENAI_MODEL=gpt-4o\n```"
        )

def _vision_ans_multi(question: str, images: list[dict]) -> str:
    """Analyze MULTIPLE images together in one response — compare, contrast, recommend."""
    names = [img["name"] for img in images]
    names_str = " and ".join(f"'{n}'" for n in names)
    prompt = (
        f"The user uploaded {len(images)} images ({names_str}) and asks:\n\n{question}\n\n"
        "Analyze ALL the images and provide a UNIFIED response:\n"
        "1. Describe each image/chart individually (axes, ranges, trends, patterns)\n"
        "2. COMPARE them — what's different, what's similar?\n"
        "3. Make a clear RECOMMENDATION if the user is choosing between them\n"
        "4. Explain your reasoning thoroughly\n"
        "Be specific with numbers, variable names, and data insights."
    )
    try:
        from langchain_core.messages import HumanMessage
        # Build content with all images
        content_parts: list = [{"type": "text", "text": prompt}]
        for i, img in enumerate(images):
            content_parts.append({
                "type": "text",
                "text": f"\n--- Image {i+1}: {img['name']} ---"
            })
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": img["b64"]},
            })
        out = B.llm.invoke([HumanMessage(content=content_parts)])
        return getattr(out, "content", str(out)).strip()
    except Exception:
        # Fallback: analyze each separately and combine
        results = []
        for i, img in enumerate(images):
            single = _vision_ans(question, img["b64"], img["name"])
            results.append(f"### Image {i+1}: {img['name']}\n\n{single}")
        combined = "\n\n---\n\n".join(results)
        combined += (
            "\n\n---\n\n**Note:** Your model analyzed images separately. "
            "For side-by-side comparison, use a multimodal model like `llava` or `gpt-4o`."
        )
        return combined
    """Inject last 8 turns as context so the AI remembers the conversation."""
    recent = [m for m in st.session_state.msgs[-10:]
              if m.get("content") not in ("__FLASHCARDS__", "__QUIZ__", "__IMGBLOCK__", "")
              and m.get("content")]
    if len(recent) < 2:
        return text
    hist = "\n".join(
        f"{'User' if m['role']=='user' else 'AI'}: {m['content'][:300]}"
        for m in recent[:-1]
    )
    return f"[Conversation context:\n{hist}\n]\n\nCurrent question: {text}"

def _ctx_query(text: str) -> str:
    """Build a rich context-aware query so the AI generates examples from the ACTUAL topic.

    Injects:
    1. The current question (main text)
    2. Recent chat topic trail — what concepts were being discussed
    3. Explicit instruction: use THIS topic for any code/worked examples
    """
    msgs = st.session_state.msgs or []
    recent = [m for m in msgs[-12:]
              if m.get("content") not in ("__FLASHCARDS__", "__QUIZ__", "__IMGBLOCK__", "")
              and m.get("content")]

    # ── Extract topic trail from recent user messages ───────────────────────
    # Find what the student has been studying (last 3 user questions)
    user_questions = []
    for m in reversed(recent):
        if m["role"] == "user" and m.get("content"):
            q = m["content"].strip()
            if q and q not in user_questions:
                user_questions.append(q)
            if len(user_questions) >= 3:
                break
    user_questions.reverse()

    # ── Extract key topic from current + recent questions ───────────────────
    # Used to tell AI: "examples must be about THIS topic"
    combined_q = " ".join(user_questions + [text]).lower()

    # Detect the study topic from keywords
    topic_hints = []
    topic_map = [
        # ── LLMs, ChatGPT, RLHF — must come first so "gpt" doesn't fall to transformers only
        (["chatgpt", "gpt-3", "gpt-4", "gpt3", "gpt4", "large language model", "llm",
          "rlhf", "reinforcement learning from human feedback", "ppo", "proximal policy",
          "reward model", "fine-tun", "token", "175 billion", "500b token",
          "conversational prompt injection", "moderation api", "openai", "prompt engineering",
          "next token", "training data", "chatbot", "fine tuned"],
         "LLMs / ChatGPT / RLHF"),
        # ── Transformers & Attention
        (["transformer", "self-attention", "attention mechanism", "multi-head", "encoder",
          "decoder", "bert", "positional encoding", "scaled dot product"],
         "Transformers / Self-Attention"),
        # ── NLP general
        (["nlp", "natural language processing", "tokeniz", "text classif", "sentiment",
          "named entity", "word2vec", "word embedding", "language model", "sequence model",
          "text generation", "corpus", "vocabulary", "bag of words", "tfidf"],
         "NLP (Natural Language Processing)"),
        # ── Computer Vision
        (["cnn", "convolutional", "image classif", "computer vision", "feature map",
          "pooling", "resnet", "vgg", "object detect", "image recognition"],
         "Computer Vision / CNN"),
        # ── RNN / Sequential
        (["rnn", "lstm", "gru", "recurrent", "sequence to sequence", "time series",
          "vanishing gradient", "hidden state"],
         "RNN / LSTM / Sequential Models"),
        # ── Reinforcement Learning
        (["reinforcement learning", "rl agent", "reward function", "policy gradient",
          "q-learning", "environment", "markov", "mdp", "gym", "gymnasium",
          "value function", "bellman"],
         "Reinforcement Learning"),
        # ── Generative Models
        (["gan", "generative adversarial", "diffusion model", "vae", "variational autoencoder",
          "image generation", "stable diffusion", "latent space"],
         "Generative Models / GAN / Diffusion"),
        # ── Regression
        (["linear regression", "logistic regression", "gradient descent", "cost function",
          "mean squared error", "hypothesis function", "basis function", "least squares"],
         "Linear/Logistic Regression"),
        # ── Neural Networks
        (["neural network", "fully connected", "activation function", "backpropagation",
          "multilayer perceptron", "deep learning", "loss function", "hidden layer",
          "sigmoid", "relu", "weight", "bias"],
         "Neural Networks / Deep Learning"),
        # ── Unsupervised
        (["clustering", "kmeans", "k-means", "dbscan", "unsupervised", "pca",
          "dimensionality reduction", "silhouette", "elbow method"],
         "Unsupervised Learning / Clustering"),
        # ── Classical ML
        (["decision tree", "random forest", "svm", "support vector machine", "ensemble",
          "boosting", "xgboost", "feature importance", "gini", "entropy"],
         "Classical ML / Tree Models"),
        # ── Bayesian
        (["bayesian", "naive bayes", "probability", "posterior", "prior", "likelihood",
          "bayes theorem", "gaussian mixture"],
         "Bayesian / Probabilistic Models"),
        # ── Classification
        (["multi-class", "multiclass", "softmax", "one-hot", "cross entropy", "nll loss",
          "classification", "confusion matrix", "precision", "recall", "f1"],
         "Multi-Class Classification"),
        # ── Statistics / Math
        (["statistics", "hypothesis test", "p-value", "confidence interval", "regression",
          "correlation", "distribution", "variance", "standard deviation"],
         "Statistics / Mathematics"),
    ]
    for keywords, label in topic_map:
        if any(kw in combined_q for kw in keywords):
            topic_hints.append(label)

    detected_topic = topic_hints[0] if topic_hints else None

    # ── Build chat history summary ──────────────────────────────────────────
    hist_parts = []
    for m in recent[:-1]:  # exclude current message (last one)
        role = "Student" if m["role"] == "user" else "NeuraPilot"
        content = m.get("content", "")
        # Skip JSON blobs and very long AI outputs
        if content.strip().startswith("[{") or content.strip().startswith('{"questions'):
            continue
        if m["role"] == "assistant" and len(content) > 350:
            content = content[:250] + "… [truncated]"
        hist_parts.append(f"{role}: {content[:300]}")

    # ── Compose final query ──────────────────────────────────────────────────
    parts = []

    if hist_parts:
        parts.append("=== RECENT CONVERSATION (for context) ===")
        parts.extend(hist_parts[-6:])  # last 6 turns max
        parts.append("=== END CONVERSATION ===")
        parts.append("")

    if detected_topic:
        parts.append(
            f"⚡ TOPIC DETECTED: The student is studying [{detected_topic}]."
        )
        parts.append(
            f"⚡ WORKED EXAMPLE RULE: Any code example you generate MUST be about [{detected_topic}]. "
            f"Use datasets and concepts relevant to {detected_topic}. "
            f"DO NOT use California Housing or GaussianMixture unless {detected_topic} explicitly requires it."
        )
        parts.append("")

    if user_questions and len(user_questions) > 1:
        parts.append(f"Student has been asking about: {' → '.join(user_questions[-3:])}")
        parts.append("")

    parts.append(f"CURRENT QUESTION: {text}")

    return "\n".join(parts)

def _save(tid: int, role: str, content: str, meta: dict) -> None:
    """Save message to DB AND session state."""
    dbmod.add_message(DB, tid, role, content, json.dumps(meta, ensure_ascii=False))
    st.session_state.msgs.append({"role": role, "content": content, "meta": meta})

def _new_thread(cid: str) -> None:
    tid = dbmod.create_thread(DB, cid)
    st.session_state.tid       = tid
    st.session_state.msgs      = []
    st.session_state.quiz_obj  = None
    st.session_state.img_queue = []

def _load_thread(tid: int) -> None:
    st.session_state.tid  = tid
    st.session_state.msgs = [
        {"role": m["role"], "content": m["content"],
         "meta": json.loads(m.get("meta_json", "{}") or "{}")}
        for m in dbmod.get_messages(DB, tid)
    ]

def _auto_title(tid: int, msg: str) -> None:
    try:
        out = B.llm.invoke(f"Write a 4-6 word title for a chat starting: '{msg[:80]}'. Return ONLY the title, no quotes.")
        title = getattr(out, "content", str(out)).strip()[:60]
        if title:
            dbmod.rename_thread(DB, tid, title)
    except Exception: pass

def _thinking_anim(placeholder: Any, step: int, intent: str = "ask") -> None:
    steps = [
        "Classifying intent & topic",
        "Rewriting query for retrieval",
        "Searching knowledge base",
        "Reranking chunks by relevance",
        {"quiz": "Generating MCQ quiz", "flashcards": "Generating flashcards",
         "plan": "Building study plan", "summarize": "Summarizing content",
         "guidance": "Extracting assignment requirements"}.get(intent, "Generating answer"),
        "Complete ✓",
    ]
    html = '<div class="think-box"><div class="think-title">⬡ Thinking</div>'
    for i, s in enumerate(steps):
        cls  = "live" if i == step else ("done" if i < step else "")
        icon = "▶" if i == step else ("✓" if i < step else "·")
        html += f'<div class="think-step {cls}">{icon} {s}</div>'
    html += '</div>'
    placeholder.markdown(html, unsafe_allow_html=True)

@st.cache_resource
def _get_sem_cache():
    """Return singleton SemanticCache (shared across all reruns)."""
    try:
        from neurapilot.core.semantic_cache import SemanticCache
        return SemanticCache(DB, B.embeddings, threshold=0.92, ttl_seconds=86_400)
    except Exception:
        return None


def _run_agent(text: str, cid: str, think_ph: Any = None) -> dict:
    """Run the RAG pipeline with semantic cache + optional live thinking animation."""
    # Detect intent for thinking display
    intent_hint = "ask"
    if re.search(r'\b(quiz|mcq|test me)\b', text, re.I):    intent_hint = "quiz"
    elif re.search(r'\b(flashcard)\b', text, re.I):           intent_hint = "flashcards"
    elif re.search(r'\b(summarize|summary)\b', text, re.I):  intent_hint = "summarize"
    elif re.search(r'\b(study plan|plan)\b', text, re.I):    intent_hint = "plan"
    elif re.search(r'\b(visuali[sz]e|diagram|draw|chart|plot)\b', text, re.I):
        intent_hint = "ask"

    # ── ⚡ Semantic cache check (skip for quiz/flashcards — always fresh) ─────
    _cache = _get_sem_cache()
    if _cache and intent_hint not in {"quiz", "flashcards", "plan"}:
        t_cache0 = time.time()
        cached = _cache.get(text, cid)
        cache_ms = int((time.time() - t_cache0) * 1000)
        if cached:
            if think_ph:
                think_ph.empty()
            real_ms = max(cache_ms, 1)
            st.toast(f"⚡ Cache hit — served in {real_ms} ms", icon="⚡")
            # Track cache hit latency in session for analytics
            if "_cache_latencies" not in st.session_state:
                st.session_state._cache_latencies = []
            st.session_state._cache_latencies.append(real_ms)
            return {
                "intent":      cached["intent"],
                "topic":       "",
                "output":      cached["answer"],
                "sources":     [],
                "latency_ms":  real_ms,
                "eval_scores": {"faithfulness": None, "answer_relevance": None,
                                "context_precision": None},
                "cache_hit":   True,
                "similarity":  cached.get("similarity", 1.0),
            }

    # ── Full pipeline ─────────────────────────────────────────────────────────
    if think_ph and st.session_state.show_think:
        for i in range(5):
            _thinking_anim(think_ph, i, intent_hint)
            time.sleep(0.28)

    pipe  = _pipe(cid, S.strict_grounding)
    t0    = time.time()
    state = pipe.invoke({"question": _ctx_query(text), "strict": bool(st.session_state.strict)})
    ms    = int((time.time() - t0) * 1000)

    if think_ph and st.session_state.show_think:
        _thinking_anim(think_ph, 5, state.get("intent", "ask"))
        time.sleep(0.3)
        think_ph.empty()

    intent  = state.get("intent", "ask")
    topic   = state.get("topic", "")
    output  = state.get("output", "")
    docs    = state.get("docs", []) or []
    sources = [{"key": f"S{i}", "source": str(d.metadata.get("source", "")),
                "page": d.metadata.get("page")} for i, d in enumerate(docs, 1)]

    ev = {"faithfulness": None, "answer_relevance": None, "context_precision": None}
    if S.eval_enabled and intent in {"ask", "summarize"}:
        try:
            ev = evaluate_response(B.llm, text, output, docs).to_dict()
        except Exception:
            pass

    dbmod.log_interaction(DB, course_id=cid, strict=bool(st.session_state.strict),
        user_text=text, intent=intent, topic=topic, output=output,
        sources=sources, latency_ms=ms, **{k: v for k, v in ev.items()})

    # ── Store in cache (only cacheable intents) ───────────────────────────────
    if _cache and intent in {"ask", "summarize"} and output:
        _cache.put(text, output, cid, intent)

    # Track full-pipeline latency in session for reduction % calc
    if "_full_latencies" not in st.session_state:
        st.session_state._full_latencies = []
    st.session_state._full_latencies.append(ms)

    return {"intent": intent, "topic": topic, "output": output,
            "sources": sources, "latency_ms": ms, "eval_scores": ev,
            "cache_hit": False}


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR — Transformer-integrated reasoning layer
# Routes student queries to the correct engine:
#   RAG (knowledge) | Code Engine | Analytics | Guidance | Study Tools
# Mirrors how a Transformer's Self-Attention weighs intent before token prediction
# ─────────────────────────────────────────────────────────────────────────────

def _orchestrate(text: str, cid: str, think_ph: Any = None) -> dict:
    """
    The NeuraPilot Orchestrator: routes queries like a Transformer routes attention.

    Architecture (mirrors ChatGPT internal flow):
      1. Tokenize intent  → classify query type (CLASSIFY_PROMPT)
      2. Route decision   → select the right engine based on intent
      3. Execute route    → RAG / Guidance / Code / Analytics / Study Tools
      4. Return result    → structured dict for _proc() to render

    This prevents the #1 bug: generating code blocks when student just wants
    to know what to DO (homework guidance, task navigation).
    """
    import re as _re

    # ── Pre-classify with fast regex (avoid full LLM call for obvious cases) ──
    _guidance_patterns = _re.compile(
        r'\b(what (do|should|must) i (do|submit|write|turn in)|'
        r'tell me what (to do|i need)|'
        r'what (is|are) (the )?(requirement|assignment|homework|task|steps?)|'
        r'how (do i|to) (complete|finish|do|start) (this|the)|'
        r'what (should i|do i need to) (do|submit|create|write)|'
        r'what is (expected|required|needed|due)|'
        r'guide me|help me (understand|complete|start) (this|the) (assignment|homework|task))\b',
        _re.I
    )
    _analytics_patterns = _re.compile(
        r'\b(how am i doing|my progress|my (quiz |mastery |)score|'
        r'performance|how (well|good) am i|my analytics|my stats)\b',
        _re.I
    )

    # Fast-path: guidance intent (no LLM call needed — saves latency)
    if _guidance_patterns.search(text):
        return _run_agent_with_intent(text, cid, think_ph, forced_intent="guidance")

    # Fast-path: analytics intent
    if _analytics_patterns.search(text):
        return _run_agent_with_intent(text, cid, think_ph, forced_intent="analytics")

    # Default: full pipeline with standard intent classification
    return _run_agent(text, cid, think_ph)


def _run_agent_with_intent(text: str, cid: str, think_ph: Any, forced_intent: str) -> dict:
    """Run pipeline but override the intent classification with a forced intent."""
    # For guidance: run RAG to get context, then format as guidance (no code)
    result = _run_agent(text, cid, think_ph)
    # Override the intent so _proc() routes to the correct renderer
    if forced_intent == "guidance":
        result["intent"] = "guidance"
    elif forced_intent == "analytics":
        result["intent"] = "analytics"
    return result


def _proc(result: dict, cid: str, topic_hint: str = "") -> None:
    """Process agent result and save to thread."""
    intent = result["intent"]
    output = result["output"]
    topic  = result["topic"] or topic_hint or "course material"
    tid    = st.session_state.tid
    base   = {"intent": intent, "topic": topic, "sources": result["sources"],
               "latency_ms": result["latency_ms"], "eval_scores": result["eval_scores"]}

    if intent == "flashcards":
        cards = []
        try:
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", output.strip(), flags=re.DOTALL)
            p = json.loads(raw)
            # Handle all JSON formats the LLM might return
            if isinstance(p, list):
                cards = p
            elif isinstance(p, dict):
                # {"cards": [...]} or {"flashcards": [...]} or {"data": [...]}
                for key in ("cards", "flashcards", "data", "items", "questions"):
                    if key in p and isinstance(p[key], list):
                        cards = p[key]; break
        except Exception:
            # Last resort: extract JSON array from anywhere in the output
            try:
                m = re.search(r'\[\s*\{.*?\}\s*\]', output, re.DOTALL)
                if m:
                    cards = json.loads(m.group())
            except Exception:
                pass
        # Ensure each card has required keys
        cards = [c for c in cards if isinstance(c, dict) and ("q" in c or "question" in c) and ("a" in c or "answer" in c)]
        # Normalize key names
        normalized = []
        for c in cards:
            normalized.append({
                "q": c.get("q") or c.get("question",""),
                "a": c.get("a") or c.get("answer",""),
                "citations": c.get("citations",[]),
                "difficulty": c.get("difficulty","medium"),
                "bloom_level": c.get("bloom_level","understand"),
            })
        cards = normalized
        if cards: dbmod.add_flashcards(DB, cid, topic, cards)
        _save(tid, "assistant", "__FLASHCARDS__", {**base, "cards": cards, "topic": topic})

    elif intent == "quiz":
        qs = []
        try:
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", output.strip(), flags=re.DOTALL)
            p = json.loads(raw)
            if isinstance(p, dict):   qs = p.get("questions", [])
            elif isinstance(p, list): qs = p
        except Exception: pass
        if qs:
            st.session_state.quiz_obj   = {"questions": qs}
            st.session_state.quiz_idx   = 0
            st.session_state.quiz_score = 0
        _save(tid, "assistant", "__QUIZ__", {**base, "n_questions": len(qs), "topic": topic})

    elif intent == "guidance":
        # ── Guidance: strip any accidental code blocks the LLM added ─────
        # Homework/task guidance should NEVER contain Python code blocks
        clean_output = re.sub(
            r'```(?:python|py|bash|shell|csv|sql|json)?\s*[\s\S]*?```',
            '*(Code example removed — see "Give me a worked example" for code)*',
            output,
            flags=re.DOTALL
        )
        st.session_state.quiz_obj  = None
        st.session_state.quiz_idx  = 0
        st.session_state.quiz_score = 0
        _save(tid, "assistant", clean_output, {**base, "intent": "guidance"})

    else:
        # ── Clear stale quiz so follow-up questions don't show old quiz ────
        st.session_state.quiz_obj  = None
        st.session_state.quiz_idx  = 0
        st.session_state.quiz_score = 0
        _save(tid, "assistant", output, base)

def _render_fc(cards: list, topic: str, kpfx: str) -> None:
    if not cards:
        st.warning("No flashcards generated. Please try again.")
        return

    # ── Header with stats ────────────────────────────────────────────────
    easy   = sum(1 for c in cards if c.get("difficulty") == "easy")
    medium = sum(1 for c in cards if c.get("difficulty") == "medium")
    hard   = sum(1 for c in cards if c.get("difficulty") == "hard")
    st.markdown(
        f'<div style="background:#1a1f2e;border:1px solid #2a3650;border-radius:14px;padding:18px 22px;margin:6px 0 14px">' +
        f'<div style="font-size:1.1em;font-weight:800;color:#60a5fa;margin-bottom:8px">🃏 {len(cards)} Flashcards — {topic}</div>' +
        f'<div style="display:flex;gap:10px;flex-wrap:wrap">' +
        f'<span style="background:#4ade8015;color:#4ade80;border:1px solid #4ade8030;padding:2px 10px;border-radius:10px;font-size:11px;font-weight:800">✅ {easy} Easy</span>' +
        f'<span style="background:#fbbf2415;color:#fbbf24;border:1px solid #fbbf2430;padding:2px 10px;border-radius:10px;font-size:11px;font-weight:800">⚡ {medium} Medium</span>' +
        f'<span style="background:#f8717115;color:#f87171;border:1px solid #f8717130;padding:2px 10px;border-radius:10px;font-size:11px;font-weight:800">🔥 {hard} Hard</span>' +
        f'</div></div>',
        unsafe_allow_html=True
    )

    # ── Render each card as a beautiful flip card ────────────────────────
    for i, c in enumerate(cards):
        q     = c.get("q", "")
        a     = c.get("a", "")
        diff  = c.get("difficulty", "medium")
        bloom = c.get("bloom_level", "understand")
        cites = c.get("citations", [])
        career = c.get("career_tip", "")

        diff_color = {"easy": "#4ade80", "medium": "#fbbf24", "hard": "#f87171"}.get(diff, "#fbbf24")
        diff_bg    = {"easy": "#4ade8015", "medium": "#fbbf2415", "hard": "#f8717115"}.get(diff, "#fbbf2415")
        bloom_icons = {"remember":"🧠","understand":"💡","apply":"🔧","analyze":"🔬","evaluate":"⚖️","create":"✨"}
        bloom_icon  = bloom_icons.get(bloom, "💡")

        # Card front (question)
        st.markdown(
            f'<div style="background:#1e1e1e;border:1px solid #383838;border-left:4px solid {diff_color};' +
            f'border-radius:12px;padding:18px 22px;margin:8px 0 2px">' +
            f'<div style="display:flex;gap:8px;margin-bottom:10px;align-items:center">' +
            f'<span style="font-size:11px;font-weight:800;color:#888">CARD {i+1}</span>' +
            f'<span style="background:{diff_bg};color:{diff_color};border:1px solid {diff_color}30;' +
            f'padding:2px 10px;border-radius:10px;font-size:11px;font-weight:800;text-transform:uppercase">{diff}</span>' +
            f'<span style="background:#a78bfa15;color:#a78bfa;border:1px solid #a78bfa30;' +
            f'padding:2px 10px;border-radius:10px;font-size:11px;font-weight:700">{bloom_icon} {bloom}</span>' +
            f'</div>' +
            f'<div style="font-size:16px;font-weight:600;color:#ececec;line-height:1.5">{q}</div>' +
            f'</div>',
            unsafe_allow_html=True
        )

        # Card back (answer) — click to reveal
        with st.expander("👁 Show Answer", expanded=False):
            st.markdown(
                f'<div style="background:#1a2620;border:1px solid #1e3a2a;border-left:4px solid #4ade80;' +
                f'border-radius:12px;padding:16px 20px;margin:4px 0 6px">' +
                f'<div style="font-size:13px;font-weight:700;color:#4ade80;margin-bottom:8px;letter-spacing:.5px">✅ ANSWER</div>' +
                f'<div style="font-size:15px;color:#d1fae5;line-height:1.6">{a}</div>' +
                (f'<div style="margin-top:12px;padding-top:10px;border-top:1px solid #1e3a2a;font-size:12px;color:#555">🚀 Career: {career}</div>' if career else '') +
                (f'<div style="margin-top:8px;font-size:11px;color:#444">📎 {", ".join(cites)}</div>' if cites else '') +
                f'</div>',
                unsafe_allow_html=True
            )
            _voice(a, f"{kpfx}_fc_{i}")

    st.success(f"✅ {len(cards)} flashcards ready! Find them in the Flashcards tab too.")

def _render_quiz_banner(n: int) -> None:
    st.markdown(f'<div class="quiz-banner">'
                f'<div style="font-size:1.2em;font-weight:800;color:#60a5fa;margin-bottom:6px">🎯 Quiz Ready — {n} Questions</div>'
                f'<div style="color:#3b6b9a;font-size:13px">Switch to the <strong>Practice</strong> tab to start.</div>'
                f'</div>', unsafe_allow_html=True)

def _merge_python_blocks(content: str) -> str:
    """Merge ALL python code blocks in a response into ONE block.
    This prevents NameError when variables defined in block 1 are needed in block 2.
    Non-python blocks (bash, sql, etc.) are left separate.
    """
    import re as _rm
    py_codes  = _rm.findall(r'```(?:python|py)\n(.*?)```', content, _rm.DOTALL)
    if len(py_codes) <= 1:
        return content  # nothing to merge

    # Combine all python blocks into one
    merged_py = "\n\n".join(block.strip() for block in py_codes)

    # Remove all individual python blocks from content
    cleaned = _rm.sub(r'```(?:python|py)\n.*?```', '__MERGED_PY__', content, flags=_rm.DOTALL)

    # Only keep the first placeholder, remove the rest
    first = True
    def replace_placeholder(m):
        nonlocal first
        if first:
            first = False
            return f'''```python\n{merged_py}\n```'''
        return ''  # remove subsequent placeholders
    cleaned = _rm.sub(r'__MERGED_PY__', replace_placeholder, cleaned)

    return cleaned


def _render_msg_body(content: str, mkey: str) -> None:
    """Render message content — merges all python blocks into ONE runnable block."""
    if "```" not in content:
        st.markdown(content)
        return

    # ── Merge all python code blocks into one ─────────────────────────────────
    content = _merge_python_blocks(content)

    parts = re.split(r'(```[\w]*\n.*?```)', content, flags=re.DOTALL)
    for pi, part in enumerate(parts):
        m = re.match(r'```([\w]*)\n(.*?)```', part, re.DOTALL)
        if m:
            lang = m.group(1).lower() or "text"
            code = m.group(2)
            st.code(code, language=lang if lang != "text" else None)
            if lang in ("python", "py"):
                if st.button("▶ Run Python", key=f"run_{pi}_{mkey}"):
                    with st.expander("📊 Output", expanded=True):
                        _run_py(code)
        elif part.strip():
            st.markdown(part)

def _render_assistant(content: str, meta: dict, mkey: str, is_last: bool, cid: str) -> None:
    """Render a full assistant message with voice, eval, sources, meta, regen."""
    _render_msg_body(content, mkey)
    _voice(content, mkey)

    # Eval bars
    ev_html = _eval_html(meta.get("eval_scores", {}))
    if ev_html: st.markdown(ev_html, unsafe_allow_html=True)

    # Sources
    src = _chips(meta.get("sources", []))
    if src: st.markdown(src, unsafe_allow_html=True)

    # Meta row
    ms     = meta.get("latency_ms", 0)
    intent = meta.get("intent", "ask")
    topic  = meta.get("topic", "")
    if ms:
        cache_badge = ('<span style="background:#f59e0b22;color:#f59e0b;font-size:10px;'
                       'font-weight:700;padding:2px 7px;border-radius:6px;margin-right:6px">'
                       '⚡ CACHE HIT</span>') if meta.get("cache_hit") else ""
        st.markdown(
            f'<div class="meta-row">'
            f'{cache_badge}'
            f'<span class="meta-txt">⏱ {ms}ms</span>'
            f'{_itag(intent)}'
            f'<span class="meta-txt">· {topic}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Actions: Bookmark + Visualize + Regenerate (last message only)
    col_bk, col_viz, col_rg, _ = st.columns([1, 1.5, 1, 7])
    with col_bk:
        if st.button("🔖", key=f"bk_{mkey}", help="Bookmark this answer"):
            dbmod.add_bookmark(DB, cid,
                title=topic or content[:60],
                content=content[:2000], tag=intent)
            st.toast("🔖 Bookmarked!")
    with col_viz:
        if st.button("📊 Visualize", key=f"viz_{mkey}", help="Generate a visual diagram of this concept"):
            st.session_state[f"_showviz_{mkey}"] = True
    if is_last:
        with col_rg:
            if st.button("↺ Regen", key=f"rg_{mkey}", help="Regenerate this response"):
                st.session_state["_regen"] = True

    # ── Visualization panel ─────────────────────────────────────────────────
    if st.session_state.get(f"_showviz_{mkey}", False):
        from viz_pipeline import render_diagram, extract_concepts, pick_dtype

        st.markdown(
            '<div style="background:#06060f;border:1.5px solid #1e3a5f;border-radius:16px;'
            'padding:22px 26px;margin:14px 0 8px 0">'
            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">'
            '<div style="width:9px;height:9px;border-radius:50%;background:#3b82f6;'
            'box-shadow:0 0 10px #3b82f6"></div>'
            '<span style="color:#60a5fa;font-size:12px;font-weight:800;letter-spacing:.9px">'
            'CONCEPT VISUALIZATION</span></div>',
            unsafe_allow_html=True,
        )

        # Diagram type selector
        viz_type_key = f"_viztype_{mkey}"
        prev_type = st.session_state.get(viz_type_key, "Auto (Best fit)")
        TYPES = ["Auto (Best fit)", "📄 AI Research Overview",
                 "Pipeline / Flowchart", "Mind Map / Concept Map",
                 "Comparison / Side-by-side", "Timeline", "Architecture Diagram"]
        viz_type = st.selectbox("Diagram style", TYPES,
                                key=f"vtype_{mkey}", label_visibility="visible")

        cache_key = f"_vizfig_{mkey}_{viz_type}"
        if viz_type != prev_type:
            st.session_state[viz_type_key] = viz_type
            st.session_state.pop(cache_key, None)

        # Render (cached per message+type)
        if cache_key not in st.session_state:
            with st.spinner("🎨 Generating visualization…"):
                try:
                    if "paper overview" in viz_type.lower():
                        from viz_arch_diagram import draw_paper_overview
                        fig = draw_paper_overview(topic or content[:80], content)
                        concepts = []
                        dtype = "paper_overview"
                    else:
                        from viz_pipeline import render_diagram
                        fig, dtype, concepts = render_diagram(
                            topic or content[:60], content, viz_type
                        )
                    st.session_state[cache_key] = (fig, dtype, concepts)
                except Exception as e:
                    st.error(f"Diagram error: {e}")
                    st.session_state[cache_key] = None

        cached = st.session_state.get(cache_key)
        if cached:
            fig, dtype, concepts = cached
            # Render the figure
            label_map = {
                "pipeline":      "🔢 Pipeline / Flowchart",
                "mindmap":       "🕸️ Mind Map",
                "comparison":    "⚖️ Comparison",
                "timeline":      "📅 Timeline",
                "architecture":  "🏗️ Architecture",
                "paper_overview":"📄 AI Research Overview",
            }
            st.caption(f"Rendering as: **{label_map.get(dtype, dtype)}** · {len(concepts)} concepts extracted")
            st.pyplot(fig, use_container_width=True)

            import matplotlib.pyplot as plt
            plt.close(fig)
            st.session_state[cache_key] = (None, dtype, concepts)  # free memory after render

            # Concept list
            with st.expander(f"📋 {len(concepts)} concepts extracted from this response"):
                for i, (t, d) in enumerate(concepts):
                    st.markdown(f"**{i+1}. {t}** — {d[:100]}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 Regenerate", key=f"reviz_{mkey}", use_container_width=True):
                st.session_state.pop(cache_key, None)
                st.rerun()
        with c2:
            if st.button("✕ Close", key=f"closeviz_{mkey}", use_container_width=True):
                st.session_state[f"_showviz_{mkey}"] = False
                st.session_state.pop(cache_key, None)
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
<div style="padding:16px 8px 12px;border-bottom:1px solid #1a1a24;margin-bottom:8px">
  <div style="display:flex;align-items:center;gap:10px">
    <div style="width:30px;height:30px;background:linear-gradient(135deg,#e8a045,#c67a1e);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:15px">⬡</div>
    <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:18px;font-weight:700;color:#e8e8f0">NeuraPilot</div>
  </div>
</div>
""", unsafe_allow_html=True)

    courses = load_courses(S)
    if not courses:
        st.info("Create a course to begin.")
        with st.expander("➕ New Course", expanded=True):
            nc_id = st.text_input("ID", key="nc_id"); nc_title = st.text_input("Title", key="nc_title")
            if st.button("Create", use_container_width=True):
                cid = nc_id.strip()
                if cid and cid.replace("-","").replace("_","").isalnum():
                    cs = load_courses(S); cs[cid] = {"title": nc_title.strip(), "description": ""}
                    save_courses(S, cs); course_upload_dir(S, cid); dbmod.upsert_course(DB, cid, nc_title.strip()); st.rerun()
        st.stop()

    # Course selector
    st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#2e2e42;padding:4px 0 6px">COURSE</div>', unsafe_allow_html=True)
    course_keys = list(courses.keys())
    # Auto-select newly created course if pending
    _pending = st.session_state.get("_pending_course")
    _default_idx = 0
    if _pending and _pending in course_keys:
        _default_idx = course_keys.index(_pending)
        st.session_state._pending_course = None  # clear after use
    course_id = st.selectbox("Course", course_keys,
        index=_default_idx,
        format_func=lambda k: f"{courses[k].get('title', k)}", label_visibility="collapsed")

    # ── Clear stale state when user switches course ───────────────────────
    if st.session_state._active_course != course_id:
        st.session_state._active_course = course_id
        st.session_state.quiz_obj   = None
        st.session_state.quiz_idx   = 0
        st.session_state.quiz_score = 0
        st.session_state.img_queue  = []
        # Reset thread so messages from old course don't show
        threads_for_new = dbmod.list_threads(DB, course_id)
        if threads_for_new:
            _load_thread(threads_for_new[0]["id"])
        else:
            _new_thread(course_id)
    # New course — no expander (avoids text overlap bug)
    if st.session_state.get("_show_nc"):
        nc_id2    = st.text_input("Course ID",    placeholder="ml101",           key="nc_id2v", label_visibility="collapsed")
        nc_title2 = st.text_input("Course Title", placeholder="Course title…",   key="nc_tt2v", label_visibility="collapsed")
        ca2, cb2 = st.columns(2)
        with ca2:
            if st.button("✅ Create", use_container_width=True, key="nc_ok2"):
                cid2 = nc_id2.strip()
                if cid2 and cid2.replace("-","").replace("_","").isalnum():
                    cs2 = load_courses(S); cs2[cid2] = {"title": nc_title2.strip() or cid2, "description": ""}
                    save_courses(S, cs2); course_upload_dir(S, cid2)
                    dbmod.upsert_course(DB, cid2, nc_title2.strip() or cid2)
                    # Store new course id so selectbox can auto-select it after rerun
                    st.session_state._pending_course = cid2
                    st.session_state._show_nc = False
                    st.success(f"✅ Course '{nc_title2.strip() or cid2}' created!")
                    st.rerun()
                else:
                    st.error("Course ID must be alphanumeric (hyphens/underscores allowed)")
        with cb2:
            if st.button("✕", use_container_width=True, key="nc_cancel2"):
                st.session_state._show_nc = False; st.rerun()
    else:
        if st.button("＋ New Course", use_container_width=True, key="nc_tog"):
            st.session_state._show_nc = True; st.rerun()

    if st.button("✏️  New Chat", use_container_width=True, type="primary"):
        _new_thread(course_id); st.rerun()

    threads = dbmod.list_threads(DB, course_id)
    if threads:
        st.caption("Recent Chats")
        for th in threads[:12]:
            is_act = (st.session_state.tid == th["id"])
            label  = th["title"][:28] + ("…" if len(th["title"]) > 28 else "")
            c1, c2 = st.columns([6, 1])
            with c1:
                if st.button(label, key=f"th_{th['id']}", use_container_width=True,
                              type="primary" if is_act else "secondary"):
                    _load_thread(th["id"]); st.rerun()
            with c2:
                if st.button("🗑", key=f"del_{th['id']}", help="Delete"):
                    dbmod.delete_thread(DB, th["id"])
                    if st.session_state.tid == th["id"]:
                        st.session_state.tid = None; st.session_state.msgs = []
                    st.rerun()

    if st.session_state.tid is None:
        _new_thread(course_id)

    st.divider()
    st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#2e2e42;padding:4px 0 6px">SETTINGS</div>', unsafe_allow_html=True)
    st.session_state.strict     = st.toggle("🔒 Strict mode",          value=bool(st.session_state.strict), help="ON=notes only. OFF=Tutor+knowledge")
    st.session_state.show_think = st.toggle("🧠 Show thinking",         value=bool(st.session_state.show_think))
    if "show_sugg" not in st.session_state:
        st.session_state.show_sugg = True
    st.session_state.show_sugg  = st.toggle("💡 Follow-up suggestions", value=bool(st.session_state.show_sugg))

    if not st.session_state.strict:
        st.caption("💡 Tutor mode — notes + general knowledge")

    st.divider()
    st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#2e2e42;padding:4px 0 6px">FILES</div>', unsafe_allow_html=True)

    IMG_EXTS = {"png","jpg","jpeg","gif","webp","bmp"}
    ALL_TYPES = ["pdf","md","txt","csv","xlsx","xls","pptx","ppt","docx","doc",
                 "json","jsonl","xml","yaml","yml","py","r","ipynb","html",
                 "png","jpg","jpeg","gif","webp"]

    # FIX 2: Accept multiple images, queue ALL of them
    up = st.file_uploader("Any files", type=ALL_TYPES, accept_multiple_files=True, label_visibility="collapsed")
    if up:
        udir = course_upload_dir(S, course_id)
        imgs_up, docs_up = [], []
        for f in up:
            ext = Path(f.name).suffix.lower().lstrip(".")
            (udir / f.name).write_bytes(f.getbuffer())
            (imgs_up if ext in IMG_EXTS else docs_up).append(f)

        if docs_up:
            st.success(f"📄 {len(docs_up)} doc(s) saved — click Ingest")
        if imgs_up:
            # Add ALL uploaded images to the queue
            new_imgs = []
            for img_f in imgs_up:
                ext  = Path(img_f.name).suffix.lower().lstrip(".")
                mime = {"jpg": "jpeg", "jpeg": "jpeg"}.get(ext, ext)
                b64  = base64.b64encode(img_f.getbuffer()).decode()
                new_imgs.append({"b64": f"data:image/{mime};base64,{b64}", "name": img_f.name})
            st.session_state.img_queue = new_imgs  # replace queue with new batch
            names = ", ".join(i["name"] for i in new_imgs)
            st.success(f"🖼️ {len(new_imgs)} image(s) queued: {names}")

    # Show image queue status
    if st.session_state.img_queue:
        st.markdown('<div style="margin:6px 0">' +
            "".join(f'<span class="img-queue-badge">🖼 {i["name"][:20]}</span>'
                    for i in st.session_state.img_queue) +
            '</div>', unsafe_allow_html=True)
        if st.button("✖ Clear image queue", use_container_width=True):
            st.session_state.img_queue = []; st.rerun()

    ci, cr = st.columns(2)
    with ci:
        if st.button("⚡ Ingest", use_container_width=True):
            pg = st.progress(0, "Starting…"); ss = st.empty()
            try:
                pg.progress(20, "Reading…")
                stats = ingest_course(course_id, settings=S, bundle=B)
                pg.progress(100, "Done!")
                ss.success(f"✅ {stats.chunks_indexed} chunks · {stats.duration_s:.1f}s")
            except Exception as e: pg.empty(); ss.error(str(e))
    with cr:
        if st.button("🔄 Reset", use_container_width=True):
            pg2 = st.progress(0, "Clearing…"); ss2 = st.empty()
            try:
                pg2.progress(10, "Clearing…")
                stats = ingest_course(course_id, settings=S, bundle=B, clear_existing=True)
                pg2.progress(100, "Done!")
                ss2.success(f"✅ {stats.chunks_indexed} fresh")
            except Exception as e: pg2.empty(); ss2.error(str(e))

    st.divider()
    st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#2e2e42;padding:4px 0 6px">QUICK ACTIONS</div>', unsafe_allow_html=True)
    qa_t = st.text_input("Topic", placeholder="e.g. neural networks", key="qa_topic", label_visibility="collapsed")
    qa1, qa2 = st.columns(2)
    with qa1:
        if st.button("📝 Summarize", use_container_width=True): st.session_state["_qa"] = ("summarize", qa_t)
        if st.button("🎯 Quiz",      use_container_width=True): st.session_state["_qa"] = ("quiz",      qa_t)
    with qa2:
        if st.button("🃏 Flashcards",use_container_width=True): st.session_state["_qa"] = ("flashcards",qa_t)
        if st.button("📅 Study Plan",use_container_width=True): st.session_state["_qa"] = ("plan",      qa_t)

    # ── Python session reset ─────────────────────────────────────────────
    st.divider()
    st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#2e2e42;padding:4px 0 6px">PYTHON NOTEBOOK</div>', unsafe_allow_html=True)
    py_vars = st.session_state.get("_py_ns", {})
    skip_default = {"pd","np","plt","matplotlib","warnings","train_test_split",
                    "StandardScaler","LinearRegression","LogisticRegression",
                    "mean_squared_error","r2_score","accuracy_score"}
    live = [k for k in py_vars if not k.startswith("_") and k not in skip_default]
    if live:
        st.caption(f"🔗 {len(live)} variable(s) in session: " + ", ".join(f"`{v}`" for v in live[:6]))
    else:
        st.caption("No variables yet — run a code block to start")
    if st.button("🗑 Clear Python Session", use_container_width=True):
        sid = _get_session_id()
        _PY_NS_STORE.pop(sid, None)
        st.session_state.pop("_py_ns", None)
        st.toast("🗑 Python session cleared — fresh start!")
        st.rerun()

    st.divider()
    fcc = dbmod.count_flashcards(DB, course_id)
    sc1, sc2 = st.columns(2)
    sc1.metric("Cards", fcc["total"]); sc2.metric("Due", fcc["due"])

    bks = dbmod.list_bookmarks(DB, course_id)
    if bks:
        st.divider()
        st.caption(f"🔖 Bookmarks ({len(bks)})")
        for bk in bks[:5]:
            with st.expander(bk["title"][:35]):
                st.markdown(bk["content"][:300])
                if st.button("🗑", key=f"rmbk_{bk['id']}"): dbmod.delete_bookmark(DB, bk["id"]); st.rerun()

    if st.session_state.msgs:
        st.divider()
        md = ["# NeuraPilot Chat Export\n\n"]
        for m in st.session_state.msgs:
            c = m.get("content","")
            if c not in ("__FLASHCARDS__","__QUIZ__","__IMGBLOCK__"):
                md.append(f"**{'You' if m['role']=='user' else 'AI'}:** {c}\n\n---\n\n")
        st.download_button("💾 Export Chat", "".join(md),
            file_name=f"chat_{course_id}_{int(time.time())}.md",
            mime="text/markdown", use_container_width=True)


    # ── License & Attribution ─────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
<div style="
    padding:14px 16px 12px;
    background:linear-gradient(135deg,#0a0a1a 0%,#0f1520 100%);
    border:1px solid #1e2a3a;
    border-radius:12px;
    text-align:center;
">
  <div style="font-size:19px;margin-bottom:5px">🧠</div>
  <div style="font-size:12px;font-weight:800;letter-spacing:1.6px;
              text-transform:uppercase;color:#3b82f6;margin-bottom:3px">
    NeuraPilot
  </div>
  <div style="font-size:11px;color:#94a3b8;margin-bottom:9px">
    Akila Lourdes Miriyala Francis
  </div>
  <div style="border-top:1px solid #1e2a3a;padding-top:8px;
              font-size:9.5px;color:#3a4a5a;line-height:1.6">
    © 2026 Akila Lourdes Miriyala Francis<br>
    <span style="letter-spacing:0.3px">— All Rights Reserved —</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Handle quick actions
# ─────────────────────────────────────────────────────────────────────────────
_QP = {
    "summarize":  lambda t: f"Summarize {'the ' + t if t else 'all uploaded notes'}. Cover all key concepts.",
    "flashcards": lambda t: f"Make flashcards on {t or 'uploaded notes'} with difficulty and Bloom levels.",
    "quiz":       lambda t: f"Create a 5-question MCQ quiz on {t or 'uploaded course material'}.",
    "plan":       lambda t: f"Create a 7-day study plan for {t or 'the course material'}.",
}

action = st.session_state.pop("_qa", None)
if action and st.session_state.tid:
    kind, topic = action
    t   = topic.strip() or "course material"
    msg = _QP[kind](t)
    _save(st.session_state.tid, "user", msg, {})
    think_ph = st.empty()
    # FIX 1: save result BEFORE rerun
    with st.spinner(""):
        result = _run_agent(msg, course_id, think_ph)
    _proc(result, course_id, topic_hint=t)
    if len(st.session_state.msgs) <= 2:
        _auto_title(st.session_state.tid, msg)
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Handle regenerate
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.pop("_regen", False):
    # Remove last assistant message
    while st.session_state.msgs and st.session_state.msgs[-1]["role"] == "assistant":
        st.session_state.msgs.pop()
    # Find last user message
    last_user = next((m for m in reversed(st.session_state.msgs) if m["role"] == "user"), None)
    if last_user:
        think_ph_rg = st.empty()
        with st.spinner("Regenerating…"):
            result_rg = _run_agent(last_user["content"], course_id, think_ph_rg)
        _proc(result_rg, course_id)
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_chat, tab_prac, tab_flash, tab_mast, tab_anal = st.tabs([
    "💬 Chat", "🎯 Practice", "🃏 Flashcards", "📊 Mastery", "📈 Analytics"
])

# ═════════════════════════════════════════════════════════════════════════════
# CHAT
# ═════════════════════════════════════════════════════════════════════════════
with tab_chat:

    # FIX 4: Show ALL queued images prominently — side by side
    if st.session_state.img_queue:
        q_imgs = st.session_state.img_queue
        n = len(q_imgs)
        st.markdown(
            f'<div style="background:#0d0d1a;border:1px solid #e8a04530;border-radius:10px;padding:14px 16px;margin-bottom:12px">'
            f'<div style="color:#e8a045;font-size:12px;font-weight:700;letter-spacing:.5px;margin-bottom:10px">'
            f'🖼️ {n} IMAGE{"S" if n>1 else ""} QUEUED — your next message will analyze {"all of them together" if n>1 else "it"}'
            f'</div>',
            unsafe_allow_html=True,
        )
        img_cols = st.columns(min(n, 4))
        for i, img_info in enumerate(q_imgs):
            with img_cols[i % 4]:
                st.markdown(
                    f'<img src="{img_info["b64"]}" style="width:100%;border-radius:6px;border:1px solid #2a2a38">',
                    unsafe_allow_html=True,
                )
                st.caption(f"Image {i+1}: {img_info['name']}")
        st.markdown('</div>', unsafe_allow_html=True)
        ca, cb = st.columns([3, 1])
        with cb:
            if st.button("✖ Clear images", use_container_width=True):
                st.session_state.img_queue = []; st.rerun()
        st.divider()

    # ── Render message history ────────────────────────────────────────────
    msgs = st.session_state.msgs
    rendered = msgs[-80:]          # show last 80 messages
    n_rendered = len(rendered)
    for mi, entry in enumerate(rendered):
        role    = entry["role"]
        content = entry["content"]
        meta    = entry.get("meta", {}) or {}
        mkey    = f"m{mi}_{id(entry)}"
        # FIX: is_last uses rendered slice index, NOT total msgs count
        is_last = (mi == n_rendered - 1)

        with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):

            # ── Flashcard block ───────────────────────────────────────────
            if role == "assistant" and content == "__FLASHCARDS__":
                _render_fc(meta.get("cards", []), meta.get("topic", ""), mkey)
                src = _chips(meta.get("sources", []))
                if src: st.markdown(src, unsafe_allow_html=True)
                st.markdown(
                    f'<div class="meta-row"><span class="meta-txt">⏱ {meta.get("latency_ms",0)}ms</span>{_itag("flashcards")}<span class="meta-txt">· {meta.get("topic","")}</span></div>',
                    unsafe_allow_html=True,
                )

            # ── Quiz block ────────────────────────────────────────────────
            elif role == "assistant" and content == "__QUIZ__":
                n_q = meta.get("n_questions", 0)
                if n_q > 0:
                    _render_quiz_banner(n_q)
                else:
                    st.warning("Quiz generation failed in Strict mode. Disable Strict mode and try again.")
                src = _chips(meta.get("sources", []))
                if src: st.markdown(src, unsafe_allow_html=True)
                st.markdown(
                    f'<div class="meta-row"><span class="meta-txt">⏱ {meta.get("latency_ms",0)}ms</span>{_itag("quiz")}</div>',
                    unsafe_allow_html=True,
                )

            # ── Image block (user sent images) ────────────────────────────
            elif role == "user" and content == "__IMGBLOCK__":
                images_in_msg = meta.get("images", [])
                question      = meta.get("question", "")
                if images_in_msg:
                    img_disp_cols = st.columns(min(len(images_in_msg), 3))
                    for ii, img_item in enumerate(images_in_msg):
                        with img_disp_cols[ii % 3]:
                            st.markdown(
                                f'<img src="{img_item["b64"]}" style="width:100%;border-radius:8px;border:1px solid #1e1e2e">',
                                unsafe_allow_html=True,
                            )
                            st.caption(img_item["name"])
                st.markdown(question)

            # ── Normal message ────────────────────────────────────────────
            else:
                if role == "assistant":
                    _render_assistant(content, meta, mkey, is_last, course_id)
                else:
                    st.markdown(content)

    # ── Suggestions (last AI message only) ───────────────────────────────
    if msgs and msgs[-1]["role"] == "assistant" and st.session_state.get("show_sugg", True):
        last_meta = msgs[-1].get("meta", {}) or {}
        intent_s  = last_meta.get("intent", "")
        if intent_s in {"ask", "summarize", "vision"}:
            topic_s = last_meta.get("topic", "this topic")
            clean_topic = re.sub(r'^image:\s*', '', topic_s)
            if intent_s == "vision":
                suggs = [
                    "What patterns or trends do you see in this data?",
                    "Compare these plots and recommend which y-axis variable to use",
                    "Write R code to recreate this visualization",
                    f"Visualize the concept of {clean_topic}",
                ]
            else:
                suggs = [
                    f"Give me a worked example of {clean_topic}",
                    f"Quiz me on {clean_topic}",
                    f"Make flashcards on {clean_topic}",
                    f"Visualize {clean_topic} as a diagram",
                ]
            scols = st.columns(len(suggs))
            for si, (scol, sq) in enumerate(zip(scols, suggs)):
                with scol:
                    if st.button(sq, key=f"sug_{si}_{len(msgs)}", use_container_width=True):
                        _save(st.session_state.tid, "user", sq, {})
                        think_ph_s = st.empty()
                        with st.spinner(""):
                            res_s = _run_agent(sq, course_id, think_ph_s)
                        _proc(res_s, course_id)
                        st.rerun()

    # ── Chat input ────────────────────────────────────────────────────────
    user_input = st.chat_input("Message NeuraPilot… · Ask about images · 'quiz on X' · 'flashcards on X'")

    if user_input:
        # ── Multi-image path ──────────────────────────────────────────────
        if st.session_state.img_queue:
            queued = st.session_state.img_queue.copy()
            st.session_state.img_queue = []  # clear queue immediately

            # Save user turn showing ALL images
            _save(st.session_state.tid, "user", "__IMGBLOCK__",
                  {"images": queued, "question": user_input})

            if len(queued) == 1:
                # Single image — direct analysis
                img = queued[0]
                with st.spinner(f"🔍 Analyzing {img['name']}…"):
                    ans = _vision_ans(user_input, img["b64"], img["name"])
                _save(st.session_state.tid, "assistant", ans, {
                    "intent": "vision", "topic": f"image: {img['name']}",
                    "sources": [], "latency_ms": 0, "eval_scores": {},
                })
            else:
                # Multiple images — analyze ALL together in one combined response
                with st.spinner(f"🔍 Analyzing all {len(queued)} images together…"):
                    ans = _vision_ans_multi(user_input, queued)
                _save(st.session_state.tid, "assistant", ans, {
                    "intent": "vision",
                    "topic": f"images: {', '.join(i['name'] for i in queued)}",
                    "sources": [], "latency_ms": 0, "eval_scores": {},
                })

            # Auto-title on first messages
            if len(st.session_state.msgs) <= 3:
                _auto_title(st.session_state.tid, user_input)

        else:
            # ── Check for "visualize overall/overview" intent first ────────
            _VIZ_OVERALL = re.compile(
                r'\b(visuali[sz]e|show|draw|diagram|overview|overall|'
                r'big picture|architecture|full picture|whole idea|'
                r'entire paper|whole paper|paper diagram)\b',
                re.I
            )
            _is_overview_req = bool(_VIZ_OVERALL.search(user_input)) and \
                any(w in user_input.lower() for w in [
                    "overall","overview","whole","entire","full","big picture",
                    "architecture","idea of","idea of the","paper","diagram of"
                ])

            # Normal text query
            _save(st.session_state.tid, "user", user_input, {})
            think_ph = st.empty()
            # Route through Orchestrator — correctly handles guidance/homework queries
            with st.spinner(""):
                result = _orchestrate(user_input, course_id, think_ph)
            _proc(result, course_id)
            if len(st.session_state.msgs) <= 2:
                _auto_title(st.session_state.tid, user_input)

            # Auto-open Paper Overview if user asked to visualize the whole paper
            if _is_overview_req:
                # Find the last assistant message key and auto-open viz panel
                last_idx = len(st.session_state.msgs) - 1
                last_entry = st.session_state.msgs[last_idx] if last_idx >= 0 else None
                if last_entry:
                    _auto_mkey = f"m{min(last_idx, 79)}_{id(last_entry)}"
                    st.session_state[f"_showviz_{_auto_mkey}"] = True
                    # Pre-select Paper Overview type
                    st.session_state[f"_viztype_{_auto_mkey}"] = "📄 AI Research Overview"

        st.rerun()

    # ── Watermark ─────────────────────────────────────────────────────────
    st.markdown(
        '<div style="text-align:center;padding:6px 0 2px;margin-top:6px">'
        '<span style="font-size:10px;color:#1e1e2e;letter-spacing:0.4px">'
        'NeuraPilot · Akila Lourdes Miriyala Francis'
        ' &nbsp;|&nbsp; '
        '© 2026 All Rights Reserved'
        '</span></div>',
        unsafe_allow_html=True,
    )

    # ── Empty state ───────────────────────────────────────────────────────
    if not msgs:
        st.markdown("""
<div style="text-align:center;padding:60px 20px;max-width:600px;margin:0 auto">
  <div style="font-size:48px;margin-bottom:16px;filter:drop-shadow(0 0 20px #e8a04540)">⬡</div>
  <div style="font-size:1.6em;font-weight:700;color:#e8e8f0;margin-bottom:8px">How can I help you learn?</div>
  <div style="color:#3a3a52;font-size:14px;line-height:1.7;margin-bottom:32px">
    Upload your notes, papers, CSV datasets, or images — then ask anything.<br>
    I can explain concepts, run code, analyze charts, generate quizzes and flashcards.
  </div>
</div>
""", unsafe_allow_html=True)
        cols = st.columns(2)
        examples = [
            ("📊", "Analyze CSV",    "What columns are in my dataset and what patterns do you see?"),
            ("🎯", "Generate quiz",  "Create a 5-question MCQ quiz on the uploaded notes"),
            ("🖼️", "Analyze chart",  "What does this chart show and what are the key insights?"),
            ("💻", "Write code",     "Write Python to visualize the data distribution and run it"),
        ]
        for i, (icon, title, prompt) in enumerate(examples):
            with cols[i % 2]:
                if st.button(f"{icon}  {title}", key=f"ex_{i}", use_container_width=True):
                    _save(st.session_state.tid, "user", prompt, {})
                    think_ex = st.empty()
                    with st.spinner(""):
                        r_ex = _run_agent(prompt, course_id, think_ex)
                    _proc(r_ex, course_id)
                    st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# PRACTICE
# ═════════════════════════════════════════════════════════════════════════════
with tab_prac:
    st.subheader("🎯 Practice Quiz")
    if not st.session_state.quiz_obj:
        st.info("No quiz loaded.\n\n- Quick Actions → **Generate Quiz**\n- Or type: *'Create a quiz on [topic]'*\n\n💡 Disable Strict mode for guaranteed quiz generation.")
    else:
        qs = st.session_state.quiz_obj.get("questions", [])
        total_q = len(qs)
        idx = st.session_state.quiz_idx; score = st.session_state.quiz_score
        if total_q == 0:
            st.warning("Quiz is empty — disable Strict mode and regenerate.")
            if st.button("Clear"): st.session_state.quiz_obj = None; st.rerun()
        elif idx >= total_q:
            pct  = int(100 * score / total_q)
            icon = "🎉" if pct >= 80 else "👍" if pct >= 60 else "📚"
            (st.success if pct >= 80 else st.info if pct >= 60 else st.warning)(
                f"{icon}  **{score}/{total_q}** ({pct}%)")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔄 Retry", use_container_width=True):
                    st.session_state.quiz_idx = 0; st.session_state.quiz_score = 0; st.rerun()
            with c2:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.quiz_obj = None; st.session_state.quiz_idx = 0; st.rerun()
        else:
            st.progress(idx / total_q, text=f"Q{idx+1}/{total_q}  ·  Score: {score}/{idx}")
            q = qs[idx]; choices = q.get("choices", []); ai = int(q.get("answer_index", 0))
            diff = q.get("difficulty",""); bloom = q.get("bloom_level","")
            di = {"easy":"🟢","medium":"🟡","hard":"🔴"}.get(diff,"⚪")
            st.markdown(f"### Q{idx+1}. {q.get('q','')}")
            if diff: st.caption(f"{di} {diff.title()}  ·  Bloom: {bloom.title()}")
            sel = st.radio("Your answer:", range(len(choices)), format_func=lambda k: choices[k], key=f"qr_{idx}")
            if st.button("✅  Submit Answer", key=f"qs_{idx}"):
                correct = (int(sel) == ai)
                if correct: st.success("✅ Correct!"); st.session_state.quiz_score += 1
                else:       st.error(f"❌ Correct answer: **{choices[ai]}**")
                exp = q.get("explanation","")
                if exp:
                    st.info(f"**Explanation:** {exp}")
                    _voice(exp, f"qv_{idx}")
                if q.get("citations"): st.caption("Sources: " + ", ".join(q["citations"]))
                dbmod.update_mastery(DB, course_id, q.get("topic", course_id), correct=correct)
                st.session_state.quiz_idx += 1
                time.sleep(0.3); st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# FLASHCARDS
# ═════════════════════════════════════════════════════════════════════════════
with tab_flash:
    st.subheader("🃏 Spaced Repetition")
    st.caption("SM-2 algorithm — optimally scheduled reviews")
    lim  = st.slider("Cards per session", 5, 30, 10)
    due  = dbmod.get_due_flashcards(DB, course_id, limit=lim)
    fcc2 = dbmod.count_flashcards(DB, course_id)
    if not due:
        if fcc2["total"] == 0:
            st.info("No cards yet. Generate flashcards in Chat or via Quick Actions.")
        else:
            st.success(f"🎉 All caught up! ({fcc2['total']} total, none due today)")
    else:
        st.caption(f"**{len(due)} cards due**  ·  0=forgot  5=perfect"); st.divider()
        for card in due:
            bi  = {"remember":"🟦","understand":"🟩","apply":"🟨","analyze":"🟧","evaluate":"🟥","create":"🟪"}.get(card.get("bloom_level",""),"⬜")
            di2 = {"easy":"🟢","medium":"🟡","hard":"🔴"}.get(card.get("difficulty",""),"⚪")
            lbl = f"{bi}{di2}  [{card['topic']}]  {card['question'][:55]}{'…' if len(card['question'])>55 else ''}"
            with st.expander(lbl):
                st.markdown(f"**{card['question']}**")
                c1,c2,c3 = st.columns(3)
                c1.metric("Reps", card["reps"]); c2.metric("Interval", f"{card['interval_days']}d"); c3.metric("Ease", f"{card['ease']:.2f}")
                if st.toggle("👁 Reveal Answer", key=f"rv_{card['id']}"):
                    st.success(card["answer"])
                    if card.get("citations"): st.caption("Sources: " + ", ".join(card["citations"]))
                    _voice(card["answer"], f"fv_{card['id']}")
                qual = st.select_slider("Rate recall:", [0,1,2,3,4,5], 3,
                    format_func=lambda v:{0:"Blackout",1:"Wrong",2:"Almost",3:"Correct",4:"Easy",5:"Perfect"}[v],
                    key=f"ql_{card['id']}")
                if st.button("Save & Next", key=f"sv_{card['id']}", use_container_width=True):
                    res = dbmod.sm2_review(DB, card["id"], qual)
                    st.success(f"Next in **{res.get('interval_days',1)}** day(s)." if qual >= 3 else "See tomorrow.")
                    time.sleep(0.2); st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# MASTERY
# ═════════════════════════════════════════════════════════════════════════════
with tab_mast:
    st.subheader("📊 Knowledge Mastery")
    st.caption("Bayesian Beta model  ·  P(mastery) = α/(α+β)")
    rows = dbmod.get_mastery(DB, course_id)
    if not rows:
        st.info("No data yet — complete quizzes in Practice to start tracking.")
    else:
        import pandas as pd
        avg  = sum(r["p_mastery"] for r in rows) / len(rows)
        high = [r for r in rows if r["p_mastery"] >= 0.8]
        low  = [r for r in rows if r["p_mastery"] < 0.5]
        m1, m2, m3 = st.columns(3)
        m1.metric("Topics",       len(rows))
        m2.metric("Avg Mastery",  f"{avg:.0%}")
        m3.metric("Mastered ≥80%",len(high))
        if low: st.warning("Needs review: " + " · ".join(f"**{r['topic']}** ({r['p_mastery']:.0%})" for r in low))
        df = pd.DataFrame(rows).sort_values("p_mastery", ascending=False)
        df["pct"] = (df["p_mastery"] * 100).round(1)
        st.bar_chart(df.set_index("topic")["pct"])
        disp = df[["topic","pct","ci_lower","ci_upper","attempts"]].copy()
        disp.columns = ["Topic","Mastery%","CI Lo","CI Hi","Attempts"]
        disp["Mastery%"] = disp["Mastery%"].apply(lambda x: f"{x:.1f}%")
        for c in ["CI Lo","CI Hi"]: disp[c] = disp[c].apply(lambda x: f"{x:.1%}")
        st.dataframe(disp, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
with tab_anal:
    try:
        from neurapilot.analytics_dashboard import render_analytics_tab
        render_analytics_tab(DB, course_id)
    except Exception as _ana_err:
        st.subheader("📈 Session Analytics")
        ixns = dbmod.get_recent_interactions(DB, course_id, limit=200)
        if not ixns:
            st.info("No interactions yet. Ask NeuraPilot some questions first!")
        else:
            import pandas as pd
            df = pd.DataFrame(ixns)
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Total Queries", len(df))
            a2.metric("Avg Latency", f"{df['latency_ms'].mean():.0f} ms")
            st.line_chart(df.set_index("ts")["latency_ms"])
            st.caption(f"Analytics note: {_ana_err}")
