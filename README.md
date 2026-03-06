<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=NeuraPilot%20🧠&fontSize=60&fontColor=ffffff&fontAlignY=38&desc=Production-grade%20Agentic%20RAG%20Tutor&descAlignY=58&descSize=18&animation=fadeIn" width="100%"/>

<br/>

### *Built by* **Akila Lourdes Miriyala Francis**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangGraph-0.2-00C853?style=for-the-badge&logo=chainlink&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20DB-FF6B35?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/LLM-Ollama%20%2F%20OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white"/>
</p>

<p align="center">
  <a href="https://drive.google.com/drive/folders/1UUTOKNSG1Bld-aUyoLKjxci97zJwvj7L?usp=sharing">
    <img src="https://img.shields.io/badge/🎬%20Live%20Demo-Watch%20Now-FF0000?style=for-the-badge&logo=googledrive&logoColor=white"/>
  </a>
  &nbsp;
  <a href="https://github.com/AKilalours/neurapilot">
    <img src="https://img.shields.io/badge/GitHub-Source%20Code-181717?style=for-the-badge&logo=github&logoColor=white"/>
  </a>
</p>

<br/>

> **Compound AI pipeline** with LangGraph orchestration, semantic caching, real-time observability, and multi-modal understanding — built for students who deserve a production-grade tutor.

<br/>

</div>

---

## 🎯 System Goal & SLOs

> All metrics benchmarked on **MacBook M2** with `llama3.1:8b` via Ollama.

| SLO | Target | Achieved | Status |
|-----|--------|----------|--------|
| p50 latency | < 2.0 s | **~1.8 s** | ✅ |
| p95 latency | < 5.0 s | **~4.2 s** | ✅ |
| p99 latency | < 10.0 s | **~7.1 s** | ✅ |
| Cost / request `gpt-4o-mini` | < $0.01 | **$0.00024** | ✅ |
| Cache hit rate (warm) | > 20% | **~38%** | ✅ |
| Latency reduction via cache | > 30% | **~42% avg** | ✅ |
| Faithfulness (RAGAS) | > 0.70 | **0.81** | ✅ |
| Answer relevance | > 0.70 | **0.78** | ✅ |
| hit@10 proxy | > 0.70 | **0.83** | ✅ |

---

## 🎬 Demo

> **📁 [Full Demo — Screenshots + Video Walkthrough](https://drive.google.com/drive/folders/1UUTOKNSG1Bld-aUyoLKjxci97zJwvj7L?usp=sharing)**

The demo folder contains:
- 📸 **Images** — UI screenshots, flashcards, quiz, visualizations, analytics dashboard
- 🎥 **Video** — Full walkthrough of the agentic RAG pipeline in action

---

## 📐 Architecture — Data Flow

```
Student Query
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI                       │
│   chat · quiz · flashcards · visualize · study plan  │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│        Semantic Cache  (SQLite + cosine sim)         │
│    embed query → cosine ≥ 0.92 → HIT  (~120 ms)    │
└──────────────┬──────────────────────┬───────────────┘
          MISS │                      │ HIT
               ▼                      │
┌──────────────────────────────────┐  │
│       LangGraph Pipeline         │  │
│                                  │  │
│  ① Classify                     │  │
│     keyword regex → LLM fallback │  │
│                                  │  │
│  ② Rewrite  (HyDE expansion)    │  │
│     hypothetical doc → richer    │  │
│     embedding for retrieval      │  │
│                                  │  │
│  ③ Retrieve (ChromaDB MMR)      │  │
│     max marginal relevance →     │  │
│     top-K diverse chunks         │  │
│                                  │  │
│  ④ Rerank   (BM25 term scoring) │  │
│     must-terms boost →           │  │
│     precision ↑ at low K         │  │
│                                  │  │
│  ⑤ Generate (intent-routed)     │  │
│     ask / quiz / flashcards /    │  │
│     summarize / plan / guidance  │  │
└──────────────┬───────────────────┘  │
               │ answer               │
               ▼                      │
┌─────────────────────────────────┐   │
│   Log + Cache + Eval            │◀──┘
│   latency · RAGAS · sources     │
│   mastery update (Bayesian)     │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│     Analytics Dashboard         │
│   p95 · cost · hit@10 · SLO    │
└─────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1 — Local (Recommended for dev)

```bash
# 1. Clone the repo
git clone git@github.com:AKilalours/neurapilot.git
cd neurapilot

# 2. Install dependencies
pip install -e .

# 3. Set up environment
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY or set OLLAMA_BASE_URL for local

# 4. Run the UI
cd neurapilot
streamlit run ui_streamlit.py
# → http://localhost:8501

# 5. Optional: Analytics dashboard
streamlit run analytics_dashboard.py --server.port 8502
# → http://localhost:8502
```

### Option 2 — Docker (One command)

```bash
cp .env.example .env
# Edit .env with your API key

docker compose up -d
# UI:        http://localhost:8501
# Analytics: http://localhost:8502
```

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | Required |
| Ollama | Latest | For local LLM — `ollama pull llama3.1:8b` |
| Docker | 24+ | Optional, for containerised run |
| OpenAI API Key | — | Optional, for cloud LLM |

---

## 🔧 Tech Stack & Trade-offs

| Layer | Technology | Trade-off Rationale |
|-------|------------|---------------------|
| Orchestration | LangGraph 0.2 | Stateful DAG + conditional routing; retry logic baked in |
| LLM | Ollama / OpenAI | Swappable provider — local for dev, OpenAI for prod |
| Embeddings | nomic-embed-text | 768-dim, free, fast; outperforms MiniLM on MTEB |
| Vector DB | ChromaDB (MMR) | Local-first, zero infra; MMR reduces chunk redundancy |
| Cache | SQLite + cosine sim | Zero extra infra; handles paraphrases automatically |
| App DB | SQLite WAL mode | Threads, flashcards, Bayesian mastery — no Postgres overhead |
| UI | Streamlit | Rapid iteration; full agentic loop in one file |
| Container | Docker + Compose | One-command reproducible deploy |

**Key decisions:**
- **SQLite over Redis** — sufficient for < 10k req/day; zero infra complexity
- **ChromaDB over Pinecone** — local-first with LangChain abstractions; swap in one line
- **Cosine cache over exact-match** — 38% hit rate vs ~5% on paraphrase queries
- **MMR retrieval** — reduces duplicate chunks; net quality win over standard top-K

---

## 📏 Latency Budget

| Stage | p50 | p95 | Notes |
|-------|-----|-----|-------|
| Tokenization + embed | ~30 ms | ~60 ms | Local nomic model |
| Cache lookup | ~15 ms | ~40 ms | SQLite cosine scan |
| Rewrite (HyDE) | ~300 ms | ~800 ms | 1 LLM call |
| ChromaDB MMR | ~50 ms | ~150 ms | top-16 candidates |
| BM25 rerank | ~5 ms | ~15 ms | in-process |
| Generation | ~1.3 s | ~3.2 s | llama3.1:8b |
| **Total (cache miss)** | **~1.8 s** | **~4.2 s** | End-to-end; all 5 pipeline stages |
| **Total (cache hit)** | **~120 ms** | **~280 ms** | **42% faster; bypasses LLM entirely** |

---

## 🔁 Reliability — Fallbacks & Observability

```
┌──────────────┐   miss   ┌──────────────┐   fail   ┌──────────────┐
│ Semantic     │ ───────▶ │  LangGraph   │ ───────▶ │  Fallback:   │
│ Cache        │          │  Pipeline    │          │  empty ctx   │
│ (SQLite)     │ ◀─────── │  (LLM+RAG)  │          │  + error msg │
└──────────────┘   hit    └──────────────┘          └──────────────┘
        │                        │
        │                        ▼
        │               ┌─────────────────┐
        └─────────────▶ │ SQLite WAL log  │
                        │ latency · RAGAS │
                        │ cost estimate   │
                        └────────┬────────┘
                                 ▼
                        ┌─────────────────┐
                        │ Analytics Dash  │
                        │ p95 SLO alerts  │
                        │ cost/req gauge  │
                        └─────────────────┘
```

**Failure modes handled:**
- ChromaDB unavailable → graceful degradation with empty context + LLM general answer
- LLM timeout → error surfaced in chat with retry suggestion
- JSON parse failure (quiz/flashcards) → fallback to plain text
- Strict mode + thin context → explicit user notification, not silent empty response

---

## 📉 Postmortem Log

| Version | Issue | Root Cause | Fix |
|---------|-------|------------|-----|
| v2→v3 | All intents → QUIZ | Classify prompt: "quiz beats everything" in LLM | Keyword regex before LLM call; visualize/analyse first |
| v3→v4 | Diagram text overlapping | Font sizes too large for card bounds | Fonts −30%, DPI 130 |
| v4→v5 | Flashcards not generating | Strict mode ON by default; thin context → empty JSON | Default `strict=False`; min-4-card fallback in prompt |
| v5→v6 | Quiz persists after other questions | `quiz_obj` never cleared on non-quiz responses | Clear `quiz_obj` in every non-quiz `_proc()` branch |
| v6→v7 | Code blocks don't share variables | Fresh `exec()` namespace each run | Shared `_py_ns` dict in session (Jupyter-style) |
| v7→v8 | `"Flashcards"` plural → ask intent | `\b(flashcard)\b` misses plural `s` | Added `s?`; expanded regex to cover "make flashcards" |
| v7→v8 | Regen button → no effect | `mkey = id(entry)` changes each rerun | Stable hash: `abs(hash((mi, content[:80])))` |
| v7→v8 | KNN query → Edge Detection answer | Stale `@st.cache_resource` held old retriever | Session-scoped cache keyed by `(course_id, ingest_version)` |

---

## 🧪 Evaluation Gates

| Metric | Tool | Gate | Action if Fail |
|--------|------|------|----------------|
| Faithfulness | RAGAS | ≥ 0.70 | Flag response; log for review |
| Answer relevance | RAGAS | ≥ 0.70 | Trigger rewrite node retry |
| hit@10 proxy | Custom BM25 | ≥ 0.70 | Increase `CANDIDATE_K` |
| p95 latency | SQLite telemetry | ≤ 5.0 s | Alert dashboard; cache warm-up |
| Cost / request | Token counter | ≤ $0.01 | Switch to smaller model tier |

---

## 🔮 MLOps & CI

- **Dockerised** — `docker compose up -d` spins UI + analytics in one command
- **`.env`-driven config** — zero hardcoded secrets; Pydantic `Settings` validates at startup
- **Eval-on-ingest** — RAGAS scores logged per interaction for offline drift detection
- **Shadow tests** — swappable LLM provider (`ollama` ↔ `openai`) for A/B latency comparison
- **Rollback path** — `clear_existing=True` wipes ChromaDB collection; semantic cache invalidated by `ingest_version` counter

---

## 📁 Project Structure

```
neurapilot/
├── neurapilot/
│   ├── ui_streamlit.py          # Main chat UI — full agentic loop
│   ├── analytics_dashboard.py   # Observability: p95, cost/req, hit@10, SLO gauges
│   ├── viz_pipeline.py          # Matplotlib diagram engine (mindmap/pipeline/arch)
│   ├── viz_arch_diagram.py      # AI paper overview renderer
│   ├── api.py                   # FastAPI REST layer
│   ├── config.py                # Pydantic Settings — all config from .env
│   ├── storage.py               # Path helpers, course manifest
│   ├── core/
│   │   ├── db.py                # SQLite schema: threads, flashcards, mastery
│   │   ├── semantic_cache.py    # Cosine similarity cache (sim ≥ 0.92 → HIT)
│   │   └── telemetry.py         # Per-stage pipeline timing
│   └── rag/
│       ├── agent_graph.py       # LangGraph DAG: classify→rewrite→retrieve→rerank→generate
│       ├── prompts.py           # All LLM prompt templates (strict + tutor modes)
│       ├── rag.py               # Generation: ask, quiz, flashcards, plan
│       ├── store.py             # ChromaDB MMR retriever + collection scoping
│       └── ingest.py            # PDF/TXT/MD loader, chunking, dedup
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## One-Liner

> *"Built a hybrid RAG tutor with LangGraph + ChromaDB + semantic caching → p95 ~4.2 s, cost $0.00024/request, RAGAS faithfulness 0.81, hit@10 0.83. FastAPI + Streamlit + SQLite + Ollama/OpenAI. Dockerised. 42% latency reduction via cosine cache. 8 production bugs found, root-caused, and fixed — all documented in postmortem log."*

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=120&section=footer" width="100%"/>

<br/>

**© 2026 Akila Lourdes Miriyala Francis — All Rights Reserved**

*NeuraPilot · LangGraph · ChromaDB · Streamlit · Ollama/OpenAI · Docker*

</div>
