# ─────────────────────────────────────────────────────────────────────────────
# NeuraPilot — Production Dockerfile
# Multi-stage build: builder (deps) → runtime (lean image)
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip + install wheel
RUN pip install --upgrade pip wheel

# Copy requirements first (layer cache)
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="NeuraPilot Team"
LABEL description="NeuraPilot — Production-grade Agentic RAG Tutor"
LABEL version="1.0.0"

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1001 neurapilot
USER neurapilot

# Install pre-built wheels from builder stage
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl

# Copy application source
COPY --chown=neurapilot:neurapilot . /app

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# ── Data volumes (persisted externally) ──────────────────────────────────────
# .data/         → SQLite DB, uploads, course manifests
# .chroma/       → ChromaDB vector store
VOLUME ["/app/.data", "/app/.chroma"]

# ── Ports ─────────────────────────────────────────────────────────────────────
# 8501 → Streamlit UI
# 8000 → FastAPI (optional)
EXPOSE 8501 8000

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Entrypoint ───────────────────────────────────────────────────────────────
# Default: run Streamlit UI
# Override with: docker run neurapilot python -m neurapilot.api   (FastAPI)
CMD ["streamlit", "run", "neurapilot/ui_streamlit.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]
