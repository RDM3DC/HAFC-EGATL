# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# Install OS-level build tools required by scipy/numpy wheel builds
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# ── Runtime ────────────────────────────────────────────────────────────────────
EXPOSE 8501

# Streamlit needs a writable home for its config cache
ENV HOME=/tmp

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
