# -----------------------------------------------------------------
# Stage 1 - Build / dependency layer
# -----------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System build deps required by faiss-cpu and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
    && rm -rf /var/lib/apt/lists/*

# Layer-cache friendly: copy requirements first
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# -----------------------------------------------------------------
# Stage 2 - Lean runtime image
# -----------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

WORKDIR /app

# Pull installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app.py   ./app.py
COPY main.py  ./main.py
COPY src/     ./src/

# Directories for FAISS indexes and HuggingFace model cache
# Mount a volume at /app/data to persist vector stores across restarts
RUN mkdir -p /app/data /app/.cache/huggingface

EXPOSE 8501

# Healthcheck via Streamlit built-in health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# Pass GROQ_API_KEY at runtime:
#   docker run -e GROQ_API_KEY=<your-key> -p 8501:8501 youtube-rag
#   docker run --env-file .env -p 8501:8501 youtube-rag
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
