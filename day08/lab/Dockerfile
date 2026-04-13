# ─────────────────────────────────────────────
# Dockerfile — Chainlit RAG App (Day 08 Lab)
# Build: docker build -t rag-chainlit .
# ─────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# Cài dependencies hệ thống
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước → tận dụng Docker layer cache
COPY requirements.txt .

# Cài đúng packages cần thiết (bỏ ragas/datasets nặng nếu chỉ chạy app)
RUN pip install --no-cache-dir \
    python-dotenv \
    openai \
    google-generativeai \
    chromadb \
    sentence-transformers \
    rank-bm25 \
    chainlit \
    tqdm

# Copy source code
COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CHAINLIT_HOST=0.0.0.0 \
    CHAINLIT_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
