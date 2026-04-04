FROM python:3.11-slim

WORKDIR /app

# ---------- System packages: nginx + supervisor + curl -----------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends nginx supervisor curl && \
    rm -rf /var/lib/apt/lists/*

# ---------- Python dependencies -----------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Application code --------------------------------------------------
COPY . .

# ---------- Pre-download sentence-transformers models -------------------------
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'); \
CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')"

# ---------- Build-time ingestion: import docs into Qdrant local ---------------
ENV QDRANT_PATH=/app/qdrant_data \
    QDRANT_URL="" \
    EMBEDDING_PROVIDER=local \
    LLM_PROVIDER=google_genai \
    API_BASE_URL=http://localhost:8000
RUN python -m scripts.ingest

# ---------- Nginx config: port 7860 reverse proxy ----------------------------
RUN rm /etc/nginx/sites-enabled/default
COPY nginx.spaces.conf /etc/nginx/conf.d/default.conf

# ---------- Supervisord config ------------------------------------------------
COPY supervisord.spaces.conf /etc/supervisor/conf.d/supervisord.conf

# ---------- Entrypoint --------------------------------------------------------
RUN chmod +x scripts/docker-entrypoint.sh

EXPOSE 7860

CMD ["supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
