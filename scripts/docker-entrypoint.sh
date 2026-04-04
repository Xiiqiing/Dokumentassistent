#!/bin/bash
set -e

# ---- Wait for Qdrant to be ready ------------------------------------------
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
COLLECTION="${COLLECTION_NAME:-ku_documents}"

echo "Waiting for Qdrant at ${QDRANT_URL} ..."
until curl -sf "${QDRANT_URL}/healthz" > /dev/null 2>&1; do
    sleep 2
done
echo "Qdrant is ready."

# ---- Check if collection already has data ----------------------------------
POINT_COUNT=$(curl -sf "${QDRANT_URL}/collections/${COLLECTION}" 2>/dev/null \
    | python -c "import sys,json; print(json.load(sys.stdin).get('result',{}).get('points_count',0))" 2>/dev/null \
    || echo "0")

if [ "${POINT_COUNT}" = "0" ] || [ -z "${POINT_COUNT}" ]; then
    echo "Collection '${COLLECTION}' is empty or missing — running ingestion ..."
    python -m scripts.ingest
    echo "Ingestion complete."
else
    echo "Collection '${COLLECTION}' already has ${POINT_COUNT} points — skipping ingestion."
fi

# ---- Start the API server --------------------------------------------------
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
