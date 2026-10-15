# Inference image for the FastAPI service (CPU only).
FROM python:3.11-slim

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY api/ api/
COPY configs/ configs/
# The checkpoint is either baked into the image (committed checkpoints/tiny_gpt.pt)
# or downloaded at start-up from $CHECKPOINT_URL.
COPY checkpoints/ checkpoints/

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT}"]
