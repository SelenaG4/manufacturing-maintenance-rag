FROM python:3.11-slim

WORKDIR /app

# faiss-cpu needs the OpenMP runtime at import time on debian-slim.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data/index ./data/index

EXPOSE 8000

# Shell form so $PORT expands -- Render assigns its own PORT; falls back to 8000.
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"
