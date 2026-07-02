FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy engine code + baked data (CSVs, model .pkl, features.json, metrics.json)
COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

EXPOSE 8080

# Cloud Run sets PORT env var (default 8080)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
