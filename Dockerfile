FROM python:3.11-slim

# Prevents Python from writing .pyc files and ensures logs are shown immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    HF_HOME=/app/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    TOKENIZERS_PARALLELISM=false

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the application
COPY . /app

# Create necessary directories at build/run time
RUN mkdir -p /data/uploads /data/faiss_indices \
    /app/.cache \
    /app/.cache/huggingface \
    /app/.cache/huggingface/hub \
    /app/.cache/huggingface/transformers \
    && chmod -R 777 /app/.cache /data/uploads /data/faiss_indices

# Expose the port used by Hugging Face Spaces
EXPOSE 7860

# Environment variables (HF Spaces sets PORT at runtime)
ENV PORT=7860 \
    HOST=0.0.0.0 \
    HF_SPACE=1

# Start the server with gunicorn, binding to $PORT
# app:app refers to Flask app object named `app` in app.py
CMD exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 8 --timeout 180 app:app


