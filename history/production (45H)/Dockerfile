# ================================================================================
# Dockerfile — CPU Build
# Project: Transformer Language Model from Scratch
# ================================================================================
# Build:   docker build -t lm-scratch .
# Run:     docker run --rm lm-scratch python run.py --prompt "Hello world"
# ================================================================================

FROM python:3.11-slim

# Metadata
LABEL maintainer="45H Build" \
      description="Transformer Language Model — CPU" \
      version="1.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 aiuser
WORKDIR /app

# Install Python dependencies first (layer cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY --chown=aiuser:aiuser . .

# Create required directories
RUN mkdir -p output/logs output/checkpoints data && \
    chown -R aiuser:aiuser /app

USER aiuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from model import LanguageModel; print('OK')"

# Default command: generate text with tiny model
CMD ["python", "run.py", \
     "--config", "config/tiny.yaml", \
     "--mode", "generate", \
     "--prompt", "The transformer architecture", \
     "--max_tokens", "50", \
     "--temperature", "0.8"]
