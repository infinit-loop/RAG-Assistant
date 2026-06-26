FROM python:3.12-slim

# System deps occasionally needed by faiss/numpy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching (only reinstalls if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install python-multipart==0.0.20 pypdf==5.1.0

# Optional PII stack (Presidio + spaCy model). Enable with:
#   docker compose build --build-arg INSTALL_PII=true
# Leaves the default image light when not requested.
ARG INSTALL_PII=false
RUN if [ "$INSTALL_PII" = "true" ]; then \
        pip install --no-cache-dir -r requirements-pii.txt && \
        python -m spacy download en_core_web_sm ; \
    fi
# Copy application code, UI, documents, and the evaluation harness
COPY app ./app
COPY ui ./ui
COPY docs ./docs
COPY eval ./eval

# Base corpus = docs/base_corpus (indexed at startup). docs/examples holds
# sample files for users to test the upload flow and is NOT indexed.
ENV DOCS_DIR=/app/docs/base_corpus \
    UPLOAD_DIR=/app/uploads \
    PYTHONUNBUFFERED=1

# Directory for persisted user uploads (also mounted as a volume in compose)
RUN mkdir -p /app/uploads

# 8000 = API, 8501 = Streamlit UI
EXPOSE 8000 8501

# Default command runs the API; docker-compose overrides this per service.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]