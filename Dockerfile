# Dockerfile per RAG System
FROM python:3.11-slim

WORKDIR /app

# Installa dipendenze di sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements e installa dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il codice dell'applicazione
COPY app/ ./app/
COPY benchmark_dataset.json .

# Crea cartelle per output e data
RUN mkdir -p /app/evaluation_results /app/data /app/scripts

# Script di avvio
COPY scripts/ /app/scripts/
RUN chmod +x /app/scripts/*.sh

# Variabili d'ambiente di default
ENV QDRANT_HOST=qdrant
ENV OLLAMA_HOST=ollama
ENV DATA_PATH=/app/data
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
