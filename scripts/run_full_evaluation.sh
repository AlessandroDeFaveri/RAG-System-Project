#!/bin/bash
# Script per eseguire la valutazione completa con più modelli
# Tutti i risultati vengono salvati nello stesso CSV

set -e

echo "RAG System - Valutazione Multi-Modello"

# Configurazione
OUTPUT_FILE="/app/evaluation_results/results.csv"
BENCHMARK_FILE="/app/benchmark_dataset.json"
MODELS=("llama3.2" "gemma2:2b" "phi3:mini")

# Determina il prossimo seed leggendo dal CSV esistente
if [ -f "$OUTPUT_FILE" ]; then
    # Trova l'ultimo seed usato e incrementa di 1
    LAST_SEED=$(tail -n 1 "$OUTPUT_FILE" | cut -d';' -f1 2>/dev/null || echo "-1")
    if [[ "$LAST_SEED" =~ ^[0-9]+$ ]]; then
        SEED=$((LAST_SEED + 1))
    else
        SEED=0
    fi
    echo "File CSV esistente trovato. Prossimo seed: $SEED"
else
    SEED=0
    echo "Nessun CSV esistente. Seed iniziale: $SEED"
fi

cd /app/app

# Funzione per verificare/scaricare un modello
ensure_model() {
    local model=$1
    echo "Verifico modello $model..."
    if ! curl -s http://${OLLAMA_HOST}:11434/api/tags | grep -q "$model"; then
        echo "  Scarico modello $model..."
        curl -X POST http://${OLLAMA_HOST}:11434/api/pull -d "{\"name\": \"$model\"}" --max-time 600
    fi
    echo "  Modello $model pronto!"
}

# Scarica tutti i modelli prima di iniziare
echo ""
echo "[1/3] Download modelli..."
for model in "${MODELS[@]}"; do
    ensure_model "$model"
done

# Esegui ingestion se necessario
echo ""
echo "[2/3] Controllo ingestion..."
if [ -d "/app/data" ] && [ "$(ls -A /app/data/*.pdf 2>/dev/null)" ]; then
    echo "  Trovati PDF, eseguo ingestion..."
    python ingestion.py
else
    echo "  Nessun PDF nuovo da indicizzare."
fi

# Non rimuovere il CSV - i nuovi risultati vengono aggiunti
echo ""
echo "[3/3] Avvio valutazione con seed=$SEED..."

# Esegui evaluation per ogni modello (sempre in append mode)
for model in "${MODELS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Valutazione modello: $model"
    echo "----------------------------------------"
    python evaluation.py "$BENCHMARK_FILE" \
        --seed "$SEED" \
        --llm "$model" \
        --output "$OUTPUT_FILE" \
        --append
done

echo ""
echo "Valutazione completata!"
echo "Risultati salvati in: $OUTPUT_FILE"

# Mostra anteprima risultati
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "Anteprima risultati:"
    cat "$OUTPUT_FILE"
fi
