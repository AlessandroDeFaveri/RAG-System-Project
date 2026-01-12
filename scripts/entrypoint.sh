#!/bin/bash
set -e

echo "RAG System - Avvio automatico"

# Attendi che Qdrant sia pronto
echo "[1/4] Attesa Qdrant..."
until curl -s http://${QDRANT_HOST}:6333/healthz > /dev/null 2>&1; do
    echo "  Qdrant non ancora pronto, attendo..."
    sleep 2
done
echo "  Qdrant OK!"

# Attendi che Ollama sia pronto
echo "[2/4] Attesa Ollama..."
until curl -s http://${OLLAMA_HOST}:11434/api/tags > /dev/null 2>&1; do
    echo "  Ollama non ancora pronto, attendo..."
    sleep 2
done
echo "  Ollama OK!"

# Funzione per scaricare un modello
pull_model() {
    local model=$1
    echo "  Controllo modello $model..."
    if ! curl -s http://${OLLAMA_HOST}:11434/api/tags | grep -q "\"name\":\"$model\""; then
        echo "  Scarico modello $model (può richiedere alcuni minuti)..."
        curl -s -X POST http://${OLLAMA_HOST}:11434/api/pull -d "{\"name\": \"$model\"}" | while read line; do
            status=$(echo "$line" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
            if [ -n "$status" ]; then
                echo "    $status"
            fi
        done
        echo "  Modello $model scaricato!"
    else
        echo "  Modello $model già presente."
    fi
}

# Verifica/scarica modelli LLM
echo "[3/4] Verifica modelli LLM..."
if [ "$1" = "full-evaluation" ]; then
    # Per full-evaluation, scarica tutti i modelli necessari
    MODELS=("llama3.2" "gemma2:2b" "phi3:mini" "qwen2.5:7b")
    for model in "${MODELS[@]}"; do
        pull_model "$model"
    done
else
    # Estrai il modello dal parametro --llm se presente
    LLM_MODEL=""
    for i in "$@"; do
        if [ "$prev_arg" = "--llm" ]; then
            LLM_MODEL="$i"
            break
        fi
        prev_arg="$i"
    done
    
    # Se non specificato, usa il default
    LLM_MODEL=${LLM_MODEL:-llama3.2}
    pull_model "$LLM_MODEL"
fi
echo "  Modelli pronti!"

# Esegui ingestion se ci sono PDF nella cartella data E Qdrant è vuoto
echo "[4/4] Controllo ingestion..."
cd /app/app

# Controlla se ci sono già punti in Qdrant
QDRANT_POINTS=$(curl -s "http://${QDRANT_HOST}:6333/collections/documenti_tesi" 2>/dev/null | grep -o '"points_count":[0-9]*' | grep -o '[0-9]*' || echo "0")

if [ "$QDRANT_POINTS" -gt 0 ]; then
    echo "  Qdrant contiene già $QDRANT_POINTS punti. Skip ingestion."
else
    if [ -d "/app/data" ] && [ "$(ls -A /app/data/*.pdf 2>/dev/null)" ]; then
        echo "  Trovati PDF e Qdrant vuoto, eseguo ingestion..."
        python ingestion.py
    else
        echo "  Nessun PDF trovato in /app/data."
    fi
fi

echo ""
echo "Sistema pronto!"

# Esegui il comando passato (evaluation, main, o shell)
cd /app/app
if [ "$1" = "evaluation" ]; then
    echo "Avvio evaluation singolo modello..."
    shift
    python evaluation.py "$@"
elif [ "$1" = "full-evaluation" ]; then
    echo "Avvio evaluation multi-modello..."
    exec /app/scripts/run_full_evaluation.sh
elif [ "$1" = "interactive" ]; then
    echo "Avvio modalità interattiva..."
    python main.py
elif [ "$1" = "shell" ]; then
    echo "Avvio shell..."
    exec /bin/bash
else
    # Default: esegui evaluation con parametri di default
    echo "Eseguo evaluation di default..."
    python evaluation.py /app/benchmark_dataset.json \
        --seed 0 \
        --llm ${LLM_MODEL:-llama3.2} \
        --output /app/evaluation_results/results.csv
fi
