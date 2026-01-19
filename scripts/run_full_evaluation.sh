#!/bin/bash
#Script per eseguire una valutazione completa su più modelli e template
# Tutti i risultati vengono salvati nello stesso CSV

set -e

echo "RAG System - Valutazione Multi-Modello e Multi-Template"

# Configurazione
OUTPUT_FILE="/app/evaluation_results/results.csv"
BENCHMARK_FILE="/app/benchmark_dataset.json"
# Modelli Ollama (locali)
OLLAMA_MODELS=("llama3.2" "qwen2.5:7b")
# Modelli Azure OpenAI (cloud) - richiedono AZURE_OPENAI_KEY
AZURE_MODELS=("gpt-5-mini")
# Tutti i modelli da testare
MODELS=("${OLLAMA_MODELS[@]}" "${AZURE_MODELS[@]}")
TEMPLATES=(1 2 3 4 5)

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

# Funzione per verificare se è un modello Azure
is_azure_model() {
    local model=$1
    for azure_model in "${AZURE_MODELS[@]}"; do
        if [[ "$model" == "$azure_model" ]]; then
            return 0
        fi
    done
    return 1
}

# Funzione per verificare/scaricare un modello Ollama
ensure_model() {
    local model=$1
    # Salta i modelli Azure (non passano da Ollama)
    if is_azure_model "$model"; then
        echo "  $model è un modello Azure OpenAI, skip download."
        return
    fi
    echo "Verifico modello $model..."
    if ! curl -s http://${OLLAMA_HOST}:11434/api/tags | grep -q "$model"; then
        echo "  Scarico modello $model..."
        curl -X POST http://${OLLAMA_HOST}:11434/api/pull -d "{\"name\": \"$model\"}" --max-time 600
    fi
    echo "  Modello $model pronto!"
}

# Scarica tutti i modelli Ollama prima di iniziare
echo ""
echo "[1/3] Download modelli Ollama..."
for model in "${OLLAMA_MODELS[@]}"; do
    ensure_model "$model"
done

# Verifica chiave Azure se ci sono modelli Azure
if [ ${#AZURE_MODELS[@]} -gt 0 ]; then
    if [ -z "$AZURE_OPENAI_KEY" ]; then
        echo ""
        echo "   ATTENZIONE: AZURE_OPENAI_KEY non configurata!"
        echo "   I modelli Azure (${AZURE_MODELS[*]}) verranno saltati."
        echo "   Per abilitarli, imposta la chiave nel file .env"
        # Rimuovi i modelli Azure dalla lista
        MODELS=("${OLLAMA_MODELS[@]}")
    else
        echo "  Azure OpenAI configurato. Modelli: ${AZURE_MODELS[*]}"
    fi
fi

# Controlla se Qdrant ha già dati
echo ""
echo "[2/3] Controllo ingestion..."
QDRANT_POINTS=$(curl -s "http://${QDRANT_HOST}:6333/collections/documenti_tesi" | grep -o '"points_count":[0-9]*' | grep -o '[0-9]*' || echo "0")

if [ "$QDRANT_POINTS" -gt 0 ] 2>/dev/null; then
    echo "  Qdrant ha già $QDRANT_POINTS documenti indicizzati. Salto ingestion."
else
    # Qdrant vuoto, controlla se ci sono PDF da indicizzare
    if [ -d "/app/data" ] && [ "$(ls -A /app/data/*.pdf 2>/dev/null)" ]; then
        echo "  Qdrant vuoto. Trovati PDF, eseguo ingestion..."
        python embeddings.py
    else
        echo "  Qdrant vuoto e nessun PDF trovato."
    fi
fi

# Non rimuovere il CSV - i nuovi risultati vengono aggiunti
echo ""
echo "[3/3] Avvio valutazione con seed=$SEED..."
echo "Modelli: ${MODELS[*]}"
echo "Template: ${TEMPLATES[*]}"
echo "Open Knowledge: No e Yes"
echo ""

# Esegui evaluation per ogni combinazione modello x template x open_knowledge
for model in "${MODELS[@]}"; do
    for template_id in "${TEMPLATES[@]}"; do
        for open_knowledge in "false" "true"; do
            if [ "$open_knowledge" = "true" ]; then
                ok_flag="--open-knowledge"
                ok_label="Yes"
            else
                ok_flag=""
                ok_label="No"
            fi
            
            echo ""
            echo "========================================"
            echo "Modello: $model | Template: $template_id | OpenKnowledge: $ok_label"
            echo "========================================"
            python evaluation.py "$BENCHMARK_FILE" \
                --seed "$SEED" \
                --llm "$model" \
                --template-id "$template_id" \
                $ok_flag \
                --output "$OUTPUT_FILE" \
                --detailed
                
        done
    done
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