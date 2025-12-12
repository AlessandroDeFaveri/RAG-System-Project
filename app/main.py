"""
RAG Pipeline Completa - Main

Questo script integra tutti i componenti:
1. Retrieval: Cerca i chunk più rilevanti in Qdrant
2. LLM: Genera la risposta usando Ollama

"""
import sys
from retrieval import load_embedding_model, connect_to_qdrant, search_similar_chunks, format_context
from llm import build_prompt, query_ollama_streaming

# CONFIGURAZIONE

TOP_K = 5  # Numero di chunk da recuperare

# FUNZIONI

def rag_query(question: str, model, qdrant_client, top_k: int = TOP_K):
    """
    Esegue una query RAG completa.
    
    1. Cerca i chunk più simili (Retrieval)
    2. Costruisce il prompt con il contesto
    3. Genera la risposta con l'LLM
    """
    print(f"\n{'='*60}")
    print(f"DOMANDA: {question}")
    print(f"{'='*60}\n")
    
    # Step 1: Retrieval
    print("[1/3] Ricerca chunk rilevanti in Qdrant...")
    chunks = search_similar_chunks(qdrant_client, model, question, top_k=top_k)
    
    if not chunks:
        print("Nessun chunk trovato!")
        return "Non ho trovato informazioni rilevanti nel database."
    
    print(f"      Trovati {len(chunks)} chunk (score max: {chunks[0]['score']:.4f})")
    
    # Mostra le fonti
    print("\n[2/3] Fonti utilizzate:")
    for i, chunk in enumerate(chunks, 1):
        print(f"      {i}. {chunk['source']} (pag. {chunk['page']}) - score: {chunk['score']:.4f}")
    
    # Step 2: Costruzione Prompt
    context = format_context(chunks)
    prompt = build_prompt(context, question)
    
    # Step 3: Query LLM
    print(f"\n[3/3] Generazione risposta con LLM...\n")
    print("-" * 40)
    print("RISPOSTA:\n")
    
    response = query_ollama_streaming(prompt)
    
    print("-" * 40)
    
    return response


def interactive_mode(model, qdrant_client):
    """Modalità interattiva: l'utente può fare più domande."""
    print("\n" + "="*60)
    print("       RAG SYSTEM - Modalità Interattiva")
    print("="*60)
    print("Scrivi la tua domanda e premi INVIO.")
    print("Digita 'exit' o 'quit' per uscire.\n")
    
    while True:
        try:
            question = input("\n📝 Domanda: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'esci', 'q']:
                print("\nArrivederci! 👋")
                break
            
            rag_query(question, model, qdrant_client)
            
        except KeyboardInterrupt:
            print("\n\nInterrotto dall'utente. Arrivederci! 👋")
            break


# MAIN

def main():
    print("Inizializzazione RAG System...")
    
    # Carica modello embedding
    model = load_embedding_model()
    
    # Connetti a Qdrant
    print("Connessione a Qdrant...")
    qdrant_client = connect_to_qdrant()
    print("Connesso!\n")
    
    # Verifica argomenti da linea di comando
    if len(sys.argv) > 1:
        # Modalità singola query
        question = " ".join(sys.argv[1:])
        rag_query(question, model, qdrant_client)
    else:
        # Modalità interattiva
        interactive_mode(model, qdrant_client)


if __name__ == "__main__":
    main()
