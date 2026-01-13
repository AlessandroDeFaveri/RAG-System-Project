"""
Step 2: Generazione Embeddings e inserimento in Qdrant (Versione Ottimizzata)
"""
import hashlib
import sys
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from tqdm import tqdm

# Importiamo le funzioni dallo Step 1
from ingestion import get_pdf_files, extract_pages_from_pdf, create_chunks

# CONFIGURAZIONE - importa da config.py
from config import (
    DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, EMBEDDING_DIM,
    QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME
)

EMBEDDING_DIMENSION = EMBEDDING_DIM


# FUNZIONI

def load_embedding_model():
    print(f"Caricamento modello '{EMBEDDING_MODEL}'...")
    return SentenceTransformer(EMBEDDING_MODEL)

def connect_to_qdrant():
    print(f"Connessione a Qdrant ({QDRANT_HOST}:{QDRANT_PORT})...")
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        # Test rapido di connessione
        client.get_collections()
        print("Connesso a Qdrant!")
        return client
    except Exception as e:
        print(f"\nERRORE CRITICO: Impossibile connettersi a Qdrant.")
        print(f"Dettaglio: {e}")
        print("Suggerimento: Hai lanciato 'docker compose up -d'?")
        sys.exit(1) # Esce dallo script

def recreate_collection(client):
    """Crea (o ricrea) la collection in Qdrant."""
    print(f"Reset collection '{COLLECTION_NAME}'...")
    
    # Elimina la collection se esiste già
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    
    # Crea la nuova collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=Distance.COSINE
        )
    )

def generate_and_insert(client, chunks, model, batch_size=64):
    """
    Genera embedding e inserisce in Qdrant a lotti per risparmiare RAM.
    """
    total_chunks = len(chunks)
    print(f"\nInizio processamento di {total_chunks} chunk...")

    # Processiamo a blocchi (batch)
    for i in tqdm(range(0, total_chunks, batch_size), desc="Embedding & Upload"):
        batch_chunks = chunks[i : i + batch_size]
        
        # 1. Estrai solo il testo per l'embedding
        batch_texts = [c["text"] for c in batch_chunks]
        
        # 2. Genera embeddings per questo lotto
        batch_vectors = model.encode(batch_texts, show_progress_bar=False)
        
        # 3. Prepara i punti Qdrant
        points = []
        for idx, (chunk, vector) in enumerate(zip(batch_chunks, batch_vectors)):
            
            
            # Creiamo un ID unico basato sul contenuto. 
            # Se rilanci lo script, l'ID sarà uguale ed eviterai duplicati fantasma.
            unique_str = f"{chunk['source']}_{chunk['page']}_{chunk['text'][:20]}"
            point_id = hashlib.md5(unique_str.encode()).hexdigest()
            
            points.append(PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload={
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "text": chunk["text"],
                    "chunk_seq_id": chunk["chunk_id"] 
                }
            ))
        
        # 4. Carica questo lotto su Qdrant
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

def run_ingestion_pipeline():
    """Riusa la logica di ingestion per ottenere i dati grezzi."""
    import os 
    
    print("--- FASE 1: LETTURA PDF ---")
    pdf_files = get_pdf_files(DATA_PATH)
    all_chunks = []
    
    for pdf_file in pdf_files:
        filename = os.path.basename(pdf_file)
        pages = extract_pages_from_pdf(pdf_file)
        
        if not pages:
            continue
            
        for page_data in pages:
            page_chunks = create_chunks(page_data["text"], CHUNK_SIZE, CHUNK_OVERLAP)
            for chunk_text in page_chunks:
                all_chunks.append({
                    "source": filename,
                    "page": page_data["page"],
                    "chunk_id": len(all_chunks),
                    "text": chunk_text
                })
    
    print(f"Trovati {len(all_chunks)} chunk totali da processare.")
    return all_chunks

# MAIN

def main():
    # 1. Check Connessione (Prima di fare lavoro pesante)
    client = connect_to_qdrant()
    
    # 2. Preparazione Dati
    raw_chunks = run_ingestion_pipeline()
    if not raw_chunks:
        print("Nessun dato trovato. Esco.")
        return

    # 3. Caricamento Modello AI
    model = load_embedding_model()

    # 4. Reset DB
    recreate_collection(client)

    # 5. Embedding + Inserimento (Ottimizzato)
    generate_and_insert(client, raw_chunks, model)

    # 6. Verifica Finale
    info = client.get_collection(COLLECTION_NAME)
    print(f"\n--- SUCCESSO ---")
    print(f"Collezione: {COLLECTION_NAME}")
    print(f"Documenti (Vettori) inseriti: {info.points_count}")

if __name__ == "__main__":
    main()
