"""
Step 3: Retrieval - Ricerca per similarità in Qdrant
"""
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL

# FUNZIONI

def load_embedding_model():
    """Carica il modello di embedding (lo stesso usato per l'ingestion!)."""
    print(f"Caricamento modello '{EMBEDDING_MODEL}'...")
    return SentenceTransformer(EMBEDDING_MODEL)


def connect_to_qdrant():
    """Connette a Qdrant."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return client


def search_similar_chunks(client, model, query: str, top_k: int = 5):
    """
    Cerca i chunk più simili alla query.
    
    Args:
        client: QdrantClient
        model: SentenceTransformer model
        query: La domanda dell'utente
        top_k: Quanti risultati restituire
    
    Returns:
        Lista di dizionari con source, page, text, score
    """
    # 1. Genera l'embedding della query
    query_vector = model.encode(query).tolist()
    
    # 2. Cerca in Qdrant
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    ).points
    
    # 3. Formatta i risultati
    chunks = []
    for result in results:
        chunks.append({
            "source": result.payload.get("source", "N/A"),
            "page": result.payload.get("page", "N/A"),
            "text": result.payload.get("text", ""),
            "score": result.score
        })
    
    return chunks


def format_context(chunks):
    """
    Formatta i chunk recuperati in un contesto per il prompt.
    
    Formato richiesto dal prof:
    [Chunk 1 – paper title, authors, etc.]
    <text...>
    """
    context_parts = []
    
    for i, chunk in enumerate(chunks, 1):
        # Creiamo un header chiaro per ogni pezzo
        header = f"--- REF ID: [{i}] ---"
        meta = f"(Source: {chunk['source']}, Page: {chunk['page']})"
        content = chunk['text']
        
        # Mettiamo tutto insieme
        entry = f"{header}\nMetadata: {meta}\nContent: {content}\n"
        context_parts.append(entry)
    
    return "\n".join(context_parts)


# TEST

if __name__ == "__main__":
    # Test del modulo
    print("--- TEST RETRIEVAL ---\n")
    
    model = load_embedding_model()
    client = connect_to_qdrant()
    
    # Query di test
    query = "What is OLAP?"
    print(f"Query: {query}\n")
    
    # Cerca
    chunks = search_similar_chunks(client, model, query, top_k=3)
    
    # Mostra risultati
    print(f"Trovati {len(chunks)} chunk rilevanti:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} (score: {chunk['score']:.4f}) ---")
        print(f"Source: {chunk['source']}, Page: {chunk['page']}")
        print(f"Text: {chunk['text'][:300]}...")
        print()
    
    # Mostra contesto formattato
    print("--- CONTESTO FORMATTATO ---")
    print(format_context(chunks))
