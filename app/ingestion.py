import os
from pypdf import PdfReader
from config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

def get_pdf_files(directory):
    """Restituisce la lista dei file PDF nella cartella."""
    pdf_files = []
    if not os.path.exists(directory):
        print(f"ERRORE: La cartella {directory} non esiste.")
        return []
        
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_files.append(os.path.join(directory, filename))
    return pdf_files

def extract_pages_from_pdf(pdf_path):
    """Estrae il testo pagina per pagina, restituendo una lista di dizionari."""
    pages = []
    try:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                pages.append({
                    "page": page_num,
                    "text": page_text
                })
    except Exception as e:
        print(f"Errore nella lettura di {pdf_path}: {e}")
    return pages

def create_chunks(text, chunk_size, overlap):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calcoliamo la fine ideale
        end = min(start + chunk_size, text_length)
        
        # Se non siamo alla fine del testo, cerchiamo un taglio pulito
        if end < text_length:
            # Salviamo il punto di taglio originale per sicurezza
            original_end = end
            
            # Cerca indietro uno spazio, punto o a-capo
            while end > start and text[end] not in ' .\n':
                end -= 1
            

            if end <= start + overlap: # Se il taglio è troppo corto rispetto all'overlap
                end = original_end     # Ignora la pulizia e taglia brutalmente
        
        # Estrai e pulisci il chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        
        # Se siamo arrivati alla fine del testo, usciamo dal ciclo
        if end == text_length:
            break
            
        # Usiamo max() per garantire che si avanzi sempre almeno di 1 carattere
        # nel caso estremo in cui end - overlap <= start
        start = max(start + 1, end - overlap)

    return chunks


def main():
    print("--- INIZIO PROCESSO DI INGESTION (PyPDF) ---")
    
    pdf_files = get_pdf_files(DATA_PATH)
    print(f"Trovati {len(pdf_files)} file PDF.")

    all_chunks = []

    for pdf_file in pdf_files:
        print(f"Elaborazione: {pdf_file}...")
        
        # 1. Estrazione pagine
        pages = extract_pages_from_pdf(pdf_file)
        
        if not pages:
            print(f"  ATTENZIONE: Nessun testo estratto da {pdf_file}")
            continue
        
        # 2. Chunking per ogni pagina
        file_chunk_count = 0
        for page_data in pages:
            page_chunks = create_chunks(page_data["text"], CHUNK_SIZE, CHUNK_OVERLAP)
            
            for chunk_text in page_chunks:
                chunk_data = {
                    "source": os.path.basename(pdf_file),
                    "page": page_data["page"],
                    "chunk_id": len(all_chunks),
                    "text": chunk_text
                }
                all_chunks.append(chunk_data)
                file_chunk_count += 1
        
        print(f"  -> Creati {file_chunk_count} chunks da {len(pages)} pagine.")

    print("\n--- RISULTATO FINALE ---")
    print(f"Totale chunks generati: {len(all_chunks)}")
    
    if all_chunks:
        print("\nEsempio primo chunk:")
        print(f"Fonte: {all_chunks[0]['source']}")
        print(f"Pagina: {all_chunks[0]['page']}")
        print(f"Testo (primi 200 car): {all_chunks[0]['text'][:200]}...")

if __name__ == "__main__":
    main()