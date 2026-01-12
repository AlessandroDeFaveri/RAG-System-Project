"""
Evaluation System per RAG Pipeline

"""
import json
import csv
import os
import re
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from retrieval import load_embedding_model, connect_to_qdrant, search_similar_chunks, format_context
from llm import build_prompt, query_ollama
from config import OLLAMA_MODEL, TOP_K


# DATA LOADING

def load_benchmark_dataset(path: str) -> List[Dict]:

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Caricato dataset con {len(data)} domande")
    return data



# RAG QUERY (versione per evaluation - non streaming)

def rag_query_for_eval(question: str, model, qdrant_client, top_k: int = TOP_K, llm_model: str = OLLAMA_MODEL, template_id: int = 1, open_knowledge: bool = False) -> Tuple[str, List[Dict]]:
    """
    Esegue una query RAG e restituisce risposta + chunk trovati.
    
    Args:
        question: La domanda dell'utente
        model: Modello di embedding
        qdrant_client: Client Qdrant
        top_k: Numero di chunk da recuperare
        llm_model: Nome del modello LLM
        template_id: ID del template di prompt (1-5)
        open_knowledge: Se True, permette all'LLM di usare conoscenza esterna
    
    Returns:
        Tuple[str, List[Dict]]: (risposta_llm, chunks_trovati)
    """
    # Retrieval
    chunks = search_similar_chunks(qdrant_client, model, question, top_k=top_k)
    
    if not chunks:
        return "Non ho trovato informazioni rilevanti.", []
    
    # Build prompt e query LLM (non streaming)
    context = format_context(chunks)
    prompt = build_prompt(context, question, template_id=template_id, open_knowledge=open_knowledge)
    response = query_ollama(prompt, model=llm_model)
    
    return response, chunks


# PARSING CITAZIONI DALLA RISPOSTA LLM

def extract_cited_refs(response: str, chunks: List[Dict]) -> List[Dict]:
    """
    Estrae i riferimenti effettivamente citati nella risposta LLM.
    
    Cerca pattern come [1], [2], [3] nella risposta e mappa ai chunk.
    
    Returns:
        Lista di chunk effettivamente citati
    """
    # Pattern per trovare citazioni [N]
    pattern = r'\[(\d+)\]'
    cited_ids = set(int(m) for m in re.findall(pattern, response))
    
    # Mappa agli indici dei chunk (1-indexed nel prompt)
    cited_chunks = []
    for ref_id in cited_ids:
        idx = ref_id - 1  # Converti a 0-indexed
        if 0 <= idx < len(chunks):
            cited_chunks.append(chunks[idx])
    
    return cited_chunks


def get_found_sources_and_pages(chunks: List[Dict]) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Estrae sorgenti e pagine dai chunk.
    
    Returns:
        Tuple: (lista_sorgenti_uniche, dizionario{sorgente: [pagine]})
    """
    sources = []
    pages_by_source = {}
    
    for chunk in chunks:
        source = chunk.get('source', '')
        page = chunk.get('page', 0)
        
        if source not in sources:
            sources.append(source)
            pages_by_source[source] = []
        
        if page not in pages_by_source[source]:
            pages_by_source[source].append(page)
    
    return sources, pages_by_source


# METRICHE

def compute_source_accuracy(expected_sources: List[str], found_sources: List[str]) -> Tuple[int, int, float]:
    """
    Calcola l'accuratezza delle sorgenti (intersezione).
    
    Returns:
        Tuple: (num_corretti, num_attesi, accuracy_decimale)
    """
    expected_set = set(expected_sources)
    found_set = set(found_sources)
    
    intersection = expected_set & found_set
    num_correct = len(intersection)
    num_expected = len(expected_set)
    
    # Gestisce domande fuori scope (expected_sources vuoto)
    if num_expected == 0:
        accuracy = 1.0 if num_correct == 0 else 0.0
    else:
        accuracy = num_correct / num_expected 
    return num_correct, num_expected, round(accuracy, 4)


def compute_page_accuracy(
    expected_pages: Dict[str, List[int]], 
    found_pages: Dict[str, List[int]]
) -> Tuple[int, int, float]:
    """
    Calcola l'accuratezza delle pagine (solo per le sorgenti trovate correttamente).
    
    Returns:
        Tuple: (num_pagine_corrette, num_pagine_attese, accuracy_decimale)
    """
    total_correct = 0
    total_expected = 0
    
    for source, expected_page_list in expected_pages.items():
        total_expected += len(expected_page_list)
        
        if source in found_pages:
            found_page_set = set(found_pages[source])
            expected_page_set = set(expected_page_list)
            total_correct += len(expected_page_set & found_page_set)
    
    # Gestisce domande fuori scope (expected_pages vuoto)
    if total_expected == 0:
        accuracy = 1.0 if total_correct == 0 else 0.0
    else:
        accuracy = total_correct / total_expected 
    return total_correct, total_expected, round(accuracy, 4)


def compute_answer_similarity(
    expected_answer: str, 
    llm_answer: str, 
    model
) -> float:
    """
    Calcola la similarità coseno tra gli embedding delle risposte.
    
    Rimuove la sezione "References" dalla risposta LLM prima del calcolo.
    """
    # Rimuovi la sezione References dalla risposta LLM
    llm_answer_clean = re.split(r'(?i)references?:?', llm_answer)[0].strip()
    
    # Genera embedding
    expected_emb = model.encode([expected_answer])
    llm_emb = model.encode([llm_answer_clean])
    
    # Calcola similarità coseno
    similarity = cosine_similarity(expected_emb, llm_emb)[0][0]
    
    return float(similarity)


# EVALUATION RUNNER

def evaluate_single_question(
    question_data: Dict,
    embedding_model,
    qdrant_client,
    top_k: int = TOP_K,
    use_cited_only: bool = False,
    llm_model: str = OLLAMA_MODEL,
    template_id: int = 1,
    open_knowledge: bool = False
) -> Dict:
    """
    Valuta una singola domanda.
    
    Args:
        question_data: Dizionario con id, question, expected_answer, expected_sources, expected_pages
        embedding_model: Modello per embedding
        qdrant_client: Client Qdrant
        top_k: Numero di chunk da recuperare
        use_cited_only: Se True, considera solo i chunk effettivamente citati nella risposta
        llm_model: Nome del modello LLM da usare
        template_id: ID del template di prompt (1-5)
        open_knowledge: Se True, permette all'LLM di usare conoscenza esterna
    
    Returns:
        Dizionario con risultati della valutazione
    """
    q_id = question_data['id']
    question = question_data['question']
    expected_answer = question_data['expected_answer']
    expected_sources = question_data['expected_sources']
    expected_pages = question_data['expected_pages']
    
    # Esegui RAG query con template specificato
    response, chunks = rag_query_for_eval(question, embedding_model, qdrant_client, top_k, llm_model, template_id, open_knowledge)
    
    # Decidi quali chunk considerare
    if use_cited_only:
        # Solo chunk effettivamente citati nella risposta
        eval_chunks = extract_cited_refs(response, chunks)
    else:
        # Tutti i chunk recuperati
        eval_chunks = chunks
    
    # Estrai sorgenti e pagine trovate
    found_sources, found_pages = get_found_sources_and_pages(eval_chunks)
    
    # Calcola metriche
    src_correct, src_total, src_str = compute_source_accuracy(expected_sources, found_sources)
    page_correct, page_total, page_str = compute_page_accuracy(expected_pages, found_pages)
    similarity = compute_answer_similarity(expected_answer, response, embedding_model)
    
    return {
        'question_id': q_id,
        'question': question,
        'llm_response': response,
        'found_sources': found_sources,
        'found_pages': found_pages,
        'source_correct': src_correct,
        'source_total': src_total,
        'source_accuracy': src_str,
        'page_correct': page_correct,
        'page_total': page_total,
        'page_accuracy': page_str,
        'similarity': round(similarity, 4)
    }


def run_evaluation(
    dataset_path: str,
    seed: int = 0,
    llm_model: str = OLLAMA_MODEL,
    top_k: int = TOP_K,
    use_cited_only: bool = False,
    template_id: int = 1,
    open_knowledge: bool = False,
    verbose: bool = True
) -> List[Dict]:
    """
    Esegue la valutazione completa su tutto il dataset.
    
    Args:
        dataset_path: Path al file JSON con le domande
        seed: Seed per riproducibilità (usato per random se necessario)
        llm_model: Nome del modello LLM da usare
        top_k: Numero di chunk da recuperare
        use_cited_only: Se True, considera solo chunk citati nella risposta
        template_id: ID del template di prompt (1-5)
        open_knowledge: Se True, permette all'LLM di usare conoscenza esterna
        verbose: Se True, stampa progress
    
    Returns:
        Lista di risultati per ogni domanda
    """
    # Set seed per riproducibilità
    random.seed(seed)
    np.random.seed(seed)
    
    # Carica risorse
    if verbose:
        print(f"\n{'='*60}")
        print(f"EVALUATION - Seed: {seed}, LLM: {llm_model}, Template: {template_id}, OpenKnowledge: {open_knowledge}")
        print(f"{'='*60}\n")
    
    embedding_model = load_embedding_model()
    qdrant_client = connect_to_qdrant()
    
    # Carica dataset
    dataset = load_benchmark_dataset(dataset_path)
    
    # Esegui valutazione
    results = []
    for i, q_data in enumerate(dataset):
        if verbose:
            print(f"\n[{i+1}/{len(dataset)}] Valutazione domanda ID={q_data['id']}: {q_data['question'][:50]}...")
        
        result = evaluate_single_question(
            q_data, 
            embedding_model, 
            qdrant_client, 
            top_k,
            use_cited_only,
            llm_model,
            template_id,
            open_knowledge
        )
        result['seed'] = seed
        result['llm'] = llm_model
        result['template_id'] = template_id
        result['open_knowledge'] = 'Yes' if open_knowledge else 'No'
        results.append(result)
        
        if verbose:
            print(f"      Source: {result['source_accuracy']}, Page: {result['page_accuracy']}, Similarity: {result['similarity']}")
    
    return results


# EXPORT CSV

def export_results_csv(
    results: List[Dict], 
    output_path: str,
    append: bool = True
) -> None:
    """
    Esporta i risultati in formato CSV.
    
    Formato output:
    seed, llm, template_id, question_id, source_accuracy, page_accuracy, similarity
    
    Aggiunge una riga vuota tra modelli diversi per separazione visiva.
    """
    fieldnames = ['seed', 'llm', 'template_id', 'question_id', 'source_accuracy', 'page_accuracy', 'similarity', 'open_knowledge']
    
    file_exists = os.path.isfile(output_path)
    mode = "a" if (file_exists and append) else "w"
    
    with open(output_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore', delimiter=';')
        
        # Scrivi header solo se file nuovo
        if not file_exists or not append:
            writer.writeheader()
        elif file_exists and append:
            # Aggiungi riga vuota prima di un nuovo blocco di risultati
            f.write('\n')
            
        for result in results:
            writer.writerow({
                'seed': result['seed'],
                'llm': result['llm'],
                'template_id': result['template_id'],
                'question_id': result['question_id'],
                'source_accuracy': result['source_accuracy'],
                'page_accuracy': result['page_accuracy'],
                'similarity': result['similarity'],
                'open_knowledge': result['open_knowledge']
            })
    
    print(f"\nRisultati salvati in: {output_path}")


def export_detailed_results_csv(
    results: List[Dict], 
    output_path: str
) -> None:
    """
    Esporta risultati dettagliati (include anche la risposta LLM).
    """
    fieldnames = [
        'seed', 'llm', 'question_id', 'question', 
        'source_accuracy', 'page_accuracy', 'similarity',
        'found_sources', 'found_pages', 'llm_response'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore', delimiter=';')
        writer.writeheader()
        
        for result in results:
            row = result.copy()
            row['found_sources'] = '; '.join(result['found_sources'])
            row['found_pages'] = json.dumps(result['found_pages'])
            writer.writerow(row)
    
    print(f"Risultati dettagliati salvati in: {output_path}")


# MAIN - CLI

def main():
    """
    Entry point per eseguire la valutazione da linea di comando.
    
    Usage:
        python evaluation.py benchmark_dataset.json [--seed 0] [--llm llama3.2] [--template-id 1] [--open-knowledge] [--output results.csv]
        
    Se --template-id non è specificato, esegue tutti i 5 template.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG Evaluation System')
    parser.add_argument('dataset', help='Path al file JSON con le domande benchmark')
    parser.add_argument('--seed', type=int, default=0, help='Seed per riproducibilità')
    parser.add_argument('--llm', type=str, default=OLLAMA_MODEL, help='Modello LLM da usare')
    parser.add_argument('--template-id', type=int, default=None, help='ID del template di prompt (1-5). Se non specificato, usa tutti.')
    parser.add_argument('--top-k', type=int, default=TOP_K, help='Numero di chunk da recuperare')
    parser.add_argument('--output', type=str, default='/app/evaluation_results/results.csv', help='Path per output CSV')
    parser.add_argument('--cited-only', action='store_true', help='Considera solo chunk citati')
    parser.add_argument('--detailed', action='store_true', help='Esporta risultati dettagliati')
    parser.add_argument('--overwrite', action='store_true', help='Sovrascrivi il file CSV invece di fare append')
    parser.add_argument('--open-knowledge', action='store_true', help='Permetti all\'LLM di usare conoscenza esterna')
    
    args = parser.parse_args()
    
    # Determina quali template usare
    if args.template_id is not None:
        # Template specifico
        templates_to_run = [args.template_id]
    else:
        # Tutti i 5 template
        templates_to_run = [1, 2, 3, 4, 5]
    
    all_results = []
    
    for template_id in templates_to_run:
        # Esegui valutazione per questo template
        results = run_evaluation(
            dataset_path=args.dataset,
            seed=args.seed,
            llm_model=args.llm,
            top_k=args.top_k,
            use_cited_only=args.cited_only,
            template_id=template_id,
            open_knowledge=args.open_knowledge
        )
        all_results.extend(results)
        
        # Output progressivo - crea la cartella se non esiste
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        if args.detailed:
            export_detailed_results_csv(results, args.output.replace('.csv', f'_template{template_id}.csv'))
        else:
            # Sempre append, a meno che non sia --overwrite E primo template
            is_first_template = (template_id == templates_to_run[0])
            should_overwrite = args.overwrite and is_first_template
            export_results_csv(results, args.output, append=not should_overwrite)
    
    # Stampa summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    avg_similarity = np.mean([r['similarity'] for r in all_results])
    total_src_correct = sum(r['source_correct'] for r in all_results)
    total_src_expected = sum(r['source_total'] for r in all_results)
    total_page_correct = sum(r['page_correct'] for r in all_results)
    total_page_expected = sum(r['page_total'] for r in all_results)
    
    print(f"Domande valutate: {len(all_results)}")
    print(f"Template usati: {templates_to_run}")
    print(f"Open Knowledge: {'Yes' if args.open_knowledge else 'No'}")
    print(f"Source accuracy totale: {total_src_correct}/{total_src_expected}")
    print(f"Page accuracy totale: {total_page_correct}/{total_page_expected}")
    print(f"Similarity media: {avg_similarity:.4f}")


if __name__ == "__main__":
    main()