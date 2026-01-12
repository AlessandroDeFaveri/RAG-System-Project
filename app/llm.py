"""
Step 3: LLM - Interazione con Ollama
"""
import requests
import json
from config import OLLAMA_HOST, OLLAMA_PORT, OLLAMA_MODEL


# Dizionario dei template di prompt (5 versioni diverse)

PROMPT_TEMPLATES = {
    # Dettagliato

    1: """You are an academic Research Assistant.
Your goal is to answer the user's question using ONLY the provided context chunks.

CONTEXT STRUCTURE:
The context consists of chunks, each labeled with a specific ID (e.g., "--- REF ID: [1] ---").
Each chunk also contains metadata: (Source: filename, Page: number).

INSTRUCTIONS:
1.  Answer: Synthesize the information to answer the question. Answer in the same language as the question.
2.  Grounding: Use ONLY information from the chunks. Do NOT use external knowledge.
3.  Inline Citations: You MUST cite the ID immediately after using information, using square brackets: [1], [2].
    - Example: "OLAP is a multidimensional analysis tool [1]."
4.  References Section: At the very end, create a section titled "References".
    - CRITICAL: The IDs in References MUST MATCH EXACTLY the IDs you used in the text.
    - If you cited [1] and [3] in your answer, list ONLY [1] and [3] in References. Do NOT list [2] if you didn't cite it.
    - Format: [ID] Source: <filename>, Page: <page>

OUTPUT FORMAT:
<Your answer text with inline citations like [1], [3]...>

References:
[1] Source: <filename>, Page: <page>
[3] Source: <filename>, Page: <page>
""",

    # Minimale
    2: """You are a Research Assistant. Answer the question using ONLY the provided context.
Cite sources using [1], [2], etc. If you cannot find the answer, say so.
""",

    # Ragionaemento passo passo
    3: """You are an academic Research Assistant.
First, analyze which chunks are relevant to the question.
Then, reason step-by-step about the answer.
Finally, provide a clear answer with citations [1], [2].

Think through this systematically before answering.
""",
    # Regole strette
    4: """You are a Research Assistant with STRICT rules:
- Use ONLY information from the provided chunks
- If the answer is NOT in the chunks, respond: "I cannot answer based on the provided documents."
- NEVER use external knowledge
- Every claim MUST have a citation [1], [2]

If unsure, say you don't know rather than guessing.
""",
    # Semplice e diretto
    5: """You are a helpful research assistant. A user is asking about academic papers.

Look through the provided context chunks and answer their question naturally.
When you use information from a chunk, mention the source like [1] or [2].

Be helpful but honest - if you can't find the answer in the chunks, just say so.
"""
}

# Template di default (per retrocompatibilità)
DEFAULT_TEMPLATE_ID = 1
SYSTEM_PROMPT = PROMPT_TEMPLATES[DEFAULT_TEMPLATE_ID]


# FUNZIONI

# Suffisso da aggiungere quando open_knowledge=True
OPEN_KNOWLEDGE_SUFFIX = """

ADDITIONAL INSTRUCTION:
You may use your general knowledge to complement the information from the chunks if it helps provide a more complete answer.
However, clearly distinguish between information from the chunks (cite with [1], [2]) and your own knowledge (no citation needed).
"""


def get_prompt_template(template_id: int = DEFAULT_TEMPLATE_ID, open_knowledge: bool = False) -> str:
    """
    Restituisce il template di prompt specificato.
    
    Args:
        template_id: ID del template (1-5)
        open_knowledge: Se True, permette all'LLM di usare conoscenza esterna
    
    Returns:
        Il testo del template (con eventuale suffisso open_knowledge)
    """
    if template_id not in PROMPT_TEMPLATES:
        print(f"Warning: Template {template_id} non trovato, uso default ({DEFAULT_TEMPLATE_ID})")
        template = PROMPT_TEMPLATES[DEFAULT_TEMPLATE_ID]
    else:
        template = PROMPT_TEMPLATES[template_id]
    
    # Aggiungi suffisso se open_knowledge è attivo
    if open_knowledge:
        template = template + OPEN_KNOWLEDGE_SUFFIX
    
    return template


def build_prompt(context: str, question: str, template_id: int = DEFAULT_TEMPLATE_ID, open_knowledge: bool = False) -> str:
    """
    Costruisce il prompt completo con contesto e domanda.
    
    Args:
        context: Il contesto con i chunk
        question: La domanda dell'utente
        template_id: ID del template da usare (1-5)
        open_knowledge: Se True, permette all'LLM di usare conoscenza esterna
    
    Formato:
    - System prompt (dal template selezionato)
    - Context con i chunk
    - Question dell'utente
    """
    system_prompt = get_prompt_template(template_id, open_knowledge)
    
    prompt = f"""{system_prompt}

Context:
{context}

Question: {question}

Answer:"""
    
    return prompt


def query_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Invia una query a Ollama e restituisce la risposta.
    
    Args:
        prompt: Il prompt completo (system + context + question)
        model: Il modello da usare
    
    Returns:
        La risposta del modello
    """
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False, 
        "options": {
            "temperature": 0.6,  
            "num_ctx": 4096      # Aumentiamo la finestra di contesto per leggere più chunk
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "Errore: nessuna risposta dal modello")
        
    except requests.exceptions.ConnectionError:
        return "Errore: Impossibile connettersi a Ollama. È in esecuzione?"
    except requests.exceptions.Timeout:
        return "Errore: Timeout nella risposta di Ollama"
    except Exception as e:
        return f"Errore: {str(e)}"


def query_ollama_streaming(prompt: str, model: str = OLLAMA_MODEL):
    """
    Invia una query a Ollama con risposta in streaming.
    Stampa la risposta man mano che arriva.
    """
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.6,  # Deterministic for reproducibility
            "num_ctx": 4096      # Context window
        }
    }
    
    try:
        response = requests.post(url, json=payload, stream=True, timeout=500)
        response.raise_for_status()
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("response", "")
                print(token, end="", flush=True)
                full_response += token
                
                if data.get("done", False):
                    print()  # Newline finale
                    break
        
        return full_response
        
    except Exception as e:
        print(f"\nErrore: {str(e)}")
        return ""


# TEST

if __name__ == "__main__":
    print("--- TEST OLLAMA ---\n")
    
    # Test semplice
    test_prompt = "What is 2 + 4? Answer in one word."
    
    print(f"Prompt: {test_prompt}")
    print("Risposta: ", end="")
    
    response = query_ollama(test_prompt)
    print(response)
