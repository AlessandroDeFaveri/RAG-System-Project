"""
Step 3: LLM - Interazione con Ollama
"""
import requests
import json
from config import OLLAMA_HOST, OLLAMA_PORT, OLLAMA_MODEL


# Prompt di sistema per il modello LLM

SYSTEM_PROMPT = """You are an academic Research Assistant.
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
"""
# FUNZIONI

def build_prompt(context: str, question: str) -> str:
    """
    Costruisce il prompt completo con contesto e domanda.
    
    Formato richiesto dal prof:
    - System prompt
    - Context con i chunk
    - Question dell'utente
    """
    prompt = f"""{SYSTEM_PROMPT}

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
            "temperature": 0.0,  
            "num_ctx": 4096      # Aumentiamo la finestra di contesto per leggere più chunk
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
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
            "temperature": 0.0,  # Deterministic for reproducibility
            "num_ctx": 4096      # Context window
        }
    }
    
    try:
        response = requests.post(url, json=payload, stream=True, timeout=300)
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
