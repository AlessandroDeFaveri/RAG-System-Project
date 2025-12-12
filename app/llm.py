"""
Step 3: LLM - Interazione con Ollama
"""
import requests
import json
from config import OLLAMA_HOST, OLLAMA_PORT, OLLAMA_MODEL


# Prompt di sistema come richiesto dal prof
SYSTEM_PROMPT = """You are an expert assistant. 
Answer the question using ONLY the context provided.
Only use provided sources, do not guess.
Cite the exact passage from the context.
If the answer is not in the context, say "I don't know."
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
        "stream": False 
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
        "stream": True
    }
    
    try:
        response = requests.post(url, json=payload, stream=True, timeout=120)
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
