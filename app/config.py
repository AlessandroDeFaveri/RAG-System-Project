import os

# Paths
DATA_PATH = os.getenv("DATA_PATH", "./data")

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = 6333
COLLECTION_NAME = "documenti_tesi"

# LLM - Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = 11434
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# LLM - Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2026-01-01-preview")

# RAG Parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
