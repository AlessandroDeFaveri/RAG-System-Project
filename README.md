# RAG System for Academic Papers

RAG (Retrieval-Augmented Generation) system for analyzing academic papers with source citations

## Quick Start

### Running with Docker Compose (Recommended)

```bash
# Start the entire system
docker compose up

# Or run in background
docker compose up -d
```

This command:
1. Starts **Qdrant** (vector database)
2. Starts **Ollama** (LLM server)
3. Downloads the LLM model if needed
4. Runs PDF ingestion (if present in `./data/`)
5. Runs evaluation with the benchmark dataset

### Execution Modes

```bash
# Single model evaluation (default)
docker compose run rag-app

# Multi-model evaluation
docker compose run rag-app full-evaluation

# Interactive mode (chat)
docker compose run rag-app interactive

# Shell for debugging
docker compose run rag-app shell
```

## Commands and Parameters

### `evaluation` Command (single evaluation)

```bash
docker compose run rag-app evaluation <dataset> [options]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `<dataset>` | (required) | Path to JSON file with questions (e.g., `/app/benchmark_dataset.json`) |
| `--llm <model>` | `llama3.2` | LLM model to use (Ollama: `qwen2.5:7b`, `gemma2:2b` / Azure: `gpt-5-mini`, `gpt-4o`) |
| `--template-id <1-5>` | All (1-5) | Prompt template ID. If not specified, runs all 5 templates |
| `--open-knowledge` | No | If present, allows the LLM to use external knowledge |
| `--seed <n>` | `0` | Seed for reproducibility |
| `--top-k <n>` | `5` | Number of chunks to retrieve |
| `--output <path>` | `/app/evaluation_results/results.csv` | Output CSV file path |
| `--overwrite` | No | Overwrites CSV instead of appending |
| `--cited-only` | No | Considers only chunks actually cited in the response |
| `--detailed` | No | Exports detailed results (includes LLM response) |

**Examples:**

```bash
# Basic evaluation with llama3.2, all templates, open_knowledge=No
docker compose run rag-app evaluation /app/benchmark_dataset.json --llm llama3.2

# Only template 1 with gemma2
docker compose run rag-app evaluation /app/benchmark_dataset.json --llm gemma2:2b --template-id 1

# With open knowledge enabled
docker compose run rag-app evaluation /app/benchmark_dataset.json --llm llama3.2 --open-knowledge

# Overwrite existing CSV
docker compose run rag-app evaluation /app/benchmark_dataset.json --llm llama3.2 --overwrite

# Detailed output with LLM response
docker compose run rag-app evaluation /app/benchmark_dataset.json --llm llama3.2 --detailed
```

### `full-evaluation` Command (complete evaluation)

Automatically runs all combinations of models, templates, and open_knowledge:

```bash
docker compose run rag-app full-evaluation
```

Current configuration:
- **Ollama Models (local)**: llama3.2, qwen2.5:7b
- **Azure OpenAI Models (cloud)**: gpt-5-mini (requires `AZURE_OPENAI_KEY`)
- **Templates**: 1, 2, 3, 4, 5
- **Open Knowledge**: No and Yes

**Total combinations**: 3 models × 5 templates × 2 open_knowledge = **30 runs** per execution

> If `AZURE_OPENAI_KEY` is not configured, Azure models are automatically skipped.

### `interactive` Command (chat)

Starts an interactive session to ask questions to the system:

```bash
docker compose run rag-app interactive
```

## Project Structure

```
RAG System Project/
├── app/
│   ├── config.py          # Global configuration
│   ├── embeddings.py      # Embeddings management
│   ├── ingestion.py       # PDF indexing
│   ├── retrieval.py       # Semantic search
│   ├── llm.py             # Ollama and Azure OpenAI interface
│   ├── main.py            # Interactive CLI
│   └── evaluation.py      # Evaluation system
├── data/                   # Folder for PDFs to index
├── evaluation_results/     # CSV output from evaluations
├── scripts/
│   ├── entrypoint.sh      # Docker startup script
│   └── run_full_evaluation.sh  # Multi-model evaluation
├── benchmark_dataset.json  # Ground truth
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Evaluation Output

Results are saved in two CSV files:

### `results.csv` (always generated)
Contains basic metrics:

| seed | llm | template_id | question_id | source_accuracy | page_accuracy | similarity | open_knowledge |
|------|-----|-------------|-------------|-----------------|---------------|------------|----------------|
| 0 | llama3.2 | 1 | 1 | 0.75 | 0.66 | 0.85 | No |

### `results_detailed.csv` (with `--detailed` flag)
Also contains the complete LLM response for debugging:

| seed | llm | template_id | question_id | question | source_accuracy | page_accuracy | similarity | open_knowledge | found_sources | found_pages | llm_response |
|------|-----|-------------|-------------|----------|-----------------|---------------|------------|----------------|---------------|-------------|--------------|

> **Note**: Results are always **appended** to existing CSVs. Use `--overwrite` to overwrite.

## Prompt Templates

The system supports 5 different prompt templates, selectable with `--template-id`:

| ID | Name | Description |
|----|------|-------------|
| 1 | Detailed | Precise instructions on citations and format (default) |
| 2 | Minimal | Very few instructions, short prompt |
| 3 | Chain-of-Thought | Step-by-step reasoning before answering |
| 4 | Strict | Anti-hallucination, very rigid rules |
| 5 | Conversational | Natural and informal tone |

```bash
# Use minimal template
python evaluation.py benchmark_dataset.json --template-id 2

# Use chain-of-thought template
python evaluation.py benchmark_dataset.json --template-id 3
```

## Open Knowledge Mode

The `--open-knowledge` flag controls whether the LLM can use external knowledge:

| Mode | Behavior |
|------|----------|
| **Without flag** (default) | LLM uses ONLY information from retrieved papers |
| **With `--open-knowledge`** | LLM can integrate with its general knowledge |

```bash
# Closed mode (papers only)
python evaluation.py benchmark_dataset.json --llm llama3.2

# Open mode (can use general knowledge)
python evaluation.py benchmark_dataset.json --llm llama3.2 --open-knowledge
```

**When to use it?**
- Without flag: To test the RAG's "faithfulness" to documents
- With flag: To see if the LLM can provide more complete answers

## Combining Parameters

```bash
# Complete example
python evaluation.py benchmark_dataset.json \
    --seed 42 \
    --llm qwen2.5:7b \
    --template-id 3 \
    --open-knowledge \
    --output results.csv \
    --detailed
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | llama3.2 | LLM model to use |
| `QDRANT_HOST` | qdrant | Qdrant database host |
| `OLLAMA_HOST` | ollama | Ollama server host |
| `DATA_PATH` | ./data | PDF folder path |
| `AZURE_OPENAI_ENDPOINT` | - | Azure OpenAI endpoint |
| `AZURE_OPENAI_KEY` | - | Azure OpenAI API Key |

### Azure OpenAI Configuration

To use Azure models (GPT-4, GPT-5), create a `.env` file in the project root:

```bash
# Copy the template
cp .env.example .env

# Edit with your key
```

`.env` file content:
```
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_KEY=your-api-key
```

Then run with an Azure model:
```bash
docker compose run rag-app evaluation /app/benchmark_dataset.json --llm gpt-5-mini --detailed
```

### Using a different model

```bash
LLM_MODEL=phi3:mini docker compose up
```

## Local Development (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start Qdrant and Ollama separately
docker run -p 6333:6333 qdrant/qdrant
docker run -p 11434:11434 ollama/ollama

# Run ingestion
cd app
python ingestion.py

# Run evaluation
python evaluation.py ../benchmark_dataset.json --seed 0 --llm llama3.2 --template-id 1 --output ../results.csv

# With open knowledge enabled
python evaluation.py ../benchmark_dataset.json --seed 0 --llm llama3.2 --open-knowledge --output ../results.csv

# Interactive mode
python main.py
```

## Full Evaluation with Docker

The `full-evaluation` script automatically tests all combinations:
- **Ollama Models**: llama3.2, qwen2.5:7b
- **Azure Models**: gpt-5-mini (if configured)
- **5 templates**: from 1 to 5
- **2 modes**: with and without open-knowledge

Total: **30 combinations** per run (with Azure configured).

```bash
# Test ALL combinations (No + Yes for open_knowledge)
docker compose run --rm rag-app full-evaluation
```

> **Note:** No need to pass `--open-knowledge` to `full-evaluation` because the script automatically tests both closed (No) and open (Yes) modes.

### Single evaluation with specific parameters

If you want to test a specific configuration:

```bash
# Closed mode only (open_knowledge=No)
docker compose run --rm rag-app evaluation /app/benchmark_dataset.json --llm llama3.2 --template-id 1

# Open mode only (open_knowledge=Yes)  
docker compose run --rm rag-app evaluation /app/benchmark_dataset.json --llm llama3.2 --template-id 1 --open-knowledge
```

## Benchmark Dataset

The `benchmark_dataset.json` file contains test questions with:
- `question`: Question to ask the system
- `expected_answer`: Expected answer
- `expected_sources`: PDF files that should be cited
- `expected_pages`: Specific expected pages

## Metrics

- **Source Accuracy**: % of expected sources retrieved by the retriever
- **Page Accuracy**: % of expected pages retrieved  
- **Similarity**: Cosine similarity between generated answer and expected answer (embedding)
