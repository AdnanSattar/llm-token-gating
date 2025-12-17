# LLM Token Gating with LangGraph

A FastAPI service implementing **token-gated LLM execution** using LangGraph. This system enforces predictable cost envelopes across planning, retrieval, generation, and quality assessment phases.

## Architecture

```
User Request
     │
     ▼
┌─────────────────┐
│  Budget Manager │  ← Token budget as first-class state
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────┐
│    Planner      │────▶│  Retriever  │
│  (800 tokens)   │     │ (dynamic k) │
└─────────────────┘     └──────┬──────┘
                               │
                               ▼
                        ┌─────────────┐
                        │  Generator  │
                        │(2500 tokens)│
                        └──────┬──────┘
                               │
                               ▼
                        ┌─────────────┐
                        │   Critic    │
                        │(800 tokens) │
                        └──────┬──────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
         [quality ≥ 0.85]  [budget low]    [loop]
              │                │                │
              ▼                ▼                ▼
           Finalize       Summarizer        Planner
              │                │            (retry)
              ▼                ▼
             END              END
```

## Features

- **Token Budget Enforcement**: Every node checks and consumes tokens from a shared budget
- **Dynamic RAG**: Retriever adjusts `top_k` based on remaining budget after reserving generation capacity
- **Graceful Degradation**: System summarizes and exits cleanly when budget is exhausted
- **Quality-Driven Termination**: Exits early when quality threshold is met
- **Loop Prevention**: Maximum step count prevents infinite agent loops
- **Full Observability**: Token usage breakdown by node in every response

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

Required environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key

Optional:

- `OPENAI_MODEL_NAME`: Model for chat completions (default: `gpt-4o`)
- `OPENAI_EMBEDDING_MODEL`: Model for embeddings (default: `text-embedding-3-small`)
- `DEFAULT_TOKEN_BUDGET`: Default budget per request (default: `10000`)
- `DEFAULT_MAX_STEPS`: Maximum agent loop iterations (default: `5`)
- `CHROMA_PERSIST_DIR`: ChromaDB storage path (default: `./chroma_db`)

### 3. Run the Server

```bash
uvicorn app.main:app --reload
```

Or:

```bash
python -m app.main
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### `GET /health`

Health check endpoint.

```bash
curl http://localhost:8000/health
```

### `POST /documents`

Ingest documents into the vector store for RAG retrieval.

```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
      "Token gating enforces budget constraints across LLM execution.",
      "RAG combines retrieval with generation for grounded responses."
    ]
  }'
```

### `POST /query`

Execute a token-gated query.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is token gating and why is it important?",
    "token_budget": 8000,
    "max_steps": 3
  }'
```

**Response:**

```json
{
  "answer": "Token gating is a budget enforcement layer...",
  "status": "COMPLETED",
  "tokens_used": {
    "planner": 450,
    "retriever": 800,
    "generator": 1200,
    "critic": 350
  },
  "total_tokens": 2800,
  "steps_executed": 1
}
```

**Status Values:**

- `COMPLETED`: Quality threshold met, full answer generated
- `COMPLETED_WITH_SUMMARY`: Budget exhausted, summarized answer
- `INSUFFICIENT_BUDGET_FOR_PLANNING`: Not enough budget to start
- `INSUFFICIENT_BUDGET_FOR_GENERATION`: Budget depleted before generation

## Project Structure

```
llm-token-gating/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app and endpoints
│   ├── config.py            # Pydantic settings
│   ├── state.py             # AgentState TypedDict
│   ├── token_accounting.py  # Centralized token tracking
│   ├── graph.py             # LangGraph assembly
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── planner.py       # Bounded planning
│   │   ├── retriever.py     # Token-aware RAG
│   │   ├── generator.py     # Budgeted generation
│   │   ├── critic.py        # Optional quality check
│   │   └── summarizer.py    # Safety exit
│   └── rag/
│       ├── __init__.py
│       ├── embeddings.py    # OpenAI embeddings
│       └── vector_store.py  # ChromaDB integration
├── requirements.txt
├── .env.example
└── README.md
```

## How Token Gating Works

### 1. Budget as State

Every request initializes with a token budget:

```python
state = initialize_state(
    user_query="...",
    total_token_budget=10000,
    max_steps=5,
)
```

### 2. Pre-flight Budget Checks

Each node checks budget before executing:

```python
if state["remaining_tokens"] < REQUIRED_BUDGET:
    state["status"] = "INSUFFICIENT_BUDGET"
    return state
```

### 3. Centralized Consumption

All token usage flows through one function:

```python
state = consume_tokens(state, "generator", actual_tokens_used)
```

### 4. Dynamic Retrieval

The retriever adjusts `top_k` based on remaining budget:

```python
available_for_context = remaining_tokens - MIN_GENERATION_BUDGET
top_k = max(1, available_for_context // TOKENS_PER_CHUNK)
```

### 5. Conditional Routing

The graph routes based on budget and quality:

```python
def should_continue(state):
    if state["remaining_tokens"] <= 500:
        return "summarize"
    if state["quality_score"] >= 0.85:
        return "end"
    return "loop"
```

## License

MIT
