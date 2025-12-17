"""FastAPI service for token-gated LLM execution."""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph import get_compiled_graph
from app.rag.vector_store import get_vector_store
from app.state import initialize_state

# =============================================================================
# Request/Response Models
# =============================================================================


class QueryRequest(BaseModel):
    """Request model for the /query endpoint."""

    query: str = Field(..., description="The user query to answer")
    token_budget: Optional[int] = Field(
        None, description="Token budget override (default from config)"
    )
    max_steps: Optional[int] = Field(
        None, description="Maximum agent loop iterations (default from config)"
    )


class QueryResponse(BaseModel):
    """Response model for the /query endpoint."""

    answer: str = Field(..., description="The generated answer")
    status: str = Field(
        ...,
        description="Execution status (COMPLETED, COMPLETED_WITH_SUMMARY, INSUFFICIENT_BUDGET_*)",
    )
    tokens_used: Dict[str, int] = Field(
        ..., description="Token usage breakdown by node"
    )
    total_tokens: int = Field(..., description="Total tokens consumed")
    steps_executed: int = Field(..., description="Number of agent loop iterations")


class DocumentsRequest(BaseModel):
    """Request model for the /documents endpoint."""

    texts: List[str] = Field(..., description="List of text chunks to ingest")
    metadatas: Optional[List[Dict]] = Field(
        None, description="Optional metadata for each chunk"
    )


class DocumentsResponse(BaseModel):
    """Response model for the /documents endpoint."""

    message: str = Field(..., description="Status message")
    count: int = Field(..., description="Number of documents ingested")


class HealthResponse(BaseModel):
    """Response model for the /health endpoint."""

    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="API version")


# =============================================================================
# FastAPI Application
# =============================================================================


app = FastAPI(
    title="Token-Gated LLM API",
    description="LLM execution with budget enforcement using LangGraph",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Execute a token-gated query.

    The system will:
    1. Plan the execution
    2. Retrieve relevant context from the vector store
    3. Generate an answer
    4. Optionally critique and iterate
    5. Summarize if budget is exhausted

    All steps are bounded by the token budget.
    """
    settings = get_settings()

    # Apply defaults
    token_budget = request.token_budget or settings.default_token_budget
    max_steps = request.max_steps or settings.default_max_steps

    # Initialize state
    initial_state = initialize_state(
        user_query=request.query,
        total_token_budget=token_budget,
        max_steps=max_steps,
    )

    # Execute the graph
    try:
        graph = get_compiled_graph()
        final_state = graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")

    # Extract results
    answer = final_state.get("final_answer") or final_state.get("draft_answer") or ""
    status = final_state.get("status", "UNKNOWN")
    tokens_used = final_state.get("tokens_used", {})
    total_tokens = sum(tokens_used.values())
    steps_executed = final_state.get("step_count", 0)

    return QueryResponse(
        answer=answer,
        status=status,
        tokens_used=tokens_used,
        total_tokens=total_tokens,
        steps_executed=steps_executed,
    )


@app.post("/documents", response_model=DocumentsResponse)
async def ingest_documents(request: DocumentsRequest):
    """
    Ingest documents into the vector store.

    Documents are embedded using OpenAI embeddings and stored in ChromaDB
    for later retrieval during query execution.
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    try:
        vector_store = get_vector_store()
        vector_store.add_texts(
            texts=request.texts,
            metadatas=request.metadatas,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")

    return DocumentsResponse(
        message="Documents ingested successfully",
        count=len(request.texts),
    )


# =============================================================================
# Entry point for uvicorn
# =============================================================================


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
