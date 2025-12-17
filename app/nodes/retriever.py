"""Token-aware retrieval node that reserves budget for generation."""

from __future__ import annotations

from app.rag.vector_store import get_vector_store
from app.state import AgentState
from app.token_accounting import consume_tokens, estimate_tokens

# Reserve this much budget for the generator
MIN_GENERATION_BUDGET = 3000

# Approximate tokens per retrieved chunk
TOKENS_PER_CHUNK = 400


def retriever_node(state: AgentState) -> AgentState:
    """
    Retrieve relevant chunks from the vector store.

    Dynamically adjusts top_k based on remaining budget after reserving
    capacity for generation. This prevents RAG cost explosions.
    """
    remaining = state.get("remaining_tokens", 0)
    available_for_context = remaining - MIN_GENERATION_BUDGET

    if available_for_context <= 0:
        # No budget for retrieval; proceed with empty context
        state["retrieved_chunks"] = []
        return state

    # Compute how many chunks we can afford
    top_k = max(1, available_for_context // TOKENS_PER_CHUNK)
    # Cap at reasonable maximum
    top_k = min(top_k, 10)

    user_query = state.get("user_query", "")

    vector_store = get_vector_store()
    chunks = vector_store.similarity_search(query=user_query, k=top_k)

    state["retrieved_chunks"] = chunks

    # Estimate token cost of retrieved chunks
    chunk_text = "\n\n".join(chunks)
    estimated_cost = estimate_tokens(chunk_text) if chunks else 0

    state = consume_tokens(state, "retriever", estimated_cost)

    return state
