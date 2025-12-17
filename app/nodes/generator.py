"""Budgeted generation node that produces answers from retrieved context."""

from __future__ import annotations

from openai import OpenAI

from app.config import get_settings
from app.state import AgentState
from app.token_accounting import consume_tokens, estimate_tokens

GENERATOR_REQUIRED_BUDGET = 2500

GENERATOR_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use the context to give accurate, well-structured answers.
If the context doesn't contain relevant information, say so clearly.
Be concise but thorough."""


def generator_node(state: AgentState) -> AgentState:
    """
    Generate an answer using retrieved context.

    Budget requirement: 2500 tokens.
    If budget is insufficient, sets status and returns early.
    """
    if state.get("remaining_tokens", 0) < GENERATOR_REQUIRED_BUDGET:
        state["status"] = "INSUFFICIENT_BUDGET_FOR_GENERATION"
        return state

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    user_query = state.get("user_query", "")
    chunks = state.get("retrieved_chunks", [])
    plan = state.get("plan", "")

    # Build context from retrieved chunks
    if chunks:
        context = "\n\n---\n\n".join(chunks)
        context_section = f"## Retrieved Context\n\n{context}"
    else:
        context_section = "No context was retrieved. Answer based on general knowledge."

    user_message = f"""## Plan
{plan}

{context_section}

## Question
{user_query}

Please provide a comprehensive answer based on the above context and plan."""

    response = client.chat.completions.create(
        model=settings.openai_model_name,
        messages=[
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=1000,
        temperature=0.5,
    )

    answer_text = response.choices[0].message.content or ""
    state["draft_answer"] = answer_text

    # Calculate actual token usage
    usage = response.usage
    if usage:
        total_tokens = usage.prompt_tokens + usage.completion_tokens
    else:
        total_tokens = estimate_tokens(
            GENERATOR_SYSTEM_PROMPT + user_message + answer_text
        )

    state = consume_tokens(state, "generator", total_tokens)

    return state
