"""Optional criticism node for quality assessment."""

from __future__ import annotations

import json

from openai import OpenAI

from app.config import get_settings
from app.state import AgentState
from app.token_accounting import consume_tokens, estimate_tokens

CRITIC_REQUIRED_BUDGET = 800

CRITIC_SYSTEM_PROMPT = """You are a quality evaluator for AI-generated answers.
Evaluate the answer on these criteria:
1. Relevance to the question
2. Accuracy based on provided context
3. Completeness
4. Clarity

Respond with a JSON object containing:
- "score": a number between 0.0 and 1.0
- "feedback": a brief explanation (1-2 sentences)

Example response:
{"score": 0.85, "feedback": "Good coverage of the topic with clear explanations."}"""


def critic_node(state: AgentState) -> AgentState:
    """
    Evaluate the quality of the draft answer.

    Budget requirement: 800 tokens.
    If budget is insufficient, assigns a default score and skips evaluation.
    This is graceful degradation - the system continues without criticism.
    """
    if state.get("remaining_tokens", 0) < CRITIC_REQUIRED_BUDGET:
        # Graceful degradation: skip criticism, use default score
        state["quality_score"] = 0.7
        return state

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    user_query = state.get("user_query", "")
    draft_answer = state.get("draft_answer", "")
    chunks = state.get("retrieved_chunks", [])

    context_summary = f"{len(chunks)} chunks retrieved" if chunks else "No context"

    user_message = f"""## Original Question
{user_query}

## Context Available
{context_summary}

## Generated Answer
{draft_answer}

Please evaluate this answer and provide a quality score."""

    response = client.chat.completions.create(
        model=settings.openai_model_name,
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=150,
        temperature=0.2,
    )

    critic_response = response.choices[0].message.content or ""

    # Parse the score from JSON response
    try:
        parsed = json.loads(critic_response)
        score = float(parsed.get("score", 0.7))
        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
    except (json.JSONDecodeError, ValueError, TypeError):
        # Fallback if parsing fails
        score = 0.7

    state["quality_score"] = score

    # Calculate actual token usage
    usage = response.usage
    if usage:
        total_tokens = usage.prompt_tokens + usage.completion_tokens
    else:
        total_tokens = estimate_tokens(
            CRITIC_SYSTEM_PROMPT + user_message + critic_response
        )

    state = consume_tokens(state, "critic", total_tokens)

    return state
