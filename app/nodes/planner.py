"""Bounded planning node with budget enforcement."""

from __future__ import annotations

from openai import OpenAI

from app.config import get_settings
from app.state import AgentState
from app.token_accounting import consume_tokens, estimate_tokens

PLANNER_REQUIRED_BUDGET = 800

PLANNER_SYSTEM_PROMPT = """You are a concise planning assistant.
Given a user query, produce a brief execution plan (2-4 steps) for answering it.
Keep your plan under 150 words. Be direct and actionable."""


def planner_node(state: AgentState) -> AgentState:
    """
    Generate an execution plan for the user query.

    Budget requirement: 800 tokens.
    If budget is insufficient, sets status and returns early.
    """
    if state.get("remaining_tokens", 0) < PLANNER_REQUIRED_BUDGET:
        state["status"] = "INSUFFICIENT_BUDGET_FOR_PLANNING"
        return state

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    user_query = state.get("user_query", "")

    response = client.chat.completions.create(
        model=settings.openai_model_name,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Create a plan to answer: {user_query}"},
        ],
        max_tokens=300,
        temperature=0.3,
    )

    plan_text = response.choices[0].message.content or ""
    state["plan"] = plan_text

    # Calculate actual token usage from response
    usage = response.usage
    if usage:
        total_tokens = usage.prompt_tokens + usage.completion_tokens
    else:
        total_tokens = estimate_tokens(PLANNER_SYSTEM_PROMPT + user_query + plan_text)

    state = consume_tokens(state, "planner", total_tokens)
    state["step_count"] = state.get("step_count", 0) + 1

    return state
