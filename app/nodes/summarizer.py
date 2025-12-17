"""Safety exit node that compresses output when budget is exhausted."""

from __future__ import annotations

from openai import OpenAI

from app.config import get_settings
from app.state import AgentState
from app.token_accounting import consume_tokens, estimate_tokens

SUMMARIZER_BUDGET = 400

SUMMARIZER_SYSTEM_PROMPT = """You are a summarization assistant.
Given a draft answer, produce a concise summary that captures the key points.
Keep your summary under 100 words."""


def summarizer_node(state: AgentState) -> AgentState:
    """
    Produce a compressed final answer when budget is depleted.

    This is the safety exit - it ensures the system always produces
    some output rather than failing silently or crashing.
    """
    draft = state.get("draft_answer", "")

    # If we have a draft and some budget, summarize it
    if draft and state.get("remaining_tokens", 0) >= SUMMARIZER_BUDGET:
        settings = get_settings()
        client = OpenAI(api_key=settings.openai_api_key)

        response = client.chat.completions.create(
            model=settings.openai_model_name,
            messages=[
                {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Summarize this answer:\n\n{draft}"},
            ],
            max_tokens=150,
            temperature=0.3,
        )

        summary = response.choices[0].message.content or draft
        state["final_answer"] = summary

        # Calculate actual token usage
        usage = response.usage
        if usage:
            total_tokens = usage.prompt_tokens + usage.completion_tokens
        else:
            total_tokens = estimate_tokens(SUMMARIZER_SYSTEM_PROMPT + draft + summary)

        state = consume_tokens(state, "summarizer", total_tokens)
    elif draft:
        # No budget for summarization, use draft as-is
        state["final_answer"] = draft
    else:
        # No draft at all
        state["final_answer"] = (
            "Unable to generate an answer due to budget constraints."
        )

    state["status"] = "COMPLETED_WITH_SUMMARY"

    return state
