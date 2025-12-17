from typing import Dict, List, TypedDict


class AgentState(TypedDict, total=False):
    """
    Core agent state for token-gated LangGraph execution.
    """

    # Input
    user_query: str

    # Artifacts
    plan: str
    retrieved_chunks: List[str]
    draft_answer: str
    final_answer: str

    # Token gating
    total_token_budget: int
    remaining_tokens: int
    tokens_used: Dict[str, int]

    # Control
    step_count: int
    max_steps: int
    quality_score: float
    status: str


def initialize_state(
    user_query: str,
    total_token_budget: int,
    max_steps: int,
) -> AgentState:
    """
    Initialize a fresh AgentState for a new request.
    """
    return AgentState(
        user_query=user_query,
        plan="",
        retrieved_chunks=[],
        draft_answer="",
        final_answer="",
        total_token_budget=total_token_budget,
        remaining_tokens=total_token_budget,
        tokens_used={},
        step_count=0,
        max_steps=max_steps,
        quality_score=0.0,
        status="INIT",
    )
