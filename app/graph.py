"""LangGraph assembly with budget-driven conditional routing."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from app.nodes import (
    critic_node,
    generator_node,
    planner_node,
    retriever_node,
    summarizer_node,
)
from app.state import AgentState

# Budget threshold for forced summarization
BUDGET_THRESHOLD = 500

# Quality threshold for completion
QUALITY_THRESHOLD = 0.85


def should_continue(state: AgentState) -> Literal["end", "loop", "summarize"]:
    """
    Budget-driven control flow router.

    Determines next action based on:
    - Remaining token budget
    - Quality score from critic
    - Step count vs max steps

    Returns:
        "end": Quality threshold met, finalize with draft answer
        "summarize": Budget depleted or max steps reached, compress and exit
        "loop": Retry planning for better result
    """
    remaining = state.get("remaining_tokens", 0)
    quality = state.get("quality_score", 0.0)
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 5)
    status = state.get("status", "")

    # Check for insufficient budget errors from nodes
    if status.startswith("INSUFFICIENT_BUDGET"):
        return "summarize"

    # Budget exhausted - force summarization
    if remaining <= BUDGET_THRESHOLD:
        return "summarize"

    # Quality threshold met - we're done
    if quality >= QUALITY_THRESHOLD:
        return "end"

    # Max steps reached - summarize what we have
    if step_count >= max_steps:
        return "summarize"

    # Otherwise, loop back and try again
    return "loop"


def finalize_node(state: AgentState) -> AgentState:
    """
    Finalization node - promotes draft to final answer.
    Called when quality threshold is met.
    """
    state["final_answer"] = state.get("draft_answer", "")
    state["status"] = "COMPLETED"
    return state


def build_graph() -> StateGraph:
    """
    Construct the token-gated agent graph.

    Flow:
        planner -> retriever -> generator -> critic
        critic -> [end | loop | summarize]

        end: finalize -> END
        loop: planner (retry)
        summarize: summarizer -> END
    """
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("generator", generator_node)
    graph.add_node("critic", critic_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("finalize", finalize_node)

    # Set entry point
    graph.set_entry_point("planner")

    # Linear flow: planner -> retriever -> generator -> critic
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "critic")

    # Conditional routing from critic
    graph.add_conditional_edges(
        "critic",
        should_continue,
        {
            "end": "finalize",
            "loop": "planner",
            "summarize": "summarizer",
        },
    )

    # Terminal edges
    graph.add_edge("finalize", END)
    graph.add_edge("summarizer", END)

    return graph


def compile_graph():
    """
    Build and compile the graph for execution.
    Returns a compiled LangGraph runnable.
    """
    graph = build_graph()
    return graph.compile()


# Singleton compiled graph instance
_COMPILED_GRAPH = None


def get_compiled_graph():
    """Get or create the compiled graph singleton."""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = compile_graph()
    return _COMPILED_GRAPH
