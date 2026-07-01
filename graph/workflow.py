import logging
import time
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from agents.executor import run_executor
from agents.planner import run_planner
from agents.researcher import run_researcher
from agents.validator import run_validator
from api.sse import push_event
from graph.state import OrchestratorState
from memory.postgres_client import upsert_task_history
from memory.redis_client import set_session_state

logger = logging.getLogger(__name__)

AGENT_ORDER = ["researcher", "planner", "executor", "validator"]


def _agent_output(state: OrchestratorState, agent: str) -> dict[str, Any]:
    if agent == "researcher":
        return state.get("findings", {})
    if agent == "planner":
        return {"steps": state.get("plan", [])}
    if agent == "executor":
        return {"results": state.get("results", {})}
    if agent == "validator":
        return {
            "final_output": state.get("final_output", ""),
            "confidence_score": state.get("confidence_score", 0.0),
        }
    return {}


async def _wrap_agent(
    agent_name: str,
    agent_fn,
    state: OrchestratorState,
) -> dict[str, Any]:
    session_id = state["session_id"]
    await push_event(
        session_id,
        {"agent": agent_name, "status": "running"},
    )
    updates = await agent_fn(state)
    merged = {**state, **updates}
    await set_session_state(session_id, _state_to_dict(merged))
    await push_event(
        session_id,
        {
            "agent": agent_name,
            "status": "done",
            "output": _agent_output(merged, agent_name),
        },
    )
    return updates


async def researcher_node(state: OrchestratorState) -> dict[str, Any]:
    return await _wrap_agent("researcher", run_researcher, state)


async def planner_node(state: OrchestratorState) -> dict[str, Any]:
    return await _wrap_agent("planner", run_planner, state)


async def executor_node(state: OrchestratorState) -> dict[str, Any]:
    return await _wrap_agent("executor", run_executor, state)


async def validator_node(state: OrchestratorState) -> dict[str, Any]:
    return await _wrap_agent("validator", run_validator, state)


async def bump_retry_node(state: OrchestratorState) -> dict[str, Any]:
    return {"retry_count": state.get("retry_count", 0) + 1}


def route_after_validator(state: OrchestratorState) -> Literal["bump_retry", "end"]:
    confidence = state.get("confidence_score", 0.0)
    retry_count = state.get("retry_count", 0)
    if confidence < 0.6 and retry_count < 2:
        return "bump_retry"
    return "end"


def _state_to_dict(state: OrchestratorState) -> dict[str, Any]:
    return {
        "session_id": state.get("session_id", ""),
        "user_task": state.get("user_task", ""),
        "findings": state.get("findings", {}),
        "plan": state.get("plan", []),
        "results": state.get("results", {}),
        "final_output": state.get("final_output", ""),
        "confidence_score": state.get("confidence_score", 0.0),
        "retry_count": state.get("retry_count", 0),
        "current_agent": state.get("current_agent", ""),
    }


def _build_graph() -> StateGraph:
    graph = StateGraph(OrchestratorState)

    graph.add_node("researcher", researcher_node)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("validator", validator_node)
    graph.add_node("bump_retry", bump_retry_node)

    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "validator")
    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "bump_retry": "bump_retry",
            "end": END,
        },
    )
    graph.add_edge("bump_retry", "executor")

    return graph


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph().compile()
    return _compiled_graph


def _determine_start_node(state: OrchestratorState) -> str:
    findings = state.get("findings", {})
    plan = state.get("plan", [])
    results = state.get("results", {})
    final_output = state.get("final_output", "")

    if findings and not plan:
        return "planner"
    if findings and plan and not results:
        return "executor"
    if findings and plan and results and not final_output:
        return "validator"
    if findings and plan and results and final_output:
        return "end"
    return "researcher"


async def run_workflow(initial_state: OrchestratorState) -> None:
    session_id = initial_state["session_id"]
    start_time = time.monotonic()

    await push_event(
        session_id,
        {
            "agent": "system",
            "status": "started",
            "session_id": session_id,
        },
    )

    start_node = _determine_start_node(initial_state)
    graph = get_compiled_graph()

    try:
        if start_node == "end":
            final_state = initial_state
        elif start_node == "researcher":
            final_state = await graph.ainvoke(initial_state)
        else:
            partial_graph = StateGraph(OrchestratorState)
            partial_graph.add_node("researcher", researcher_node)
            partial_graph.add_node("planner", planner_node)
            partial_graph.add_node("executor", executor_node)
            partial_graph.add_node("validator", validator_node)
            partial_graph.add_node("bump_retry", bump_retry_node)
            partial_graph.set_entry_point(start_node)
            partial_graph.add_edge("researcher", "planner")
            partial_graph.add_edge("planner", "executor")
            partial_graph.add_edge("executor", "validator")
            partial_graph.add_conditional_edges(
                "validator",
                route_after_validator,
                {"bump_retry": "bump_retry", "end": END},
            )
            partial_graph.add_edge("bump_retry", "executor")
            compiled = partial_graph.compile()
            final_state = await compiled.ainvoke(initial_state)

        latency_ms = int((time.monotonic() - start_time) * 1000)

        agent_outputs = {
            "researcher": _agent_output(final_state, "researcher"),
            "planner": _agent_output(final_state, "planner"),
            "executor": _agent_output(final_state, "executor"),
            "validator": _agent_output(final_state, "validator"),
        }

        await upsert_task_history(
            session_id=session_id,
            user_task=final_state.get("user_task", ""),
            agent_outputs=agent_outputs,
            final_output=final_state.get("final_output", ""),
            confidence_score=final_state.get("confidence_score", 0.0),
            completed=True,
        )

        await set_session_state(session_id, _state_to_dict(final_state))

        await push_event(
            session_id,
            {
                "agent": "validator",
                "status": "complete",
                "final_output": final_state.get("final_output", ""),
                "confidence_score": final_state.get("confidence_score", 0.0),
                "latency_ms": latency_ms,
            },
        )
    except Exception as exc:
        logger.exception("Workflow failed for session %s", session_id)
        await push_event(
            session_id,
            {
                "agent": "system",
                "status": "error",
                "message": str(exc),
            },
        )
