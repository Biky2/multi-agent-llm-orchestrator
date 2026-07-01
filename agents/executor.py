import json
import logging
from datetime import datetime, timezone

from core.llm import get_llm_response, parse_llm_json
from graph.state import OrchestratorState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert writer and analyst. Execute each step of the plan thoroughly. "
    "Return JSON with key: results (object where each key is the step name and "
    "value is the detailed output)."
)

DEFAULT_OUTPUT = {
    "results": {
        "step1": "Executed plan steps based on available context.",
    }
}


async def run_executor(state: OrchestratorState) -> dict:
    agent_name = "executor"
    started = datetime.now(timezone.utc).isoformat()
    logger.info("%s started at %s", agent_name, started)

    state["current_agent"] = agent_name
    plan = state.get("plan", [])
    findings = state.get("findings", {})
    prompt = (
        f"User task: {state['user_task']}\n\n"
        f"Research findings:\n{json.dumps(findings, indent=2)}\n\n"
        f"Execution plan:\n{json.dumps(plan, indent=2)}"
    )

    raw = await get_llm_response(prompt, SYSTEM_PROMPT)
    parsed = parse_llm_json(raw, DEFAULT_OUTPUT)

    results = parsed.get("results", DEFAULT_OUTPUT["results"])
    if not isinstance(results, dict):
        results = {"output": str(results)}

    finished = datetime.now(timezone.utc).isoformat()
    logger.info("%s finished at %s", agent_name, finished)

    return {
        "results": results,
        "current_agent": agent_name,
    }
