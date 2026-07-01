import json
import logging
from datetime import datetime, timezone

from core.llm import get_llm_response, parse_llm_json
from graph.state import OrchestratorState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a project planner. Given research findings, create a clear numbered "
    "execution plan with 3-6 concrete steps. Return JSON with key: steps (list of strings)."
)

DEFAULT_OUTPUT = {
    "steps": [
        "Review the research findings",
        "Draft the main deliverable",
        "Review and refine the output",
    ]
}


async def run_planner(state: OrchestratorState) -> dict:
    agent_name = "planner"
    started = datetime.now(timezone.utc).isoformat()
    logger.info("%s started at %s", agent_name, started)

    state["current_agent"] = agent_name
    findings = state.get("findings", {})
    prompt = (
        f"User task: {state['user_task']}\n\n"
        f"Research findings:\n{json.dumps(findings, indent=2)}"
    )

    raw = await get_llm_response(prompt, SYSTEM_PROMPT)
    parsed = parse_llm_json(raw, DEFAULT_OUTPUT)

    steps = parsed.get("steps", DEFAULT_OUTPUT["steps"])
    if not isinstance(steps, list):
        steps = [str(steps)]

    finished = datetime.now(timezone.utc).isoformat()
    logger.info("%s finished at %s", agent_name, finished)

    return {
        "plan": steps,
        "current_agent": agent_name,
    }
