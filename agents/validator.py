import json
import logging
from datetime import datetime, timezone

from core.llm import get_llm_response, parse_llm_json
from graph.state import OrchestratorState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a quality reviewer. Review the executed results for completeness, "
    "accuracy, and coherence. Produce a polished final output and score your confidence. "
    "Return JSON with keys: final_output (string) and confidence_score (float 0.0-1.0)."
)

DEFAULT_OUTPUT = {
    "final_output": "The task was processed but the validator could not parse a structured response.",
    "confidence_score": 0.5,
}


async def run_validator(state: OrchestratorState) -> dict:
    agent_name = "validator"
    started = datetime.now(timezone.utc).isoformat()
    logger.info("%s started at %s", agent_name, started)

    state["current_agent"] = agent_name
    results = state.get("results", {})
    prompt = (
        f"User task: {state['user_task']}\n\n"
        f"Executed results:\n{json.dumps(results, indent=2)}"
    )

    raw = await get_llm_response(prompt, SYSTEM_PROMPT)
    parsed = parse_llm_json(raw, DEFAULT_OUTPUT)

    final_output = parsed.get("final_output", DEFAULT_OUTPUT["final_output"])
    if not isinstance(final_output, str):
        final_output = str(final_output)

    confidence_score = parsed.get("confidence_score", DEFAULT_OUTPUT["confidence_score"])
    try:
        confidence_score = float(confidence_score)
        confidence_score = max(0.0, min(1.0, confidence_score))
    except (TypeError, ValueError):
        confidence_score = 0.5

    finished = datetime.now(timezone.utc).isoformat()
    logger.info("%s finished at %s", agent_name, finished)

    return {
        "final_output": final_output,
        "confidence_score": confidence_score,
        "current_agent": agent_name,
    }
