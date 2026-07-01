import logging
from datetime import datetime, timezone

from core.llm import get_llm_response, parse_llm_json
from graph.state import OrchestratorState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a research analyst. Break the user task into 3-5 sub-questions, "
    "answer each thoroughly using your knowledge, and return a JSON object with keys: "
    "findings (list of strings) and sources (list of strings)."
)

DEFAULT_OUTPUT = {
    "findings": ["Unable to parse research findings from LLM response."],
    "sources": ["General knowledge"],
}


async def run_researcher(state: OrchestratorState) -> dict:
    agent_name = "researcher"
    started = datetime.now(timezone.utc).isoformat()
    logger.info("%s started at %s", agent_name, started)

    state["current_agent"] = agent_name
    prompt = f"User task: {state['user_task']}"

    raw = await get_llm_response(prompt, SYSTEM_PROMPT)
    parsed = parse_llm_json(raw, DEFAULT_OUTPUT)

    findings = parsed.get("findings", DEFAULT_OUTPUT["findings"])
    sources = parsed.get("sources", DEFAULT_OUTPUT["sources"])
    if not isinstance(findings, list):
        findings = [str(findings)]
    if not isinstance(sources, list):
        sources = [str(sources)]

    output = {"findings": findings, "sources": sources}

    finished = datetime.now(timezone.utc).isoformat()
    logger.info("%s finished at %s", agent_name, finished)

    return {
        "findings": output,
        "current_agent": agent_name,
    }
