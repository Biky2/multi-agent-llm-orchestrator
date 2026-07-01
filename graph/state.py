from typing import TypedDict


class OrchestratorState(TypedDict):
    session_id: str
    user_task: str
    findings: dict
    plan: list[str]
    results: dict
    final_output: str
    confidence_score: float
    retry_count: int
    current_agent: str
