from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalContext:
    task_type: str
    agent_role: str
    situation: str
    query_text: str
    confidence: float
    retry_count: int
    latency_budget_ms: int
    token_budget: int = 600
