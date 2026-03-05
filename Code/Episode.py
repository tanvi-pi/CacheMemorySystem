from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Episode:
    """Canonical memory unit persisted across all cache tiers."""

    episode_id: str
    task_type: str
    agent_role: str
    outcome: str  # success | failure | warning
    abstract: str
    full_trace: str
    cost_tokens: int
    cost_latency_ms: int
    created_at_ms: int
    tags: Optional[List[str]] = None
    warning_flags: Optional[List[str]] = None
