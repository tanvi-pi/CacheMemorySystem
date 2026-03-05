from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

try:
    from .Episode import Episode
    from .MemoryWriter import MemoryWriter
    from .RetrievalContext import RetrievalContext
    from .TieredRetrievalPolicy import TieredRetrievalPolicy
except ImportError:
    from Episode import Episode
    from MemoryWriter import MemoryWriter
    from RetrievalContext import RetrievalContext
    from TieredRetrievalPolicy import TieredRetrievalPolicy


class Agent:
    def __init__(self, agent_role: str, policy: TieredRetrievalPolicy, writer: MemoryWriter):
        self.role = agent_role
        self.policy = policy
        self.writer = writer

    def step(
        self,
        task_type: str,
        situation: str,
        user_query: str,
        confidence: float,
        retry_count: int,
        latency_budget_ms: int = 40,
        token_budget: int = 600,
    ) -> Dict[str, Any]:
        ctx = RetrievalContext(
            task_type=task_type,
            agent_role=self.role,
            situation=situation,
            query_text=user_query,
            confidence=confidence,
            retry_count=retry_count,
            latency_budget_ms=latency_budget_ms,
            token_budget=token_budget,
        )
        return self.policy.retrieve(ctx)

    def finalize_episode(
        self,
        episode_id: str,
        task_type: str,
        outcome: str,
        abstract: str,
        full_trace: str,
        situation_signature: str,
        cost_tokens: int,
        cost_latency_ms: int,
        tags: Optional[List[str]] = None,
        warning_flags: Optional[List[str]] = None,
    ) -> None:
        ep = Episode(
            episode_id=episode_id,
            task_type=task_type,
            agent_role=self.role,
            outcome=outcome,
            abstract=abstract,
            full_trace=full_trace,
            cost_tokens=cost_tokens,
            cost_latency_ms=cost_latency_ms,
            created_at_ms=int(time.time() * 1000),
            tags=tags,
            warning_flags=warning_flags,
        )
        self.writer.store_episode(ep, situation_signature)
