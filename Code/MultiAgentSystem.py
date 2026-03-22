from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from .Agent import Agent
    from .Embedder import Embedder
    from .MemoryStore import MemoryStore
    from .MemoryWriter import MemoryWriter
    from .TieredRetrievalPolicy import TieredRetrievalPolicy
except ImportError:
    from Agent import Agent
    from Embedder import Embedder
    from MemoryStore import MemoryStore
    from MemoryWriter import MemoryWriter
    from TieredRetrievalPolicy import TieredRetrievalPolicy


class MultiAgentSystem:
    """Role-aware coordinator that shares one memory backend across agents."""

    def __init__(
        self,
        roles: List[str],
        task_role_map: Optional[Dict[str, str]] = None,
        db_path: str = "memory.db",
        embed_dim: int = 512,
        embedder: Any = None,
        abstract_generator: Any = None,
        always_generate_abstracts: bool = False,
    ):
        if not roles:
            raise ValueError("roles must contain at least one agent role")
        self.store = MemoryStore(db_path)
        self.embedder = embedder if embedder is not None else Embedder(dim=embed_dim)
        self.writer = MemoryWriter(
            self.store,
            self.embedder,
            abstract_generator=abstract_generator,
            always_generate=always_generate_abstracts,
        )
        self.task_role_map = task_role_map or {}

        self.agents: Dict[str, Agent] = {}
        for role in roles:
            policy = TieredRetrievalPolicy(self.store, self.embedder)
            self.agents[role] = Agent(role, policy, self.writer)

    def _resolve_role(self, task_type: str, role: Optional[str]) -> Agent:
        if role is not None:
            if role not in self.agents:
                raise ValueError(f"Unknown role '{role}'. Valid roles: {sorted(self.agents.keys())}")
            return self.agents[role]

        mapped_role = self.task_role_map.get(task_type)
        if mapped_role is not None:
            if mapped_role not in self.agents:
                raise ValueError(
                    f"Task '{task_type}' maps to unknown role '{mapped_role}'. "
                    f"Valid roles: {sorted(self.agents.keys())}"
                )
            return self.agents[mapped_role]

        return next(iter(self.agents.values()))

    def step(
        self,
        task_type: str,
        situation: str,
        user_query: str,
        confidence: float,
        retry_count: int,
        role: Optional[str] = None,
        latency_budget_ms: int = 40,
        token_budget: int = 600,
        query_type: str = "",
    ) -> Dict[str, Any]:
        agent = self._resolve_role(task_type, role)
        out = agent.step(
            task_type=task_type,
            situation=situation,
            user_query=user_query,
            confidence=confidence,
            retry_count=retry_count,
            latency_budget_ms=latency_budget_ms,
            token_budget=token_budget,
            query_type=query_type,
        )
        out["agent_role"] = agent.role
        return out

    def finalize_episode(
        self,
        role: str,
        episode_id: str,
        task_type: str,
        outcome: str,
        abstract: str,
        full_trace: str,
        situation_signature: str,
        cost_tokens: int,
        cost_latency_ms: int,
    ) -> None:
        agent = self._resolve_role(task_type, role)
        agent.finalize_episode(
            episode_id=episode_id,
            task_type=task_type,
            outcome=outcome,
            abstract=abstract,
            full_trace=full_trace,
            situation_signature=situation_signature,
            cost_tokens=cost_tokens,
            cost_latency_ms=cost_latency_ms,
        )
