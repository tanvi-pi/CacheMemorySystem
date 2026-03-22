from __future__ import annotations

import dataclasses
import os
from typing import Any, Optional

try:
    from .Episode import Episode
    from .MemoryStore import MemoryStore, stable_hash_key
except ImportError:
    from Episode import Episode
    from MemoryStore import MemoryStore, stable_hash_key


class LLMAbstractGenerator:
    """Generates cause-effect, role-specific L1 abstracts via OpenAI chat completion."""

    _SYSTEM = (
        "You are an episodic memory encoder for an AI agent. "
        "Your output is stored as a concise abstract in the agent's long-term memory. "
        "Focus strictly on cause-effect relationships and role-specific lessons learned."
    )

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        self.client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def generate(self, task_type: str, agent_role: str, full_trace: str, outcome: str) -> str:
        user_content = (
            f"Role: {agent_role}\n"
            f"Task type: {task_type}\n"
            f"Outcome: {outcome}\n\n"
            f"Execution trace (truncated to 3000 chars):\n{full_trace[:3000]}\n\n"
            "Write a 2-3 sentence abstract covering:\n"
            "1. What specifically caused the outcome\n"
            "2. The key decision or action taken by this role\n"
            "3. What should be remembered for similar future situations"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._SYSTEM},
                {"role": "user", "content": user_content},
            ],
            max_tokens=220,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()


class MemoryWriter:
    """Writes finalized episodes into L0, L1 and L2-backed storage."""

    def __init__(
        self,
        store: MemoryStore,
        embedder: Any,
        abstract_generator: Optional[LLMAbstractGenerator] = None,
        always_generate: bool = False,
    ):
        self.store = store
        self.embedder = embedder
        self.abstract_generator = abstract_generator
        self.always_generate = always_generate

    def store_episode(
        self,
        ep: Episode,
        situation_signature: str,
        deduplicate: bool = True,
    ) -> Optional[str]:
        """
        Write an episode to memory. Returns the episode_id that was written,
        or the id of an existing duplicate if deduplication fired (NOOP).

        deduplicate=True mirrors Mem0's NOOP operation: if an episode with
        the same role, task_type, situation_signature, and near-identical
        abstract already exists (cosine sim >= 0.92), the write is skipped
        and the existing episode_id is returned.
        """
        self.store.check_embedder_compat(self.embedder.model_id)
        abstract = ep.abstract
        if self.abstract_generator and (self.always_generate or not abstract.strip()):
            abstract = self.abstract_generator.generate(
                task_type=ep.task_type,
                agent_role=ep.agent_role,
                full_trace=ep.full_trace,
                outcome=ep.outcome,
            )
            ep = dataclasses.replace(ep, abstract=abstract)
        embedding = self.embedder.embed(abstract)

        if deduplicate:
            existing_id = self.store.find_duplicate(
                agent_role=ep.agent_role,
                task_type=ep.task_type,
                situation_signature=situation_signature,
                abstract_embedding=embedding,
            )
            if existing_id is not None:
                return existing_id  # NOOP — duplicate already stored

        self.store.upsert_episode(ep, embedding, situation_signature=situation_signature)
        key = stable_hash_key(ep.task_type, ep.agent_role, situation_signature)
        self.store.upsert_l0_signal(key, ep.task_type, ep.agent_role, ep.outcome, ep.episode_id)
        return ep.episode_id
