from __future__ import annotations

import time
from typing import Any, Dict, List

try:
    from .Embedder import Embedder
    from .MemoryStore import MemoryStore
    from .RetrievalContext import RetrievalContext
except ImportError:
    from Embedder import Embedder
    from MemoryStore import MemoryStore
    from RetrievalContext import RetrievalContext


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class TieredRetrievalPolicy:
    """Latency and budget-aware L0->L1->L2 retrieval.

    L0 (canonical situation match):
        Embeds ctx.situation and finds the nearest canonical situation vector
        using cosine similarity. If similarity >= l0_sim_threshold, returns the
        best past episode for that situation immediately — skipping L1 entirely.
        This avoids the broader abstract search when the situation is already
        well-recognised, saving tokens and latency.

    L1 (abstract vector search):
        Runs only when L0 misses. Embeds the full query text and searches
        role-partitioned episode abstracts.

    L2 (full trace escalation):
        Triggered when confidence is low AND L1 results are weak, or when
        retry_count reaches retry_escalate_to_l2. Fetches full execution traces
        for the top L1 hits. For open_domain queries (ctx.query_type="open_domain"),
        L2 escalation is always forced and a wider candidate set is used.
    """

    # open_domain retrieval parameters
    OPEN_DOMAIN_L2_TOPK = 10
    OPEN_DOMAIN_TOKEN_BUDGET = 5000
    OPEN_DOMAIN_L1_LIMIT = 10
    OPEN_DOMAIN_L1_MAX_CANDIDATES = 1200

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        l0_sim_threshold: float = 0.65,
        min_confidence_for_l1_only: float = 0.55,
        retry_escalate_to_l2: int = 2,
        l2_topk: int = 2,
    ):
        self.store = store
        self.embedder = embedder
        self.l0_sim_threshold = l0_sim_threshold
        self.min_confidence_for_l1_only = min_confidence_for_l1_only
        self.retry_escalate_to_l2 = retry_escalate_to_l2
        self.l2_topk = l2_topk

    def retrieve(self, ctx: RetrievalContext) -> Dict[str, Any]:
        t0 = time.perf_counter()
        used = {"l0": False, "l1": False, "l2": False}
        token_spend = 0
        result: Dict[str, Any] = {
            "tier": None,
            "signals": None,
            "abstract_hits": [],
            "full_hits": [],
            "debug": {"used": used, "latency_ms": 0, "retrieval_tokens": 0},
        }

        self.store.check_embedder_compat(self.embedder.model_id)
        is_open_domain = ctx.query_type == "open_domain"

        # ── L0: canonical situation embedding match ───────────────────────────
        # Embed ctx.situation and find the nearest canonical situation vector.
        # If cosine similarity >= l0_sim_threshold, the situation is recognised —
        # return the best past episode for it immediately without running L1.
        # Open-domain queries skip L0 because canonical situations are scoped
        # per task_type + agent_role and can't cover cross-domain queries.
        if not is_open_domain:
            sit_emb = self.embedder.embed(ctx.situation)
            canonical_label = self.store.resolve_situation(
                ctx.task_type, ctx.agent_role, sit_emb, threshold=self.l0_sim_threshold
            )
            used["l0"] = True
            if canonical_label:
                episode = self.store.get_best_episode_for_situation(
                    ctx.task_type, ctx.agent_role, canonical_label
                )
                if episode:
                    token_spend += _estimate_tokens(episode["abstract"])
                    result["tier"] = "L0"
                    result["signals"] = {"canonical_label": canonical_label}
                    result["abstract_hits"] = [{"sim": self.l0_sim_threshold, **episode}]
                    result["debug"]["latency_ms"] = int((time.perf_counter() - t0) * 1000)
                    result["debug"]["retrieval_tokens"] = token_spend
                    result["debug"]["used"] = used
                    return result

        # ── L1: role-partitioned abstract search ─────────────────────────────
        # Runs when L0 misses (no canonical match above threshold) or for
        # open_domain queries.
        l1_limit = self.OPEN_DOMAIN_L1_LIMIT if is_open_domain else 8
        l1_max_candidates = self.OPEN_DOMAIN_L1_MAX_CANDIDATES if is_open_domain else 800
        qemb = self.embedder.embed(ctx.query_text)
        hits = self.store.search_l1(
            task_type=ctx.task_type,
            agent_role=ctx.agent_role,
            query_embedding=qemb,
            limit=l1_limit,
            min_sim=0.15,
            max_candidates=l1_max_candidates,
        )
        used["l1"] = True
        abstract_hits = [{"sim": s, **meta} for (s, meta) in hits]
        token_spend += sum(_estimate_tokens(h["abstract"]) for h in abstract_hits)
        result["tier"] = "L1"
        result["abstract_hits"] = abstract_hits

        # ── L2: full trace escalation ─────────────────────────────────────────
        # Escalate when L1 similarity is weak AND (confidence is low OR query is
        # open_domain), or when the agent is in a retry loop.
        low_conf = ctx.confidence < self.min_confidence_for_l1_only
        retry_loop = ctx.retry_count >= self.retry_escalate_to_l2
        weak_l1 = not abstract_hits or abstract_hits[0]["sim"] < 0.25
        should_escalate = (weak_l1 and (is_open_domain or low_conf)) or retry_loop

        effective_token_budget = (
            self.OPEN_DOMAIN_TOKEN_BUDGET if is_open_domain else ctx.token_budget
        )
        l2_topk = self.OPEN_DOMAIN_L2_TOPK if is_open_domain else self.l2_topk

        if should_escalate and token_spend < effective_token_budget:
            top_ids = [h["episode_id"] for h in abstract_hits[:l2_topk]]
            full = self.store.fetch_l2_full_traces(top_ids)

            allowed: List[Dict[str, Any]] = []
            for hit in full:
                trace_tokens = _estimate_tokens(hit["full_trace"])
                if token_spend + trace_tokens > effective_token_budget:
                    break
                token_spend += trace_tokens
                allowed.append(hit)

            used["l2"] = len(allowed) > 0
            if allowed:
                result["tier"] = "L2"
                result["full_hits"] = allowed

        result["debug"]["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        result["debug"]["retrieval_tokens"] = token_spend
        return result


class FlatRetrievalPolicy:
    """Baseline: one-step global similarity then full trace pull on top hits."""

    def __init__(self, store: MemoryStore, embedder: Embedder, topk: int = 3):
        self.store = store
        self.embedder = embedder
        self.topk = topk

    def retrieve(self, ctx: RetrievalContext) -> Dict[str, Any]:
        t0 = time.perf_counter()
        qemb = self.embedder.embed(ctx.query_text)
        hits = self.store.search_l1_flat(query_embedding=qemb, limit=self.topk, min_sim=0.15, max_candidates=3000)
        abstract_hits = [{"sim": s, **meta} for (s, meta) in hits]
        top_ids = [h["episode_id"] for h in abstract_hits]
        full_hits = self.store.fetch_l2_full_traces(top_ids)
        token_spend = sum(_estimate_tokens(h["abstract"]) for h in abstract_hits)
        token_spend += sum(_estimate_tokens(h["full_trace"]) for h in full_hits)

        return {
            "tier": "flat",
            "signals": None,
            "abstract_hits": abstract_hits,
            "full_hits": full_hits,
            "debug": {
                "used": {"flat": True},
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "retrieval_tokens": token_spend,
            },
        }
