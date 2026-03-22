from __future__ import annotations

import time
from typing import Any, Dict, List

try:
    from .Embedder import Embedder
    from .MemoryStore import MemoryStore, stable_hash_key
    from .RetrievalContext import RetrievalContext
except ImportError:
    from Embedder import Embedder
    from MemoryStore import MemoryStore, stable_hash_key
    from RetrievalContext import RetrievalContext


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class TieredRetrievalPolicy:
    """Latency and budget-aware L0->L1->L2 retrieval.

    L0 (hash signal):
        Checks the historical failure rate for the exact (task_type, agent_role,
        situation) triple. If the rate exceeds l0_fail_threshold, the query is
        flagged and — by default — returned immediately without running L1 or L2.
        This prevents wasted embedding and generation calls on known-failing
        situations. Set l0_skip_l1=False to disable early exit and treat L0 as
        a warning signal only.

    L1 (abstract vector search):
        Embeds the query and searches role-partitioned episode abstracts.

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
        l0_fail_threshold: float = 0.75,
        l0_skip_l1: bool = True,
        min_confidence_for_l1_only: float = 0.55,
        retry_escalate_to_l2: int = 2,
        l2_topk: int = 2,
    ):
        self.store = store
        self.embedder = embedder
        self.l0_fail_threshold = l0_fail_threshold
        self.l0_skip_l1 = l0_skip_l1
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

        # ── L0: hash signal lookup ────────────────────────────────────────────
        key = stable_hash_key(ctx.task_type, ctx.agent_role, ctx.situation)
        sig = self.store.get_l0_signal(key)
        used["l0"] = True
        result["signals"] = sig

        l0_fired = False
        if sig:
            total = sig["success_count"] + sig["failure_count"] + sig["warning_count"]
            fail_rate = (sig["failure_count"] / total) if total else 0.0
            if fail_rate >= self.l0_fail_threshold:
                l0_fired = True
                result["tier"] = "L0"
                result["l0_recommendation"] = {
                    "message": "High historical failure rate for this role/task/situation signature.",
                    "fail_rate": fail_rate,
                    "last_episode_id": sig["last_episode_id"],
                    "last_outcome": sig["last_outcome"],
                }

        if l0_fired and self.l0_skip_l1:
            # Early exit: skip the full L1 search. Fetch the most recent episode
            # for this situation from the L0 signal so the agent receives the
            # last known context (typically a failure trace) rather than nothing.
            # The caller should inspect result["l0_recommendation"] and decide
            # whether to proceed or abort.
            last_ep = self.store.fetch_l2_full_traces([sig["last_episode_id"]])
            if last_ep:
                token_spend += _estimate_tokens(last_ep[0].get("full_trace", ""))
                result["full_hits"] = last_ep
                result["tier"] = "L0"
            result["debug"]["latency_ms"] = int((time.perf_counter() - t0) * 1000)
            result["debug"]["retrieval_tokens"] = token_spend
            return result

        # ── L1: role-partitioned abstract search ─────────────────────────────
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        if elapsed_ms < ctx.latency_budget_ms:
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
            result["tier"] = result["tier"] or "L1"
            result["abstract_hits"] = abstract_hits

        # ── L2: full trace escalation ─────────────────────────────────────────
        # open_domain always escalates to L2; other types escalate on low
        # confidence + weak L1, or on retry.
        low_conf = ctx.confidence < self.min_confidence_for_l1_only
        retry_loop = ctx.retry_count >= self.retry_escalate_to_l2
        weak_l1 = not result["abstract_hits"] or result["abstract_hits"][0]["sim"] < 0.25
        should_escalate = is_open_domain or (low_conf and weak_l1) or retry_loop

        effective_token_budget = (
            self.OPEN_DOMAIN_TOKEN_BUDGET if is_open_domain else ctx.token_budget
        )
        l2_topk = self.OPEN_DOMAIN_L2_TOPK if is_open_domain else self.l2_topk

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        if should_escalate and elapsed_ms < ctx.latency_budget_ms and token_spend < effective_token_budget:
            top_ids = [h["episode_id"] for h in result["abstract_hits"][:l2_topk]]
            if not top_ids and sig:
                top_ids = [sig["last_episode_id"]]
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
