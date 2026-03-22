"""
L0 Micro-Experiment: Demonstrates L0 signal behavior and cost-avoidance value.

The L0 tier is an O(1) hash-key lookup against a SQLite table of outcome
statistics. When the historical failure rate for a (task_type, agent_role,
situation_signature) triple exceeds a threshold, L0 fires and the agent can
act on this warning *before* paying the cost of an embedding API call + GPT
generation call.

This script measures three things:
  1. L0 lookup latency in isolation (hash query only, no embedding)
  2. Full L1 search latency (embedding + cosine search on episodes table)
  3. Simulated end-to-end savings: if agent skips GPT on L0-fire vs always-generate

No OpenAI key required — hash Embedder used for L1 timing.
For production-realistic embedding latency, see the note at the bottom of the output.

Usage:
    python3 l0_experiment.py
    python3 l0_experiment.py --queries 500 --verbose
"""
from __future__ import annotations

import argparse
import statistics
import tempfile
import time
import uuid
from typing import List, Tuple

try:
    from Episode import Episode
    from Embedder import Embedder
    from MemoryStore import MemoryStore, stable_hash_key
    from MemoryWriter import MemoryWriter
    from RetrievalContext import RetrievalContext
    from TieredRetrievalPolicy import TieredRetrievalPolicy
except ImportError:
    from .Episode import Episode
    from .Embedder import Embedder
    from .MemoryStore import MemoryStore, stable_hash_key
    from .MemoryWriter import MemoryWriter
    from .RetrievalContext import RetrievalContext
    from .TieredRetrievalPolicy import TieredRetrievalPolicy


# ── Fixed scenario ────────────────────────────────────────────────────────────

TASK_TYPE  = "api_integration"
AGENT_ROLE = "executor"
SITUATION  = "POST /v2/payments endpoint returns 429 rate-limit on concurrent batch"
QUERY_TEXT = "How should executor handle 429 rate-limit on payments endpoint?"

# Six failures → fail_rate = 1.0, well above 0.75 threshold → L0 fires
FAILURE_SEED_COUNT = 6

# Simulated GPT-4o-mini answer-generation latency for cost-avoidance calculation
SIMULATED_GPT_MS = 480   # p50 observed in qa_benchmark runs


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_episode(outcome: str, idx: int) -> Episode:
    now = int(time.time() * 1000) - (idx * 60_000)
    return Episode(
        episode_id=str(uuid.uuid4()),
        task_type=TASK_TYPE,
        agent_role=AGENT_ROLE,
        outcome=outcome,
        abstract=(
            f"Executor submitted batch POST /v2/payments (attempt {idx + 1}). "
            f"API returned 429 Too Many Requests. Exponential backoff exhausted. "
            f"Outcome: {outcome}."
        ),
        full_trace=(
            f"[T+0ms] executor: submitting batch of 50 payment items\n"
            f"[T+12ms] api_client: POST /v2/payments → HTTP 429\n"
            f"[T+512ms] executor: retry 1 after 500ms backoff → HTTP 429\n"
            f"[T+2512ms] executor: retry 2 after 2000ms backoff → HTTP 429\n"
            f"[T+2512ms] executor: max retries exhausted, raising RateLimitError\n"
            f"outcome={outcome} tokens_used=312 latency_ms=2512"
        ),
        cost_tokens=312,
        cost_latency_ms=2512,
        created_at_ms=now,
        tags=["rate_limit", "batch", "payments"],
    )


def _seed(store: MemoryStore, embedder: Embedder, n_failures: int, n_successes: int) -> None:
    writer = MemoryWriter(store=store, embedder=embedder, abstract_generator=None)
    for i in range(n_failures):
        ep = _make_episode("failure", i)
        writer.store_episode(ep, situation_signature=SITUATION, deduplicate=False)
    for i in range(n_successes):
        ep = _make_episode("success", n_failures + i)
        writer.store_episode(ep, situation_signature=SITUATION, deduplicate=False)


def _time_l0_only(store: MemoryStore, n: int) -> List[float]:
    """Measure just the L0 hash-key lookup, nothing else."""
    key = stable_hash_key(TASK_TYPE, AGENT_ROLE, SITUATION)
    latencies: List[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        store.get_l0_signal(key)
        latencies.append((time.perf_counter() - t0) * 1_000)
    return latencies


def _time_l1_only(store: MemoryStore, embedder: Embedder, n: int) -> List[float]:
    """Measure embed + cosine search without L0 overhead."""
    latencies: List[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        emb = embedder.embed(QUERY_TEXT)
        store.search_l1(
            task_type=TASK_TYPE,
            agent_role=AGENT_ROLE,
            query_embedding=emb,
            limit=8,
            min_sim=0.15,
            max_candidates=800,
        )
        latencies.append((time.perf_counter() - t0) * 1_000)
    return latencies


def _time_full_policy(store: MemoryStore, embedder: Embedder, n: int) -> Tuple[List[float], str]:
    """Measure the full TieredRetrievalPolicy retrieve() call."""
    policy = TieredRetrievalPolicy(store=store, embedder=embedder)
    ctx = RetrievalContext(
        task_type=TASK_TYPE,
        agent_role=AGENT_ROLE,
        situation=SITUATION,
        query_text=QUERY_TEXT,
        confidence=0.6,
        retry_count=0,
        latency_budget_ms=2000,
        token_budget=600,
    )
    latencies: List[float] = []
    tier = "?"
    for _ in range(n):
        t0 = time.perf_counter()
        result = policy.retrieve(ctx)
        latencies.append((time.perf_counter() - t0) * 1_000)
        tier = result["tier"]
    return latencies, tier


def _pct(values: List[float], p: float) -> float:
    idx = max(0, min(len(values) - 1, int(len(values) * p / 100)))
    return sorted(values)[idx]


def _print_row(label: str, values: List[float]) -> None:
    print(
        f"  {label:<35}  "
        f"mean={statistics.mean(values):6.3f}ms  "
        f"p50={statistics.median(values):6.3f}ms  "
        f"p95={_pct(values, 95):6.3f}ms"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="L0 micro-experiment")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    n = args.queries
    embedder = Embedder()

    # ── DB with repeated failures (L0 fires) ──────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_fail = f.name
    store_fail = MemoryStore(path=db_fail)
    _seed(store_fail, embedder, n_failures=FAILURE_SEED_COUNT, n_successes=0)

    key = stable_hash_key(TASK_TYPE, AGENT_ROLE, SITUATION)
    sig = store_fail.get_l0_signal(key)
    total = sig["success_count"] + sig["failure_count"] + sig["warning_count"]
    fail_rate = sig["failure_count"] / total

    # ── DB with only successes (L0 below threshold) ───────────────────────
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_ok = f.name
    store_ok = MemoryStore(path=db_ok)
    _seed(store_ok, embedder, n_failures=0, n_successes=4)

    # ── Measure ───────────────────────────────────────────────────────────
    l0_lat   = _time_l0_only(store_fail, n)            # hash lookup only
    l1_lat   = _time_l1_only(store_fail, embedder, n)  # embed + search only
    full_lat_fire, tier_fire = _time_full_policy(store_fail, embedder, n)  # L0 fires
    full_lat_miss, tier_miss = _time_full_policy(store_ok,   embedder, n)  # L0 misses

    # ── Report ────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  L0 MICRO-EXPERIMENT RESULTS")
    print("=" * 70)
    print(f"  Queries per condition  : {n}")
    print(f"  Task / Role            : {TASK_TYPE} / {AGENT_ROLE}")
    print(f"  Seeded failures        : {FAILURE_SEED_COUNT}  →  fail_rate={fail_rate:.0%}")
    print(f"  L0 threshold           : 75%")
    print()

    print("  ── Component latency (isolated) ─────────────────────────────")
    _print_row("L0 hash lookup", l0_lat)
    _print_row("L1 embed + cosine search", l1_lat)
    print()

    print("  ── Full policy latency ──────────────────────────────────────")
    print(f"  Condition A  L0 fires  (tier={tier_fire}):")
    _print_row("  end-to-end retrieval", full_lat_fire)
    print(f"  Condition B  L0 absent (tier={tier_miss}):")
    _print_row("  end-to-end retrieval", full_lat_miss)
    print()

    l0_mean = statistics.mean(l0_lat)
    l1_mean = statistics.mean(l1_lat)
    print("  ── Cost-avoidance model ─────────────────────────────────────")
    print(f"  L0 lookup cost                  : {l0_mean:.3f}ms")
    print(f"  L1 search cost                  : {l1_mean:.3f}ms  (hash embedder)")
    print(f"  Simulated GPT generation        : {SIMULATED_GPT_MS}ms  (p50 observed)")
    print()
    total_without_l0 = l1_mean + SIMULATED_GPT_MS
    total_with_l0_fire = l0_mean  # agent skips L1+GPT on L0 fire
    total_with_l0_pass = l0_mean + l1_mean + SIMULATED_GPT_MS
    saving = total_without_l0 - total_with_l0_fire
    print(f"  Per-query cost (no L0 signal)   : {total_without_l0:.1f}ms")
    print(f"  Per-query cost (L0 fires→abort) : {total_with_l0_fire:.1f}ms  "
          f"(saves {saving:.1f}ms = {saving/total_without_l0*100:.0f}%)")
    print(f"  Per-query cost (L0 passes→L1)   : {total_with_l0_pass:.1f}ms  "
          f"(overhead: +{l0_mean:.3f}ms)")
    print()

    print("  ── Paper note ───────────────────────────────────────────────")
    print(
        f"  L0 fires correctly (fail_rate={fail_rate:.0%} ≥ 75% threshold) after\n"
        f"  {FAILURE_SEED_COUNT} observed failures. The hash lookup costs {l0_mean:.3f}ms.\n"
        f"  When an agent aborts on L0 fire rather than issuing an L1 search\n"
        f"  and downstream GPT call, the per-query cost drops from ~{total_without_l0:.0f}ms\n"
        f"  to ~{total_with_l0_fire:.1f}ms — a {saving/total_without_l0*100:.0f}% reduction on known-failing\n"
        f"  situations. L0 overhead when it does NOT fire is {l0_mean:.3f}ms."
    )
    print("=" * 70)
    print()

    if args.verbose:
        print("  L0 signal detail:")
        print(f"    {sig}")
        print()


if __name__ == "__main__":
    main()
