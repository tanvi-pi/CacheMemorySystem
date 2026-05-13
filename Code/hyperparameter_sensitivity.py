#!/usr/bin/env python3
"""
hyperparameter_sensitivity.py — One-at-a-time sensitivity analysis for the
TieredRetrievalPolicy thresholds.

Five thresholds are swept independently while holding all others at their
default values.  Each configuration is evaluated on a held-out tuning set
(seed=42, 100 queries) that is disjoint from the main evaluation seed (seed=7).

Usage:
    python3 hyperparameter_sensitivity.py
    python3 hyperparameter_sensitivity.py --output sensitivity.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from Embedder import Embedder
from MemoryStore import MemoryStore
from RetrievalContext import RetrievalContext
from TieredRetrievalPolicy import TieredRetrievalPolicy, FlatRetrievalPolicy
from terminal_benchmark import (
    TASKS,
    PATTERNS,
    _build_system,
    _seed_episodes,
    _seed_canonical_situations,
    _generate_queries,
    _score_accuracy,
)


# ── Defaults (values used in the paper) ──────────────────────────────────────
DEFAULTS = {
    "l0_sim_threshold":          0.65,
    "l1_min_sim":                0.15,
    "min_confidence_for_l1_only": 0.55,
    "weak_l1_sim_threshold":     0.25,
    "retry_escalate_to_l2":      2,
}

# Sweep grids — centred on the default value
SWEEP = {
    "l0_sim_threshold":           [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
    "l1_min_sim":                 [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "min_confidence_for_l1_only": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    "weak_l1_sim_threshold":      [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "retry_escalate_to_l2":       [1, 2, 3, 4, 5],
}

EPISODES_PER_TASK = 20
NUM_QUERIES       = 100
TUNING_SEED       = 42   # held-out seed, distinct from evaluation seed (7)
DB_PATH_TEMPLATE  = "/tmp/sensitivity_{param}_{i}.db"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_db(path: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        p = path + suffix
        if os.path.exists(p):
            os.remove(p)


def _build_policy(store: MemoryStore, embedder: Embedder, kwargs: Dict) -> TieredRetrievalPolicy:
    return TieredRetrievalPolicy(store, embedder, **kwargs)


def _run_single(
    db_path: str,
    embedder: Embedder,
    queries: List[Dict],
    policy_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Seed memory and run *queries* under the given policy, return summary."""
    _clean_db(db_path)
    rng = random.Random(TUNING_SEED)

    system = _build_system(db_path, embedder=embedder)
    _seed_episodes(system, EPISODES_PER_TASK, rng)
    store = MemoryStore(db_path)
    _seed_canonical_situations(store, embedder)

    policy = _build_policy(store, embedder, policy_kwargs)

    top1_correct = 0
    evaluable = 0
    total_tokens = 0
    tier_counts: Dict[str, int] = {"L0": 0, "L1": 0, "L2": 0}

    for q in queries:
        raw_sit = str(q["situation"])
        canonical_sit = str(q.get("ground_truth_situation", raw_sit))

        ctx = RetrievalContext(
            task_type=str(q["task_type"]),
            agent_role=str(q["role"]),
            situation=raw_sit,
            query_text=str(q["query"]),
            confidence=float(q["confidence"]),
            retry_count=int(q["retry_count"]),
            latency_budget_ms=200,
            token_budget=2000,
        )
        result = policy.retrieve(ctx)
        hits = result.get("abstract_hits", [])
        tier = str(result.get("tier", "L1"))
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        total_tokens += result["debug"].get("retrieval_tokens", 0)

        if bool(q.get("known", True)):
            acc = _score_accuracy(canonical_sit, hits, True, str(q["task_type"]))
            if acc["evaluable"]:
                evaluable += 1
                if acc["top1_correct"]:
                    top1_correct += 1

    n = evaluable or 1
    return {
        "top1_pct":   round(100 * top1_correct / n, 1),
        "avg_tokens": round(total_tokens / max(len(queries), 1), 1),
        "tier_counts": tier_counts,
        "evaluable":  evaluable,
    }


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_sensitivity(output_path: Optional[str]) -> Dict[str, Any]:
    embedder = Embedder(dim=512)

    # Pre-generate the held-out query set once (seeded, so it's reproducible)
    rng_q = random.Random(TUNING_SEED)
    queries = _generate_queries(NUM_QUERIES, rng_q)

    all_results: Dict[str, List[Dict]] = {}

    for param, values in SWEEP.items():
        print(f"\n[sweep] {param}")
        param_results = []

        for val in values:
            kwargs = dict(DEFAULTS)
            kwargs[param] = val

            # retry_escalate_to_l2 must be int
            kwargs["retry_escalate_to_l2"] = int(kwargs["retry_escalate_to_l2"])

            db_path = f"/tmp/sensitivity_{param}_{val}.db"
            summary = _run_single(db_path, embedder, queries, kwargs)
            is_default = (val == DEFAULTS[param])
            flag = " ← default" if is_default else ""
            print(
                f"  {param}={val:<6}  top-1={summary['top1_pct']:>5}%  "
                f"avg_tokens={summary['avg_tokens']:>7.1f}{flag}"
            )
            param_results.append({"value": val, "is_default": is_default, **summary})

        all_results[param] = param_results

    _print_report(all_results)

    report = {
        "config": {
            "episodes_per_task": EPISODES_PER_TASK,
            "num_queries": NUM_QUERIES,
            "tuning_seed": TUNING_SEED,
            "defaults": DEFAULTS,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": (
                "One-at-a-time sensitivity sweep. Tuning seed (42) is disjoint "
                "from the evaluation seed (7) used in Tables 2-5."
            ),
        },
        "results": all_results,
        "stability_summary": _stability_summary(all_results),
    }

    if output_path:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(os.path.basename(output_path))
        out = os.path.join(results_dir, f"{base}_{ts}{ext}")
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[output] Saved to {out}")

    return report


def _stability_summary(all_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    summary = {}
    for param, rows in all_results.items():
        top1_vals = [r["top1_pct"] for r in rows]
        tok_vals  = [r["avg_tokens"] for r in rows]
        def _std(vals: List[float]) -> float:
            mu = sum(vals) / len(vals)
            return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
        summary[param] = {
            "top1_range_pp":    round(max(top1_vals) - min(top1_vals), 1),
            "top1_std_pp":      round(_std(top1_vals), 1),
            "token_range_pct":  round(100 * (max(tok_vals) - min(tok_vals)) / max(max(tok_vals), 1), 1),
        }
    return summary


def _print_report(all_results: Dict[str, List[Dict]]) -> None:
    print(f"\n{'='*72}")
    print("  HYPERPARAMETER SENSITIVITY REPORT")
    print(f"  Tuning seed={TUNING_SEED}  queries={NUM_QUERIES}  episodes/task={EPISODES_PER_TASK}")
    print(f"{'='*72}")

    for param, rows in all_results.items():
        default_row = next((r for r in rows if r["is_default"]), rows[0])
        top1_vals   = [r["top1_pct"]   for r in rows]
        tok_vals    = [r["avg_tokens"] for r in rows]
        top1_range  = round(max(top1_vals) - min(top1_vals), 1)
        tok_range   = round(100 * (max(tok_vals) - min(tok_vals)) / max(max(tok_vals), 1), 1)

        print(f"\n  {param} (default={DEFAULTS[param]})")
        print(f"  {'Value':<10} {'Top-1 %':>8} {'Avg Tokens':>12} {'Tier L0/L1/L2':>18}")
        print(f"  {'-'*52}")
        for r in rows:
            marker = " *" if r["is_default"] else "  "
            tc = r["tier_counts"]
            tier_str = f"{tc.get('L0',0)}/{tc.get('L1',0)}/{tc.get('L2',0)}"
            print(f"  {str(r['value']):<10} {r['top1_pct']:>7}% {r['avg_tokens']:>12.1f}{marker} {tier_str:>16}")
        print(f"  range: top-1 ±{top1_range}pp,  tokens ±{tok_range}%")

    print(f"\n{'='*72}")
    print("  STABILITY SUMMARY (across full sweep range per parameter)")
    print(f"  {'Parameter':<32} {'Top-1 range (pp)':>18} {'Token range (%)':>16}")
    print(f"  {'-'*68}")
    for param, rows in all_results.items():
        top1_vals = [r["top1_pct"] for r in rows]
        tok_vals  = [r["avg_tokens"] for r in rows]
        top1_range = round(max(top1_vals) - min(top1_vals), 1)
        tok_range  = round(100 * (max(tok_vals) - min(tok_vals)) / max(max(tok_vals), 1), 1)
        print(f"  {param:<32} {top1_range:>18.1f} {tok_range:>16.1f}")
    print(f"{'='*72}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-at-a-time sensitivity analysis for retrieval policy thresholds."
    )
    parser.add_argument("--output", default="sensitivity.json",
                        help="Output JSON filename (saved into results/)")
    args = parser.parse_args()
    run_sensitivity(args.output)


if __name__ == "__main__":
    main()
