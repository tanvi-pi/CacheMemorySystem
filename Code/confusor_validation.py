#!/usr/bin/env python3
"""
confusor_validation.py — Independent validation of role-partitioning advantage.

Generates confusor queries via GPT without any knowledge of the system's
architecture, then evaluates tiered vs flat retrieval on both the original
hand-written set and the LLM-generated set side by side.

Addresses reviewer concern: the 200-query benchmark's confusor queries were
authored by the same team that designed the system. LLM-generated confusors
are independent — the generation prompt describes only the task categories,
not the role-partitioning architecture being tested.

Usage:
    python3 confusor_validation.py --output confusor_validation.json
    (uses OPENAI_API_KEY env var)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from Embedder import Embedder
from MemoryStore import MemoryStore
from RetrievalContext import RetrievalContext
from TieredRetrievalPolicy import FlatRetrievalPolicy
from terminal_benchmark import (
    TASKS,
    PATTERNS,
    CROSS_TASK_CONFUSORS,
    _build_system,
    _seed_episodes,
    _seed_canonical_situations,
)


# ── LLM confusor generation ───────────────────────────────────────────────────

_GENERATION_PROMPT = """\
You are generating evaluation queries for a multi-agent software engineering assistant.

The assistant has five specialized agents, each responsible for a different task category:
  1. db_query_debug         — database query performance, index tuning, connection issues
  2. api_integration        — external API auth, rate limiting, pagination, webhooks
  3. code_review            — race conditions, memory leaks, security vulnerabilities, test coverage
  4. incident_investigation — production outages, error spikes, infrastructure failures
  5. release_validation     — deployment pipelines, canary releases, smoke tests, rollbacks

Your task: generate {n} "confusor" queries. A confusor query describes a situation that
truly belongs to one task category but is worded so that it superficially sounds like a
different category. The goal is to test whether an agent can correctly identify the true
task category despite misleading surface phrasing.

Rules:
- Each confusor must have a clearly defined true category and a clearly defined disguise category
- The query text must NOT contain the name of the true category
- The query text should use vocabulary associated with the disguise category
- Situations must be realistic technical scenarios (not generic)
- Cover all 5 task types as both source (true_task) and target (disguise_as)
- Do not repeat the same true_task/disguise_as combination more than twice

Output a JSON array of objects. Each object must have exactly these fields:
  "true_task"   : one of the 5 category names above
  "disguise_as" : category the query superficially resembles (different from true_task)
  "situation"   : 3-6 word technical scenario label (e.g. "connection pool exhaustion")
  "query"       : 1-2 sentence confusor query text

Output valid JSON only. No explanation, no markdown fences.
"""

_ROLE_MAP = {t: r for t, r in TASKS.items()}
_VALID_TASKS = set(_ROLE_MAP)


def _generate_llm_confusors(n: int, client: Any, model: str) -> List[Dict]:
    prompt = _GENERATION_PROMPT.format(n=n)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.9,
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:] if lines[0].startswith("```") else lines)
    if raw.endswith("```"):
        raw = "\n".join(raw.split("\n")[:-1])

    parsed = json.loads(raw.strip())
    valid = []
    for c in parsed:
        true_task = c.get("true_task", "").strip()
        if true_task not in _VALID_TASKS:
            continue
        valid.append({
            "task_type": true_task,
            "role": _ROLE_MAP[true_task],
            "situation": c.get("situation", "llm_generated_confusor").strip(),
            "query": c.get("query", "").strip(),
            "disguise_as": c.get("disguise_as", "unknown"),
            "source": "llm_generated",
        })
    return valid


# ── Retrieval + scoring ───────────────────────────────────────────────────────

def _retrieve_tiered(system: Any, c: Dict) -> List[Dict]:
    result = system.step(
        task_type=c["task_type"],
        role=c["role"],
        situation=c["situation"],
        user_query=c["query"],
        confidence=0.5,
        retry_count=0,
        latency_budget_ms=200,
        token_budget=2000,
    )
    return result.get("abstract_hits", [])


def _retrieve_flat(flat: Any, c: Dict) -> List[Dict]:
    ctx = RetrievalContext(
        task_type=c["task_type"],
        agent_role=c["role"],
        situation=c["situation"],
        query_text=c["query"],
        confidence=0.5,
        retry_count=0,
        latency_budget_ms=200,
        token_budget=2000,
    )
    return flat.retrieve(ctx).get("abstract_hits", [])


def _score(hits: List[Dict], true_task: str) -> Dict[str, Any]:
    if not hits:
        return {"top1_correct": False, "top3_correct": False, "top1_task": None}
    tasks = [h.get("task_type") for h in hits]
    return {
        "top1_correct": tasks[0] == true_task,
        "top3_correct": true_task in tasks[:3],
        "top1_task": tasks[0],
    }


def _eval_set(
    confusors: List[Dict],
    system: Any,
    flat: Any,
) -> Tuple[List[Dict], List[Dict]]:
    tiered_scores, flat_scores = [], []
    for c in confusors:
        t_hits = _retrieve_tiered(system, c)
        f_hits = _retrieve_flat(flat, c)
        tiered_scores.append({**c, **_score(t_hits, c["task_type"])})
        flat_scores.append({**c, **_score(f_hits, c["task_type"])})
    return tiered_scores, flat_scores


# ── Reporting ─────────────────────────────────────────────────────────────────

def _summarize_set(label: str, tiered: List[Dict], flat: List[Dict]) -> Dict:
    n = len(tiered)
    t1 = sum(1 for r in tiered if r["top1_correct"])
    f1 = sum(1 for r in flat  if r["top1_correct"])
    adv = round((t1 - f1) / max(n, 1) * 100, 1)
    return {
        "label": label,
        "n": n,
        "tiered_top1_pct": round(100 * t1 / n, 1),
        "flat_top1_pct":   round(100 * f1 / n, 1),
        "advantage_pp":    adv,
    }


def _print_report(hw: Dict, llm: Dict) -> None:
    print(f"\n{'='*66}")
    print("  CONFUSOR ROBUSTNESS VALIDATION")
    print(f"{'='*66}")
    print(f"  {'Set':<36} {'Tiered':>8} {'Flat':>8} {'Adv (pp)':>10}")
    print(f"  {'-'*62}")
    for s in (hw, llm):
        print(
            f"  {s['label']:<36} {s['tiered_top1_pct']:>7}% "
            f"{s['flat_top1_pct']:>7}% {s['advantage_pp']:>+10}"
        )
    gap = round(abs(hw["advantage_pp"] - llm["advantage_pp"]), 1)
    print(f"\n  Advantage gap between sets: {gap}pp")
    if gap <= 10:
        verdict = "PASS — advantage is stable across independent confusors"
    else:
        verdict = "REVIEW — advantage differs more than 10pp between sets"
    print(f"  Verdict: {verdict}")
    print(f"{'='*66}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_validation(
    n_confusors: int,
    episodes_per_task: int,
    seed: int,
    api_key: str,
    model: str,
    db_path: str,
    output_path: Optional[str],
) -> None:
    import openai as openai_lib
    client = openai_lib.OpenAI(api_key=api_key)
    embedder = Embedder(dim=512)

    for suffix in ("", "-wal", "-shm"):
        p = db_path + suffix
        if os.path.exists(p):
            os.remove(p)

    rng = random.Random(seed)
    system = _build_system(db_path, embedder=embedder)
    print(f"[seeding] {episodes_per_task} episodes × {len(TASKS)} tasks...")
    _seed_episodes(system, episodes_per_task, rng)
    store = MemoryStore(db_path)
    _seed_canonical_situations(store, embedder)

    flat = FlatRetrievalPolicy(store, embedder, topk=3)
    print("[seeding] done\n")

    # Hand-written confusors from the original benchmark
    hw_confusors = [
        {
            "task_type": c["task_type"],
            "role": c["role"],
            "situation": c["situation"],
            "query": c["query"],
            "disguise_as": c.get("confusor_type", "unknown"),
            "source": "hand_written",
        }
        for c in CROSS_TASK_CONFUSORS
    ]
    print(f"[hand-written] {len(hw_confusors)} confusors from original benchmark")

    # LLM-generated confusors — prompt contains no system architecture details
    print(f"[llm] Generating {n_confusors} independent confusors via {model}...")
    llm_confusors = _generate_llm_confusors(n_confusors, client, model)
    print(f"[llm] {len(llm_confusors)} valid confusors received\n")

    print("[eval] Hand-written set...")
    hw_tiered, hw_flat = _eval_set(hw_confusors, system, flat)

    print("[eval] LLM-generated set...")
    llm_tiered, llm_flat = _eval_set(llm_confusors, system, flat)

    hw_summary  = _summarize_set("Hand-written (original benchmark)", hw_tiered, hw_flat)
    llm_summary = _summarize_set("LLM-generated (independent)",       llm_tiered, llm_flat)
    _print_report(hw_summary, llm_summary)

    if output_path:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        ts  = time.strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(os.path.basename(output_path))
        out = os.path.join(results_dir, f"{base}_{ts}{ext}")
        report = {
            "config": {
                "n_confusors_requested": n_confusors,
                "n_confusors_generated": len(llm_confusors),
                "episodes_per_task": episodes_per_task,
                "seed": seed,
                "model": model,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "note": (
                    "LLM generation prompt describes only task categories, "
                    "not the role-partitioning architecture under test."
                ),
            },
            "summary": {
                "hand_written": hw_summary,
                "llm_generated": llm_summary,
            },
            "hand_written_detail": {"tiered": hw_tiered, "flat": hw_flat},
            "llm_generated_detail": {"tiered": llm_tiered, "flat": llm_flat},
            "llm_confusors_text": llm_confusors,
        }
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[output] Saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate confusor robustness with independently-generated queries."
    )
    parser.add_argument(
        "--n-confusors", type=int, default=40,
        help="LLM-generated confusors to produce (default 40)",
    )
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--openai-key", default=None)
    parser.add_argument("--db-path", default="confusor_val.db")
    parser.add_argument("--output", default="confusor_validation.json")
    args = parser.parse_args()

    key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass --openai-key.")

    run_validation(
        n_confusors=args.n_confusors,
        episodes_per_task=args.episodes_per_task,
        seed=args.seed,
        api_key=key,
        model=args.model,
        db_path=args.db_path,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
