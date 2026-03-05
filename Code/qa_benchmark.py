"""
qa_benchmark.py — HotpotQA-based benchmark comparing 4 memory conditions.

This tests whether your tiered memory system produces BETTER answers than
competing approaches, using a real public dataset with ground-truth answers.

Conditions compared:
  1. no_memory      — GPT answers the question with zero memory (pure baseline)
  2. buffer_memory  — Last K stored episodes regardless of relevance
                      (simulates LangChain ConversationBufferMemory)
  3. flat_memory    — Vector search across all agents, no role/task filter
                      (simulates a standard RAG/vector-DB approach)
  4. tiered_memory  — Your L0/L1/L2 tiered system (the system under test)

Metrics (standard HotpotQA evaluation):
  - Exact Match (EM) %  : predicted answer == gold answer after normalization
  - F1 %                : token-level overlap between prediction and gold
  - Avg tokens used     : total tokens consumed per query (cost proxy)
  - Avg latency ms      : end-to-end time per query

Dataset: HotpotQA distractor dev split (public, no auth required)
Install:  pip install datasets openai

Run:
  python3 qa_benchmark.py --n-seed 100 --n-test 50 --output qa_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from Embedder import OpenAIEmbedder
from MemoryStore import MemoryStore
from MultiAgentSystem import MultiAgentSystem
from RetrievalContext import RetrievalContext
from TieredRetrievalPolicy import FlatRetrievalPolicy


# ── Dataset loading ───────────────────────────────────────────────────────────

def _load_hotpotqa(n_seed: int, n_test: int) -> Tuple[List[Dict], List[Dict]]:
    """
    Download HotpotQA dev set via HuggingFace datasets library.
    Results are cached locally so subsequent runs are instant.
    """
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hotpotqa_cache")
    cache_file = os.path.join(cache_dir, "hotpotqa_dev.json")

    if os.path.exists(cache_file):
        print(f"[dataset] Loading HotpotQA from cache ({cache_file})")
        with open(cache_file) as f:
            data = json.load(f)
    else:
        try:
            from datasets import load_dataset
        except ImportError:
            print("ERROR: The 'datasets' library is required.")
            print("       Run:  pip install datasets")
            sys.exit(1)

        print("[dataset] Downloading HotpotQA dev set from HuggingFace (one-time, ~50 MB)...")
        ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
        data = [dict(item) for item in ds]
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(data, f)
        print(f"[dataset] Cached {len(data)} questions to {cache_file}")

    total_needed = n_seed + n_test
    if len(data) < total_needed:
        print(f"WARNING: Only {len(data)} questions available, using all of them.")
        n_seed = int(len(data) * 0.66)
        n_test = len(data) - n_seed

    # Stratified split: take proportionally from each question type so both
    # seed and test sets cover all difficulty/type combinations
    by_type: Dict[str, List[Dict]] = {}
    for q in data:
        key = f"{q.get('type', 'bridge')}_{q.get('level', 'medium')}"
        by_type.setdefault(key, []).append(q)

    seed_qs: List[Dict] = []
    test_qs: List[Dict] = []
    for questions in by_type.values():
        n = len(questions)
        n_s = max(1, round(n * n_seed / total_needed))
        n_t = max(1, round(n * n_test / total_needed))
        seed_qs.extend(questions[:n_s])
        test_qs.extend(questions[n_s: n_s + n_t])

    return seed_qs[:n_seed], test_qs[:n_test]


# ── Question → agent mapping ──────────────────────────────────────────────────

# Map HotpotQA question properties to task_type and agent_role.
# bridge   = multi-hop factual lookup (researcher traces evidence chains)
# comparison = comparative analysis (reviewer compares two entities)
_QTYPE_MAP: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("bridge",     "easy"):   ("fact_retrieval",        "executor"),
    ("bridge",     "medium"): ("multi_hop_research",    "planner"),
    ("bridge",     "hard"):   ("complex_investigation", "researcher"),
    ("comparison", "easy"):   ("entity_comparison",     "reviewer"),
    ("comparison", "medium"): ("entity_comparison",     "reviewer"),
    ("comparison", "hard"):   ("deep_comparison",       "reviewer"),
}

_TASK_ROLE_MAP = {
    "fact_retrieval":        "executor",
    "multi_hop_research":    "planner",
    "complex_investigation": "researcher",
    "entity_comparison":     "reviewer",
    "deep_comparison":       "reviewer",
}


def _classify(q: Dict) -> Tuple[str, str, str]:
    """Return (task_type, agent_role, situation_signature) for a question."""
    qtype = q.get("type", "bridge")
    level = q.get("level", "medium")
    task_type, role = _QTYPE_MAP.get((qtype, level), ("multi_hop_research", "researcher"))
    situation_sig = f"{qtype}_{level}"
    return task_type, role, situation_sig


# ── Answer evaluation (standard HotpotQA metrics) ────────────────────────────

def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_toks = _normalize(pred).split()
    gold_toks = _normalize(gold).split()
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if not num_same:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


# ── Memory seeding ────────────────────────────────────────────────────────────

def _seed_memory(system: MultiAgentSystem, seed_qs: List[Dict]) -> None:
    """
    Store each seed Q&A as an episodic memory.
    The agent remembers: what type of question this was, what the answer was,
    and what reasoning strategy led to the answer.
    """
    print(f"[seeding] Storing {len(seed_qs)} HotpotQA Q&A pairs as agent episodes...")
    for i, q in enumerate(seed_qs):
        task_type, role, situation_sig = _classify(q)
        question = q["question"]
        gold_answer = q["answer"]
        qtype = q.get("type", "bridge")
        level = q.get("level", "medium")

        # Extract topic titles from supporting facts
        sf = q.get("supporting_facts", {})
        if isinstance(sf, dict):
            topics = list(set(sf.get("title", [])))[:3]
        elif isinstance(sf, list):
            topics = list({s[0] for s in sf})[:3]
        else:
            topics = []
        topic_str = ", ".join(topics) if topics else "multiple sources"

        abstract = (
            f"Successfully answered a {level}-difficulty {qtype} question "
            f"involving evidence from: {topic_str}. "
            f"The question was: '{question[:120]}'. "
            f"The correct answer is '{gold_answer}'. "
            f"Strategy: cross-reference multiple sources and identify the bridging entity."
        )
        full_trace = (
            f"[TASK] {task_type} | [ROLE] {role} | [LEVEL] {level} | [TYPE] {qtype}\n"
            f"[QUESTION] {question}\n"
            f"[SUPPORTING TOPICS] {topic_str}\n"
            f"[GOLD ANSWER] {gold_answer}\n"
            f"[OUTCOME] success\n"
            f"[STRATEGY] Traced evidence through supporting documents, "
            f"identified key bridging facts, arrived at answer '{gold_answer}'.\n"
        )

        system.finalize_episode(
            role=role,
            episode_id=f"hotpot_seed_{i}",
            task_type=task_type,
            outcome="success",
            abstract=abstract,
            full_trace=full_trace,
            situation_signature=situation_sig,
            cost_tokens=len(question.split()) * 2,
            cost_latency_ms=80,
        )

    print(f"[seeding] Done — {len(seed_qs)} episodes stored.\n")


# ── LLM answer generation ─────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a precise question-answering agent. "
    "Answer questions as concisely as possible — typically 1 to 5 words for factual questions. "
    "Do not explain your reasoning. Output only the answer."
)


def _call_gpt(
    question: str,
    memory_context: str,
    client: Any,
    model: str,
) -> Tuple[str, int, int]:
    """
    Call GPT to answer a question.
    Returns (answer, total_tokens, latency_ms).
    """
    if memory_context:
        user_content = (
            f"You have the following relevant past experiences from memory:\n\n"
            f"{memory_context}\n\n"
            f"Using these experiences to inform your reasoning, answer this question "
            f"in 1-5 words:\n"
            f"Question: {question}"
        )
    else:
        user_content = (
            f"Answer this question in 1-5 words:\n"
            f"Question: {question}"
        )

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=30,
        temperature=0,
    )
    latency_ms = int((time.time() - t0) * 1000)
    answer = resp.choices[0].message.content.strip()
    tokens = resp.usage.total_tokens
    return answer, tokens, latency_ms


def _format_hits(hits: List[Dict], max_chars: int = 300) -> str:
    """Format retrieved memory hits into a readable context string."""
    lines = []
    for i, h in enumerate(hits[:3], 1):
        text = h.get("abstract", h.get("full_trace", ""))[:max_chars]
        lines.append(f"[Memory {i}]: {text}")
    return "\n\n".join(lines)


# ── The 4 conditions ──────────────────────────────────────────────────────────

def _condition_no_memory(
    test_qs: List[Dict],
    client: Any,
    model: str,
) -> List[Dict]:
    """Condition 1: No memory — GPT answers each question cold."""
    results = []
    for q in test_qs:
        answer, tokens, latency = _call_gpt(q["question"], "", client, model)
        results.append({
            "question": q["question"],
            "gold": q["answer"],
            "predicted": answer,
            "em": exact_match(answer, q["answer"]),
            "f1": f1_score(answer, q["answer"]),
            "tokens": tokens,
            "latency_ms": latency,
        })
    return results


def _condition_buffer_memory(
    test_qs: List[Dict],
    store: MemoryStore,
    client: Any,
    model: str,
    buffer_k: int = 8,
) -> List[Dict]:
    """
    Condition 2: Buffer memory — last K stored episodes, no search.
    Simulates LangChain ConversationBufferMemory: always uses the most
    recent memories regardless of whether they are relevant to the question.
    """
    cur = store.conn.execute(
        "SELECT abstract FROM episodes ORDER BY created_at_ms DESC LIMIT ?",
        (buffer_k,),
    )
    buffer = [r[0][:250] for r in cur.fetchall()]
    memory_context = "\n\n".join(f"[Memory {i+1}]: {a}" for i, a in enumerate(buffer))

    results = []
    for q in test_qs:
        answer, tokens, latency = _call_gpt(q["question"], memory_context, client, model)
        results.append({
            "question": q["question"],
            "gold": q["answer"],
            "predicted": answer,
            "em": exact_match(answer, q["answer"]),
            "f1": f1_score(answer, q["answer"]),
            "tokens": tokens,
            "latency_ms": latency,
        })
    return results


def _condition_flat_memory(
    test_qs: List[Dict],
    store: MemoryStore,
    embedder: Any,
    client: Any,
    model: str,
) -> List[Dict]:
    """
    Condition 3: Flat memory — vector search across all agents with no
    role or task-type filtering. Simulates a standard RAG / vector-DB approach.
    """
    flat_policy = FlatRetrievalPolicy(store, embedder, topk=3)
    results = []
    for q in test_qs:
        task_type, role, situation_sig = _classify(q)
        ctx = RetrievalContext(
            task_type=task_type,
            agent_role=role,
            situation=situation_sig,
            query_text=q["question"],
            confidence=0.5,
            retry_count=0,
            latency_budget_ms=300,
            token_budget=1200,
        )
        flat_result = flat_policy.retrieve(ctx)
        hits = flat_result.get("abstract_hits", [])
        memory_context = _format_hits(hits)
        answer, tokens, latency = _call_gpt(q["question"], memory_context, client, model)
        retrieval_tokens = flat_result["debug"].get("retrieval_tokens", 0)
        retrieval_ms = flat_result["debug"].get("latency_ms", 0)
        results.append({
            "question": q["question"],
            "gold": q["answer"],
            "predicted": answer,
            "em": exact_match(answer, q["answer"]),
            "f1": f1_score(answer, q["answer"]),
            "tokens": tokens + retrieval_tokens,
            "latency_ms": latency + retrieval_ms,
            "hits_returned": len(hits),
        })
    return results


def _condition_tiered_memory(
    test_qs: List[Dict],
    system: MultiAgentSystem,
    client: Any,
    model: str,
) -> List[Dict]:
    """
    Condition 4: Tiered memory — your full L0/L1/L2 system.
    Role-filtered vector search with L0 fast-path for known failure patterns.
    """
    results = []
    tier_counts: Dict[str, int] = {}
    for q in test_qs:
        task_type, role, situation_sig = _classify(q)
        tiered = system.step(
            task_type=task_type,
            role=role,
            situation=situation_sig,
            user_query=q["question"],
            confidence=0.5,
            retry_count=0,
            latency_budget_ms=300,
            token_budget=1200,
        )
        tier = tiered.get("tier", "L1")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        hits = tiered.get("abstract_hits", []) + tiered.get("full_hits", [])
        memory_context = _format_hits(hits)
        answer, tokens, latency = _call_gpt(q["question"], memory_context, client, model)
        retrieval_tokens = tiered["debug"].get("retrieval_tokens", 0)
        retrieval_ms = tiered["debug"].get("latency_ms", 0)
        results.append({
            "question": q["question"],
            "gold": q["answer"],
            "predicted": answer,
            "em": exact_match(answer, q["answer"]),
            "f1": f1_score(answer, q["answer"]),
            "tokens": tokens + retrieval_tokens,
            "latency_ms": latency + retrieval_ms,
            "tier_used": tier,
            "hits_returned": len(hits),
        })

    print(f"         Tier breakdown: {tier_counts}")
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def _summarize(name: str, results: List[Dict]) -> Dict:
    n = len(results) or 1
    return {
        "condition": name,
        "n_questions": n,
        "exact_match_pct": round(100 * sum(r["em"] for r in results) / n, 1),
        "f1_pct": round(100 * sum(r["f1"] for r in results) / n, 1),
        "avg_tokens": round(sum(r["tokens"] for r in results) / n, 1),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / n, 1),
    }


def _print_report(summaries: List[Dict]) -> None:
    print(f"\n{'='*68}")
    print("  HOTPOTQA COMPARATIVE BENCHMARK — MEMORY SYSTEM COMPARISON")
    print(f"{'='*68}")
    print(f"  {'Condition':<22} {'EM %':>7} {'F1 %':>7} {'Tokens':>9} {'Latency':>11}")
    print(f"  {'-'*58}")
    for s in summaries:
        print(
            f"  {s['condition']:<22} {s['exact_match_pct']:>7} {s['f1_pct']:>7} "
            f"{s['avg_tokens']:>9} {s['avg_latency_ms']:>10}ms"
        )

    baseline = summaries[0]  # no_memory
    print(f"\n  Improvement over no_memory baseline:")
    print(f"  {'-'*58}")
    for s in summaries[1:]:
        em_d = round(s["exact_match_pct"] - baseline["exact_match_pct"], 1)
        f1_d = round(s["f1_pct"] - baseline["f1_pct"], 1)
        tok_d = round(
            (s["avg_tokens"] - baseline["avg_tokens"]) / max(baseline["avg_tokens"], 1) * 100, 1
        )
        em_sign = "+" if em_d >= 0 else ""
        f1_sign = "+" if f1_d >= 0 else ""
        tok_sign = "+" if tok_d >= 0 else ""
        print(
            f"  {s['condition']:<22}  EM: {em_sign}{em_d}%   "
            f"F1: {f1_sign}{f1_d}%   Tokens: {tok_sign}{tok_d}%"
        )

    print(f"\n  Published HotpotQA baselines (for context):")
    print(f"  {'GPT-4o zero-shot':<22} ~55-65% F1  (no retrieval, no memory)")
    print(f"  {'GPT-4o-mini zero-shot':<22} ~40-50% F1  (no retrieval, no memory)")
    print(f"  {'RAG + GPT-4':<22} ~65-75% F1  (full document retrieval)")
    print(f"{'='*68}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_qa_benchmark(
    n_seed: int,
    n_test: int,
    openai_key: Optional[str],
    model: str,
    output: Optional[str],
    db_path: str,
) -> None:
    import openai as openai_lib

    key = openai_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("ERROR: OpenAI API key required. Set OPENAI_API_KEY or pass --openai-key")
        sys.exit(1)

    client = openai_lib.OpenAI(api_key=key)

    # Clean stale DB files
    for suffix in ("", "-wal", "-shm"):
        p = db_path + suffix
        if os.path.exists(p):
            os.remove(p)

    # Load dataset
    seed_qs, test_qs = _load_hotpotqa(n_seed, n_test)
    print(f"[dataset] {len(seed_qs)} seed questions | {len(test_qs)} test questions")

    # Build embedder and system
    embedder = OpenAIEmbedder(api_key=key)
    print(f"[embedder] {embedder.model} (dim={embedder.dim})")

    roles = ["planner", "coder", "reviewer", "researcher", "executor"]
    system = MultiAgentSystem(
        roles=roles,
        task_role_map=_TASK_ROLE_MAP,
        db_path=db_path,
        embedder=embedder,
    )
    store = MemoryStore(db_path)

    # Seed memory
    _seed_memory(system, seed_qs)

    # Estimate API cost
    est_calls = n_test * 4  # 4 conditions × n_test questions
    est_cost = round(est_calls * 0.001, 2)  # rough gpt-4o-mini estimate
    print(f"[cost] Estimated ~{est_calls} GPT calls, ~${est_cost} at gpt-4o-mini rates")
    print(f"[testing] Running 4 conditions on {len(test_qs)} questions...\n")

    # Run conditions
    print("[1/4] no_memory       — GPT answers cold, no retrieval")
    r_none = _condition_no_memory(test_qs, client, model)

    print("[2/4] buffer_memory   — Last 8 episodes, no search (LangChain-style)")
    r_buf = _condition_buffer_memory(test_qs, store, client, model)

    print("[3/4] flat_memory     — Vector search, no role/task filter (standard RAG)")
    r_flat = _condition_flat_memory(test_qs, store, embedder, client, model)

    print("[4/4] tiered_memory   — L0/L1/L2 tiered system (your system)")
    r_tiered = _condition_tiered_memory(test_qs, system, client, model)

    # Build summaries and print report
    summaries = [
        _summarize("no_memory",     r_none),
        _summarize("buffer_memory", r_buf),
        _summarize("flat_memory",   r_flat),
        _summarize("tiered_memory", r_tiered),
    ]
    _print_report(summaries)

    # Save to file
    if output:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(os.path.basename(output))
        out_path = os.path.join(results_dir, f"{base}_{ts}{ext}")

        report = {
            "config": {
                "dataset": "HotpotQA distractor dev",
                "n_seed": len(seed_qs),
                "n_test": len(test_qs),
                "model": model,
                "embedder": embedder.model,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "summary": summaries,
            "published_baselines": {
                "gpt4o_zero_shot_f1": "55-65%",
                "gpt4o_mini_zero_shot_f1": "40-50%",
                "rag_gpt4_f1": "65-75%",
                "source": "HotpotQA leaderboard + published papers",
            },
            "per_question": {
                "no_memory":     r_none,
                "buffer_memory": r_buf,
                "flat_memory":   r_flat,
                "tiered_memory": r_tiered,
            },
        }
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[output] Results saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HotpotQA benchmark comparing no_memory / buffer / flat / tiered memory systems."
    )
    parser.add_argument(
        "--n-seed", type=int, default=100,
        help="Questions to store in agent memory (default 100)",
    )
    parser.add_argument(
        "--n-test", type=int, default=50,
        help="Questions to test on (default 50)",
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI model for answer generation (default gpt-4o-mini)",
    )
    parser.add_argument("--openai-key", default=None)
    parser.add_argument(
        "--output", default="qa_results.json",
        help="Save results to this file inside results/ folder",
    )
    parser.add_argument("--db-path", default="qa_benchmark.db")
    args = parser.parse_args()

    run_qa_benchmark(
        n_seed=args.n_seed,
        n_test=args.n_test,
        openai_key=args.openai_key,
        model=args.model,
        output=args.output,
        db_path=args.db_path,
    )


if __name__ == "__main__":
    main()
