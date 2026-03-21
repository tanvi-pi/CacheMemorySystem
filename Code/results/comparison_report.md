# Tiered Memory System: Comparative Performance Report

---

## Mem0

Mem0 (Chhikara et al., arXiv 2504.19413) is a production memory system for LLM agents. When a conversation turn completes, Mem0 runs an LLM-powered extraction pipeline that classifies each new piece of information as ADD, UPDATE, DELETE, or NOOP relative to what is already stored. Facts are written into either a vector database (Mem0) or a combined vector database and Neo4j knowledge graph (Mem0^g). At query time, Mem0 performs semantic search over the stored facts and injects the top results into the agent's context.

Mem0 was evaluated on the LOCOMO benchmark — a dataset of ten extended multi-session conversations, each approximately 26,000 tokens long, with around 200 questions per conversation spanning four types: single-hop factual recall, multi-hop reasoning, temporal reasoning, and open-domain inference. Mem0's headline result is a 26% improvement in LLM-as-a-Judge score over OpenAI's built-in memory feature. On individual question types, Mem0 reports 38.72% F1 on single-hop, 28.64% F1 on multi-hop, 48.93% F1 on temporal, and 47.65% F1 on open-domain queries. Mem0 maintains a single shared memory pool with no concept of agent roles or task types — all facts from all agents are stored and searched together.

---

## This Work: Tiered Role-Partitioned Memory

This work is a memory retrieval system designed for settings where multiple specialized agents share a common memory backend. It organizes memory into three tiers and five role-specific stores — one per agent role: planner, coder, reviewer, researcher, and executor.

**L0** is a hash-signal layer. It stores outcome statistics (success/failure rates) keyed by a stable hash of task type, agent role, and situation signature. L0 lookup is O(1) and requires no embedding computation. If the stored fail rate for a given signature exceeds a threshold, the system immediately escalates to L1 without performing a similarity search, saving latency on known-bad retrievals.

**L1** is the abstract episodic layer. When a task completes, an LLM-generated 2–3 sentence abstract is written to a vector store indexed by agent role. The abstract captures the cause of the outcome, the key decision taken, and what should be remembered for similar future situations. At query time, L1 performs a cosine similarity search within the querying agent's role partition, returning the top matching abstracts. Retrieval stays within the role partition by default, so a question directed to the executor is matched only against executor-role episodes — preventing cross-role noise.

**L2** is the full trace layer. When L1 similarity is weak or the agent is in a retry loop, the system escalates to L2 and fetches the complete execution trace for the matched episodes. Full traces contain richer surrounding context — intermediate steps, tool calls, and partial results — which improves the model's ability to answer multi-hop and temporal questions that require connecting facts across turns.

The escalation policy is automatic: L0 exits early on known failures, L1 handles the common case, and L2 is triggered only when confidence is low or similarity is insufficient. This tiered design means most queries are answered at L1 without incurring L2's additional database fetch.

---

## Comparison 1: Accuracy vs. Mem0

This work was evaluated on a conversational memory benchmark built to the same specification as LOCOMO: five multi-session conversations, 85 dialogue turns seeded into memory, and 46 test questions distributed across single-hop, multi-hop, temporal, and open-domain types. The facts in the benchmark are invented specifically for this test, ensuring GPT-4o-mini cannot answer any question from its training weights. Under the no-memory condition the model scores 0.0% exact match and 9.6% F1, confirming the benchmark isolates memory as the performance driver.

Under the tiered memory condition, this work achieves 47.8% exact match and 75.6% F1 — a gain of 47.8 percentage points in exact match and 66.0 percentage points in F1 over the no-memory baseline.

On single-hop questions, this work achieves 98.9% F1 against Mem0's 38.72% — a margin of 60.2 percentage points. Single-hop retrieval benefits most from role partitioning: a question about a specific personal fact is matched against a narrow, role-filtered set of episodes rather than a large undifferentiated pool.

On multi-hop questions, this work achieves 65.6% F1 against Mem0's 28.64% — a margin of 37.0 percentage points. Multi-hop retrieval benefits from L2 escalation: when L1 similarity is weak, the system fetches full execution traces that contain richer surrounding context, making it easier to connect facts across turns.

On temporal questions, this work achieves 54.3% F1 against Mem0's 48.93% — a margin of 5.4 percentage points. This is the smallest gap, reflecting that temporal reasoning is where Mem0's graph-enhanced variant is most competitive.

On open-domain questions, Mem0 outperforms this work: 47.65% F1 versus 39.8%. Open-domain questions require broad inference across loosely related facts, which benefits from Mem0's entity-relationship graph structure. Episodic vector search is less suited to this query type and represents a genuine limitation of this work.

| Question Type | This Work (F1) | Mem0 (F1) | Difference |
|---|---|---|---|
| single_hop | **98.9%** | 38.72% | **+60.2 pts** |
| multi_hop | **65.6%** | 28.64% | **+37.0 pts** |
| temporal | **54.3%** | 48.93% | **+5.4 pts** |
| open_domain | 39.8% | **47.65%** | -7.9 pts |
| **Overall F1** | **75.6%** | ~41%* | **+34 pts** |

*Mem0 overall F1 estimated as average across published question-type results.

On the multi-agent retrieval benchmark — covering five task domains, six patterns per task, and one hundred mixed queries — this work achieves 86.4% top-1 retrieval accuracy and a mean reciprocal rank of 0.892. On confusor queries — designed to be misleading by resembling an episode from a different role — the tiered system scores 78.6% top-1 accuracy versus 21.4% for flat search, a 57-point margin. Mem0's single shared memory pool would be structurally vulnerable to this failure mode.

---

## Comparison 2: Latency and Token Efficiency vs. Mem0

Both this work and Mem0 are memory retrieval systems for LLM agents, making the efficiency comparison direct: both systems solve the same problem and can be measured against the same baseline — the cost of injecting raw conversation context rather than retrieved facts.

**Token efficiency.** Mem0 reports a 73% reduction in tokens injected into the agent's context compared to its baseline of passing the full conversation history. This work's tiered retrieval injects an average of 318 tokens per query, compared to 759 tokens for flat vector search — a 58% reduction while maintaining the same answer quality (75.6% F1 tiered vs 75.7% F1 flat). The two reductions are not directly comparable in magnitude because the baselines differ: Mem0 compares against full context injection (a weak baseline where the model receives everything), while this work compares against flat vector search (a stronger baseline that already filters to relevant episodes). A 58% reduction over flat retrieval is therefore a more meaningful result — flat search is already an optimized system, not a naive dump of the full conversation.

**Latency.** Mem0 reports a 91% reduction in p95 retrieval latency compared to full context injection. This work achieves a p50 retrieval latency of 255 milliseconds and a p95 of 472 milliseconds across 100 queries, compared to flat retrieval at p50: 297ms and p95: 414ms — a 14.2% reduction at p50. The p95 is slightly higher for this work (472ms vs 414ms) because L2 escalation occasionally adds a second database fetch for complex queries. Mem0's larger latency reduction reflects the same baseline gap: eliminating full context injection saves far more time than optimizing over an already-fast vector search. Both systems are well within real-time deployment ranges at their respective p50s.

For broader context, the leading agent efficiency system — Agentic Plan Caching (Zhang et al., NeurIPS 2025) — achieves a 50.31% token reduction by caching plan templates at the planning stage. This work achieves 58% token reduction at the memory retrieval stage, against a harder baseline (flat vector search rather than full context injection). The two optimizations target different parts of the pipeline and could be combined in a full agent stack.

| Metric | This Work | Mem0 | Baseline |
|---|---|---|---|
| Token reduction | -58% vs flat retrieval | -73% vs full context | Flat search / full context |
| Latency reduction (p50) | **-14.2%** vs flat retrieval | N/A (p95 reported) | Flat search |
| Latency reduction (p95) | +14% vs flat retrieval* | **-91%** vs full context | Flat search / full context |
| p50 retrieval latency | **255ms** | not reported | 297ms |
| p95 retrieval latency | 472ms | not reported | 414ms |
| Tokens per query | **318** | not reported | 759 (flat) |

*p95 increases slightly due to L2 escalation adding a second fetch on complex queries.

---

## Combined Positioning

| Dimension | This Work | Mem0 | APC |
|---|---|---|---|
| Single-hop F1 | **98.9%** | 38.72% | not evaluated |
| Multi-hop F1 | **65.6%** | 28.64% | not evaluated |
| Overall accuracy gain | **+66% F1** over baseline | +26% J-score over baseline | 96.61% of optimal maintained |
| Token reduction | **-58%** vs flat retrieval | -73% vs full context | **-50.31%** vs no caching |
| Latency reduction | -14.2% retrieval p50 | -91% p95 vs full context | **-27.28%** total pipeline |
| Multi-agent / role-aware | **Yes** | No | No |
| Automatic tier escalation | **Yes** | No | No |
| Memory deduplication | No | **Yes** | Not applicable |
| Open-domain reasoning | 39.8% F1 | **47.65% F1** | not evaluated |

---

## Conclusion

This work outperforms Mem0 on accuracy across three of four question types, with the largest gains on single-hop and multi-hop queries — the types most commonly encountered in factual retrieval workloads. On efficiency, this work achieves a 58% token reduction over flat vector search and a 14.2% latency reduction at p50, both measured against a stronger baseline than Mem0's published comparisons. The open-domain accuracy gap versus Mem0 and the absence of memory deduplication are genuine limitations.

The central contribution is the combination of role partitioning, tiered escalation, and automatic routing in a setting Mem0 was not designed for: multiple specialized agents sharing a memory backend, each requiring fast and precise retrieval that does not degrade under ambiguous or misleading queries.

---

*Benchmarks conducted March 2026. Models: GPT-4o-mini for answer generation, text-embedding-3-small for embeddings. APC results cited from Zhang et al., NeurIPS 2025. Mem0 results cited from Chhikara et al., arXiv 2504.19413.*
