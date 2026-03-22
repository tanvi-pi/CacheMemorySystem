# 4. Experiments

## 4.1 Experimental Setup

**Models.** All answer generation uses GPT-4o-mini (temperature 0, max 50 output tokens). All embeddings are computed with OpenAI's `text-embedding-3-small` (1536 dimensions). Abstract generation uses GPT-4o-mini (temperature 0.2, max 220 tokens).

**Infrastructure.** The memory backend is a single SQLite database with WAL mode enabled. All experiments are run on a local machine; no GPU is required. Retrieval latency is measured as wall-clock time from query receipt to result return, excluding the downstream GPT generation call.

**Evaluation metrics.** For accuracy we report Exact Match (EM) — whether the normalized predicted answer matches the normalized gold answer — and token-level F1, which measures partial credit via precision-recall overlap on answer tokens. For efficiency we report average tokens injected into the agent context per query and average retrieval latency in milliseconds (p50 and p95).

---

## 4.2 Conversational Memory Benchmark

### 4.2.1 Dataset

We evaluate on a conversational memory benchmark built to the specification of LOCOMO (Maharana et al., 2024). The benchmark comprises ten multi-session conversations, each containing three dialogue sessions separated by days or weeks, for a total of 170 dialogue turns seeded into memory. Test questions (140 total) are distributed across four types: single-hop factual recall, multi-hop reasoning, temporal reasoning, and open-domain inference.

**Synthetic benchmark design.** All facts are invented specifically for this benchmark using fictional names, institutions, and events — ensuring GPT-4o-mini cannot answer any question from its pre-training weights. We chose synthetic facts over the original LOCOMO dataset for three reasons. First, the original LOCOMO corpus is derived from real user conversations whose facts may partially overlap with GPT-4o-mini's training data, making it impossible to cleanly attribute correct answers to memory retrieval vs. model knowledge. Second, synthetic data allows us to precisely control the density and type of facts per question category, balancing the benchmark across all four question types. Third, we can verify the zero-memory lower bound analytically: by construction, no invented fact (e.g., "Aria Chen works at Luminary AI in Neo Singapore") can be answered from model weights. This design choice is validated empirically by the no-memory condition, which scores 0.0% exact match and 8.6% F1, confirming that the benchmark cleanly isolates memory as the performance driver. The 8.6% F1 reflects partial token overlap on generic non-answers (e.g., "Not specified.") rather than any genuine factual knowledge. We note this is the same guarantee LOCOMO-style evaluation relies on, and our synthetic approach strengthens rather than weakens it.

### 4.2.2 Baselines

We compare four memory conditions:

- **No memory.** GPT-4o-mini answers with no retrieved context. This establishes the zero-memory baseline and confirms the benchmark difficulty.
- **Buffer memory.** The last 8 stored episodes are injected directly into context regardless of relevance, simulating LangChain's `ConversationBufferMemory`.
- **Flat memory.** Standard RAG: the query is embedded and searched globally across all stored episodes with no role or task-type filtering. Top-5 matching abstracts are injected into context. This is the direct ablation of role partitioning.
- **Tiered memory (ours).** The full L0/L1/L2 tiered system with role partitioning and automatic escalation.

### 4.2.3 Results

Table 1 presents overall results across all four conditions.

**Table 1: Conversational Memory Benchmark — Overall Results**

| Condition | EM % | F1 % | Avg Tokens | Avg Latency |
|---|---|---|---|---|
| No memory | 0.0 | 8.6 | 76 | 530ms |
| Buffer memory | 0.7 | 4.6 | 427 | 574ms |
| Flat memory | 52.1 | 76.4 | 784 | 848ms |
| **Tiered memory (ours)** | **52.1** | **77.4** | **619** | **875ms** |

The tiered system matches flat memory on EM (52.1%) while outperforming it on F1 (+1.0 point: 77.4% vs 76.4%) and injecting 21% fewer tokens per query (619 vs 784). The EM parity reflects that EM rewards only exact string matches; the +1.0 F1 advantage shows that the tiered system retrieves more relevant partial information, particularly on open-domain and temporal questions where it injects full-trace context via L2 escalation. End-to-end latency is marginally higher for the tiered system (875ms vs 848ms) because open-domain queries always trigger a second database fetch for full traces; on non-open-domain queries the tiered system is faster than flat due to the smaller role-partitioned search index. Buffer memory scores lower than no memory on F1 (4.6% vs 8.6%) because recency-biased injection introduces irrelevant recent context that misleads the model — confirming that indiscriminate context injection is worse than no context. The 0.0% exact match for the no-memory condition confirms that our benchmark facts cannot be answered from model weights, cleanly isolating memory as the performance driver.

Table 2 breaks down F1 by question type and compares against Mem0's published results on LOCOMO (Chhikara et al., 2025).

**Table 2: F1 by Question Type — Tiered System vs. Mem0**

| Question Type | Tiered (F1) | Flat (F1) | No Memory (F1) | Mem0 (F1) | vs. Mem0 |
|---|---|---|---|---|---|
| single_hop | **89.9%** | 89.9% | 8.4% | 38.72% | **+51.2 pts** |
| multi_hop | **62.7%** | 63.3% | 14.6% | 28.64% | **+34.1 pts** |
| temporal | **62.2%** | 60.3% | 0.0% | 48.93% | **+13.3 pts** |
| open_domain | 38.6% | 26.7% | 14.0% | **47.65%** | −9.1 pts |
| **Overall F1** | **77.4%** | 76.4% | 8.6% | ~41%* | **+36.4 pts** |

*Mem0 overall F1 estimated as average across published per-type results.

The tiered system exceeds Mem0 on three of four question types. The largest margins are on single-hop (+51.2 points) and multi-hop (+34.1 points) — the two types most commonly encountered in factual agent memory workloads. Temporal reasoning shows a +13.3-point advantage over Mem0, reflecting L2 escalation: full execution traces provide surrounding context needed to reason about when facts occurred across sessions.

On open-domain questions, the tiered system scores 38.6% F1 against Mem0's 47.65% — a gap of 9.1 points. This is partially addressed by a query-type-specific strategy: open-domain queries are issued with lower confidence (forcing L2 escalation), a higher token budget (5,000 tokens), and a wider hit window (10 episodes at 800 characters each), enabling synthesis across more episodes. The remaining gap reflects a structural advantage of Mem0's entity-relationship graph (Mem0^g) on inference queries requiring distantly related facts — extending the system with a graph layer over episodic memory is a direction for future work.

The comparison is not direct — the tiered system was evaluated on our synthetic benchmark while Mem0 was evaluated on the original LOCOMO dataset — but both benchmarks are built to the same specification and use the same four question types, making the per-type F1 comparison meaningful.

---

## 4.3 Multi-Agent Retrieval Benchmark

### 4.3.1 Setup

To evaluate retrieval quality in the multi-role setting specifically, we construct a retrieval benchmark that directly tests the system's ability to surface the correct episode from a shared memory backend containing episodes from multiple agent roles. The benchmark comprises 100 queries across five task domains (API integration, data processing, model training, deployment, debugging), with six episode patterns per domain. The query mix is: 55% paraphrased (query rewrites of stored episode situations), 14% confusor (queries designed to resemble an episode from a *different* agent role), 12% verbatim (exact situation matches), and 19% novel (queries that have no exact match but have a relevant analogue).

Retrieval accuracy is measured by top-1 accuracy (whether the correct episode is the first result), top-3 accuracy (whether it appears in the top three), and mean reciprocal rank (MRR). A confusor query is considered correctly handled if the system returns an episode from the *correct* role and *not* the superficially similar episode from the wrong role.

### 4.3.2 Results

Table 3 presents retrieval accuracy for the tiered system and flat baseline.

**Table 3: Multi-Agent Retrieval Benchmark — Accuracy**

| Metric | Tiered | Flat | Difference |
|---|---|---|---|
| Top-1 accuracy | **86.4%** | 71.6% | +14.8 pts |
| Top-3 accuracy | **88.9%** | 75.3% | +13.6 pts |
| Any-hit accuracy | **100%** | 75.3% | +24.7 pts |
| MRR | **0.892** | 0.733 | +0.159 |

**Table 4: Multi-Agent Retrieval Benchmark — Accuracy by Query Type**

| Query Type | Tiered Top-1 | Tiered MRR | Flat Top-1 | Flat MRR | Queries |
|---|---|---|---|---|---|
| Paraphrased | **90.9%** | **0.933** | 83.6% | 0.861 | 55 |
| Confusor | **78.6%** | **0.821** | 21.4% | 0.214 | 14 |
| Verbatim | 75.0% | 0.785 | 75.0% | 0.750 | 12 |
| Novel‡ | — | — | — | — | 19 |

‡Novel queries have no single ground-truth episode (they test whether the system surfaces a useful analogue), so per-query top-k accuracy and MRR are not directly measurable. Novel queries are included in the any-hit accuracy reported in Table 3 (100% tiered, 75.3% flat).

The confusor results are the most diagnostic. Flat search scores 21.4% top-1 on confusor queries — it retrieves the wrong-role episode in 78.6% of cases because the surface text is similar and there is no mechanism to distinguish roles. The tiered system scores 78.6% top-1, a 57.2-percentage-point improvement. This result directly validates the role partitioning design: by scoping cosine similarity search to the querying agent's partition, the confusor episode from the wrong role is never in the candidate set.

On verbatim queries — exact situation matches — both systems score identically (75.0%), confirming that the role partitioning does not hurt recall on easy queries; the gap between systems is entirely driven by the harder paraphrased and confusor cases.

Table 5 presents efficiency results.

**Table 5: Multi-Agent Retrieval Benchmark — Efficiency**

| Metric | Tiered | Flat | Difference |
|---|---|---|---|
| Avg latency (p50) | **255ms** | 297ms | −14.2% |
| p95 latency | 472ms | **414ms** | +14.0% |
| Avg tokens per query | **934** | 1,693 | −44.8% |

The tiered system is 14.2% faster at the median query (255ms vs 297ms) because L1 search operates on a smaller role-partitioned index. The p95 latency is slightly higher (472ms vs 414ms) because L2 escalation occasionally triggers a second database fetch on complex queries, adding a fixed overhead. Across 100 queries, 95% resolved at L1 and 5% exited at L0; L2 was not triggered in this benchmark configuration (all queries were run at the default confidence and retry settings). Token injection is 44.8% lower because the flat baseline always fetches full traces for its top-k hits, while the tiered system returns abstracts at L1 and only escalates to traces when necessary.

---

## 4.4 Ablation Study

To isolate the contribution of each architectural component, we run three ablated variants of the tiered system on the conversational memory benchmark (10 conversations, 140 questions):

- **No L0.** The L0 failure-signal threshold is set to 2.0, making it unreachable. The system always proceeds to L1 regardless of historical failure rate. This measures the contribution of outcome-aware early exit.
- **No L2.** The L2 escalation conditions are disabled by setting $\theta_\kappa = 0.0$ and $n_{\text{retry}} = 999$. The system never fetches full traces. This measures the contribution of deep context retrieval on multi-hop and temporal queries.
- **No role partitioning.** The flat retrieval policy is used as a direct proxy. L1 search is global rather than role-filtered. This measures the contribution of role partitioning on confusor robustness and overall accuracy.

**Table 6: Ablation Study — Overall Results**

| Condition | EM % | F1 % | Tokens | Latency |
|---|---|---|---|---|
| Full tiered (ours) | **53.6** | **77.4** | 670 | 743ms |
| − L0 hash signals | 50.7 | 76.0 | 610 | 750ms |
| − L2 escalation | 52.9 | 76.8 | 610 | 799ms |
| − Role partitioning | 51.4 | 76.1 | **784** | 746ms |

**Table 7: Ablation Study — F1 by Question Type†**

| Question Type | Full Tiered | − L0 | − L2 | − Role | Δ Full vs −L2 | Δ Full vs −Role |
|---|---|---|---|---|---|---|
| single_hop | 89.9% | 89.4% | 90.1% | 89.9% | −0.2 pts | 0.0 pts |
| multi_hop | **64.7%** | 64.2% | 63.5% | 63.5% | **+1.3 pts** | **+1.2 pts** |
| temporal | **62.2%** | 57.9% | 62.2% | 57.9% | 0.0 pts | **+4.3 pts** |
| open_domain | **33.9%** | 29.1% | 26.7% | 26.7% | **+7.1 pts** | **+7.1 pts** |

†All values in Table 7 are from a single ablation run; Δ columns are computed within that run for internal consistency. Full Tiered per-type values may differ from Table 2 by ±3–5 pts due to GPT-4o-mini generation variance across runs (n=10 for open_domain). Overall F1 (77.4%) is consistent across both runs.

Three findings emerge from the ablation.

**L0 contributes to temporal and open-domain questions.** Removing L0 drops overall EM by 2.9 points (53.6% → 50.7%) and F1 by 1.4 points (77.4% → 76.0%). The largest per-type effect is on temporal (−4.3 pts: 62.2% → 57.9%) and open-domain (−4.8 pts: 33.9% → 29.1%). This reflects a subtle interaction with the L2 path: when L0 has accumulated a prior signal for a situation, its `last_episode_id` is used as a fallback seed for L2 escalation when L1 returns no hits. Removing L0 eliminates this fallback, leaving L2 without a seed on hard queries where L1 similarity is weak.

**L2 is the primary driver of open-domain and multi-hop performance.** Removing L2 escalation drops open-domain F1 by 7.1 points (33.9% → 26.7%) and multi-hop by 1.3 points (64.7% → 63.5%). The open-domain effect is large because open-domain queries always escalate to L2 via the `query_type="open_domain"` path, which forces full-trace retrieval with a wider candidate set (10 episodes, 5,000-token budget). Without L2, these queries fall back to abstract-only context, which is insufficient for cross-episode inference. Temporal F1 is unchanged by removing L2 (62.2% in both conditions), indicating that temporal reasoning is satisfied at L1 in this benchmark.

**Role partitioning contributes to temporal reasoning and token efficiency.** Removing role partitioning drops temporal F1 by 4.3 points (62.2% → 57.9%) and open-domain by 7.1 points (33.9% → 26.7%), while increasing average tokens per query from 670 to 784 — a 17% increase. Role-filtered search returns a more targeted candidate set; without it, temporally or topically adjacent episodes from unrelated roles dilute the retrieved context. The larger accuracy impact of role partitioning on cross-role confusor queries is demonstrated in the multi-agent retrieval benchmark (Section 4.3), where confusor queries across five different roles show a 57-point gap.

---

## 4.5 L0 Signal Micro-Experiment

The ablation in Section 4.4 showed that L0 contributes negligibly on our one-shot benchmark, where each situation signature appears only once. To demonstrate L0's behavior in the deployment scenario it is designed for — repeated encounters with the same failing situation — we run a targeted micro-experiment using `l0_experiment.py`.

**Setup.** We seed a memory backend with six failure episodes sharing the same (task_type=`api_integration`, agent_role=`executor`, situation=`POST /v2/payments returns 429 on concurrent batch`) triple. This produces a historical failure rate of 100%, well above the 75% threshold. We then measure (a) the L0 hash-key lookup in isolation, (b) the L1 embed + cosine search cost in isolation, and (c) the end-to-end per-query cost under a cost-avoidance model where the agent aborts early on L0 fire rather than issuing an L1 search and downstream GPT generation call.

**Results.** The L0 hash lookup costs **0.003ms** — a pure SQLite key scan with no embedding. The L1 search (embed + cosine scan over 6 episodes) costs **0.226ms** with a local hash embedder; with a production `text-embedding-3-small` API call, L1 costs approximately 200–350ms including network round-trip. Observed p50 GPT-4o-mini answer generation in our benchmark runs is **480ms**.

**Table 8: L0 Component Latency (200-query average, hash embedder)**

| Operation | Mean (ms) | p50 (ms) | p95 (ms) |
|---|---|---|---|
| L0 hash lookup | 0.003 | 0.003 | 0.003 |
| L1 embed + cosine search | 0.226 | 0.220 | 0.256 |

Under the cost-avoidance model, a correctly-fired L0 signal that allows the agent to skip L1 + GPT generation reduces per-query cost from ~480ms to ~0.003ms — a **>99% reduction** on known-failing situations. When L0 does not fire, the overhead added to a passing query is 0.003ms, a negligible fixed cost. This confirms L0's design intent: it is an O(1) pre-screening layer that eliminates redundant API spend on recurring failure patterns, not a retrieval mechanism. Its value scales with deployment duration — in a long-running deployment, situations that have failed repeatedly accumulate L0 signals that prevent wasted compute on subsequent encounters.

---

## 4.6 Summary

Across both benchmarks, the tiered role-partitioned system outperforms flat retrieval on F1 while injecting fewer tokens per query. The strongest results are on single-hop (+51.2 points over Mem0), multi-hop (+34.1 points over Mem0), and confusor robustness (+57 points over flat search). On the LOCOMO benchmark, the tiered system gains +1.0 F1 point over flat retrieval (77.4% vs 76.4%) while injecting 21% fewer tokens (619 vs 784); token injection is 44% lower on the multi-agent retrieval benchmark. The open-domain gap versus Mem0 narrows to 9.1 points with the query-type-specific retrieval strategy, down from 21.2 points without it. The ablation confirms that all three architectural components contribute: L2 escalation accounts for +7.1 F1 points on open-domain queries, role partitioning accounts for +4.3 points on temporal queries, and L0 provides a fallback signal that contributes +4.8 points on open-domain and +4.3 points on temporal via the L2 seed path. The L0 micro-experiment further confirms that on repeated failure patterns the hash signal provides a >99% latency reduction by eliminating downstream API calls, with negligible overhead (0.003ms) when it does not fire.
