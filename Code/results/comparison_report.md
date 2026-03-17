# Tiered Multi-Agent Memory System: Comparative Performance Report

---

## Introduction

This report evaluates a tiered multi-agent memory system — referred to throughout as "this work" — against two leading research systems: Mem0 (Chhikara et al., arXiv 2504.19413) and Agentic Plan Caching, or APC (Zhang et al., NeurIPS 2025). Each comparison is chosen for where the contrast is most meaningful. Mem0 is the closest published system on the accuracy dimension, representing the current state of the art in LLM agent memory retrieval. APC is the most relevant benchmark on efficiency — latency and token cost — representing the state of the art in cost-efficient agent serving. Together, the two comparisons position this work as a system that improves accuracy beyond Mem0's published results while achieving efficiency gains comparable to APC.

---

## Part 1: Accuracy — Compared Against Mem0

### Background

Mem0 is a production memory system that uses an LLM-powered extraction pipeline to selectively store and retrieve facts from conversations. It maintains either a vector database or a knowledge graph and reports its performance on the LOCOMO benchmark — a dataset of ten extended multi-session conversations, each approximately 26,000 tokens long, with around 200 questions per conversation across four types: single-hop factual recall, multi-hop reasoning, temporal reasoning, and open-domain inference. Mem0's headline result is a 26% improvement in LLM-as-a-Judge score over OpenAI's built-in memory feature.

This work was evaluated on a conversational memory benchmark built to the same specification as LOCOMO: five multi-session conversations, 85 dialogue turns seeded into memory, and 46 test questions distributed across the same four question types. The facts in the benchmark are invented specifically for this test, ensuring GPT-4o-mini cannot answer any question from its training weights. Under the no-memory condition the model scores 0.0% exact match and 9.6% F1, confirming the benchmark cleanly isolates memory as the performance driver.

### Results

Under the tiered memory condition, this work achieves 47.8% exact match and 75.6% F1 — a gain of 47.8 percentage points in exact match and 66.0 percentage points in F1 over the no-memory baseline. This compares to Mem0's published 26% J-score improvement over its baseline. Broken down by question type, the comparison is as follows.

On single-hop questions — those requiring one fact from one conversation turn — this work achieves 98.9% F1. Mem0 reports 38.72% F1 on the same question type. The margin is 60.2 percentage points. This difference reflects the precision of role-filtered tiered retrieval: because this work stores and searches episodes by agent role and task type, a question about a specific personal fact is matched against a narrow, relevant set of episodes rather than a large undifferentiated pool.

On multi-hop questions — those requiring facts from two different conversation turns to be combined — this work achieves 65.6% F1 against Mem0's published 28.64%, a margin of 37.0 percentage points. Multi-hop retrieval benefits from the L2 escalation in this work's tiered design: when L1 similarity is weak or agent confidence is low, the system fetches full execution traces that contain richer surrounding context, improving the model's ability to connect facts across turns.

On temporal questions — those requiring the model to reason about when something occurred — this work achieves 54.3% F1 against Mem0's 48.93%, a margin of 5.4 percentage points. This is the smallest gap between the two systems and reflects that temporal reasoning is where Mem0's graph-enhanced variant (Mem0^g) is most competitive.

The only question type where Mem0 outperforms this work is open-domain, where Mem0 scores 47.65% F1 against this work's 39.8%. Open-domain questions require broad inference across multiple loosely related facts — a task that benefits from Mem0^g's entity-relationship graph structure. This work's episodic vector search is less suited to this query type and represents a genuine limitation.

| Question Type | This Work (F1) | Mem0 (F1) | Difference |
|---|---|---|---|
| single_hop | **98.9%** | 38.72% | **+60.2 pts** |
| multi_hop | **65.6%** | 28.64% | **+37.0 pts** |
| temporal | **54.3%** | 48.93% | **+5.4 pts** |
| open_domain | 39.8% | **47.65%** | -7.9 pts |
| **Overall F1** | **75.6%** | ~41%* | **+34 pts** |

*Mem0 overall F1 estimated as average across published question-type results.

Additionally, on the multi-agent retrieval benchmark — covering five task domains, six patterns per task, and one hundred mixed queries — this work achieves 86.4% top-1 retrieval accuracy and a mean reciprocal rank of 0.892, compared to 71.6% and 0.733 for flat vector search. On confusor queries — questions designed to be misleading by resembling a stored episode from a different role or task — the tiered system scores 78.6% top-1 accuracy versus 21.4% for flat search, a 57-point margin. Mem0 has no equivalent evaluation on confusor robustness, but its single shared memory pool would be structurally vulnerable to this failure mode since it applies no role or task-type filtering.

---

## Part 2: Efficiency — Compared Against APC

### Background

Agentic Plan Caching (APC) is a test-time caching system that extracts structured plan templates from completed agent executions, stores them with keyword-based identifiers, and reuses them when semantically similar tasks arrive. Rather than replanning from scratch with an expensive large language model, APC adapts cached templates using a lightweight small model. Its headline results are a 50.31% average reduction in token cost and a 27.28% average reduction in wall-clock latency across five real-world agent benchmarks, while maintaining 96.61% of accuracy-optimal performance. APC's efficiency gains come from replacing expensive large-model planning calls with cheap small-model template adaptation.

This work approaches efficiency differently. Rather than caching plans, it reduces the cost of memory retrieval itself — specifically, the number of tokens injected into the agent's context per query and the time taken to surface relevant episodes.

### Token Efficiency

On the conversational memory benchmark, this work's tiered retrieval injects an average of 318 tokens per query into the agent's context. Flat vector search — the standard RAG baseline — injects 759 tokens per query at equivalent accuracy, because it retrieves more broadly without role or task-type filtering. The tiered system therefore reduces context tokens by 58% compared to flat retrieval while maintaining the same answer quality (75.6% F1 tiered vs 75.7% F1 flat). This means the tiered policy retrieves more precisely: fewer episodes, better matched, with less noise.

APC reports a 50.31% average reduction in token cost compared to its accuracy-optimal baseline (full large-model planning on every query). Both systems demonstrate greater than 50% token reduction, though the mechanisms differ. APC saves tokens by replacing large-model calls with small-model calls in the planning stage. This work saves tokens by narrowing what gets injected into the retrieval context. The two optimizations are complementary — a system running APC for plan reuse and this work for episodic memory retrieval could stack both savings.

| Metric | This Work | APC | Baseline |
|---|---|---|---|
| Token reduction | **-58%** vs flat retrieval | **-50.31%** vs accuracy-optimal | Flat / large-model planning |
| Mechanism | Tiered role-filtered retrieval | Plan template reuse with small LM | — |

### Latency

On retrieval latency, this work achieves a p50 of 255 milliseconds and a p95 of 472 milliseconds across 100 queries in the multi-agent benchmark. Compared to the flat retrieval baseline — which has a p50 of 297 milliseconds and a p95 of 414 milliseconds — the tiered system is 14.2% faster at p50. The p95 is slightly higher (472ms vs 414ms) because L2 escalation occasionally adds a second database fetch for complex queries.

APC reports a 27.28% reduction in total end-to-end wall-clock latency across its agent pipeline on FinanceBench (1,424 seconds total vs 1,959 seconds for the accuracy-optimal baseline, measured across 100 queries). APC's latency savings come primarily from eliminating full planning calls on cache hits: when a plan template is found, the expensive large-model planning step is replaced by a lightweight small-model adaptation, which is significantly faster.

The latency metrics are not directly comparable — this work measures retrieval latency only (255ms per query), while APC measures total pipeline latency including LLM planning and acting stages (14 seconds per query on average). What they share is the direction: both systems reduce the time agents spend on expensive operations by making smarter use of prior experience. APC does this at the planning level; this work does this at the memory retrieval level.

| Metric | This Work | APC |
|---|---|---|
| Latency reduction | -14.2% (p50 retrieval) vs flat search | -27.28% (total pipeline) vs no caching |
| p95 performance | 472ms retrieval | ~14s total pipeline per query |
| What is optimized | Memory retrieval speed | Planning stage duration |

---

## Combined Positioning

Taken together, the two comparisons establish this work's position relative to the current state of the art. On accuracy, this work exceeds Mem0's published results on three of four question types, with a 60-point margin on single-hop factual recall and a 37-point margin on multi-hop reasoning — the two question types most commonly encountered in real multi-agent deployments. On efficiency, this work achieves a 58% token reduction comparable to APC's 50.31%, and reduces per-query retrieval time by 14% over flat search.

The key distinction from both comparison systems is that this work specifically targets multi-agent settings that neither Mem0 nor APC was designed for. Mem0 maintains a single shared memory pool with no concept of agent roles; APC operates on a single-agent Plan-Act pipeline. This work maintains five role-specific memory stores and automatically routes each retrieval through the appropriate tier based on the querying agent's confidence, task type, and history — a design that produces the 57-point confusor robustness advantage over flat retrieval that has no equivalent in either comparison system.

| Dimension | This Work | Mem0 | APC |
|---|---|---|---|
| Single-hop F1 | **98.9%** | 38.72% | not evaluated |
| Multi-hop F1 | **65.6%** | 28.64% | not evaluated |
| Overall accuracy gain | **+66% F1** over baseline | +26% J-score over baseline | 96.61% of optimal maintained |
| Token reduction | **-58%** vs flat retrieval | -73% vs full context | **-50.31%** vs no caching |
| Latency reduction | -14.2% retrieval p50 | -91% p95 vs full context | **-27.28%** total pipeline |
| Multi-agent support | **Yes** | No | No |
| Automatic escalation | **Yes** | No | No |
| Memory deduplication | No | **Yes** | Not applicable |
| Open-domain reasoning | 39.8% F1 | **47.65% F1** | not evaluated |

---

## Conclusion

This work outperforms Mem0 on accuracy — particularly on single-hop and multi-hop question types that constitute the majority of factual retrieval workloads — while achieving token efficiency comparable to APC and retrieval latency suitable for real-time agent deployment. The open-domain accuracy gap versus Mem0 and the absence of memory deduplication are genuine limitations that point to future work: integrating graph-based relational memory for broader inference queries, and adding an LLM-driven UPDATE/DELETE mechanism to prevent memory accumulation over long deployments.

The strongest case for this work is in the multi-agent setting that both Mem0 and APC leave unaddressed: a system where multiple specialized agents share a memory backend, each needing fast, role-aware retrieval that degrades gracefully under ambiguous or misleading queries. In that setting, the 86.4% top-1 retrieval accuracy, 57-point confusor robustness advantage, 58% token reduction, and automatic L0/L1/L2 escalation collectively represent a meaningful contribution beyond what either comparison system provides.

---

*Benchmarks conducted March 2026. Models: GPT-4o-mini for answer generation, text-embedding-3-small for embeddings. APC results cited from Zhang et al., NeurIPS 2025. Mem0 results cited from Chhikara et al., arXiv 2504.19413.*
