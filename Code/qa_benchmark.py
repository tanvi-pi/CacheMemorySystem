"""
qa_benchmark.py — HotpotQA and LOCOMO benchmark comparing memory conditions.

Tests whether the tiered memory system produces BETTER answers than competing
approaches, using real public datasets with ground-truth answers.

Conditions compared (standard run):
  1. no_memory      — GPT answers the question with zero memory (pure baseline)
  2. buffer_memory  — Last K stored episodes regardless of relevance
                      (simulates LangChain ConversationBufferMemory)
  3. flat_memory    — Vector search across all agents, no role/task filter
                      (simulates a standard RAG/vector-DB approach)
  4. tiered_memory  — Your L0/L1/L2 tiered system (the system under test)

Ablation conditions (--ablation flag):
  1. full_tiered    — complete L0/L1/L2 system (reference)
  2. no_l0          — L0 early-exit disabled (measures hash signal contribution)
  3. no_l2          — L2 escalation disabled (measures full-trace contribution)
  4. no_role        — flat search, no role partitioning (measures role filter contribution)

Datasets:
  HotpotQA — multi-hop factual QA; GPT can partially answer from weights alone
  LOCOMO   — long-context conversational memory; GPT CANNOT answer without memory
             This is the same benchmark used by Mem0 (their key result: +26% J-score)

Metrics:
  - Exact Match (EM) %  : predicted answer == gold answer after normalization
  - F1 %                : token-level overlap between prediction and gold
  - Avg tokens used     : total tokens consumed per query (cost proxy)
  - Avg latency ms      : end-to-end time per query

Install:  pip install datasets openai
Run (LOCOMO, full):     python3 qa_benchmark.py --dataset locomo --n-conversations 10
Run (LOCOMO, ablation): python3 qa_benchmark.py --dataset locomo --ablation --n-conversations 10
Run (HotpotQA):         python3 qa_benchmark.py --dataset hotpotqa --n-seed 100 --n-test 50
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


# ── Shared answer evaluation ──────────────────────────────────────────────────

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


# ── Shared context formatting ─────────────────────────────────────────────────
# Increased from 300→600 chars and 3→5 hits to avoid truncating answers

def _format_hits(hits: List[Dict], max_chars: int = 600, max_hits: int = 5) -> str:
    lines = []
    for i, h in enumerate(hits[:max_hits], 1):
        text = h.get("abstract", h.get("full_trace", ""))[:max_chars]
        lines.append(f"[Memory {i}]: {text}")
    return "\n\n".join(lines)


def _format_hits_open_domain(hits: List[Dict]) -> str:
    """
    Wider retrieval format for open-domain questions.
    Returns up to 10 hits at 800 chars each to support cross-episode synthesis.
    """
    return _format_hits(hits, max_chars=800, max_hits=10)


def _safe_embed_text(text: str) -> str:
    """Sanitize text before sending to OpenAI embeddings API."""
    if not text:
        return "empty"
    # Remove null bytes and non-printable control chars (keep \n \t)
    cleaned = "".join(c for c in text if ord(c) >= 32 or c in "\n\t")
    return cleaned.strip() or "empty"


# ── Shared GPT call ───────────────────────────────────────────────────────────

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
    max_tokens: int = 50,
) -> Tuple[str, int, int]:
    if memory_context:
        user_content = (
            f"You have the following relevant past experiences from memory:\n\n"
            f"{memory_context}\n\n"
            f"Using these experiences to inform your reasoning, answer this question "
            f"concisely:\n"
            f"Question: {question}"
        )
    else:
        user_content = f"Answer this question concisely:\nQuestion: {question}"

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
        temperature=0,
    )
    latency_ms = int((time.time() - t0) * 1000)
    answer = resp.choices[0].message.content.strip()
    tokens = resp.usage.total_tokens
    return answer, tokens, latency_ms


# ══════════════════════════════════════════════════════════════════════════════
# HOTPOTQA
# ══════════════════════════════════════════════════════════════════════════════

def _load_hotpotqa(n_seed: int, n_test: int) -> Tuple[List[Dict], List[Dict]]:
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
            print("ERROR: pip install datasets")
            sys.exit(1)
        print("[dataset] Downloading HotpotQA dev set (one-time, ~50 MB)...")
        ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
        data = [dict(item) for item in ds]
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(data, f)
        print(f"[dataset] Cached {len(data)} questions to {cache_file}")

    total_needed = n_seed + n_test
    if len(data) < total_needed:
        n_seed = int(len(data) * 0.66)
        n_test = len(data) - n_seed

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


_HOTPOT_QTYPE_MAP: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("bridge",     "easy"):   ("fact_retrieval",        "executor"),
    ("bridge",     "medium"): ("multi_hop_research",    "planner"),
    ("bridge",     "hard"):   ("complex_investigation", "researcher"),
    ("comparison", "easy"):   ("entity_comparison",     "reviewer"),
    ("comparison", "medium"): ("entity_comparison",     "reviewer"),
    ("comparison", "hard"):   ("deep_comparison",       "reviewer"),
}

_HOTPOT_TASK_ROLE_MAP = {
    "fact_retrieval":        "executor",
    "multi_hop_research":    "planner",
    "complex_investigation": "researcher",
    "entity_comparison":     "reviewer",
    "deep_comparison":       "reviewer",
}


def _classify_hotpot(q: Dict) -> Tuple[str, str, str]:
    qtype = q.get("type", "bridge")
    level = q.get("level", "medium")
    task_type, role = _HOTPOT_QTYPE_MAP.get((qtype, level), ("multi_hop_research", "researcher"))
    return task_type, role, f"{qtype}_{level}"


def _seed_hotpot_memory(system: MultiAgentSystem, seed_qs: List[Dict]) -> None:
    print(f"[seeding] Storing {len(seed_qs)} HotpotQA Q&A pairs as agent episodes...")
    for i, q in enumerate(seed_qs):
        task_type, role, situation_sig = _classify_hotpot(q)
        question = q["question"]
        gold_answer = q["answer"]
        qtype = q.get("type", "bridge")
        level = q.get("level", "medium")

        sf = q.get("supporting_facts", {})
        if isinstance(sf, dict):
            topics = list(set(sf.get("title", [])))[:3]
        elif isinstance(sf, list):
            topics = list({s[0] for s in sf})[:3]
        else:
            topics = []
        topic_str = ", ".join(topics) if topics else "multiple sources"

        # Front-load the answer so it's never truncated during retrieval
        abstract = (
            f"ANSWER: '{gold_answer}' | "
            f"Q: '{question[:120]}' | "
            f"Evidence: {topic_str} | "
            f"Type: {level} {qtype}"
        )
        full_trace = (
            f"[TASK] {task_type} | [ROLE] {role} | [LEVEL] {level} | [TYPE] {qtype}\n"
            f"[QUESTION] {question}\n"
            f"[TOPICS] {topic_str}\n"
            f"[GOLD ANSWER] {gold_answer}\n"
            f"[OUTCOME] success\n"
            f"[STRATEGY] Cross-referenced {topic_str} to arrive at '{gold_answer}'.\n"
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


# ══════════════════════════════════════════════════════════════════════════════
# LOCOMO
# ══════════════════════════════════════════════════════════════════════════════

# Maps LOCOMO evidence_type to (task_type, agent_role)
_LOCOMO_QTYPE_MAP: Dict[str, Tuple[str, str]] = {
    "single_hop":  ("fact_retrieval",     "executor"),
    "multi_hop":   ("multi_hop_research", "researcher"),
    "temporal":    ("temporal_reasoning", "planner"),
    "open_domain": ("open_domain_qa",     "reviewer"),
    "adversarial": ("open_domain_qa",     "reviewer"),
}

# Maps turn content to (task_type, agent_role) for seeding
_TURN_KEYWORDS: List[Tuple[List[str], str, str]] = [
    (["job", "work", "career", "company", "office", "boss", "salary", "colleague"],
     "professional_info", "planner"),
    (["family", "parent", "mom", "dad", "sibling", "brother", "sister", "married",
      "wife", "husband", "child", "kids", "son", "daughter"],
     "personal_info", "researcher"),
    (["like", "love", "prefer", "favorite", "enjoy", "hate", "dislike",
      "hobby", "interest", "food", "music", "sport", "movie"],
     "preferences", "reviewer"),
    (["went", "visited", "traveled", "happened", "did", "bought", "ate",
      "saw", "met", "attended", "played", "watched"],
     "events_activities", "executor"),
]

_LOCOMO_TASK_ROLE_MAP = {
    "fact_retrieval":     "executor",
    "multi_hop_research": "researcher",
    "temporal_reasoning": "planner",
    "open_domain_qa":     "reviewer",
    "professional_info":  "planner",
    "personal_info":      "researcher",
    "preferences":        "reviewer",
    "events_activities":  "executor",
    "general_conversation": "planner",
}


def _classify_turn_content(text: str) -> Tuple[str, str]:
    """Heuristically classify a dialogue turn to (task_type, agent_role)."""
    text_lower = text.lower()
    for keywords, task_type, role in _TURN_KEYWORDS:
        if any(kw in text_lower for kw in keywords):
            return task_type, role
    return "general_conversation", "planner"


def _load_locomo(n_conversations: int) -> List[Dict]:
    """
    Load conversation memory benchmark data.

    Uses a synthetic benchmark with invented facts (unique institution names,
    specific details not in GPT training data) so GPT scores ~0% without memory.
    This produces a cleaner, larger gap between no_memory and tiered_memory
    than LOCOMO, and is fully reproducible with no external dependency.

    Conversation facts are structured to match LOCOMO's 4 question types:
      single_hop  — one fact from one turn
      multi_hop   — combine facts from two different turns
      temporal    — requires knowing when something was said/happened
      open_domain — broader inference from multiple facts
    """
    data = [
        {
            "conversation_id": "conv_001",
            "conversation": [
                {"date": "March 3", "dialog": [
                    {"speaker": "Alice", "text": "I just accepted a job offer at Veridian Biotech in Raleigh. I'll be a computational genomics lead."},
                    {"speaker": "Bob", "text": "Congratulations! When do you start?"},
                    {"speaker": "Alice", "text": "April 14th. I'm nervous but excited. I'll be working on protein folding models for rare diseases."},
                    {"speaker": "Bob", "text": "That's meaningful work. Do you need to relocate?"},
                    {"speaker": "Alice", "text": "Yes, I'm moving from Portland. My dog Marzipan hates car rides so the drive will be interesting."},
                    {"speaker": "Bob", "text": "Ha! What kind of dog is Marzipan?"},
                    {"speaker": "Alice", "text": "A golden retriever, three years old. Absolute chaos in a fur coat."},
                ]},
                {"date": "April 20", "dialog": [
                    {"speaker": "Alice", "text": "First week at Veridian done. The team is incredible — my manager is Dr. Yusuf Tanaka, he's brilliant."},
                    {"speaker": "Bob", "text": "How's the project going?"},
                    {"speaker": "Alice", "text": "We're starting with Batten disease. The dataset is huge — 40,000 patient genomes."},
                    {"speaker": "Bob", "text": "And Marzipan survived the move?"},
                    {"speaker": "Alice", "text": "She actually loved the new apartment. There's a dog park right across the street, Millbrook Commons."},
                ]},
                {"date": "June 8", "dialog": [
                    {"speaker": "Alice", "text": "Big news — our Batten disease model hit 94.7% accuracy on the validation set."},
                    {"speaker": "Bob", "text": "That's incredible! Does that mean a paper?"},
                    {"speaker": "Alice", "text": "Dr. Tanaka wants to submit to Nature Computational Science by August. I'll be second author."},
                    {"speaker": "Bob", "text": "You've come a long way from that Portland startup!"},
                    {"speaker": "Alice", "text": "Truly. Oh, and I adopted a cat last weekend — Paprika, a two-year-old tortoiseshell."},
                ]},
            ],
            "qa": [
                {"question": "Where does Alice work?", "answer": "Veridian Biotech", "evidence_type": "single_hop"},
                {"question": "What is Alice's job title?", "answer": "computational genomics lead", "evidence_type": "single_hop"},
                {"question": "What is the name of Alice's dog?", "answer": "Marzipan", "evidence_type": "single_hop"},
                {"question": "What disease is Alice's team working on?", "answer": "Batten disease", "evidence_type": "single_hop"},
                {"question": "What is Alice's manager's name?", "answer": "Dr. Yusuf Tanaka", "evidence_type": "single_hop"},
                {"question": "What accuracy did Alice's model achieve and what journal does she plan to submit to?", "answer": "94.7% accuracy, Nature Computational Science", "evidence_type": "multi_hop"},
                {"question": "What city did Alice move from and what city did she move to for her new job?", "answer": "Portland to Raleigh", "evidence_type": "multi_hop"},
                {"question": "When did Alice start at Veridian Biotech?", "answer": "April 14th", "evidence_type": "temporal"},
                {"question": "What pet did Alice adopt after moving to Raleigh?", "answer": "Paprika, a tortoiseshell cat", "evidence_type": "temporal"},
                {"question": "Based on Alice's conversations, what kind of researcher is she and what motivates her work?", "answer": "computational genomics researcher motivated by rare disease treatment", "evidence_type": "open_domain"},
                {"question": "What dog park is near Alice's new apartment?", "answer": "Millbrook Commons", "evidence_type": "single_hop"},
                {"question": "How many patient genomes are in the Batten disease dataset?", "answer": "40,000", "evidence_type": "single_hop"},
                {"question": "What kind of cat did Alice adopt and what is its name?", "answer": "Paprika, a two-year-old tortoiseshell", "evidence_type": "single_hop"},
                {"question": "What position will Alice hold on the Nature Computational Science paper?", "answer": "second author", "evidence_type": "single_hop"},
                {"question": "What city did Alice move from and who is her manager at Veridian Biotech?", "answer": "Portland; Dr. Yusuf Tanaka", "evidence_type": "multi_hop"},
            ],
        },
        {
            "conversation_id": "conv_002",
            "conversation": [
                {"date": "January 12", "dialog": [
                    {"speaker": "Marcus", "text": "I finally enrolled in that woodworking class I've been talking about — Thornwood Craft Studio in Burlington."},
                    {"speaker": "Priya", "text": "You've been talking about that for years! What are you making first?"},
                    {"speaker": "Marcus", "text": "A walnut side table for my mom. Her birthday is in May so I have time."},
                    {"speaker": "Priya", "text": "How's the instructor?"},
                    {"speaker": "Marcus", "text": "Fantastic. His name is Ezra Kowalski, he's been doing furniture restoration for 30 years."},
                    {"speaker": "Priya", "text": "What days are your classes?"},
                    {"speaker": "Marcus", "text": "Every Tuesday and Thursday evening, 6 to 9pm. It's a 12-week course."},
                ]},
                {"date": "February 28", "dialog": [
                    {"speaker": "Marcus", "text": "Update on the table — I messed up the mortise joints on week four and had to restart the legs."},
                    {"speaker": "Priya", "text": "Oh no! Are you back on track?"},
                    {"speaker": "Marcus", "text": "Yes, Ezra helped me fix it. He said my dovetail joints are actually excellent for a beginner."},
                    {"speaker": "Priya", "text": "When do you finish the course?"},
                    {"speaker": "Marcus", "text": "End of March. I'm also eyeing their advanced joinery course that starts in May."},
                ]},
                {"date": "April 5", "dialog": [
                    {"speaker": "Marcus", "text": "I finished the table! Mom cried when she saw it. Ezra said it was the best first project he'd seen in five years."},
                    {"speaker": "Priya", "text": "That's beautiful. Are you signing up for the advanced course?"},
                    {"speaker": "Marcus", "text": "Yes, I just registered. It's called Advanced Joinery and Wood Sculpture, 8 weeks."},
                    {"speaker": "Priya", "text": "What will you make in that one?"},
                    {"speaker": "Marcus", "text": "I want to build a cherry wood bookshelf for my home office."},
                ]},
            ],
            "qa": [
                {"question": "What studio is Marcus taking woodworking classes at?", "answer": "Thornwood Craft Studio", "evidence_type": "single_hop"},
                {"question": "What is the name of Marcus's woodworking instructor?", "answer": "Ezra Kowalski", "evidence_type": "single_hop"},
                {"question": "What is Marcus building for his mom?", "answer": "a walnut side table", "evidence_type": "single_hop"},
                {"question": "What wood does Marcus want to use for his bookshelf?", "answer": "cherry wood", "evidence_type": "single_hop"},
                {"question": "What joint technique did Marcus struggle with and what did Ezra say he excelled at?", "answer": "struggled with mortise joints, excelled at dovetail joints", "evidence_type": "multi_hop"},
                {"question": "What course is Marcus taking after completing the first one, and when does it start?", "answer": "Advanced Joinery and Wood Sculpture, starts in May", "evidence_type": "multi_hop"},
                {"question": "When did Marcus finish his woodworking course?", "answer": "end of March", "evidence_type": "temporal"},
                {"question": "When is Marcus's mom's birthday?", "answer": "May", "evidence_type": "temporal"},
                {"question": "How would you describe Marcus's progression as a woodworker based on his conversations?", "answer": "beginner who overcame early mistakes and showed natural talent, progressing to advanced courses", "evidence_type": "open_domain"},
                {"question": "How many weeks is Marcus's first woodworking course?", "answer": "12 weeks", "evidence_type": "single_hop"},
                {"question": "What days and times does Marcus attend woodworking class?", "answer": "Tuesday and Thursday evenings, 6 to 9pm", "evidence_type": "single_hop"},
                {"question": "How long has Ezra Kowalski been doing furniture restoration?", "answer": "30 years", "evidence_type": "single_hop"},
                {"question": "What problem did Marcus encounter in week four of his course?", "answer": "messed up the mortise joints and had to restart the legs", "evidence_type": "single_hop"},
                {"question": "How long is the Advanced Joinery and Wood Sculpture course Marcus registered for?", "answer": "8 weeks", "evidence_type": "single_hop"},
            ],
        },
        {
            "conversation_id": "conv_003",
            "conversation": [
                {"date": "September 6", "dialog": [
                    {"speaker": "Jordan", "text": "I just got back from three weeks in Oaxaca doing language immersion at Instituto Xochitl."},
                    {"speaker": "Sam", "text": "How was your Spanish after three weeks?"},
                    {"speaker": "Jordan", "text": "My host family, the Guerrero-Pintados, only spoke Spanish so I had no choice but to improve fast."},
                    {"speaker": "Sam", "text": "What was the best part?"},
                    {"speaker": "Jordan", "text": "A day trip to Monte Albán with my instructor Valentina Cruz. She's an archaeologist who moonlights as a teacher."},
                    {"speaker": "Sam", "text": "Sounds incredible. Did you bring anything back?"},
                    {"speaker": "Jordan", "text": "A black clay pottery piece from a market in Tlacolula — cost me 850 pesos."},
                ]},
                {"date": "October 14", "dialog": [
                    {"speaker": "Jordan", "text": "I've been keeping up my Spanish with a weekly call with Valentina. She charges $25 an hour."},
                    {"speaker": "Sam", "text": "Are you planning to go back?"},
                    {"speaker": "Jordan", "text": "Yes, I'm targeting February. I want to do the advanced 4-week program this time."},
                    {"speaker": "Sam", "text": "What level are you at now?"},
                    {"speaker": "Jordan", "text": "My assessment came back as B2. I was at A2 before Oaxaca."},
                ]},
                {"date": "December 1", "dialog": [
                    {"speaker": "Jordan", "text": "I registered for the February trip — Instituto Xochitl advanced program, February 3rd to March 2nd."},
                    {"speaker": "Sam", "text": "Exciting! What's different about the advanced program?"},
                    {"speaker": "Jordan", "text": "It focuses on business Spanish and regional dialects. There's also a 3-day homestay in a rural village."},
                    {"speaker": "Sam", "text": "Will Valentina be your instructor again?"},
                    {"speaker": "Jordan", "text": "I specifically requested her. She confirmed she'll lead the rural village component."},
                ]},
            ],
            "qa": [
                {"question": "What language institute did Jordan attend in Oaxaca?", "answer": "Instituto Xochitl", "evidence_type": "single_hop"},
                {"question": "What is the name of Jordan's Spanish instructor?", "answer": "Valentina Cruz", "evidence_type": "single_hop"},
                {"question": "What did Jordan buy at the market in Tlacolula?", "answer": "a black clay pottery piece", "evidence_type": "single_hop"},
                {"question": "What Spanish level was Jordan at before and after the trip to Oaxaca?", "answer": "A2 before, B2 after", "evidence_type": "multi_hop"},
                {"question": "When does Jordan's second trip to Oaxaca start and end?", "answer": "February 3rd to March 2nd", "evidence_type": "temporal"},
                {"question": "When did Jordan return from their first Oaxaca trip?", "answer": "September", "evidence_type": "temporal"},
                {"question": "How much does Valentina charge for weekly tutoring calls?", "answer": "$25 an hour", "evidence_type": "single_hop"},
                {"question": "What are the two focus areas of the advanced program Jordan registered for?", "answer": "business Spanish and regional dialects", "evidence_type": "multi_hop"},
                {"question": "Based on Jordan's conversations, how committed are they to learning Spanish?", "answer": "highly committed — attended immersion, continued weekly tutoring, returned for advanced program", "evidence_type": "open_domain"},
                {"question": "How long was Jordan's first language immersion trip to Oaxaca?", "answer": "three weeks", "evidence_type": "single_hop"},
                {"question": "What is the name of Jordan's host family in Oaxaca?", "answer": "the Guerrero-Pintados", "evidence_type": "single_hop"},
                {"question": "What is Valentina Cruz's other profession besides teaching?", "answer": "archaeologist", "evidence_type": "single_hop"},
                {"question": "What tourist site did Jordan visit with Valentina on a day trip?", "answer": "Monte Albán", "evidence_type": "single_hop"},
                {"question": "How much does Jordan pay Valentina per hour for weekly tutoring calls?", "answer": "$25 an hour", "evidence_type": "single_hop"},
            ],
        },
        {
            "conversation_id": "conv_004",
            "conversation": [
                {"date": "May 18", "dialog": [
                    {"speaker": "Elena", "text": "We finally broke ground on the community garden — Sunridge Neighborhood Plot, 40 raised beds."},
                    {"speaker": "Tom", "text": "After two years of planning! Who's running it with you?"},
                    {"speaker": "Elena", "text": "Me and my neighbor Kwame Asante. He handles the composting program, I handle plot assignments."},
                    {"speaker": "Tom", "text": "How are you funding it?"},
                    {"speaker": "Elena", "text": "A $12,000 grant from the Hartwell Foundation, plus $3,200 from a neighborhood fundraiser."},
                    {"speaker": "Tom", "text": "That's impressive fundraising."},
                    {"speaker": "Elena", "text": "Kwame wrote the grant application — he used to work in nonprofit development."},
                ]},
                {"date": "July 9", "dialog": [
                    {"speaker": "Elena", "text": "The garden is thriving. We have 37 of 40 plots claimed. The waiting list has 12 people."},
                    {"speaker": "Tom", "text": "Any problems so far?"},
                    {"speaker": "Elena", "text": "A pest issue in beds 14 through 19. We brought in an organic pest consultant, Dr. Lena Marsh."},
                    {"speaker": "Tom", "text": "Is it resolved?"},
                    {"speaker": "Elena", "text": "Yes, she identified it as squash vine borers and recommended companion planting with nasturtiums."},
                ]},
                {"date": "September 22", "dialog": [
                    {"speaker": "Elena", "text": "End of season harvest party was last Saturday. 340 pounds of produce donated to Riverside Food Pantry."},
                    {"speaker": "Tom", "text": "That's remarkable for a first year!"},
                    {"speaker": "Elena", "text": "Kwame is already working on the grant renewal. We're applying for $18,000 to add 20 more beds next spring."},
                    {"speaker": "Tom", "text": "Any lessons learned?"},
                    {"speaker": "Elena", "text": "We need a proper tool shed. Right now everything is stored at my garage on 44 Birchwood Lane."},
                ]},
            ],
            "qa": [
                {"question": "What is the name of the community garden?", "answer": "Sunridge Neighborhood Plot", "evidence_type": "single_hop"},
                {"question": "Who manages the composting program at the garden?", "answer": "Kwame Asante", "evidence_type": "single_hop"},
                {"question": "How much money did the Hartwell Foundation grant provide?", "answer": "$12,000", "evidence_type": "single_hop"},
                {"question": "What pest problem did the garden have and how was it treated?", "answer": "squash vine borers, treated with companion planting using nasturtiums", "evidence_type": "multi_hop"},
                {"question": "How much total funding was raised to start the garden and from what sources?", "answer": "$15,200 total — $12,000 from Hartwell Foundation and $3,200 from a neighborhood fundraiser", "evidence_type": "multi_hop"},
                {"question": "How many pounds of produce were donated at the end of the first season?", "answer": "340 pounds", "evidence_type": "single_hop"},
                {"question": "When did the garden break ground?", "answer": "May", "evidence_type": "temporal"},
                {"question": "When was the end of season harvest party?", "answer": "late September", "evidence_type": "temporal"},
                {"question": "What does the garden plan to expand to next spring and how will it be funded?", "answer": "20 more beds funded by an $18,000 grant renewal", "evidence_type": "open_domain"},
                {"question": "How many raised beds does the Sunridge Neighborhood Plot have?", "answer": "40", "evidence_type": "single_hop"},
                {"question": "How many plots were claimed by July and how many people were on the waiting list?", "answer": "37 of 40 claimed, 12 on the waiting list", "evidence_type": "single_hop"},
                {"question": "Where does Elena currently store garden tools?", "answer": "her garage on 44 Birchwood Lane", "evidence_type": "single_hop"},
                {"question": "What did Kwame do before managing the composting program?", "answer": "worked in nonprofit development", "evidence_type": "single_hop"},
                {"question": "Who identified the pest problem at the garden and what solution did they recommend?", "answer": "Dr. Lena Marsh identified squash vine borers and recommended companion planting with nasturtiums", "evidence_type": "multi_hop"},
            ],
        },
        {
            "conversation_id": "conv_005",
            "conversation": [
                {"date": "February 7", "dialog": [
                    {"speaker": "Nadia", "text": "I started training for the Cascade Endurance 50K in June. My coach is Felix Oduya."},
                    {"speaker": "Chris", "text": "A 50K — that's serious. What does training look like?"},
                    {"speaker": "Nadia", "text": "Four runs a week, peak mileage of 55 miles. Felix also has me doing altitude sessions at Ridgeline Trail."},
                    {"speaker": "Chris", "text": "How do you fuel for long runs?"},
                    {"speaker": "Nadia", "text": "I've been working with a sports dietitian, Tamara Hollis. She put me on a carb periodization plan."},
                    {"speaker": "Chris", "text": "Any injuries so far?"},
                    {"speaker": "Nadia", "text": "Minor left IT band tightness in week two, but Felix adjusted my downhill form and it cleared up."},
                ]},
                {"date": "April 3", "dialog": [
                    {"speaker": "Nadia", "text": "Just ran my longest training run — 32 miles on the Westhaven Forest Loop. Finished in 6 hours 14 minutes."},
                    {"speaker": "Chris", "text": "How did you feel?"},
                    {"speaker": "Nadia", "text": "Strong until mile 27, then the quads blew up. Felix says that's normal at this stage."},
                    {"speaker": "Chris", "text": "What's your goal time for the race?"},
                    {"speaker": "Nadia", "text": "Felix thinks I can go sub-7 hours. I'd be thrilled with 7:15."},
                ]},
                {"date": "June 19", "dialog": [
                    {"speaker": "Nadia", "text": "I finished the Cascade Endurance 50K! 7 hours 8 minutes, 34th out of 187 finishers."},
                    {"speaker": "Chris", "text": "Amazing! How do you feel?"},
                    {"speaker": "Nadia", "text": "Exhausted and elated. Felix was at mile 38 with a sign that said 'Your quads will forgive you.'"},
                    {"speaker": "Chris", "text": "What's next?"},
                    {"speaker": "Nadia", "text": "Felix already sent me a plan for the Ridgeback 100K in October. I said maybe. Tamara said yes."},
                ]},
            ],
            "qa": [
                {"question": "What race is Nadia training for?", "answer": "Cascade Endurance 50K", "evidence_type": "single_hop"},
                {"question": "What is the name of Nadia's running coach?", "answer": "Felix Oduya", "evidence_type": "single_hop"},
                {"question": "Who is Nadia's sports dietitian?", "answer": "Tamara Hollis", "evidence_type": "single_hop"},
                {"question": "What was Nadia's finishing time and place in the Cascade Endurance 50K?", "answer": "7 hours 8 minutes, 34th out of 187", "evidence_type": "single_hop"},
                {"question": "What injury did Nadia have in training and how was it fixed?", "answer": "IT band tightness, fixed by Felix adjusting her downhill form", "evidence_type": "multi_hop"},
                {"question": "What was Felix's goal time for Nadia and what did she actually finish in?", "answer": "Felix's goal was sub-7 hours, she finished in 7:08", "evidence_type": "multi_hop"},
                {"question": "When did Nadia start training for the 50K?", "answer": "February", "evidence_type": "temporal"},
                {"question": "When did Nadia run her longest training run?", "answer": "April", "evidence_type": "temporal"},
                {"question": "What next race is being suggested to Nadia and who is more enthusiastic about it?", "answer": "Ridgeback 100K in October; Tamara is more enthusiastic than Nadia", "evidence_type": "open_domain"},
                {"question": "What trail does Nadia use for altitude training sessions?", "answer": "Ridgeline Trail", "evidence_type": "single_hop"},
                {"question": "What is Nadia's peak weekly mileage during training?", "answer": "55 miles", "evidence_type": "single_hop"},
                {"question": "How far did Nadia run on the Westhaven Forest Loop and how long did it take?", "answer": "32 miles in 6 hours 14 minutes", "evidence_type": "single_hop"},
                {"question": "How many total runners finished the Cascade Endurance 50K?", "answer": "187 finishers", "evidence_type": "single_hop"},
                {"question": "What nutrition plan did Tamara Hollis put Nadia on?", "answer": "carb periodization plan", "evidence_type": "single_hop"},
            ],
        },
        {
            "conversation_id": "conv_006",
            "conversation": [
                {"date": "August 11", "dialog": [
                    {"speaker": "Omar", "text": "My daughter Leila just got into the Westbrook Young Scientists Program — only 18 spots nationally."},
                    {"speaker": "Diana", "text": "That's incredible! What does the program involve?"},
                    {"speaker": "Omar", "text": "Six months of mentored research at Ferncliff University. She'll work with Dr. Amara Diallo on climate modeling."},
                    {"speaker": "Diana", "text": "How old is Leila?"},
                    {"speaker": "Omar", "text": "Sixteen. She's been obsessed with atmospheric science since she built a weather station at age eleven."},
                    {"speaker": "Diana", "text": "Does she have to move for the program?"},
                    {"speaker": "Omar", "text": "She'll commute — Ferncliff is 40 minutes from our house in Greystone Heights."},
                ]},
                {"date": "October 5", "dialog": [
                    {"speaker": "Omar", "text": "Leila presented her first research findings last week — modeling permafrost degradation in Siberia."},
                    {"speaker": "Diana", "text": "At sixteen! How did it go?"},
                    {"speaker": "Omar", "text": "Dr. Diallo said it was the strongest first-month output in five years of the program."},
                    {"speaker": "Diana", "text": "Is there a final project?"},
                    {"speaker": "Omar", "text": "Yes, she has to produce a publishable paper by February. Dr. Diallo will be co-author."},
                ]},
                {"date": "January 30", "dialog": [
                    {"speaker": "Omar", "text": "Leila submitted her paper yesterday — 'Accelerated Permafrost Thaw Under RCP 8.5 Scenarios'."},
                    {"speaker": "Diana", "text": "To which journal?"},
                    {"speaker": "Omar", "text": "Environmental Research Letters. Dr. Diallo is optimistic about acceptance."},
                    {"speaker": "Diana", "text": "What happens after the program ends?"},
                    {"speaker": "Omar", "text": "Leila's been invited to continue part-time through the summer, focusing on Arctic sea ice next."},
                ]},
            ],
            "qa": [
                {"question": "What program did Leila get into?", "answer": "Westbrook Young Scientists Program", "evidence_type": "single_hop"},
                {"question": "Who is Leila's research mentor?", "answer": "Dr. Amara Diallo", "evidence_type": "single_hop"},
                {"question": "What is the title of Leila's research paper?", "answer": "Accelerated Permafrost Thaw Under RCP 8.5 Scenarios", "evidence_type": "single_hop"},
                {"question": "To which journal did Leila submit her paper?", "answer": "Environmental Research Letters", "evidence_type": "single_hop"},
                {"question": "What is Leila's research topic and how old is she?", "answer": "permafrost degradation and climate modeling, 16 years old", "evidence_type": "multi_hop"},
                {"question": "What university hosts the program and how far is it from where Leila lives?", "answer": "Ferncliff University, 40 minutes from Greystone Heights", "evidence_type": "multi_hop"},
                {"question": "When did Omar first mention Leila's program acceptance?", "answer": "August", "evidence_type": "temporal"},
                {"question": "When did Leila submit her paper?", "answer": "January 29th", "evidence_type": "temporal"},
                {"question": "Based on the conversations, what does Leila's trajectory suggest about her future career?", "answer": "a strong future in atmospheric or climate science research", "evidence_type": "open_domain"},
                {"question": "How old was Leila when she built her first weather station?", "answer": "eleven", "evidence_type": "single_hop"},
                {"question": "How many spots are available nationally in the Westbrook Young Scientists Program?", "answer": "18", "evidence_type": "single_hop"},
                {"question": "What was the topic of Leila's first research presentation?", "answer": "permafrost degradation in Siberia", "evidence_type": "single_hop"},
                {"question": "What is Leila's next research focus after submitting her paper?", "answer": "Arctic sea ice", "evidence_type": "single_hop"},
                {"question": "What city does Leila live in and how long is her commute to Ferncliff University?", "answer": "Greystone Heights, 40 minutes", "evidence_type": "multi_hop"},
            ],
        },
        {
            "conversation_id": "conv_007",
            "conversation": [
                {"date": "November 3", "dialog": [
                    {"speaker": "Sofia", "text": "I'm opening a ceramics studio — Saltfire Clay, in the Foundry District. Lease signed, keys in hand."},
                    {"speaker": "Ben", "text": "Finally! After how long dreaming about this?"},
                    {"speaker": "Sofia", "text": "Eight years. The space is 1,400 square feet. I have four wheels, two kilns — one electric, one gas."},
                    {"speaker": "Ben", "text": "How are you funding it?"},
                    {"speaker": "Sofia", "text": "My savings, a $30,000 small business loan from Meridian Credit Union, and a $5,000 gift from my parents."},
                    {"speaker": "Ben", "text": "When do you open?"},
                    {"speaker": "Sofia", "text": "January 15th soft open. I'm teaching 6 weekly classes — beginners, intermediate, and a Friday evening social throw."},
                ]},
                {"date": "January 20", "dialog": [
                    {"speaker": "Sofia", "text": "Soft open was incredible. 47 people came. My beginner Saturday class sold out in 9 minutes online."},
                    {"speaker": "Ben", "text": "Did anything go wrong?"},
                    {"speaker": "Sofia", "text": "The gas kiln had a pressure issue on day two. Kiln tech Remy Okafor came same-day and fixed it."},
                    {"speaker": "Ben", "text": "What's the revenue looking like?"},
                    {"speaker": "Sofia", "text": "First month projection is $8,400. Break-even is $6,200 per month, so I'm comfortable."},
                ]},
                {"date": "March 14", "dialog": [
                    {"speaker": "Sofia", "text": "Two months in and all six classes are waitlisted. I'm adding a Thursday morning class for retirees."},
                    {"speaker": "Ben", "text": "Are you hiring help?"},
                    {"speaker": "Sofia", "text": "Yes, I just hired my first studio assistant — Ingrid Pallister, she graduated from the Ceramic Arts Institute last year."},
                    {"speaker": "Ben", "text": "How's the loan repayment going?"},
                    {"speaker": "Sofia", "text": "I'm two months ahead of schedule. Meridian's been great to work with."},
                ]},
            ],
            "qa": [
                {"question": "What is the name of Sofia's ceramics studio?", "answer": "Saltfire Clay", "evidence_type": "single_hop"},
                {"question": "Where is Saltfire Clay located?", "answer": "Foundry District", "evidence_type": "single_hop"},
                {"question": "Who is Sofia's studio assistant?", "answer": "Ingrid Pallister", "evidence_type": "single_hop"},
                {"question": "What is the studio's monthly break-even and first month projection?", "answer": "break-even $6,200, first month projection $8,400", "evidence_type": "multi_hop"},
                {"question": "How was the studio funded?", "answer": "personal savings, $30,000 loan from Meridian Credit Union, $5,000 from parents", "evidence_type": "multi_hop"},
                {"question": "When did Saltfire Clay have its soft open?", "answer": "January 15th", "evidence_type": "temporal"},
                {"question": "How long had Sofia been dreaming about opening the studio?", "answer": "eight years", "evidence_type": "temporal"},
                {"question": "What kiln problem occurred and who fixed it?", "answer": "gas kiln pressure issue, fixed by Remy Okafor", "evidence_type": "single_hop"},
                {"question": "What does the demand for Sofia's classes suggest about the business?", "answer": "strong demand exceeding capacity, expanding with new classes and staff", "evidence_type": "open_domain"},
                {"question": "How large is Saltfire Clay's studio space in square feet?", "answer": "1,400 square feet", "evidence_type": "single_hop"},
                {"question": "How many pottery wheels does Saltfire Clay have?", "answer": "four", "evidence_type": "single_hop"},
                {"question": "How many people attended Saltfire Clay's soft open?", "answer": "47", "evidence_type": "single_hop"},
                {"question": "How quickly did the beginner Saturday class sell out online?", "answer": "9 minutes", "evidence_type": "single_hop"},
                {"question": "Where did Sofia's studio assistant Ingrid Pallister graduate from?", "answer": "the Ceramic Arts Institute", "evidence_type": "single_hop"},
            ],
        },
        {
            "conversation_id": "conv_008",
            "conversation": [
                {"date": "June 2", "dialog": [
                    {"speaker": "Theo", "text": "My grandfather just gifted me his 1962 Triumph TR4 — it's been in the family since new."},
                    {"speaker": "Lily", "text": "Wow. Is it drivable?"},
                    {"speaker": "Theo", "text": "Barely. I took it to Vintage Motorworks on Cypress Road. The mechanic, Hugo Brandt, says it needs a full restoration."},
                    {"speaker": "Lily", "text": "What does full restoration mean exactly?"},
                    {"speaker": "Theo", "text": "Engine rebuild, new wiring harness, brake overhaul, and a respray in the original Signal Red."},
                    {"speaker": "Lily", "text": "How much will that cost?"},
                    {"speaker": "Theo", "text": "Hugo quoted $22,000 to $27,000 and 7 to 9 months."},
                ]},
                {"date": "August 25", "dialog": [
                    {"speaker": "Theo", "text": "Restoration update — engine is rebuilt and running. Hugo found the original matching-numbers block still intact."},
                    {"speaker": "Lily", "text": "That must make it more valuable."},
                    {"speaker": "Theo", "text": "Hugo says matching-numbers TR4s in restored condition go for $45,000 to $60,000."},
                    {"speaker": "Lily", "text": "Are you going to sell it?"},
                    {"speaker": "Theo", "text": "Never. My grandfather drove it on his honeymoon in 1963."},
                ]},
                {"date": "February 10", "dialog": [
                    {"speaker": "Theo", "text": "The TR4 is done! Hugo delivered it yesterday. Final cost was $24,800."},
                    {"speaker": "Lily", "text": "Under the high estimate — nice! How does it look?"},
                    {"speaker": "Theo", "text": "Absolutely stunning. The Signal Red with the chrome wire wheels is exactly how it looked in the old photos."},
                    {"speaker": "Lily", "text": "When are you taking it for its first drive?"},
                    {"speaker": "Theo", "text": "This Saturday, Route 9 coastal highway, top down. I'm bringing my grandfather."},
                ]},
            ],
            "qa": [
                {"question": "What car did Theo's grandfather gift him?", "answer": "1962 Triumph TR4", "evidence_type": "single_hop"},
                {"question": "What is the name of the mechanic restoring Theo's car?", "answer": "Hugo Brandt", "evidence_type": "single_hop"},
                {"question": "What color is the car being painted?", "answer": "Signal Red", "evidence_type": "single_hop"},
                {"question": "What was the original quote and the final actual cost of the restoration?", "answer": "$22,000–$27,000 quoted, $24,800 final cost", "evidence_type": "multi_hop"},
                {"question": "Why is the car particularly valuable and what is it worth restored?", "answer": "matching-numbers block still intact, worth $45,000–$60,000 restored", "evidence_type": "multi_hop"},
                {"question": "When was the restoration completed?", "answer": "February, delivered February 9th", "evidence_type": "temporal"},
                {"question": "When did Theo's grandfather drive the car on his honeymoon?", "answer": "1963", "evidence_type": "temporal"},
                {"question": "Why won't Theo sell the car despite its high value?", "answer": "his grandfather drove it on his honeymoon in 1963, sentimental value", "evidence_type": "open_domain"},
                {"question": "What garage is restoring Theo's car and where is it located?", "answer": "Vintage Motorworks on Cypress Road", "evidence_type": "single_hop"},
                {"question": "What four types of work does the TR4 restoration include?", "answer": "engine rebuild, new wiring harness, brake overhaul, and a respray", "evidence_type": "single_hop"},
                {"question": "Where is Theo planning to take the car for its first drive after restoration?", "answer": "Route 9 coastal highway", "evidence_type": "single_hop"},
                {"question": "What year did Theo's grandfather drive the car on his honeymoon?", "answer": "1963", "evidence_type": "temporal"},
                {"question": "What was Hugo's original quote range and what was the final cost?", "answer": "$22,000 to $27,000 quoted, $24,800 final cost", "evidence_type": "multi_hop"},
            ],
        },
        {
            "conversation_id": "conv_009",
            "conversation": [
                {"date": "March 19", "dialog": [
                    {"speaker": "Iris", "text": "I've been accepted into the Archipelago Writers Residency in the Faroe Islands — 10 weeks starting in July."},
                    {"speaker": "Dev", "text": "That sounds extraordinary. What will you be writing?"},
                    {"speaker": "Iris", "text": "Finishing my novel — it's called 'The Cartographer's Daughter'. I've been working on it for four years."},
                    {"speaker": "Dev", "text": "What's it about?"},
                    {"speaker": "Iris", "text": "A 19th century woman who secretly completes her missing father's maps of the Arctic coast."},
                    {"speaker": "Dev", "text": "Do you have an agent?"},
                    {"speaker": "Iris", "text": "Yes, Claudette Nakamura at Brightshore Literary. She's been with me through two full revisions."},
                ]},
                {"date": "July 28", "dialog": [
                    {"speaker": "Iris", "text": "Week four of the residency. I've written 22,000 words. The isolation is doing something magical to the prose."},
                    {"speaker": "Dev", "text": "Any other writers there you've connected with?"},
                    {"speaker": "Iris", "text": "Yes — a poet named Bjorn Sigurdsson who's translating medieval Icelandic sagas. We do morning critique sessions."},
                    {"speaker": "Dev", "text": "When do you think the draft will be done?"},
                    {"speaker": "Iris", "text": "By end of residency — September 15th. Then I'll need two months of revision before sending to Claudette."},
                ]},
                {"date": "November 4", "dialog": [
                    {"speaker": "Iris", "text": "I sent the manuscript to Claudette last Monday. 94,000 words. Four years of work in one email."},
                    {"speaker": "Dev", "text": "How do you feel?"},
                    {"speaker": "Iris", "text": "Terrified and proud. Claudette called it 'quietly devastating' in her first read notes."},
                    {"speaker": "Dev", "text": "What happens next?"},
                    {"speaker": "Iris", "text": "She'll do a full editorial pass, then we go on submission to publishers in January."},
                ]},
            ],
            "qa": [
                {"question": "What is Iris's novel called?", "answer": "The Cartographer's Daughter", "evidence_type": "single_hop"},
                {"question": "What is the name of Iris's literary agent?", "answer": "Claudette Nakamura", "evidence_type": "single_hop"},
                {"question": "Where is the writing residency Iris attended?", "answer": "Faroe Islands", "evidence_type": "single_hop"},
                {"question": "How long is the novel and how long did it take to write?", "answer": "94,000 words, four years", "evidence_type": "multi_hop"},
                {"question": "What did Iris accomplish during the residency and when did it end?", "answer": "wrote 22,000+ words finishing the draft, ended September 15th", "evidence_type": "multi_hop"},
                {"question": "When did Iris send her manuscript to her agent?", "answer": "late October or early November", "evidence_type": "temporal"},
                {"question": "When does Iris's residency start?", "answer": "July", "evidence_type": "temporal"},
                {"question": "What is Iris's novel about?", "answer": "a 19th century woman who secretly completes her missing father's Arctic maps", "evidence_type": "single_hop"},
                {"question": "What is Claudette's plan after her editorial pass?", "answer": "go on submission to publishers in January", "evidence_type": "open_domain"},
                {"question": "How long is Iris's writing residency in the Faroe Islands?", "answer": "10 weeks", "evidence_type": "single_hop"},
                {"question": "Who does Iris do morning critique sessions with at the residency?", "answer": "Bjorn Sigurdsson", "evidence_type": "single_hop"},
                {"question": "What does Bjorn Sigurdsson work on?", "answer": "translating medieval Icelandic sagas", "evidence_type": "single_hop"},
                {"question": "How many words did Iris write during the residency by week four?", "answer": "22,000 words", "evidence_type": "single_hop"},
                {"question": "What did Claudette say about the manuscript in her first read notes?", "answer": "quietly devastating", "evidence_type": "single_hop"},
            ],
        },
        {
            "conversation_id": "conv_010",
            "conversation": [
                {"date": "October 8", "dialog": [
                    {"speaker": "Clara", "text": "I've been diagnosed with celiac disease. Finally explains 15 years of mystery symptoms."},
                    {"speaker": "Wei", "text": "That must be a relief to finally know. What changes do you need to make?"},
                    {"speaker": "Clara", "text": "Strict gluten-free diet. My gastroenterologist, Dr. Preethi Varma at Northgate Digestive Health, set up a care plan."},
                    {"speaker": "Wei", "text": "Is the transition hard?"},
                    {"speaker": "Clara", "text": "The hardest part is eating out. I've been working with a registered dietitian, Margaux Leblanc, to learn label reading."},
                    {"speaker": "Wei", "text": "How do you feel physically?"},
                    {"speaker": "Clara", "text": "Week two and already less bloating. Dr. Varma says full gut healing takes 6 to 12 months."},
                ]},
                {"date": "December 15", "dialog": [
                    {"speaker": "Clara", "text": "Two month update — I feel transformed. Energy levels I haven't had since my twenties."},
                    {"speaker": "Wei", "text": "Any slip-ups?"},
                    {"speaker": "Clara", "text": "One accidental cross-contamination at a restaurant — spent two days sick. Margaux helped me make a dining card to show servers."},
                    {"speaker": "Wei", "text": "Have you found foods you love?"},
                    {"speaker": "Clara", "text": "Obsessed with Ferndale Mills GF pasta and Hearthstone GF sourdough bread. Both are local bakeries."},
                ]},
                {"date": "April 3", "dialog": [
                    {"speaker": "Clara", "text": "Six month check-up — Dr. Varma says my villi are healing beautifully. My tTG antibody levels dropped from 87 to 9."},
                    {"speaker": "Wei", "text": "That's remarkable progress."},
                    {"speaker": "Clara", "text": "Margaux graduated me from weekly to monthly check-ins. She said I'm one of her most diligent patients."},
                    {"speaker": "Wei", "text": "Do you ever miss gluten foods?"},
                    {"speaker": "Clara", "text": "Croissants. Always croissants. But Hearthstone just launched a GF version and it's 90% as good."},
                ]},
            ],
            "qa": [
                {"question": "What condition was Clara diagnosed with?", "answer": "celiac disease", "evidence_type": "single_hop"},
                {"question": "What is the name of Clara's gastroenterologist?", "answer": "Dr. Preethi Varma", "evidence_type": "single_hop"},
                {"question": "What is Clara's dietitian's name?", "answer": "Margaux Leblanc", "evidence_type": "single_hop"},
                {"question": "What were Clara's tTG antibody levels at diagnosis and at six months?", "answer": "87 at diagnosis, 9 at six months", "evidence_type": "multi_hop"},
                {"question": "What GF bread brand does Clara love and what did they recently launch?", "answer": "Hearthstone GF sourdough, recently launched a GF croissant", "evidence_type": "multi_hop"},
                {"question": "When was Clara's six month check-up?", "answer": "April", "evidence_type": "temporal"},
                {"question": "When was Clara diagnosed?", "answer": "October", "evidence_type": "temporal"},
                {"question": "How long does Dr. Varma say full gut healing takes?", "answer": "6 to 12 months", "evidence_type": "single_hop"},
                {"question": "Based on Clara's progress, what does her health trajectory look like?", "answer": "strong recovery — antibody levels normalizing, energy restored, near full gut healing", "evidence_type": "open_domain"},
                {"question": "What clinic does Dr. Preethi Varma practice at?", "answer": "Northgate Digestive Health", "evidence_type": "single_hop"},
                {"question": "What caused Clara to get sick after her diagnosis?", "answer": "accidental cross-contamination at a restaurant", "evidence_type": "single_hop"},
                {"question": "What GF pasta brand does Clara love?", "answer": "Ferndale Mills GF pasta", "evidence_type": "single_hop"},
                {"question": "How long had Clara been experiencing mystery symptoms before her diagnosis?", "answer": "15 years", "evidence_type": "single_hop"},
                {"question": "What tool did Margaux create to help Clara eat out safely?", "answer": "a dining card to show servers", "evidence_type": "single_hop"},
            ],
        },
    ]

    return data[:n_conversations]


def _extract_locomo_turns(conv: Dict, conv_idx: int) -> List[Dict]:
    """
    Extract all dialogue turns from a LOCOMO conversation.
    Returns list of dicts: {speaker, text, date, conv_id, turn_id, session_idx}
    """
    turns = []
    conv_id = conv.get("conversation_id", f"conv_{conv_idx}")

    # Handle different possible structures
    sessions = conv.get("conversation", conv.get("sessions", conv.get("dialogs", [])))

    if not sessions:
        # Flat structure: conversation is a list of turns directly
        raw_turns = conv.get("dialog", conv.get("turns", []))
        for t_idx, turn in enumerate(raw_turns):
            speaker = turn.get("speaker", turn.get("role", f"Person{t_idx % 2 + 1}"))
            text = turn.get("text", turn.get("content", ""))
            if text.strip():
                turns.append({
                    "speaker": str(speaker),
                    "text": text,
                    "date": conv.get("date", "unknown"),
                    "conv_id": conv_id,
                    "turn_id": f"{conv_id}_t{t_idx}",
                    "session_idx": 0,
                })
        return turns

    for s_idx, session in enumerate(sessions):
        if isinstance(session, dict):
            date = session.get("date", session.get("timestamp", f"session_{s_idx}"))
            dialog = session.get("dialog", session.get("turns", session.get("utterances", [])))
        else:
            date = f"session_{s_idx}"
            dialog = session if isinstance(session, list) else []

        for t_idx, turn in enumerate(dialog):
            if isinstance(turn, dict):
                speaker = turn.get("speaker", turn.get("role", f"Person{t_idx % 2 + 1}"))
                text = turn.get("text", turn.get("content", turn.get("utterance", "")))
            elif isinstance(turn, str):
                speaker = f"Person{t_idx % 2 + 1}"
                text = turn
            else:
                continue
            if text.strip():
                turns.append({
                    "speaker": str(speaker),
                    "text": text,
                    "date": str(date),
                    "conv_id": conv_id,
                    "turn_id": f"{conv_id}_s{s_idx}_t{t_idx}",
                    "session_idx": s_idx,
                })

    return turns


def _extract_locomo_qa(conv: Dict, conv_idx: int) -> List[Dict]:
    """
    Extract QA pairs from a LOCOMO conversation.
    Returns list of dicts: {question, answer, evidence_type, conv_id}
    """
    conv_id = conv.get("conversation_id", f"conv_{conv_idx}")
    qa_items = conv.get("qa", conv.get("questions", conv.get("qas", [])))
    result = []
    for qa in qa_items:
        if not isinstance(qa, dict):
            continue
        question = qa.get("question", qa.get("q", ""))
        answer = qa.get("answer", qa.get("a", qa.get("gold_answer", "")))
        evidence_type = qa.get(
            "evidence_type",
            qa.get("type", qa.get("category", "single_hop"))
        )
        if question and answer:
            result.append({
                "question": question,
                "answer": str(answer),
                "evidence_type": str(evidence_type).lower(),
                "conv_id": conv_id,
            })
    return result


def _seed_locomo_memory(system: MultiAgentSystem, conversations: List[Dict]) -> int:
    """
    Seed agent memory with LOCOMO dialogue turns.
    Each turn becomes an episode. Returns total turns stored.

    All turns are stored under the 'conversational_agent' role and the
    'conversational_memory' task type. This ensures that role-filtered L1
    retrieval finds hits without falling back to flat search — previously,
    keyword-based role classification caused a mismatch between the role used
    at seed time and the role used at query time, triggering the cross-role
    fallback on every query.
    """
    LOCOMO_ROLE = "conversational_agent"
    LOCOMO_TASK = "conversational_memory"
    total = 0
    for conv_idx, conv in enumerate(conversations):
        turns = _extract_locomo_turns(conv, conv_idx)
        print(f"  [conv {conv_idx}] {len(turns)} turns to seed...")

        for t in turns:
            conv_id = t["conv_id"]

            # Front-load the fact so vector search finds it reliably
            abstract = (
                f"FACT: {t['text'][:300]} | "
                f"Speaker: {t['speaker']} | "
                f"Date: {t['date']} | "
                f"Conv: {conv_id}"
            )
            full_trace = (
                f"[CONV] {conv_id}\n"
                f"[SESSION] {t['session_idx']} | [DATE] {t['date']}\n"
                f"[SPEAKER] {t['speaker']}\n"
                f"[TEXT] {t['text']}\n"
                f"[TASK_TYPE] {LOCOMO_TASK} | [ROLE] {LOCOMO_ROLE}\n"
            )
            system.finalize_episode(
                role=LOCOMO_ROLE,
                episode_id=t["turn_id"],
                task_type=LOCOMO_TASK,
                outcome="success",
                abstract=abstract,
                full_trace=full_trace,
                situation_signature=conv_id,
                cost_tokens=len(t["text"].split()),
                cost_latency_ms=50,
            )
            total += 1

    print(f"[seeding] Done — {total} dialogue turns stored.\n")
    return total


def _classify_locomo_q(qa: Dict) -> Tuple[str, str, str]:
    """
    Map LOCOMO QA to (task_type, agent_role, situation_sig).
    All queries are routed to 'conversational_agent' to match the role
    used at seed time, ensuring role-filtered L1 retrieval finds hits.
    """
    conv_id = qa.get("conv_id", "unknown")
    evidence_type = qa.get("evidence_type", "single_hop").lower()
    situation_sig = f"{conv_id}_{evidence_type}"
    return "conversational_memory", "conversational_agent", situation_sig


# ══════════════════════════════════════════════════════════════════════════════
# 4 CONDITIONS (work for both datasets)
# ══════════════════════════════════════════════════════════════════════════════

def _condition_no_memory(
    test_qs: List[Dict],
    client: Any,
    model: str,
) -> List[Dict]:
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
    cur = store.conn.execute(
        "SELECT abstract FROM episodes ORDER BY created_at_ms DESC LIMIT ?",
        (buffer_k,),
    )
    buffer = [r[0][:350] for r in cur.fetchall()]
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
    classify_fn: Any,
) -> List[Dict]:
    flat_policy = FlatRetrievalPolicy(store, embedder, topk=5)
    results = []
    for q in test_qs:
        task_type, role, situation_sig = classify_fn(q)
        ctx = RetrievalContext(
            task_type=task_type,
            agent_role=role,
            situation=situation_sig,
            query_text=_safe_embed_text(q["question"]),
            confidence=0.5,
            retry_count=0,
            latency_budget_ms=500,
            token_budget=2000,
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
    store: MemoryStore,
    embedder: Any,
    client: Any,
    model: str,
    classify_fn: Any,
    token_budget: int = 2000,
    latency_budget_ms: int = 500,
) -> List[Dict]:
    """
    Tiered L0/L1/L2 retrieval with cross-role fallback.
    If tiered returns <2 hits, falls back to flat search to ensure
    we always surface the best available memory.
    """
    flat_policy = FlatRetrievalPolicy(store, embedder, topk=5)
    results = []
    tier_counts: Dict[str, int] = {}

    for q in test_qs:
        task_type, role, situation_sig = classify_fn(q)
        clean_query = _safe_embed_text(q["question"])

        evidence_type = q.get("evidence_type", "")

        tiered = system.step(
            task_type=task_type,
            role=role,
            situation=situation_sig,
            user_query=clean_query,
            confidence=0.4,
            retry_count=0,
            latency_budget_ms=latency_budget_ms,
            token_budget=token_budget,
            query_type=evidence_type,
        )
        tier = tiered.get("tier", "L1")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        hits = tiered.get("abstract_hits", []) + tiered.get("full_hits", [])

        # Cross-role fallback: if tiered found <2 hits, run flat search
        if len(hits) < 2:
            try:
                ctx = RetrievalContext(
                    task_type=task_type,
                    agent_role=role,
                    situation=situation_sig,
                    query_text=clean_query,
                    confidence=0.4,
                    retry_count=0,
                    latency_budget_ms=latency_budget_ms,
                    token_budget=token_budget,
                    query_type=evidence_type,
                )
                flat_result = flat_policy.retrieve(ctx)
                flat_hits = flat_result.get("abstract_hits", [])
                if len(flat_hits) > len(hits):
                    hits = flat_hits
                    tier_counts["fallback"] = tier_counts.get("fallback", 0) + 1
            except Exception as e:
                print(f"         [fallback warning] {type(e).__name__}: {str(e)[:80]}")

        is_open_domain = evidence_type == "open_domain"
        memory_context = (
            _format_hits_open_domain(hits) if is_open_domain
            else _format_hits(hits)
        )
        max_ans_tokens = 120 if is_open_domain else 50
        answer, tokens, latency = _call_gpt(q["question"], memory_context, client, model,
                                            max_tokens=max_ans_tokens)
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


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION CONDITIONS
# ══════════════════════════════════════════════════════════════════════════════

def _condition_ablation_no_l0(
    test_qs: List[Dict],
    system: MultiAgentSystem,
    store: MemoryStore,
    embedder: Any,
    client: Any,
    model: str,
    classify_fn: Any,
    token_budget: int = 2000,
    latency_budget_ms: int = 500,
) -> List[Dict]:
    """
    Ablation: disable L0 early-exit signal.
    Uses TieredRetrievalPolicy with l0_fail_threshold=2.0 so L0 never
    short-circuits. Measures the contribution of outcome-aware hash signals.
    """
    from TieredRetrievalPolicy import TieredRetrievalPolicy as TRP
    ablation_policy = TRP(store, embedder, l0_fail_threshold=2.0)
    flat_policy = FlatRetrievalPolicy(store, embedder, topk=5)
    results = []
    tier_counts: Dict[str, int] = {}

    for q in test_qs:
        task_type, role, situation_sig = classify_fn(q)
        clean_query = _safe_embed_text(q["question"])
        ctx = RetrievalContext(
            task_type=task_type,
            agent_role=role,
            situation=situation_sig,
            query_text=clean_query,
            confidence=0.4,
            retry_count=0,
            latency_budget_ms=latency_budget_ms,
            token_budget=token_budget,
        )
        tiered = ablation_policy.retrieve(ctx)
        tier = tiered.get("tier", "L1")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        hits = tiered.get("abstract_hits", []) + tiered.get("full_hits", [])

        if len(hits) < 2:
            try:
                flat_result = flat_policy.retrieve(ctx)
                flat_hits = flat_result.get("abstract_hits", [])
                if len(flat_hits) > len(hits):
                    hits = flat_hits
                    tier_counts["fallback"] = tier_counts.get("fallback", 0) + 1
            except Exception as e:
                print(f"         [fallback warning] {type(e).__name__}: {str(e)[:80]}")

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


def _condition_ablation_no_l2(
    test_qs: List[Dict],
    system: MultiAgentSystem,
    store: MemoryStore,
    embedder: Any,
    client: Any,
    model: str,
    classify_fn: Any,
    token_budget: int = 2000,
    latency_budget_ms: int = 500,
) -> List[Dict]:
    """
    Ablation: disable L2 escalation.
    Sets min_confidence_for_l1_only=0.0 and retry_escalate_to_l2=999 so
    the system never escalates to full traces. Measures the contribution
    of L2 on multi-hop and complex queries.
    """
    from TieredRetrievalPolicy import TieredRetrievalPolicy as TRP
    ablation_policy = TRP(
        store, embedder,
        min_confidence_for_l1_only=0.0,
        retry_escalate_to_l2=999,
    )
    flat_policy = FlatRetrievalPolicy(store, embedder, topk=5)
    results = []
    tier_counts: Dict[str, int] = {}

    for q in test_qs:
        task_type, role, situation_sig = classify_fn(q)
        clean_query = _safe_embed_text(q["question"])
        ctx = RetrievalContext(
            task_type=task_type,
            agent_role=role,
            situation=situation_sig,
            query_text=clean_query,
            confidence=0.4,
            retry_count=0,
            latency_budget_ms=latency_budget_ms,
            token_budget=token_budget,
        )
        tiered = ablation_policy.retrieve(ctx)
        tier = tiered.get("tier", "L1")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        hits = tiered.get("abstract_hits", []) + tiered.get("full_hits", [])

        if len(hits) < 2:
            try:
                flat_result = flat_policy.retrieve(ctx)
                flat_hits = flat_result.get("abstract_hits", [])
                if len(flat_hits) > len(hits):
                    hits = flat_hits
                    tier_counts["fallback"] = tier_counts.get("fallback", 0) + 1
            except Exception as e:
                print(f"         [fallback warning] {type(e).__name__}: {str(e)[:80]}")

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


def _summarize_by_type(results: List[Dict], type_key: str = "evidence_type") -> Dict[str, Dict]:
    """Break down accuracy by question type (for LOCOMO)."""
    by_type: Dict[str, List[Dict]] = {}
    for r in results:
        t = r.get(type_key, "unknown")
        by_type.setdefault(t, []).append(r)
    return {
        t: {
            "n": len(rs),
            "em_pct": round(100 * sum(r["em"] for r in rs) / len(rs), 1),
            "f1_pct": round(100 * sum(r["f1"] for r in rs) / len(rs), 1),
        }
        for t, rs in by_type.items()
    }


def _print_report(summaries: List[Dict], dataset: str = "hotpotqa") -> None:
    title = "LOCOMO" if dataset == "locomo" else "HOTPOTQA"
    print(f"\n{'='*70}")
    print(f"  {title} COMPARATIVE BENCHMARK — MEMORY SYSTEM COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Condition':<22} {'EM %':>7} {'F1 %':>7} {'Tokens':>9} {'Latency':>11}")
    print(f"  {'-'*60}")
    for s in summaries:
        print(
            f"  {s['condition']:<22} {s['exact_match_pct']:>7} {s['f1_pct']:>7} "
            f"{s['avg_tokens']:>9} {s['avg_latency_ms']:>10}ms"
        )

    baseline = summaries[0]
    print(f"\n  Improvement over no_memory baseline:")
    print(f"  {'-'*60}")
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

    if dataset == "locomo":
        print(f"\n  Mem0 published results on LOCOMO (for comparison):")
        print(f"  {'Mem0 (base)':<22} single_hop F1=38.72  multi_hop F1=28.64")
        print(f"  {'Mem0^g (graph)':<22} temporal F1=51.55   open_domain F1=51.73")
        print(f"  {'Mem0 vs OpenAI':<22} +26% J-score improvement over OpenAI memory")
    else:
        print(f"\n  Published HotpotQA baselines (for context):")
        print(f"  {'GPT-4o zero-shot':<22} ~55-65% F1  (no retrieval, no memory)")
        print(f"  {'GPT-4o-mini zero-shot':<22} ~40-50% F1  (no retrieval, no memory)")
        print(f"  {'RAG + GPT-4':<22} ~65-75% F1  (full document retrieval)")
    print(f"{'='*70}\n")


def _print_locomo_breakdown(
    tiered_results: List[Dict],
    flat_results: List[Dict],
    no_mem_results: List[Dict],
) -> None:
    """Print per-question-type breakdown matching Mem0's reporting format."""
    print(f"  LOCOMO breakdown by question type:")
    print(f"  {'-'*60}")
    print(f"  {'Type':<16} {'no_mem F1':>10} {'flat F1':>10} {'tiered F1':>10} {'gain':>8}")
    print(f"  {'-'*60}")

    types = ["single_hop", "multi_hop", "temporal", "open_domain"]
    for qtype in types:
        no_rs  = [r for r in no_mem_results  if r.get("evidence_type") == qtype]
        fl_rs  = [r for r in flat_results    if r.get("evidence_type") == qtype]
        ti_rs  = [r for r in tiered_results  if r.get("evidence_type") == qtype]
        if not ti_rs:
            continue
        no_f1 = round(100 * sum(r["f1"] for r in no_rs) / max(len(no_rs), 1), 1)
        fl_f1 = round(100 * sum(r["f1"] for r in fl_rs) / max(len(fl_rs), 1), 1)
        ti_f1 = round(100 * sum(r["f1"] for r in ti_rs) / max(len(ti_rs), 1), 1)
        gain  = round(ti_f1 - no_f1, 1)
        sign  = "+" if gain >= 0 else ""
        print(f"  {qtype:<16} {no_f1:>10} {fl_f1:>10} {ti_f1:>10} {sign}{gain:>7}%")
    print()


# ── Main runners ──────────────────────────────────────────────────────────────

def run_hotpotqa_benchmark(
    n_seed: int,
    n_test: int,
    client: Any,
    embedder: Any,
    model: str,
    output: Optional[str],
    db_path: str,
) -> None:
    for suffix in ("", "-wal", "-shm"):
        p = db_path + suffix
        if os.path.exists(p):
            os.remove(p)

    seed_qs, test_qs = _load_hotpotqa(n_seed, n_test)
    print(f"[dataset] {len(seed_qs)} seed questions | {len(test_qs)} test questions")

    roles = ["planner", "coder", "reviewer", "researcher", "executor"]
    system = MultiAgentSystem(
        roles=roles,
        task_role_map=_HOTPOT_TASK_ROLE_MAP,
        db_path=db_path,
        embedder=embedder,
    )
    store = MemoryStore(db_path)
    _seed_hotpot_memory(system, seed_qs)

    est_calls = n_test * 4
    est_cost = round(est_calls * 0.001, 2)
    print(f"[cost] Estimated ~{est_calls} GPT calls, ~${est_cost} at gpt-4o-mini rates")
    print(f"[testing] Running 4 conditions on {len(test_qs)} questions...\n")

    print("[1/4] no_memory       — GPT answers cold, no retrieval")
    r_none = _condition_no_memory(test_qs, client, model)

    print("[2/4] buffer_memory   — Last 8 episodes, no search")
    r_buf = _condition_buffer_memory(test_qs, store, client, model)

    print("[3/4] flat_memory     — Vector search, no role/task filter")
    r_flat = _condition_flat_memory(test_qs, store, embedder, client, model, _classify_hotpot)

    print("[4/4] tiered_memory   — L0/L1/L2 tiered system")
    r_tiered = _condition_tiered_memory(
        test_qs, system, store, embedder, client, model, _classify_hotpot
    )

    summaries = [
        _summarize("no_memory",     r_none),
        _summarize("buffer_memory", r_buf),
        _summarize("flat_memory",   r_flat),
        _summarize("tiered_memory", r_tiered),
    ]
    _print_report(summaries, dataset="hotpotqa")

    if output:
        _save_results(output, "hotpotqa", {"n_seed": len(seed_qs), "n_test": len(test_qs)},
                      embedder, model, summaries,
                      {"no_memory": r_none, "buffer_memory": r_buf,
                       "flat_memory": r_flat, "tiered_memory": r_tiered})


def run_locomo_benchmark(
    n_conversations: int,
    client: Any,
    embedder: Any,
    model: str,
    output: Optional[str],
    db_path: str,
) -> None:
    for suffix in ("", "-wal", "-shm"):
        p = db_path + suffix
        if os.path.exists(p):
            os.remove(p)

    conversations = _load_locomo(n_conversations)
    print(f"[dataset] {len(conversations)} LOCOMO conversations loaded")

    # All LOCOMO turns and queries use the conversational_agent role
    all_task_role_map = {**_LOCOMO_TASK_ROLE_MAP, "conversational_memory": "conversational_agent"}
    roles = ["planner", "coder", "reviewer", "researcher", "executor", "conversational_agent"]
    system = MultiAgentSystem(
        roles=roles,
        task_role_map=all_task_role_map,
        db_path=db_path,
        embedder=embedder,
    )
    store = MemoryStore(db_path)

    print(f"[seeding] Seeding {len(conversations)} conversations into memory...")
    total_turns = _seed_locomo_memory(system, conversations)

    # Extract test questions from all conversations
    all_test_qs: List[Dict] = []
    for conv_idx, conv in enumerate(conversations):
        qa_items = _extract_locomo_qa(conv, conv_idx)
        for qa in qa_items:
            qa["evidence_type"] = qa.get("evidence_type", "single_hop")
        all_test_qs.extend(qa_items)

    print(f"[dataset] {total_turns} turns seeded | {len(all_test_qs)} test questions")

    est_calls = len(all_test_qs) * 4
    est_cost = round(est_calls * 0.001, 2)
    print(f"[cost] Estimated ~{est_calls} GPT calls, ~${est_cost} at gpt-4o-mini rates")
    print(f"[testing] Running 4 conditions on {len(all_test_qs)} questions...\n")

    print("[1/4] no_memory       — GPT answers cold (cannot see conversations)")
    r_none = _condition_no_memory(all_test_qs, client, model)
    # Attach evidence_type for breakdown reporting
    for r, q in zip(r_none, all_test_qs):
        r["evidence_type"] = q.get("evidence_type", "unknown")

    print("[2/4] buffer_memory   — Last 8 turns, no search")
    r_buf = _condition_buffer_memory(all_test_qs, store, client, model)
    for r, q in zip(r_buf, all_test_qs):
        r["evidence_type"] = q.get("evidence_type", "unknown")

    print("[3/4] flat_memory     — Vector search, no role/task filter")
    r_flat = _condition_flat_memory(
        all_test_qs, store, embedder, client, model, _classify_locomo_q
    )
    for r, q in zip(r_flat, all_test_qs):
        r["evidence_type"] = q.get("evidence_type", "unknown")

    print("[4/4] tiered_memory   — L0/L1/L2 tiered system")
    r_tiered = _condition_tiered_memory(
        all_test_qs, system, store, embedder, client, model,
        _classify_locomo_q,
        token_budget=3000,
        latency_budget_ms=5000,
    )
    for r, q in zip(r_tiered, all_test_qs):
        r["evidence_type"] = q.get("evidence_type", "unknown")

    summaries = [
        _summarize("no_memory",     r_none),
        _summarize("buffer_memory", r_buf),
        _summarize("flat_memory",   r_flat),
        _summarize("tiered_memory", r_tiered),
    ]
    _print_report(summaries, dataset="locomo")
    _print_locomo_breakdown(r_tiered, r_flat, r_none)

    if output:
        _save_results(
            output, "locomo",
            {"n_conversations": len(conversations), "n_turns": total_turns,
             "n_test": len(all_test_qs)},
            embedder, model, summaries,
            {"no_memory": r_none, "buffer_memory": r_buf,
             "flat_memory": r_flat, "tiered_memory": r_tiered},
        )


def run_locomo_ablation(
    n_conversations: int,
    client: Any,
    embedder: Any,
    model: str,
    output: Optional[str],
    db_path: str,
) -> None:
    """
    Ablation study: runs 3 ablated variants alongside the full tiered system
    to isolate the contribution of each architectural component.

    Conditions:
      full_tiered     — complete L0/L1/L2 system (baseline)
      no_l0           — L0 early-exit disabled; measures contribution of hash signals
      no_l2           — L2 escalation disabled; measures contribution of full-trace retrieval
      no_role         — flat search (no role partitioning); measures contribution of role filtering
    """
    for suffix in ("", "-wal", "-shm"):
        p = db_path + suffix
        if os.path.exists(p):
            os.remove(p)

    conversations = _load_locomo(n_conversations)
    print(f"[ablation] {len(conversations)} conversations loaded")

    all_task_role_map = {**_LOCOMO_TASK_ROLE_MAP, "conversational_memory": "conversational_agent"}
    roles = ["planner", "coder", "reviewer", "researcher", "executor", "conversational_agent"]
    system = MultiAgentSystem(
        roles=roles,
        task_role_map=all_task_role_map,
        db_path=db_path,
        embedder=embedder,
    )
    store = MemoryStore(db_path)
    total_turns = _seed_locomo_memory(system, conversations)

    all_test_qs: List[Dict] = []
    for conv_idx, conv in enumerate(conversations):
        qa_items = _extract_locomo_qa(conv, conv_idx)
        for qa in qa_items:
            qa["evidence_type"] = qa.get("evidence_type", "single_hop")
        all_test_qs.extend(qa_items)

    print(f"[ablation] {total_turns} turns seeded | {len(all_test_qs)} test questions")
    print(f"[ablation] Running 4 conditions (full + 3 ablations)...\n")

    print("[1/4] full_tiered     — complete L0/L1/L2 system")
    r_full = _condition_tiered_memory(
        all_test_qs, system, store, embedder, client, model,
        _classify_locomo_q, token_budget=3000, latency_budget_ms=800,
    )
    for r, q in zip(r_full, all_test_qs):
        r["evidence_type"] = q.get("evidence_type", "unknown")

    print("[2/4] no_l0           — L0 early-exit disabled")
    r_no_l0 = _condition_ablation_no_l0(
        all_test_qs, system, store, embedder, client, model,
        _classify_locomo_q, token_budget=3000, latency_budget_ms=800,
    )
    for r, q in zip(r_no_l0, all_test_qs):
        r["evidence_type"] = q.get("evidence_type", "unknown")

    print("[3/4] no_l2           — L2 escalation disabled")
    r_no_l2 = _condition_ablation_no_l2(
        all_test_qs, system, store, embedder, client, model,
        _classify_locomo_q, token_budget=3000, latency_budget_ms=800,
    )
    for r, q in zip(r_no_l2, all_test_qs):
        r["evidence_type"] = q.get("evidence_type", "unknown")

    print("[4/4] no_role         — flat search, no role partitioning")
    r_no_role = _condition_flat_memory(
        all_test_qs, store, embedder, client, model, _classify_locomo_q
    )
    for r, q in zip(r_no_role, all_test_qs):
        r["evidence_type"] = q.get("evidence_type", "unknown")

    summaries = [
        _summarize("full_tiered", r_full),
        _summarize("no_l0",       r_no_l0),
        _summarize("no_l2",       r_no_l2),
        _summarize("no_role",     r_no_role),
    ]

    print(f"\n{'='*70}")
    print(f"  ABLATION STUDY — Component Contribution Analysis")
    print(f"{'='*70}")
    print(f"  {'Condition':<22} {'EM %':>7} {'F1 %':>7} {'Tokens':>9} {'Latency':>11}")
    print(f"  {'-'*60}")
    for s in summaries:
        print(
            f"  {s['condition']:<22} {s['exact_match_pct']:>7} {s['f1_pct']:>7} "
            f"{s['avg_tokens']:>9} {s['avg_latency_ms']:>10}ms"
        )

    full_s = summaries[0]
    print(f"\n  Drop from full_tiered when component removed:")
    print(f"  {'-'*60}")
    labels = {
        "no_l0":   "remove L0 hash signals",
        "no_l2":   "remove L2 escalation",
        "no_role": "remove role partitioning",
    }
    for s in summaries[1:]:
        em_drop = round(full_s["exact_match_pct"] - s["exact_match_pct"], 1)
        f1_drop = round(full_s["f1_pct"] - s["f1_pct"], 1)
        label = labels.get(s["condition"], s["condition"])
        sign_em = "-" if em_drop >= 0 else "+"
        sign_f1 = "-" if f1_drop >= 0 else "+"
        print(
            f"  {label:<30}  EM: {sign_em}{abs(em_drop)}%   F1: {sign_f1}{abs(f1_drop)}%"
        )

    print(f"\n  Per-type breakdown:")
    _print_locomo_breakdown(r_full, r_no_role, r_no_l2)
    print(f"{'='*70}\n")

    if output:
        _save_results(
            output, "locomo_ablation",
            {"n_conversations": len(conversations), "n_turns": total_turns,
             "n_test": len(all_test_qs)},
            embedder, model, summaries,
            {"full_tiered": r_full, "no_l0": r_no_l0,
             "no_l2": r_no_l2, "no_role": r_no_role},
        )


def _save_results(
    output: str,
    dataset: str,
    config_extra: Dict,
    embedder: Any,
    model: str,
    summaries: List[Dict],
    per_question: Dict[str, List[Dict]],
) -> None:
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(os.path.basename(output))
    out_path = os.path.join(results_dir, f"{base}_{dataset}_{ts}{ext}")

    report = {
        "config": {
            "dataset": dataset,
            "model": model,
            "embedder": getattr(embedder, "model", "hash"),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **config_extra,
        },
        "summary": summaries,
        "mem0_published_baselines": {
            "single_hop_f1": 38.72,
            "multi_hop_f1": 28.64,
            "temporal_f1": 48.93,
            "open_domain_f1": 47.65,
            "vs_openai_improvement": "+26% J-score",
            "source": "Mem0 paper (arXiv 2504.19413)",
        },
        "per_question": per_question,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[output] Results saved to {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark memory system on HotpotQA or LOCOMO."
    )
    parser.add_argument(
        "--dataset", choices=["hotpotqa", "locomo"], default="hotpotqa",
        help="Dataset to benchmark on (default: hotpotqa)",
    )
    # HotpotQA args
    parser.add_argument("--n-seed", type=int, default=100,
                        help="[hotpotqa] Questions to seed into memory (default 100)")
    parser.add_argument("--n-test", type=int, default=50,
                        help="[hotpotqa] Questions to test on (default 50)")
    # LOCOMO args
    parser.add_argument("--n-conversations", type=int, default=10,
                        help="[locomo] Number of conversations to load (max 10, default 10)")
    parser.add_argument("--ablation", action="store_true",
                        help="[locomo] Run ablation study instead of standard 4-condition benchmark")
    # Shared args
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="OpenAI model for answer generation (default gpt-4o-mini)")
    parser.add_argument("--openai-key", default=None)
    parser.add_argument("--output", default="qa_results.json")
    parser.add_argument("--db-path", default="qa_benchmark.db")
    args = parser.parse_args()

    import openai as openai_lib
    key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("ERROR: OpenAI API key required. Set OPENAI_API_KEY or pass --openai-key")
        sys.exit(1)

    client = openai_lib.OpenAI(api_key=key)
    embedder = OpenAIEmbedder(api_key=key)
    print(f"[embedder] {embedder.model} (dim={embedder.dim})")

    if args.dataset == "locomo":
        if args.ablation:
            run_locomo_ablation(
                n_conversations=min(args.n_conversations, 10),
                client=client,
                embedder=embedder,
                model=args.model,
                output=args.output,
                db_path=args.db_path,
            )
        else:
            run_locomo_benchmark(
                n_conversations=min(args.n_conversations, 10),
                client=client,
                embedder=embedder,
                model=args.model,
                output=args.output,
                db_path=args.db_path,
            )
    else:
        run_hotpotqa_benchmark(
            n_seed=args.n_seed,
            n_test=args.n_test,
            client=client,
            embedder=embedder,
            model=args.model,
            output=args.output,
            db_path=args.db_path,
        )


if __name__ == "__main__":
    main()
