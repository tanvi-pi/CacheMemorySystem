from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from .Embedder import cosine
    from .Episode import Episode
except ImportError:
    from Embedder import cosine
    from Episode import Episode


class MemoryStore:
    """SQLite-backed storage for L0/L1/L2 memory tiers."""

    def __init__(self, path: str = "memory.db"):
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS _meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                agent_role TEXT NOT NULL,
                outcome TEXT NOT NULL,
                abstract TEXT NOT NULL,
                full_trace TEXT NOT NULL,
                cost_tokens INTEGER NOT NULL,
                cost_latency_ms INTEGER NOT NULL,
                created_at_ms INTEGER NOT NULL,
                tags_json TEXT,
                warning_flags_json TEXT,
                abstract_embedding_json TEXT NOT NULL,
                situation_signature TEXT
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS l0_signals (
                key TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                agent_role TEXT NOT NULL,
                success_count INTEGER NOT NULL,
                failure_count INTEGER NOT NULL,
                warning_count INTEGER NOT NULL,
                last_outcome TEXT NOT NULL,
                last_episode_id TEXT NOT NULL,
                updated_at_ms INTEGER NOT NULL
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS canonical_situations (
                id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                agent_role TEXT NOT NULL,
                label TEXT NOT NULL,
                embedding_json TEXT NOT NULL
            );
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_canonical_task_role
            ON canonical_situations(task_type, agent_role);
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_episodes_task_role_time
            ON episodes(task_type, agent_role, created_at_ms DESC);
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_episodes_time
            ON episodes(created_at_ms DESC);
            """
        )
        self.conn.commit()

    def check_embedder_compat(self, model_id: str) -> None:
        """Verify the embedder matches what was used to populate this database.

        On first call for a fresh database, records the model_id.
        On subsequent calls, raises ValueError if the model_id differs.
        This prevents silent corruption when switching embedders.
        """
        cur = self.conn.execute("SELECT value FROM _meta WHERE key = 'embedder_model'")
        row = cur.fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO _meta (key, value) VALUES ('embedder_model', ?)", (model_id,)
            )
            self.conn.commit()
        elif row[0] != model_id:
            raise ValueError(
                f"Embedder mismatch: this database was built with '{row[0]}' but "
                f"you are using '{model_id}'. Embeddings are incompatible — "
                f"create a new database or reseed with the correct embedder."
            )

    def upsert_episode(self, ep: Episode, abstract_embedding: List[float], situation_signature: Optional[str] = None) -> None:
        self.conn.execute(
            """
            INSERT INTO episodes (
                episode_id, task_type, agent_role, outcome, abstract, full_trace,
                cost_tokens, cost_latency_ms, created_at_ms,
                tags_json, warning_flags_json, abstract_embedding_json, situation_signature
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(episode_id) DO UPDATE SET
                task_type=excluded.task_type,
                agent_role=excluded.agent_role,
                outcome=excluded.outcome,
                abstract=excluded.abstract,
                full_trace=excluded.full_trace,
                cost_tokens=excluded.cost_tokens,
                cost_latency_ms=excluded.cost_latency_ms,
                created_at_ms=excluded.created_at_ms,
                tags_json=excluded.tags_json,
                warning_flags_json=excluded.warning_flags_json,
                abstract_embedding_json=excluded.abstract_embedding_json,
                situation_signature=excluded.situation_signature
            """,
            (
                ep.episode_id,
                ep.task_type,
                ep.agent_role,
                ep.outcome,
                ep.abstract,
                ep.full_trace,
                ep.cost_tokens,
                ep.cost_latency_ms,
                ep.created_at_ms,
                json.dumps(ep.tags) if ep.tags else None,
                json.dumps(ep.warning_flags) if ep.warning_flags else None,
                json.dumps(abstract_embedding),
                situation_signature,
            ),
        )
        self.conn.commit()

    def find_duplicate(
        self,
        agent_role: str,
        task_type: str,
        situation_signature: str,
        abstract_embedding: List[float],
        sim_threshold: float = 0.92,
        max_candidates: int = 50,
    ) -> Optional[str]:
        """
        Check if a near-identical episode already exists before inserting.
        Returns the existing episode_id if a duplicate is found, else None.

        Deduplication criteria: same role + task_type + situation_signature,
        and abstract cosine similarity >= sim_threshold. This mirrors Mem0's
        NOOP operation — if the fact is already stored, skip the write.
        """
        cur = self.conn.execute(
            """
            SELECT episode_id, abstract_embedding_json
            FROM episodes
            WHERE agent_role = ? AND task_type = ? AND situation_signature = ?
            ORDER BY created_at_ms DESC
            LIMIT ?
            """,
            (agent_role, task_type, situation_signature, max_candidates),
        )
        for (episode_id, emb_json) in cur.fetchall():
            emb = json.loads(emb_json)
            if cosine(abstract_embedding, emb) >= sim_threshold:
                return episode_id
        return None

    def search_l1(
        self,
        task_type: str,
        agent_role: str,
        query_embedding: List[float],
        limit: int = 8,
        min_sim: float = 0.15,
        max_candidates: int = 800,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        cur = self.conn.execute(
            """
            SELECT episode_id, task_type, agent_role, outcome, abstract, created_at_ms, abstract_embedding_json, situation_signature
            FROM episodes
            WHERE task_type = ? AND agent_role = ?
            ORDER BY created_at_ms DESC
            LIMIT ?
            """,
            (task_type, agent_role, max_candidates),
        )

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for (episode_id, tt, role, outcome, abstract, created_at_ms, emb_json, sit_sig) in cur.fetchall():
            emb = json.loads(emb_json)
            sim = cosine(query_embedding, emb)
            if sim >= min_sim:
                scored.append(
                    (
                        sim,
                        {
                            "episode_id": episode_id,
                            "task_type": tt,
                            "agent_role": role,
                            "outcome": outcome,
                            "abstract": abstract,
                            "created_at_ms": created_at_ms,
                            "situation_signature": sit_sig,
                        },
                    )
                )

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:limit]

    def search_l1_flat(
        self,
        query_embedding: List[float],
        limit: int = 8,
        min_sim: float = 0.15,
        max_candidates: int = 3000,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Baseline flat retrieval without task/role filtering."""
        cur = self.conn.execute(
            """
            SELECT episode_id, task_type, agent_role, outcome, abstract, created_at_ms, abstract_embedding_json, situation_signature
            FROM episodes
            ORDER BY created_at_ms DESC
            LIMIT ?
            """,
            (max_candidates,),
        )

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for (episode_id, tt, role, outcome, abstract, created_at_ms, emb_json, sit_sig) in cur.fetchall():
            emb = json.loads(emb_json)
            sim = cosine(query_embedding, emb)
            if sim >= min_sim:
                scored.append(
                    (
                        sim,
                        {
                            "episode_id": episode_id,
                            "task_type": tt,
                            "agent_role": role,
                            "outcome": outcome,
                            "abstract": abstract,
                            "created_at_ms": created_at_ms,
                            "situation_signature": sit_sig,
                        },
                    )
                )

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:limit]

    def fetch_l2_full_traces(self, episode_ids: List[str]) -> List[Dict[str, Any]]:
        if not episode_ids:
            return []
        qmarks = ",".join(["?"] * len(episode_ids))
        cur = self.conn.execute(
            f"""
            SELECT episode_id, task_type, agent_role, outcome, abstract, full_trace, cost_tokens, cost_latency_ms, created_at_ms
            FROM episodes
            WHERE episode_id IN ({qmarks})
            """,
            tuple(episode_ids),
        )
        rows = cur.fetchall()
        by_id = {r[0]: r for r in rows}
        out: List[Dict[str, Any]] = []
        for eid in episode_ids:
            r = by_id.get(eid)
            if not r:
                continue
            out.append(
                {
                    "episode_id": r[0],
                    "task_type": r[1],
                    "agent_role": r[2],
                    "outcome": r[3],
                    "abstract": r[4],
                    "full_trace": r[5],
                    "cost_tokens": r[6],
                    "cost_latency_ms": r[7],
                    "created_at_ms": r[8],
                }
            )
        return out

    def count_canonicals(self, task_type: str, agent_role: str) -> int:
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM canonical_situations WHERE task_type = ? AND agent_role = ?",
            (task_type, agent_role),
        )
        return cur.fetchone()[0]

    def merge_canonical_embedding(
        self,
        task_type: str,
        agent_role: str,
        label: str,
        new_embedding: List[float],
    ) -> None:
        """Average the existing canonical embedding with new_embedding and update in place."""
        cur = self.conn.execute(
            "SELECT embedding_json FROM canonical_situations WHERE task_type = ? AND agent_role = ? AND label = ?",
            (task_type, agent_role, label),
        )
        row = cur.fetchone()
        if not row:
            return
        existing = json.loads(row[0])
        merged = [0.5 * a + 0.5 * b for a, b in zip(existing, new_embedding)]
        norm = sum(x * x for x in merged) ** 0.5 or 1.0
        merged = [x / norm for x in merged]
        self.conn.execute(
            "UPDATE canonical_situations SET embedding_json = ? WHERE task_type = ? AND agent_role = ? AND label = ?",
            (json.dumps(merged), task_type, agent_role, label),
        )
        self.conn.commit()

    def register_canonical_situation(
        self,
        task_type: str,
        agent_role: str,
        label: str,
        embedding: List[float],
    ) -> str:
        """Register a canonical situation for semantic L0 matching.

        The label becomes the stable string used in the L0 hash key when a
        query situation is resolved to this canonical. Multiple surface forms
        (e.g. 'postgres timeout on replica', 'DB read timeout under load') will
        all map to the same label and accumulate signal together.

        Returns the canonical id (stable_hash_key of task_type/role/label).
        """
        cid = stable_hash_key(task_type, agent_role, label)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO canonical_situations
                (id, task_type, agent_role, label, embedding_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (cid, task_type, agent_role, label, json.dumps(embedding)),
        )
        self.conn.commit()
        return cid

    def resolve_situation(
        self,
        task_type: str,
        agent_role: str,
        situation_embedding: List[float],
        threshold: float = 0.55,
    ) -> Optional[str]:
        """Find the nearest canonical situation label for a given embedding.

        Returns the canonical label if the best cosine similarity exceeds
        threshold, otherwise returns None (caller falls back to raw situation).
        """
        cur = self.conn.execute(
            "SELECT label, embedding_json FROM canonical_situations WHERE task_type = ? AND agent_role = ?",
            (task_type, agent_role),
        )
        rows = cur.fetchall()
        if not rows:
            return None

        best_label, best_sim = None, -1.0
        for label, emb_json in rows:
            sim = cosine(situation_embedding, json.loads(emb_json))
            if sim > best_sim:
                best_sim = sim
                best_label = label

        return best_label if best_sim >= threshold else None

    def get_best_episode_for_situation(
        self,
        task_type: str,
        agent_role: str,
        situation_label: str,
        prefer_outcome: str = "success",
    ) -> Optional[Dict[str, Any]]:
        """Return the best past episode for a canonical situation label.

        Prefers the most recent episode with prefer_outcome; falls back to
        any outcome if none exists.
        """
        cur = self.conn.execute(
            """
            SELECT episode_id, task_type, agent_role, outcome, abstract, created_at_ms, situation_signature
            FROM episodes
            WHERE task_type = ? AND agent_role = ? AND situation_signature = ?
            ORDER BY
                CASE WHEN outcome = ? THEN 0 ELSE 1 END,
                created_at_ms DESC
            LIMIT 1
            """,
            (task_type, agent_role, situation_label, prefer_outcome),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "episode_id": row[0],
            "task_type": row[1],
            "agent_role": row[2],
            "outcome": row[3],
            "abstract": row[4],
            "created_at_ms": row[5],
            "situation_signature": row[6],
        }

    def upsert_l0_signal(self, key: str, task_type: str, agent_role: str, outcome: str, episode_id: str) -> None:
        now = int(time.time() * 1000)
        cur = self.conn.execute(
            "SELECT success_count, failure_count, warning_count FROM l0_signals WHERE key = ?",
            (key,),
        )
        row = cur.fetchone()
        sc, fc, wc = (0, 0, 0) if not row else row

        if outcome == "success":
            sc += 1
        elif outcome == "failure":
            fc += 1
        else:
            wc += 1

        self.conn.execute(
            """
            INSERT INTO l0_signals (
                key, task_type, agent_role, success_count, failure_count, warning_count,
                last_outcome, last_episode_id, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                task_type=excluded.task_type,
                agent_role=excluded.agent_role,
                success_count=excluded.success_count,
                failure_count=excluded.failure_count,
                warning_count=excluded.warning_count,
                last_outcome=excluded.last_outcome,
                last_episode_id=excluded.last_episode_id,
                updated_at_ms=excluded.updated_at_ms
            """,
            (key, task_type, agent_role, sc, fc, wc, outcome, episode_id, now),
        )
        self.conn.commit()

    def get_l0_signal(self, key: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT key, task_type, agent_role, success_count, failure_count, warning_count,
                   last_outcome, last_episode_id, updated_at_ms
            FROM l0_signals
            WHERE key = ?
            """,
            (key,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "key": row[0],
            "task_type": row[1],
            "agent_role": row[2],
            "success_count": row[3],
            "failure_count": row[4],
            "warning_count": row[5],
            "last_outcome": row[6],
            "last_episode_id": row[7],
            "updated_at_ms": row[8],
        }


def _normalize_situation(text: str) -> str:
    """Normalize situation text before hashing.

    Handles the most common surface variations that should map to the same
    L0 signal: case differences, extra whitespace, and leading/trailing
    whitespace. Does not handle true paraphrases (e.g. 'rate limit error'
    vs 'HTTP 429 Too Many Requests') — those require embedding-based fuzzy
    matching, which is a planned future extension.
    """
    import re
    return re.sub(r"\s+", " ", text.lower().strip())


def stable_hash_key(task_type: str, agent_role: str, situation: str) -> str:
    normalized = _normalize_situation(situation)
    raw = f"{task_type}||{agent_role}||{normalized}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]
