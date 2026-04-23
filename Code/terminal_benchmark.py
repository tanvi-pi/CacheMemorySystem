from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Optional

from Embedder import Embedder
from MemoryStore import MemoryStore
from MultiAgentSystem import MultiAgentSystem
from RetrievalContext import RetrievalContext
from TieredRetrievalPolicy import FlatRetrievalPolicy


TASKS: Dict[str, str] = {
    "db_query_debug": "planner",
    "api_integration": "coder",
    "code_review": "reviewer",
    "incident_investigation": "researcher",
    "release_validation": "executor",
}

# Expanded: 6 patterns per task (was 3) — harder search space
PATTERNS = {
    "db_query_debug": [
        "postgres timeout index",
        "high cardinality join",
        "missing analyze stats",
        "deadlock on concurrent writes",
        "bloated table vacuum needed",
        "connection pool exhaustion",
    ],
    "api_integration": [
        "oauth refresh token",
        "rate limit handling",
        "schema mismatch",
        "webhook signature validation",
        "pagination cursor drift",
        "ssl certificate pinning failure",
    ],
    "code_review": [
        "race condition",
        "unsafe cast",
        "missing test coverage",
        "sql injection vector",
        "memory leak in event listener",
        "unhandled promise rejection",
    ],
    "incident_investigation": [
        "service flapping alerts",
        "error spike after deploy",
        "network saturation",
        "disk iops throttling",
        "pod oom killed in cluster",
        "dns resolution failures",
    ],
    "release_validation": [
        "canary rollback needed",
        "e2e smoke failures",
        "migration checks pending",
        "feature flag misconfiguration",
        "rollout stuck at threshold",
        "health check endpoint down",
    ],
}

# Natural-language seed abstracts — describe the situation semantically WITHOUT
# repeating the exact pattern phrase. Forces real semantic matching.
SEED_ABSTRACTS: Dict[str, str] = {
    # db_query_debug
    "postgres timeout index": (
        "A high-traffic read endpoint began timing out after a traffic spike. Investigation "
        "revealed a full sequential scan on a heavily queried column with no supporting index. "
        "Adding a targeted B-tree index resolved the latency. Always verify index coverage before "
        "deploying endpoints that touch large tables."
    ),
    "high cardinality join": (
        "A reporting query joining two multi-million-row tables exceeded the statement timeout. "
        "The query planner chose a nested loop due to stale statistics. Rewriting with a CTE and "
        "running ANALYZE immediately restored a fast execution plan."
    ),
    "missing analyze stats": (
        "After a bulk data load the optimizer made poor index choices, causing major slowdowns. "
        "Table row-count estimates were wildly off because statistics had not been refreshed. "
        "Running ANALYZE immediately restored optimal plan selection across all affected queries."
    ),
    "deadlock on concurrent writes": (
        "Write throughput dropped to zero as two transactions blocked each other indefinitely. "
        "Both were acquiring row locks in opposite order under concurrent load, creating a cycle. "
        "Standardizing lock acquisition order across all write paths eliminated the deadlock entirely."
    ),
    "bloated table vacuum needed": (
        "Table size on disk grew 8x while live row count stayed flat, causing query slowdowns. "
        "Dead tuples from frequent deletes accumulated faster than the background cleanup could process. "
        "Forcing a full vacuum and tuning the autovacuum thresholds resolved the bloat immediately."
    ),
    "connection pool exhaustion": (
        "Service deployments started failing as the database refused new connection attempts. "
        "All available pool slots were held by long-running transactions that were never released. "
        "Reducing transaction scope, adding timeouts, and right-sizing the pool resolved the issue."
    ),
    # api_integration
    "oauth refresh token": (
        "Users were silently logged out after one hour when their access tokens expired. "
        "The credential renewal flow was not being triggered automatically before the expiry window. "
        "Implementing proactive token refresh five minutes before expiration resolved all logouts."
    ),
    "rate limit handling": (
        "Bursts of upstream calls resulted in rejection responses from the external service. "
        "The client had no backoff logic and failed immediately on quota errors without retry. "
        "Adding exponential backoff with jitter and a token bucket limiter resolved the failures."
    ),
    "schema mismatch": (
        "The integration silently dropped data after the upstream vendor updated their API contract. "
        "Field names were renamed and the parser discarded all unrecognized keys without alerting. "
        "Adding schema versioning and strict validation with drift alerts prevented future silent failures."
    ),
    "webhook signature validation": (
        "Incoming event notifications were accepted without verifying the sender's authenticity. "
        "Any caller could forge payloads and trigger internal business logic without authorization. "
        "Implementing HMAC-SHA256 signature validation using the shared secret closed the security gap."
    ),
    "pagination cursor drift": (
        "Resource listings across pages returned duplicate items and skipped others intermittently. "
        "Offset-based pagination became inconsistent when items were inserted between page fetches. "
        "Switching to cursor-based keyset pagination with a stable sort key eliminated the drift."
    ),
    "ssl certificate pinning failure": (
        "HTTPS connections to the external service failed after the vendor rotated their certificate. "
        "The hardcoded fingerprint no longer matched and connections were rejected at the TLS layer. "
        "Moving from static cert pinning to CA-chain validation with a rotation checklist fixed this."
    ),
    # code_review
    "race condition": (
        "A data race between two concurrent goroutines caused intermittent failures under load. "
        "Shared mutable state was being read and written concurrently without any synchronization. "
        "Adding a mutex around the critical section eliminated the non-deterministic behavior."
    ),
    "unsafe cast": (
        "A runtime panic occurred in production when an interface value held an unexpected type. "
        "The type assertion was written without a comma-ok guard and panicked on any mismatch. "
        "Adding checked assertions with graceful error handling resolved the crash entirely."
    ),
    "missing test coverage": (
        "A regression was introduced in error handling code that had never been exercised by tests. "
        "The happy path was well covered but all failure branches were completely untested. "
        "Adding table-driven tests for every error condition caught the bug before the next release."
    ),
    "sql injection vector": (
        "A search endpoint constructed queries by string-formatting user input without parameterization. "
        "An attacker could escape the string context and execute arbitrary SQL commands on the database. "
        "Replacing all formatted queries with parameterized statements closed the vulnerability completely."
    ),
    "memory leak in event listener": (
        "Heap memory grew without bound as a component mounted and unmounted repeatedly in tests. "
        "Event handlers added on initialization were never removed on teardown, retaining object references. "
        "Adding proper listener cleanup in the component teardown lifecycle resolved the unbounded growth."
    ),
    "unhandled promise rejection": (
        "An async error was silently swallowed because the returned promise had no error handler. "
        "The failure propagated nowhere, leaving partial state written to the database undetected. "
        "Adding explicit catch handlers and a global unhandled rejection hook surfaced all failures."
    ),
    # incident_investigation
    "service flapping alerts": (
        "On-call was paged repeatedly as the service oscillated between healthy and unhealthy states. "
        "A memory pressure loop caused the process to crash and restart every few minutes under load. "
        "Increasing memory limits and fixing the underlying leak finally stabilized the service."
    ),
    "error spike after deploy": (
        "Error rate jumped from near-zero to 30% within minutes of the deployment completing. "
        "A breaking dependency change was introduced without a feature flag or rollback plan. "
        "Immediately rolling back the deployment restored baseline error rates across all endpoints."
    ),
    "network saturation": (
        "All services degraded simultaneously as inter-service call latency spiked to several seconds. "
        "A misconfigured batch job was saturating the primary network interface with bulk data transfers. "
        "Throttling the batch job and moving it to a dedicated interface restored normal traffic."
    ),
    "disk iops throttling": (
        "Write latency jumped 10x as the application stalled waiting on storage operations to complete. "
        "The cloud volume's provisioned IOPS budget was exhausted by a runaway debug logging loop. "
        "Disabling the log loop and upgrading the volume's IOPS allocation resolved the stalls immediately."
    ),
    "pod oom killed in cluster": (
        "The service kept restarting every few minutes due to out-of-memory termination by the kernel. "
        "Memory limits were set far below the actual working set size under production traffic levels. "
        "Profiling memory usage, fixing a cache eviction bug, and raising limits resolved the restarts."
    ),
    "dns resolution failures": (
        "Cross-service calls failed with hostname resolution errors after a cluster network change. "
        "CoreDNS pods were overloaded and dropping queries under the new traffic distribution pattern. "
        "Scaling CoreDNS replicas and fixing the upstream forwarder configuration resolved the failures."
    ),
    # release_validation
    "canary rollback needed": (
        "The new version serving 5% of traffic showed error rates significantly above the baseline. "
        "A subtle behavior change caused failures only for users in a specific account state. "
        "Rolling back the canary and fixing the edge case before re-promoting resolved the issue."
    ),
    "e2e smoke failures": (
        "The post-deployment smoke suite failed on the critical checkout user journey immediately. "
        "Investigation revealed both a broken test selector and an actual regression in the API response. "
        "Both the test and the API format were fixed before re-running end-to-end validation."
    ),
    "migration checks pending": (
        "The new service version started but errored immediately because expected schema columns were absent. "
        "The migration script had not been executed before deploying the updated application code. "
        "Adding a pre-deploy migration verification gate in the pipeline prevented any recurrence."
    ),
    "feature flag misconfiguration": (
        "An incomplete feature was exposed to all production users because a kill-switch was in the wrong state. "
        "Configuration drift between staging and production left the flag enabled after environment promotion. "
        "Auditing flag states as a release checklist step and using environment-specific overrides fixed this."
    ),
    "rollout stuck at threshold": (
        "Progressive delivery paused at 20% traffic because success metrics were just below the SLO gate. "
        "The new version had a subtle performance regression visible only under sustained production load. "
        "Profiling and fixing the regression allowed the rollout to proceed through all health gates."
    ),
    "health check endpoint down": (
        "Newly deployed instances never joined the load balancer pool because the readiness probe kept failing. "
        "A missing environment variable caused the health check handler to throw an error on every call. "
        "Adding the variable to the deployment spec and verifying all probe paths in CI resolved the issue."
    ),
}

# Situation paraphrases: alternative free-text descriptions of the same situation.
# Used to test semantic L0 canonical resolution — a paraphrased situation should
# resolve to the same canonical label as the exact pattern string.
SITUATION_PARAPHRASES: Dict[str, List[str]] = {
    "postgres timeout index":        ["database read hanging on unindexed column", "slow query on high-traffic table"],
    "high cardinality join":         ["massive cross-table join producing slow plans", "nested loop on large result set"],
    "missing analyze stats":         ["planner using stale row estimates", "bad execution plan after bulk load"],
    "deadlock on concurrent writes": ["circular lock wait between transactions", "transactions blocking each other on write"],
    "bloated table vacuum needed":   ["dead row accumulation slowing queries", "table disk usage far exceeds live rows"],
    "connection pool exhaustion":    ["no available DB connections under load", "pool slots all occupied, new requests hang"],
    "oauth refresh token":           ["credential expiring mid-session", "token renewal not triggered after expiry"],
    "rate limit handling":           ["quota errors from upstream API", "external service rejecting bursts of requests"],
    "schema mismatch":               ["API response structure changed unexpectedly", "vendor contract update broke parsing"],
    "webhook signature validation":  ["incoming payloads accepted without auth check", "forged callbacks not being rejected"],
    "pagination cursor drift":       ["duplicate results across page fetches", "listing inconsistent when data changes mid-page"],
    "ssl certificate pinning failure": ["TLS handshake failing after cert rotation", "pinned fingerprint outdated after renewal"],
    "race condition":                ["shared state accessed without synchronization", "flaky test under concurrent load"],
    "unsafe cast":                   ["narrowing type coercion without bounds check", "type assertion panicking at runtime"],
    "missing test coverage":         ["error branches never exercised by tests", "retry logic has no automated validation"],
    "sql injection vector":          ["unsanitized user input passed to query", "string concatenation into SQL statement"],
    "memory leak in event listener": ["listener not removed causing heap growth", "event callbacks accumulating over time"],
    "unhandled promise rejection":   ["async error silently swallowed", "rejected promise with no catch handler"],
    "service flapping alerts":       ["service repeatedly cycling up and down", "health check oscillating between pass and fail"],
    "error spike after deploy":      ["errors increased immediately after release", "rollout correlated with failure rate jump"],
    "network saturation":            ["bandwidth exhausted causing packet loss", "all services slow due to link congestion"],
    "disk iops throttling":          ["storage throughput capped causing latency", "disk ops hitting provider rate limit"],
    "pod oom killed in cluster":     ["container killed due to memory limit", "OOM event in kubernetes workload"],
    "dns resolution failures":       ["hostname lookup failing intermittently", "service discovery broken due to DNS"],
    "failed smoke test":             ["post-deploy validation failing", "basic health check not passing after release"],
    "rollback triggered":            ["release reverted due to errors", "deployment rolled back after failure detection"],
    "canary rollback needed":        ["canary showing elevated error rate", "new version performing worse than baseline"],
    "config drift in staging":       ["staging environment diverged from production", "env var mismatch between environments"],
    "dependency version conflict":   ["package version incompatibility blocking deploy", "conflicting library versions in release"],
}

# Paraphrases: 3 phrasings per pattern that share NO keywords with the pattern phrase.
# This is the core of the hard test — the system must match on semantics alone.
PARAPHRASES: Dict[str, List[str]] = {
    # db_query_debug
    "postgres timeout index": [
        "database reads are hanging on a frequently queried column — no supporting structure appears to exist",
        "slow query log shows full table scans on a high-traffic table, latency is climbing fast",
        "the API response time degraded and it traced back to unoptimized lookups causing lock waits",
    ],
    "high cardinality join": [
        "two large tables being combined are producing extremely slow query plans with nested loops",
        "the query planner picked the worst possible execution path, result set is huge before filtering",
        "cross-table aggregation that should take milliseconds is now running for several minutes",
    ],
    "missing analyze stats": [
        "the optimizer is making poor choices — table row-count estimates look wildly wrong",
        "after a bulk load the planner is picking the wrong index based on incorrect estimates",
        "autovacuum has not run recently and the execution plan quality has regressed significantly",
    ],
    "deadlock on concurrent writes": [
        "two transactions are blocking each other and neither can commit, rollbacks are constant",
        "concurrent update operations are acquiring locks in opposite order causing circular waits",
        "write throughput has dropped to zero and logs show repeated blocking errors between services",
    ],
    "bloated table vacuum needed": [
        "table size on disk is enormous despite relatively few live rows, something is accumulating",
        "background cleanup is not keeping up with deletion volume and queries are slowing from bloat",
        "disk usage grew 10x without proportional data growth — dead row accumulation is suspected",
    ],
    "connection pool exhaustion": [
        "new database connections are being refused, all available slots appear to be occupied",
        "service is hanging trying to acquire a DB connection, the pool is undersized for traffic",
        "application startup failing because no connection can be obtained from the shared pool under load",
    ],
    # api_integration
    "oauth refresh token": [
        "users are being logged out mid-session, the credential is expiring and not being renewed",
        "authentication is failing after an hour — the token renewal flow is not being triggered",
        "getting 401 errors periodically after initial login succeeds, suspect the credential lifecycle is broken",
    ],
    "rate limit handling": [
        "third-party API requests are failing with quota errors, we are not backing off correctly",
        "hitting the external service's request ceiling and the client is not retrying with delays",
        "bursts of upstream calls are getting rejected — no throttling or exponential delays in place",
    ],
    "schema mismatch": [
        "the external API response changed its structure and our parser is silently dropping data",
        "integration broke after the vendor updated their contract — field types no longer align",
        "deserialization fails intermittently because the upstream payload format shifted unexpectedly",
    ],
    "webhook signature validation": [
        "incoming event notifications are being accepted without verifying the sender's identity",
        "the cryptographic check on incoming payloads is missing — any caller can forge events",
        "we are processing callbacks that have not been authenticated — a security gap in the handler",
    ],
    "pagination cursor drift": [
        "listing resources across pages returns duplicates and sometimes skips items entirely",
        "the page position becomes invalid between requests because underlying data is changing",
        "offset-based listing is inconsistent when new items are inserted between page fetches",
    ],
    "ssl certificate pinning failure": [
        "HTTPS connections to the external service fail whenever their certificate is rotated",
        "the hardcoded certificate fingerprint no longer matches after the vendor renewed theirs",
        "TLS handshake is being rejected because the pinned certificate hash is outdated",
    ],
    # code_review
    "race condition": [
        "two goroutines are reading and writing shared state without any synchronization mechanism",
        "test passes in isolation but flakes under concurrent load — shared mutable variable is suspect",
        "timing-dependent bug where the outcome depends on which thread gets scheduled first",
    ],
    "unsafe cast": [
        "integer is being coerced to a narrower type without checking bounds first, could truncate silently",
        "type assertion made without a guard — panics at runtime when the interface holds an unexpected value",
        "coercing a floating point to integer discards the fractional part causing calculation errors",
    ],
    "missing test coverage": [
        "the error handling branches in this module have never been exercised by any automated test",
        "happy path is covered but all edge cases and failure modes remain completely untested",
        "the code coverage report shows the retry and fallback logic is entirely without test validation",
    ],
    "sql injection vector": [
        "user input is being concatenated directly into a query string without any sanitization",
        "the search handler builds the database command by string formatting — attacker can inject commands",
        "raw query construction with unsanitized input found in the user-facing API endpoint",
    ],
    "memory leak in event listener": [
        "objects are being kept alive because handlers hold references and are never deregistered",
        "heap grows continuously as the component mounts and unmounts — listeners are never cleaned up",
        "subscription is added on initialization but no corresponding teardown causes unbounded growth",
    ],
    "unhandled promise rejection": [
        "async error is swallowed silently because the returned promise has no catch attached",
        "failure in the async call chain propagates nowhere — the process crashed on an unhandled rejection",
        "missing await causes the exception to be lost, leaving partial state written to the database",
    ],
    # incident_investigation
    "service flapping alerts": [
        "the health check is oscillating rapidly between passing and failing, paging on-call repeatedly",
        "the service keeps cycling between up and down without fully recovering to stable state",
        "monitoring shows repeated brief outages spaced minutes apart — something keeps crashing and restarting",
    ],
    "error spike after deploy": [
        "error rate jumped immediately after the last push, likely a regression was introduced",
        "5xx responses went from near-zero to 30% right after the release, the correlation is obvious",
        "the deployment finished successfully but production began throwing exceptions within minutes",
    ],
    "network saturation": [
        "bandwidth utilization on the primary link is maxed out and packets are being dropped",
        "throughput has degraded across all services — the network interface appears to be the bottleneck",
        "TCP retransmission rate is spiking and latency is extreme on all internal service calls",
    ],
    "disk iops throttling": [
        "storage layer is hitting its throughput ceiling and write latency has jumped dramatically",
        "application is stalling because the underlying volume is being throttled by the provider",
        "the storage operations budget is exhausted — reads and writes are queued and everything touching disk is slow",
    ],
    "pod oom killed in cluster": [
        "containers are being terminated by the kernel for exceeding their configured memory limit",
        "the workload keeps restarting — the OOM killer is evicting it before it can finish",
        "out-of-memory events in the cluster are causing the service to restart every few minutes",
    ],
    "dns resolution failures": [
        "services cannot resolve each other by name, internal traffic is failing with hostname lookup errors",
        "name lookups are timing out intermittently — the cluster DNS component appears overloaded",
        "cross-service calls failing with host-not-found errors after a recent cluster network change",
    ],
    # release_validation
    "canary rollback needed": [
        "the new version is serving a small percentage of traffic and the error rate is elevated",
        "canary metrics are outside acceptable bounds — success rate dropped below the threshold",
        "the partial rollout is underperforming baseline and needs to be reverted before full promotion",
    ],
    "e2e smoke failures": [
        "the automated end-to-end tests that run after deployment are failing on critical user flows",
        "post-release synthetic monitoring detected a broken user journey — checkout or login is broken",
        "the smoke suite immediately caught a regression after the release was pushed to production",
    ],
    "migration checks pending": [
        "the schema changes have not been applied yet and the new code already expects them to exist",
        "the release is blocked because the pre-flight database script has not been verified to succeed",
        "structural changes need to be executed before the new service version can function correctly",
    ],
    "feature flag misconfiguration": [
        "a flag that was supposed to be disabled in production is active, exposing incomplete work",
        "wrong feature gate value in the config caused unreleased code to be visible to all users",
        "configuration drift between environments left a kill-switch in the wrong state after promotion",
    ],
    "rollout stuck at threshold": [
        "automated progressive delivery has paused because the success metric fell just below the limit",
        "the deployment pipeline is waiting because health check percentages are borderline failing",
        "the staged rollout has halted because the SLO is being violated by the new version",
    ],
    "health check endpoint down": [
        "the liveness probe is returning errors and the load balancer has marked the instance unhealthy",
        "the readiness check is failing post-deploy — instances never join the serving pool",
        "the status endpoint is unreachable and is blocking the deployment from completing successfully",
    ],
}

# Cross-task confusors: queries that could plausibly look like a different task type.
# Tests whether the system correctly anchors retrieval to the right agent/task.
CROSS_TASK_CONFUSORS: List[Dict[str, Any]] = [
    {
        "task_type": "db_query_debug",
        "role": "planner",
        "situation": "postgres timeout index",
        "query": "the system is degraded and users are getting errors — what should I check first?",
        "confusor_type": "db_phrased_as_incident",
    },
    {
        "task_type": "api_integration",
        "role": "coder",
        "situation": "rate limit handling",
        "query": "I spotted a pattern in this code where upstream calls might overwhelm the external service — how do we fix it?",
        "confusor_type": "api_phrased_as_code_review",
    },
    {
        "task_type": "code_review",
        "role": "reviewer",
        "situation": "race condition",
        "query": "production is flapping intermittently and restarts are not fixing it — something seems non-deterministic",
        "confusor_type": "code_phrased_as_incident",
    },
    {
        "task_type": "incident_investigation",
        "role": "researcher",
        "situation": "error spike after deploy",
        "query": "should we roll back the release? errors started right after the push — how do we diagnose and decide?",
        "confusor_type": "incident_phrased_as_release",
    },
    {
        "task_type": "release_validation",
        "role": "executor",
        "situation": "canary rollback needed",
        "query": "the new version is returning unexpected responses for some users — is this a code bug or a rollout problem?",
        "confusor_type": "release_phrased_as_code",
    },
    {
        "task_type": "db_query_debug",
        "role": "planner",
        "situation": "connection pool exhaustion",
        "query": "the API is timing out for everyone — infrastructure team says the application is the culprit",
        "confusor_type": "db_phrased_as_api",
    },
    {
        "task_type": "incident_investigation",
        "role": "researcher",
        "situation": "network saturation",
        "query": "all our services are slow at once — could this be related to the deployment we just did?",
        "confusor_type": "incident_phrased_as_release",
    },
]


def _build_system(
    db_path: str,
    embedder: Any = None,
    abstract_generator: Any = None,
    always_generate_abstracts: bool = False,
) -> MultiAgentSystem:
    roles = ["planner", "coder", "reviewer", "researcher", "executor"]
    return MultiAgentSystem(
        roles=roles,
        task_role_map=TASKS,
        db_path=db_path,
        embed_dim=512,
        embedder=embedder,
        abstract_generator=abstract_generator,
        always_generate_abstracts=always_generate_abstracts,
    )


def _seed_canonical_situations(store: MemoryStore, embedder: Any) -> None:
    """Register each PATTERN as a canonical situation for semantic L0 matching.

    This allows surface variations of a known situation (e.g. 'postgres read
    timeout under load') to resolve to the canonical label ('postgres timeout
    index') and share the same L0 signal rather than creating a new key each
    time.
    """
    for task_type, role in TASKS.items():
        for pattern in PATTERNS[task_type]:
            emb = embedder.embed(pattern)
            store.register_canonical_situation(task_type, role, pattern, emb)
    print(f"[canonicals] Registered {sum(len(v) for v in PATTERNS.values())} canonical situations.")


def _seed_episodes(system: MultiAgentSystem, episodes_per_task: int, rng: random.Random) -> None:
    idx = 0
    for task_type, role in TASKS.items():
        patterns = PATTERNS[task_type]
        for _ in range(episodes_per_task):
            pat = rng.choice(patterns)
            fail_heavy = pat in (
                "postgres timeout index", "rate limit handling", "race condition",
                "deadlock on concurrent writes", "pod oom killed in cluster",
                "dns resolution failures",
            )
            if fail_heavy:
                outcome = rng.choices(["failure", "warning", "success"], weights=[0.80, 0.15, 0.05])[0]
            else:
                outcome = rng.choices(["success", "warning", "failure"], weights=[0.70, 0.20, 0.10])[0]

            base_abstract = SEED_ABSTRACTS.get(pat, f"Handled {pat} for {task_type}. Outcome: {outcome}.")
            abstract = f"{base_abstract} Final outcome: {outcome}."

            full_trace = (
                f"[{role.upper()}] Task: {task_type} | Situation: {pat} | Outcome: {outcome}\n"
                f"Step 1: Received task. Context: {abstract[:150]}\n"
                f"Step 2: Ran diagnostics, reviewed logs, identified root cause.\n"
                f"Step 3: Applied resolution strategy. Monitored results.\n"
                f"Step 4: Confirmed outcome={outcome}. Documented findings.\n"
            ) * 4

            system.finalize_episode(
                role=role,
                episode_id=f"ep_{task_type}_{idx}",
                task_type=task_type,
                outcome=outcome,
                abstract=abstract,
                full_trace=full_trace,
                situation_signature=pat,
                cost_tokens=rng.randint(800, 2400),
                cost_latency_ms=rng.randint(80, 420),
            )
            idx += 1


def _generate_queries(num_queries: int, rng: random.Random) -> List[Dict[str, object]]:
    """
    Query distribution (aggressive testing):
      50% paraphrased known  — semantic match required, no keyword leakage
      15% verbatim known     — easy baseline (old behavior)
      15% cross-task confusor — query sounds like a different task type
      20% novel              — genuinely unseen situations
    """
    queries: List[Dict[str, object]] = []
    tasks = list(TASKS.keys())

    for i in range(num_queries):
        roll = rng.random()

        if roll < 0.50:
            # Paraphrased known — hard
            # Use a paraphrased situation string (not the exact pattern) so that
            # semantic canonical resolution is exercised on the L0 path.
            task_type = rng.choice(tasks)
            role = TASKS[task_type]
            situation = rng.choice(PATTERNS[task_type])
            sit_paraphrases = SITUATION_PARAPHRASES.get(situation, [])
            situation_text = rng.choice(sit_paraphrases) if sit_paraphrases else situation
            paraphrases = PARAPHRASES.get(situation, [])
            query = rng.choice(paraphrases) if paraphrases else f"I'm dealing with a variant of {situation}"
            queries.append({
                "task_type": task_type,
                "role": role,
                "situation": situation_text,   # paraphrased — canonical resolution needed
                "ground_truth_situation": situation,  # canonical label for scoring
                "query": query,
                "confidence": rng.uniform(0.25, 0.9),
                "retry_count": rng.choice([0, 0, 1, 2, 3]),
                "known": True,
                "query_type": "paraphrased",
            })

        elif roll < 0.65:
            # Verbatim known — easy baseline
            task_type = rng.choice(tasks)
            role = TASKS[task_type]
            situation = rng.choice(PATTERNS[task_type])
            query = f"Need help with {task_type}: {situation}. What should I do next?"
            queries.append({
                "task_type": task_type,
                "role": role,
                "situation": situation,
                "query": query,
                "confidence": rng.uniform(0.25, 0.9),
                "retry_count": rng.choice([0, 0, 1, 2, 3]),
                "known": True,
                "query_type": "verbatim",
            })

        elif roll < 0.80:
            # Cross-task confusor
            confusor = rng.choice(CROSS_TASK_CONFUSORS)
            queries.append({
                "task_type": confusor["task_type"],
                "role": confusor["role"],
                "situation": confusor["situation"],
                "query": confusor["query"],
                "confidence": rng.uniform(0.25, 0.75),
                "retry_count": rng.choice([0, 1, 2]),
                "known": True,
                "query_type": "confusor",
                "confusor_type": confusor["confusor_type"],
            })

        else:
            # Novel — genuinely unseen
            task_type = rng.choice(tasks)
            role = TASKS[task_type]
            novel_queries = [
                f"intermittent data corruption is appearing in the {task_type} pipeline",
                f"performance regression after a library upgrade is affecting {task_type}",
                f"unexpected behavior only under high concurrency in the {task_type} system",
                f"third-party breaking changes are affecting {task_type} operations",
                f"a rarely-triggered code path in {task_type} is producing wrong results",
            ]
            queries.append({
                "task_type": task_type,
                "role": role,
                "situation": f"novel_{task_type}_{i}",
                "query": rng.choice(novel_queries),
                "confidence": rng.uniform(0.25, 0.9),
                "retry_count": rng.choice([0, 0, 1, 2, 3]),
                "known": False,
                "query_type": "novel",
            })

    return queries


def _collect_agent_memory(store: MemoryStore, roles: List[str]) -> Dict[str, Any]:
    memory: Dict[str, Any] = {}
    for role in roles:
        cur = store.conn.execute(
            """SELECT key, task_type, success_count, failure_count, warning_count,
                      last_outcome, last_episode_id, updated_at_ms
               FROM l0_signals WHERE agent_role = ? ORDER BY updated_at_ms DESC""",
            (role,),
        )
        l0 = [
            {
                "hash_key": r[0], "task_type": r[1],
                "success_count": r[2], "failure_count": r[3], "warning_count": r[4],
                "fail_rate": round(r[3] / max(r[2]+r[3]+r[4], 1), 3),
                "last_outcome": r[5], "last_episode_id": r[6],
            }
            for r in cur.fetchall()
        ]
        cur = store.conn.execute(
            """SELECT episode_id, task_type, outcome, abstract, full_trace,
                      cost_tokens, cost_latency_ms, created_at_ms
               FROM episodes WHERE agent_role = ? ORDER BY created_at_ms DESC""",
            (role,),
        )
        l1, l2 = [], []
        for r in cur.fetchall():
            l1.append({
                "episode_id": r[0], "task_type": r[1],
                "outcome": r[2], "abstract": r[3],
                "cost_tokens": r[5], "cost_latency_ms": r[6],
            })
            l2.append({
                "episode_id": r[0], "task_type": r[1],
                "outcome": r[2], "full_trace": r[4],
            })
        memory[role] = {"l0_signals": l0, "l1_abstracts": l1, "l2_full_traces": l2}
    return memory


def _score_accuracy(
    situation: str,
    hits: List[Dict[str, Any]],
    known: bool,
    task_type: str = "",
) -> Dict[str, Any]:
    """
    Score retrieval quality.
    - known queries: exact situation_signature match, returns top1/top3/MRR/rank
    - novel queries: analog score — did top-3 hits come from the right task_type?
    """
    if not known:
        analog = any(h.get("task_type") == task_type for h in hits[:3]) if hits else False
        return {
            "evaluable": False,
            "reason": "novel pattern — no ground truth",
            "analog_top3_same_task": analog,
            "top1_task": hits[0].get("task_type") if hits else None,
        }
    if not hits:
        return {
            "evaluable": True,
            "top1_correct": False,
            "top3_correct": False,
            "any_correct": False,
            "mrr": 0.0,
            "rank": None,
            "reason": "no hits returned",
        }

    rank = None
    for i, h in enumerate(hits):
        if h.get("situation_signature") == situation:
            rank = i + 1
            break

    top1 = rank == 1
    top3 = rank is not None and rank <= 3
    any_correct = rank is not None
    mrr = round(1.0 / rank, 4) if rank else 0.0

    return {
        "evaluable": True,
        "top1_correct": top1,
        "top3_correct": top3,
        "any_correct": any_correct,
        "mrr": mrr,
        "rank": rank,
        "top1_similarity": round(hits[0]["sim"], 4),
        "correct_similarity": round(hits[rank - 1]["sim"], 4) if rank else None,
        "reason": f"correct episode at rank {rank}" if rank else "correct episode not in top results",
    }


def _build_trace(idx: int, q: Dict[str, Any], tiered: Dict[str, Any], flat: Dict[str, Any]) -> Dict[str, Any]:
    tier = str(tiered.get("tier", "L1"))

    if tier == "L0":
        sig = tiered.get("signals", {}) or {}
        total = max(sig.get("success_count", 0) + sig.get("failure_count", 0) + sig.get("warning_count", 0), 1)
        source = (
            f"L0 signal matched (fail_rate="
            f"{round(sig.get('failure_count', 0) / total, 2)})"
            f" → agent warned, no embedding search needed"
        )
    elif tier == "L1":
        hits = tiered.get("abstract_hits", [])
        top_sim = round(hits[0]["sim"], 3) if hits else 0.0
        source = f"L1 abstract retrieved (top similarity={top_sim}) → abstract summary used"
    elif tier == "L2":
        hits = tiered.get("abstract_hits", [])
        top_sim = round(hits[0]["sim"], 3) if hits else 0.0
        source = f"L2 full trace retrieved (L1 similarity={top_sim} triggered escalation)"
    else:
        source = "unknown"

    known = bool(q.get("known", True))
    t_hits = tiered.get("abstract_hits", [])
    f_hits = flat.get("abstract_hits", [])

    # Use ground_truth_situation for scoring when available (paraphrased queries
    # have a paraphrased situation string but the stored episode uses the canonical).
    score_situation = str(q.get("ground_truth_situation", q["situation"]))
    t_acc = _score_accuracy(score_situation, t_hits, known, str(q["task_type"]))
    f_acc = _score_accuracy(score_situation, f_hits, known, str(q["task_type"]))

    tiered_correct = t_acc["top1_correct"] if t_acc["evaluable"] else None
    flat_correct = f_acc["top1_correct"] if f_acc["evaluable"] else None

    return {
        "query_id": idx,
        "task_type": q["task_type"],
        "agent_role": q["role"],
        "situation": q["situation"],
        "query": q["query"],
        "query_type": q.get("query_type", "unknown"),
        "confusor_type": q.get("confusor_type"),
        "known_pattern": known,
        "confidence": round(float(q["confidence"]), 3),
        "retry_count": int(q["retry_count"]),
        "tiered_correct": tiered_correct,
        "flat_correct": flat_correct,
        "tiered_retrieval": {
            "tier_used": tier,
            "memory_source": source,
            "accuracy": t_acc,
            "l0_signal": tiered.get("signals"),
            "l0_recommendation": tiered.get("l0_recommendation"),
            "l1_abstract_hits": [
                {
                    "episode_id": h["episode_id"],
                    "task_type": h["task_type"],
                    "outcome": h["outcome"],
                    "similarity": round(h["sim"], 4),
                    "abstract": h["abstract"],
                    "situation_signature": h.get("situation_signature"),
                }
                for h in t_hits
            ],
            "l2_full_traces": [
                {
                    "episode_id": h["episode_id"],
                    "task_type": h["task_type"],
                    "outcome": h["outcome"],
                    "full_trace": h["full_trace"],
                }
                for h in tiered.get("full_hits", [])
            ],
            "latency_ms": tiered["debug"]["latency_ms"],
            "retrieval_tokens": tiered["debug"]["retrieval_tokens"],
            "tiers_checked": tiered["debug"]["used"],
        },
        "flat_retrieval": {
            "tier_used": "flat",
            "memory_source": "global similarity search (no role/task filtering)",
            "accuracy": f_acc,
            "l1_abstract_hits": [
                {
                    "episode_id": h["episode_id"],
                    "task_type": h["task_type"],
                    "outcome": h["outcome"],
                    "similarity": round(h["sim"], 4),
                    "abstract": h["abstract"],
                    "situation_signature": h.get("situation_signature"),
                }
                for h in f_hits
            ],
            "l2_full_traces": [
                {
                    "episode_id": h["episode_id"],
                    "task_type": h["task_type"],
                    "outcome": h["outcome"],
                    "full_trace": h["full_trace"],
                }
                for h in flat.get("full_hits", [])
            ],
            "latency_ms": flat["debug"]["latency_ms"],
            "retrieval_tokens": flat["debug"]["retrieval_tokens"],
        },
    }


def _summarize(results: List[Dict[str, object]], label: str) -> None:
    latencies = [int(r["debug"]["latency_ms"]) for r in results]
    tokens = [int(r["debug"].get("retrieval_tokens", 0)) for r in results]
    tier_counts: Dict[str, int] = {}
    for r in results:
        t = str(r.get("tier"))
        tier_counts[t] = tier_counts.get(t, 0) + 1

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_tokens = sum(tokens) / len(tokens) if tokens else 0.0
    p95_idx = max(0, int(len(latencies) * 0.95) - 1)
    p95_latency = sorted(latencies)[p95_idx] if latencies else 0

    print(f"\n=== {label} ===")
    print(f"queries={len(results)}")
    print(f"avg_latency_ms={avg_latency:.2f}")
    print(f"p95_latency_ms={p95_latency}")
    print(f"avg_retrieval_tokens={avg_tokens:.2f}")
    print(f"tier_counts={tier_counts}")


def _print_accuracy_report(traces: List[Dict[str, Any]]) -> None:
    """Print a detailed accuracy report to the terminal."""
    evaluable = [t for t in traces if t["tiered_retrieval"]["accuracy"]["evaluable"]]
    novel = [t for t in traces if not t["tiered_retrieval"]["accuracy"]["evaluable"]]

    print(f"\n{'='*60}")
    print("  ACCURACY REPORT")
    print(f"{'='*60}")
    print(f"  Total queries : {len(traces)}")
    print(f"  Evaluable     : {len(evaluable)}  (known patterns)")
    print(f"  Novel         : {len(novel)}  (no ground truth)")

    if evaluable:
        top1 = sum(1 for t in evaluable if t["tiered_retrieval"]["accuracy"]["top1_correct"])
        top3 = sum(1 for t in evaluable if t["tiered_retrieval"]["accuracy"]["top3_correct"])
        mrr_vals = [t["tiered_retrieval"]["accuracy"]["mrr"] for t in evaluable]
        avg_mrr = sum(mrr_vals) / len(mrr_vals)

        f_top1 = sum(1 for t in evaluable if t["flat_retrieval"]["accuracy"]["top1_correct"])
        f_top3 = sum(1 for t in evaluable if t["flat_retrieval"]["accuracy"]["top3_correct"])
        f_mrr_vals = [t["flat_retrieval"]["accuracy"]["mrr"] for t in evaluable]
        f_avg_mrr = sum(f_mrr_vals) / len(f_mrr_vals)

        n = len(evaluable)
        print(f"\n  {'Metric':<20} {'Tiered':>10} {'Flat':>10}")
        print(f"  {'-'*40}")
        print(f"  {'Top-1 accuracy':<20} {top1}/{n} ({100*top1//n}%) {f_top1}/{n} ({100*f_top1//n}%)")
        print(f"  {'Top-3 accuracy':<20} {top3}/{n} ({100*top3//n}%) {f_top3}/{n} ({100*f_top3//n}%)")
        print(f"  {'MRR':<20} {avg_mrr:.3f}{'':>6} {f_avg_mrr:.3f}")

        # Breakdown by query_type
        for qtype in ("paraphrased", "verbatim", "confusor"):
            sub = [t for t in evaluable if t.get("query_type") == qtype]
            if not sub:
                continue
            t1 = sum(1 for t in sub if t["tiered_retrieval"]["accuracy"]["top1_correct"])
            print(f"\n  [{qtype.upper()}]  n={len(sub)}")
            print(f"    Tiered top-1: {t1}/{len(sub)} ({100*t1//len(sub)}%)")

    # Novel analog scoring
    analog_hits = sum(
        1 for t in novel
        if t["tiered_retrieval"]["accuracy"].get("analog_top3_same_task", False)
    )
    if novel:
        print(f"\n  [NOVEL]  analog top-3 same-task: {analog_hits}/{len(novel)} ({100*analog_hits//len(novel)}%)")

    print(f"{'='*60}\n")


def run_benchmark(
    db_path: str,
    episodes_per_task: int,
    num_queries: int,
    seed: int,
    use_openai: bool = False,
    llm_abstracts: bool = False,
    openai_key: Optional[str] = None,
    output_path: Optional[str] = None,
) -> None:
    for suffix in ("", "-wal", "-shm"):
        p = db_path + suffix
        if os.path.exists(p):
            os.remove(p)

    if use_openai:
        from Embedder import OpenAIEmbedder
        active_embedder: Any = OpenAIEmbedder(api_key=openai_key)
        print(f"[embedder] OpenAI {active_embedder.model} (dim={active_embedder.dim})")
    else:
        active_embedder = Embedder(dim=512)
        print("[embedder] local hash-based (dim=512)")

    abstract_generator: Any = None
    if llm_abstracts:
        from MemoryWriter import LLMAbstractGenerator
        abstract_generator = LLMAbstractGenerator(api_key=openai_key)
        print(f"[abstracts] LLM-generated via {abstract_generator.model}")
    else:
        print("[abstracts] manually constructed (no LLM)")

    total_patterns = sum(len(v) for v in PATTERNS.values())
    print(f"[benchmark] {total_patterns} patterns across {len(TASKS)} tasks")
    print(f"[benchmark] query mix: 50% paraphrased / 15% verbatim / 15% confusor / 20% novel")

    rng = random.Random(seed)
    system = _build_system(
        db_path,
        embedder=active_embedder,
        abstract_generator=abstract_generator,
        always_generate_abstracts=llm_abstracts,
    )
    print(f"[seeding] {episodes_per_task} episodes × {len(TASKS)} tasks = {episodes_per_task * len(TASKS)} total…")
    _seed_episodes(system, episodes_per_task, rng)
    print("[seeding] done\n")

    store = MemoryStore(db_path)
    _seed_canonical_situations(store, active_embedder)
    flat = FlatRetrievalPolicy(store, active_embedder, topk=3)

    queries = _generate_queries(num_queries, rng)

    tiered_results: List[Dict[str, object]] = []
    flat_results: List[Dict[str, object]] = []
    execution_traces: List[Dict[str, Any]] = []

    for idx, q in enumerate(queries):
        # Pass the raw (possibly paraphrased) situation to the tiered system so
        # L0's embedding-based canonical resolution is exercised. Scoring uses
        # ground_truth_situation (the canonical label) to check correctness.
        raw_situation = str(q["situation"])
        canonical_situation = str(q.get("ground_truth_situation", raw_situation))

        tiered = system.step(
            task_type=str(q["task_type"]),
            role=str(q["role"]),
            situation=raw_situation,
            user_query=str(q["query"]),
            confidence=float(q["confidence"]),
            retry_count=int(q["retry_count"]),
            latency_budget_ms=40,
            token_budget=2000,
        )
        tiered_results.append(tiered)

        flat_ctx = RetrievalContext(
            task_type=str(q["task_type"]),
            agent_role=str(q["role"]),
            situation=canonical_situation,
            query_text=str(q["query"]),
            confidence=float(q["confidence"]),
            retry_count=int(q["retry_count"]),
            latency_budget_ms=40,
            token_budget=2000,
        )
        flat_result = flat.retrieve(flat_ctx)
        flat_results.append(flat_result)

        execution_traces.append(_build_trace(idx, q, tiered, flat_result))

    _summarize(tiered_results, "Tiered Policy (L0/L1/L2)")
    _summarize(flat_results, "Flat Baseline")
    _print_accuracy_report(execution_traces)

    if output_path:
        roles = list(set(TASKS.values()))
        agent_memory = _collect_agent_memory(store, roles)

        def _tier_summary(results: List[Dict[str, object]]) -> Dict[str, Any]:
            latencies = [int(r["debug"]["latency_ms"]) for r in results]
            tokens = [int(r["debug"].get("retrieval_tokens", 0)) for r in results]
            tier_counts: Dict[str, int] = {}
            for r in results:
                t = str(r.get("tier"))
                tier_counts[t] = tier_counts.get(t, 0) + 1
            avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
            p95 = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0
            return {
                "total_queries": len(results),
                "avg_latency_ms": round(avg_lat, 2),
                "p95_latency_ms": p95,
                "avg_retrieval_tokens": round(sum(tokens) / len(tokens) if tokens else 0, 2),
                "tier_counts": tier_counts,
            }

        def _accuracy_summary(traces: List[Dict[str, Any]], policy_key: str) -> Dict[str, Any]:
            evaluable = [t for t in traces if t[policy_key]["accuracy"]["evaluable"]]
            top1 = sum(1 for t in evaluable if t[policy_key]["accuracy"]["top1_correct"])
            top3 = sum(1 for t in evaluable if t[policy_key]["accuracy"]["top3_correct"])
            any_c = sum(1 for t in evaluable if t[policy_key]["accuracy"]["any_correct"])
            mrr_vals = [t[policy_key]["accuracy"]["mrr"] for t in evaluable]
            avg_mrr = round(sum(mrr_vals) / len(mrr_vals), 4) if mrr_vals else 0.0
            n = len(evaluable) or 1

            by_qtype: Dict[str, Any] = {}
            for t in evaluable:
                qt = t.get("query_type", "unknown")
                by_qtype.setdefault(qt, {"top1": 0, "top3": 0, "total": 0, "mrr_sum": 0.0})
                by_qtype[qt]["total"] += 1
                by_qtype[qt]["top1"] += t[policy_key]["accuracy"]["top1_correct"]
                by_qtype[qt]["top3"] += t[policy_key]["accuracy"]["top3_correct"]
                by_qtype[qt]["mrr_sum"] += t[policy_key]["accuracy"]["mrr"]

            novel = [t for t in traces if not t[policy_key]["accuracy"]["evaluable"]]
            analog = sum(
                1 for t in novel
                if t[policy_key]["accuracy"].get("analog_top3_same_task", False)
            )

            return {
                "evaluable_queries": len(evaluable),
                "top1_accuracy_pct": round(100 * top1 / n, 1),
                "top3_accuracy_pct": round(100 * top3 / n, 1),
                "any_hit_accuracy_pct": round(100 * any_c / n, 1),
                "mean_reciprocal_rank": avg_mrr,
                "novel_analog_top3_pct": round(100 * analog / len(novel), 1) if novel else None,
                "by_query_type": {
                    qt: {
                        "top1_pct": round(100 * v["top1"] / v["total"], 1),
                        "top3_pct": round(100 * v["top3"] / v["total"], 1),
                        "mrr": round(v["mrr_sum"] / v["total"], 4),
                        "queries": v["total"],
                    }
                    for qt, v in by_qtype.items()
                },
            }

        def _task_accuracy_analysis(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
            by_task: Dict[str, Any] = {}
            for t in traces:
                task = t["task_type"]
                by_task.setdefault(task, {
                    "tiered": {"correct": 0, "incorrect": 0, "novel": 0, "queries": []},
                    "flat": {"correct": 0, "incorrect": 0, "novel": 0, "queries": []},
                })
                for policy, key in [("tiered", "tiered_correct"), ("flat", "flat_correct")]:
                    val = t[key]
                    entry = {
                        "query_id": t["query_id"],
                        "situation": t["situation"],
                        "agent_role": t["agent_role"],
                        "known": t["known_pattern"],
                        "query_type": t.get("query_type"),
                        "tier_used": t[f"{policy}_retrieval"]["tier_used"],
                        "correct": val,
                        "mrr": t[f"{policy}_retrieval"]["accuracy"].get("mrr"),
                        "rank": t[f"{policy}_retrieval"]["accuracy"].get("rank"),
                    }
                    by_task[task][policy]["queries"].append(entry)
                    if val is True:
                        by_task[task][policy]["correct"] += 1
                    elif val is False:
                        by_task[task][policy]["incorrect"] += 1
                    else:
                        by_task[task][policy]["novel"] += 1

            for task, data in by_task.items():
                for policy in ("tiered", "flat"):
                    d = data[policy]
                    evaluable = d["correct"] + d["incorrect"]
                    d["accuracy_pct"] = round(100 * d["correct"] / evaluable, 1) if evaluable else None
            return by_task

        canonical_events = system.writer.canonical_events
        # Summary: new situations added and merges per agent
        canonical_summary: Dict[str, Any] = {}
        for ev in canonical_events:
            role = ev["agent_role"]
            canonical_summary.setdefault(role, {"added": [], "merged": []})
            if ev["type"] == "add":
                canonical_summary[role]["added"].append({
                    "label": ev["normalized_label"],
                    "raw_situation": ev["raw_situation"],
                    "task_type": ev["task_type"],
                    "episode_id": ev["episode_id"],
                })
            elif ev["type"] == "merge":
                canonical_summary[role]["merged"].append({
                    "canonical_label": ev["canonical_label"],
                    "raw_situation": ev["raw_situation"],
                    "task_type": ev["task_type"],
                    "outcome": ev["outcome"],
                    "episode_id": ev["episode_id"],
                })

        report = {
            "config": {
                "episodes_per_task": episodes_per_task,
                "num_queries": num_queries,
                "seed": seed,
                "use_openai": use_openai,
                "llm_abstracts": llm_abstracts,
                "db_path": db_path,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "patterns_per_task": 6,
                "query_mix": "50% paraphrased / 15% verbatim / 15% confusor / 20% novel",
            },
            "summary": {
                "tiered_policy": {
                    **_tier_summary(tiered_results),
                    "accuracy": _accuracy_summary(execution_traces, "tiered_retrieval"),
                },
                "flat_baseline": {
                    **_tier_summary(flat_results),
                    "accuracy": _accuracy_summary(execution_traces, "flat_retrieval"),
                },
            },
            "canonical_evolution": {
                "total_events": len(canonical_events),
                "total_added": sum(1 for e in canonical_events if e["type"] == "add"),
                "total_merged": sum(1 for e in canonical_events if e["type"] == "merge"),
                "by_agent": canonical_summary,
                "full_event_log": canonical_events,
            },
            "accuracy_analysis": _task_accuracy_analysis(execution_traces),
            "execution_traces": execution_traces,
            "agent_memory": agent_memory,
        }

        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        base, ext = os.path.splitext(os.path.basename(output_path))

        results_path = os.path.join(results_dir, f"{base}_{ts}{ext}")
        with open(results_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[output] Results saved to {results_path}")

        memory_lines: List[str] = []
        memory_lines.append("=" * 70)
        memory_lines.append("  AGENT MEMORY SNAPSHOT")
        memory_lines.append(f"  Generated : {report['config']['generated_at']}")
        memory_lines.append(f"  Episodes  : {episodes_per_task} per task")
        memory_lines.append(f"  OpenAI    : embeddings={use_openai}  llm_abstracts={llm_abstracts}")
        memory_lines.append("=" * 70)

        # Canonical evolution summary
        ce = report["canonical_evolution"]
        memory_lines.append("")
        memory_lines.append("=" * 70)
        memory_lines.append("  CANONICAL SITUATION EVOLUTION")
        memory_lines.append(f"  Total events: {ce['total_events']}  "
                            f"(added={ce['total_added']}  merged={ce['total_merged']})")
        memory_lines.append("=" * 70)
        for role, ev_data in ce["by_agent"].items():
            memory_lines.append(f"\n  Agent: {role.upper()}")
            added = ev_data["added"]
            merged = ev_data["merged"]
            if added:
                memory_lines.append(f"  + New situations added ({len(added)}):")
                for item in added:
                    memory_lines.append(f"    • [{item['task_type']}] \"{item['label']}\"")
                    memory_lines.append(f"      raw: \"{item['raw_situation']}\"  (ep {item['episode_id']})")
            else:
                memory_lines.append("  + No new situations added.")
            if merged:
                memory_lines.append(f"  ~ Embeddings merged into existing canonicals ({len(merged)}):")
                for item in merged:
                    memory_lines.append(f"    • [{item['task_type']}] \"{item['raw_situation']}\" → \"{item['canonical_label']}\"")
            else:
                memory_lines.append("  ~ No merges performed.")
        memory_lines.append("")
        memory_lines.append("=" * 70)

        for role, mem in agent_memory.items():
            memory_lines.append(f"\n{'─' * 70}")
            memory_lines.append(f"  AGENT: {role.upper()}")
            memory_lines.append(f"{'─' * 70}")

            memory_lines.append("\n  [L0] WHAT I WATCH OUT FOR")
            memory_lines.append("  ─────────────────────────")
            if mem["l0_signals"]:
                for sig in mem["l0_signals"]:
                    fail_rate = sig["fail_rate"]
                    if fail_rate >= 0.75:
                        warning = "⚠ HIGH FAILURE — avoid this approach"
                    elif fail_rate >= 0.4:
                        warning = "~ UNRELIABLE — proceed with caution"
                    else:
                        warning = "✓ Generally safe"
                    memory_lines.append(f"  Task: {sig['task_type']}")
                    memory_lines.append(
                        f"  Outcomes seen: {sig['success_count']} success, "
                        f"{sig['failure_count']} failure, {sig['warning_count']} warning  "
                        f"(fail rate: {int(fail_rate*100)}%)  →  {warning}"
                    )
                    memory_lines.append(f"  Last experience: {sig['last_outcome']} (episode {sig['last_episode_id']})")
                    memory_lines.append("")
            else:
                memory_lines.append("  No signals recorded yet.\n")

            memory_lines.append("  [L1] WHAT I REMEMBER  (cause-effect abstracts)")
            memory_lines.append("  ───────────────────────────────────────────────")
            if mem["l1_abstracts"]:
                for ep in mem["l1_abstracts"]:
                    outcome_tag = {"success": "✓", "failure": "✗", "warning": "~"}.get(ep["outcome"], "?")
                    memory_lines.append(f"  {outcome_tag} [{ep['outcome'].upper()}]  {ep['task_type']}  ({ep['episode_id']})")
                    memory_lines.append(f"  \"{ep['abstract']}\"")
                    memory_lines.append("")
            else:
                memory_lines.append("  No abstracts recorded yet.\n")

            memory_lines.append("  [L2] MY DETAILED EXPERIENCES  (full traces)")
            memory_lines.append("  ────────────────────────────────────────────")
            if mem["l2_full_traces"]:
                for ep in mem["l2_full_traces"]:
                    outcome_tag = {"success": "✓", "failure": "✗", "warning": "~"}.get(ep["outcome"], "?")
                    memory_lines.append(f"  {outcome_tag} [{ep['outcome'].upper()}]  {ep['task_type']}  ({ep['episode_id']})")
                    preview = ep["full_trace"][:300].replace("\n", " ")
                    memory_lines.append(f"  {preview}...")
                    memory_lines.append("")
            else:
                memory_lines.append("  No full traces recorded yet.\n")

        memory_path = os.path.join(results_dir, f"memory_{ts}.txt")
        with open(memory_path, "w") as f:
            f.write("\n".join(memory_lines))
        print(f"[output] Memory  saved to {memory_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggressive terminal benchmark for tiered memory retrieval.")
    parser.add_argument("--db-path", default="benchmark_memory.db")
    parser.add_argument("--episodes-per-task", type=int, default=10,
                        help="Episodes seeded per task type (default 10; use 50+ for full run)")
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-openai", action="store_true",
                        help="Disable OpenAI embeddings and fall back to local hash embedder")
    parser.add_argument("--no-llm-abstracts", action="store_true",
                        help="Disable LLM-generated abstracts")
    parser.add_argument("--openai-key", default=None,
                        help="OpenAI API key (falls back to OPENAI_API_KEY env var)")
    parser.add_argument("--output", default=None,
                        help="Save detailed results to this JSON file (e.g. results.json)")
    args = parser.parse_args()

    run_benchmark(
        db_path=args.db_path,
        episodes_per_task=args.episodes_per_task,
        num_queries=args.queries,
        seed=args.seed,
        use_openai=not args.no_openai,
        llm_abstracts=not args.no_llm_abstracts,
        openai_key=args.openai_key,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
