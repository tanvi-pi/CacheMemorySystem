from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

try:
    from .MultiAgentSystem import MultiAgentSystem
except ImportError:
    from MultiAgentSystem import MultiAgentSystem

try:
    from harbor.agents.base import BaseAgent
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "harbor is required for this adapter. Install with: pip install harbor"
    ) from exc


TASK_ROLE_MAP: Dict[str, str] = {
    "db_query_debug": "planner",
    "api_integration": "coder",
    "code_review": "reviewer",
    "incident_investigation": "researcher",
    "release_validation": "executor",
}


class TieredMemoryHarborAgent(BaseAgent):
    """External Harbor agent adapter backed by your tiered multi-agent memory system."""

    def __init__(
        self,
        db_path: str = "/tmp/tiered_memory_harbor.db",
        max_steps: int = 16,
        latency_budget_ms: int = 60,
        token_budget: int = 900,
    ):
        self.system = MultiAgentSystem(
            roles=["planner", "coder", "reviewer", "researcher", "executor"],
            task_role_map=TASK_ROLE_MAP,
            db_path=db_path,
            embed_dim=512,
        )
        self.max_steps = max_steps
        self.latency_budget_ms = latency_budget_ms
        self.token_budget = token_budget

    @staticmethod
    def name() -> str:
        return "tiered-memory-agent"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: Any) -> None:
        await self._exec(environment, "pwd")
        await self._exec(environment, "ls -la")

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        start_ms = int(time.time() * 1000)
        task_type = self._infer_task_type(instruction)
        situation = self._situation_signature(instruction)
        role = TASK_ROLE_MAP[task_type]

        command_history: List[Tuple[str, int, str, str]] = []
        confidence = 0.65
        outcome = "warning"
        final_summary = ""

        for step in range(self.max_steps):
            retrieval = self.system.step(
                task_type=task_type,
                role=role,
                situation=situation,
                user_query=instruction,
                confidence=confidence,
                retry_count=step,
                latency_budget_ms=self.latency_budget_ms,
                token_budget=self.token_budget,
            )

            cmd = self._choose_command(instruction, retrieval, step, command_history)
            if not cmd:
                break

            exit_code, stdout, stderr = await self._exec(environment, cmd)
            command_history.append((cmd, exit_code, stdout, stderr))

            if exit_code == 0:
                outcome = "success"
                confidence = min(0.95, confidence + 0.04)
            else:
                outcome = "failure"
                confidence = max(0.2, confidence - 0.1)

            if self._looks_complete(cmd, exit_code, stdout, stderr):
                break

        full_trace = self._render_trace(command_history)
        final_summary = self._build_summary(task_type, role, outcome, command_history)

        self.system.finalize_episode(
            role=role,
            episode_id=f"harbor_{start_ms}",
            task_type=task_type,
            outcome=outcome,
            abstract=final_summary,
            full_trace=full_trace,
            situation_signature=situation,
            cost_tokens=0,
            cost_latency_ms=max(1, int(time.time() * 1000) - start_ms),
        )

        self._populate_context(context, final_summary, command_history)

    def _infer_task_type(self, instruction: str) -> str:
        s = instruction.lower()
        if any(x in s for x in ["sql", "query", "database", "postgres", "mysql"]):
            return "db_query_debug"
        if any(x in s for x in ["api", "endpoint", "http", "oauth", "token"]):
            return "api_integration"
        if any(x in s for x in ["review", "bug", "race", "test coverage", "security"]):
            return "code_review"
        if any(x in s for x in ["incident", "outage", "alert", "investigate", "latency spike"]):
            return "incident_investigation"
        return "release_validation"

    def _situation_signature(self, instruction: str) -> str:
        cleaned = re.sub(r"\s+", " ", instruction.strip().lower())
        return cleaned[:160]

    def _choose_command(
        self,
        instruction: str,
        retrieval: Dict[str, Any],
        step: int,
        history: List[Tuple[str, int, str, str]],
    ) -> Optional[str]:
        if step == 0:
            return "pwd && ls -la"
        if step == 1:
            return "find . -maxdepth 3 -type f | head -200"

        s = instruction.lower()
        if any(x in s for x in ["python", "pytest", "test"]):
            if not any("pytest" in h[0] for h in history):
                return "pytest -q"
        if any(x in s for x in ["node", "npm", "javascript", "typescript"]):
            if not any("npm test" in h[0] for h in history):
                return "npm test --silent"

        if step == 2:
            return "git status --short"

        return None

    def _looks_complete(self, cmd: str, exit_code: int, stdout: str, stderr: str) -> bool:
        if cmd.startswith("pytest") and exit_code == 0:
            return True
        if "npm test" in cmd and exit_code == 0:
            return True
        text = f"{stdout}\n{stderr}".lower()
        return "all tests passed" in text or "0 failed" in text

    async def _exec(self, environment: Any, command: str) -> Tuple[int, str, str]:
        if hasattr(environment, "exec"):
            out = await environment.exec(command)
        elif hasattr(environment, "execute"):
            out = await environment.execute(command)
        elif hasattr(environment, "run"):
            out = await environment.run(command)
        else:
            raise AttributeError("Environment does not expose exec/execute/run")

        if isinstance(out, dict):
            return (
                int(out.get("exit_code", out.get("returncode", 1))),
                str(out.get("stdout", "")),
                str(out.get("stderr", "")),
            )

        exit_code = int(getattr(out, "exit_code", getattr(out, "returncode", 1)))
        stdout = str(getattr(out, "stdout", ""))
        stderr = str(getattr(out, "stderr", ""))
        return exit_code, stdout, stderr

    def _render_trace(self, history: List[Tuple[str, int, str, str]]) -> str:
        chunks: List[str] = []
        for idx, (cmd, code, stdout, stderr) in enumerate(history):
            chunks.append(
                f"STEP {idx + 1}\nCMD: {cmd}\nEXIT: {code}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}\n"
            )
        return "\n".join(chunks)

    def _build_summary(
        self,
        task_type: str,
        role: str,
        outcome: str,
        history: List[Tuple[str, int, str, str]],
    ) -> str:
        if not history:
            return f"Role {role} on {task_type}: no commands executed; outcome={outcome}."
        success_count = sum(1 for _, code, _, _ in history if code == 0)
        return (
            f"Role {role} handled {task_type}. outcome={outcome}. "
            f"Executed {len(history)} commands with {success_count} successful steps."
        )

    def _populate_context(
        self,
        context: Any,
        summary: str,
        history: List[Tuple[str, int, str, str]],
    ) -> None:
        rendered = summary + "\n\n" + self._render_trace(history)

        for field in ["final_response", "submission", "answer", "output", "result"]:
            if hasattr(context, field):
                setattr(context, field, rendered)

        if hasattr(context, "metadata") and isinstance(getattr(context, "metadata"), dict):
            context.metadata["agent"] = self.name()
            context.metadata["steps"] = len(history)

        if hasattr(context, "messages") and isinstance(getattr(context, "messages"), list):
            context.messages.append({"role": "assistant", "content": summary})
