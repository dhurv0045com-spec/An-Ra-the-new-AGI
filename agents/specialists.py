from __future__ import annotations

import sys
import time
from pathlib import Path

from anra_paths import ROOT
from execution.sandbox import CodeSandbox
from training.verifier import VerifierHierarchy

DEFAULT_CODE_PATH = "generated.py"


class BaseAgent:
    def __init__(self, agent_id: str, model, tokenizer, sandbox, fs_agent, memory_router, bus) -> None:
        self.agent_id = agent_id
        self.model = model
        self.tokenizer = tokenizer
        self.sandbox = sandbox
        self.fs_agent = fs_agent
        self.memory_router = memory_router
        self.bus = bus

    def _generate(self, prompt: str) -> str:
        if hasattr(self.model, "generate"):
            out = self.model.generate(prompt)
            if isinstance(out, list):
                return str(out[0])
            return str(out)
        return prompt


class CoderAgent(BaseAgent):
    async def run(self, task: dict) -> dict:
        prompt = f"You are An-Ra's Coder.\n{task.get('prompt','')}"
        test_code = task.get("test_code", "")
        path = task.get("path", DEFAULT_CODE_PATH)

        attempts = 0
        code = ""
        last = None
        success = False
        while attempts < 3:
            attempts += 1
            code = self._generate(prompt)
            last = self.sandbox.execute(code + ("\n\n" + test_code if test_code else ""))
            if last.success:
                success = True
                break
            prompt += f"\nFix this error:\n{last.stderr}"

        self.fs_agent.write(path, code)
        return {"code": code, "stdout": (last.stdout if last else ""), "success": success, "attempts": attempts}


class ResearcherAgent(BaseAgent):
    async def run(self, task: dict) -> dict:
        files = task.get("files", [])
        corpus = []
        for f in files:
            try:
                corpus.append(self.fs_agent.read(f))
            except Exception:
                continue
        prompt = f"Summarize findings:\n{chr(10).join(corpus)[:12000]}"
        summary = self._generate(prompt)
        self.memory_router.write(summary, metadata={"type": "research"}, tier="episodic")
        return {"summary": summary, "stored": True}


class MemoryAgent(BaseAgent):
    async def run(self, task: dict) -> dict:
        q = task.get("query", "")
        results = self.memory_router.read(q, n=int(task.get("n", 8)), tier=task.get("tier", "episodic"))
        context = "\n".join([str(r) for r in results])
        return {"context": context, "n_results": len(results)}


class CriticAgent(BaseAgent):
    def __init__(self, *args, verifier: VerifierHierarchy | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.verifier = verifier or VerifierHierarchy()

    async def run(self, task: dict) -> dict:
        task_type = task.get("task_type", "open")
        res = self.verifier.score(
            task_type,
            code=task.get("code", ""),
            test_code=task.get("test_code", ""),
            response=task.get("response", task.get("code", "")),
            task=task.get("prompt", ""),
            expression=task.get("expression", ""),
            expected=task.get("expected", ""),
            check_fn=task.get("check_fn", lambda: False),
            pattern=task.get("pattern", ".*"),
        )
        return {"score": float(res.score), "tier": int(res.tier), "reason": res.reason, "approved": res.score >= 0.7}


# AN: Specialist wrappers connect existing tool infrastructure to previously inert agent roles.
class CodeSpecialist:
    def __init__(self, sandbox: CodeSandbox | None = None) -> None:
        self.sandbox = sandbox or CodeSandbox()

    def run(self, task) -> dict:
        start = time.perf_counter()
        task_description = task.get("prompt", task.get("code", "")) if isinstance(task, dict) else str(task)
        code = task.get("code", task_description) if isinstance(task, dict) else task_description
        result = self.sandbox.execute(code)
        return {
            "agent": "code",
            "task": task_description,
            "result": {"stdout": result.stdout, "stderr": result.stderr, "return_code": result.return_code},
            "verified": bool(result.success),
            "tool_used": "execution/sandbox.py",
            "time_taken": time.perf_counter() - start,
        }


class MathSpecialist:
    def run(self, task) -> dict:
        start = time.perf_counter()
        task_description = task.get("prompt", task.get("expression", "")) if isinstance(task, dict) else str(task)
        expression = task.get("expression", task_description) if isinstance(task, dict) else task_description
        try:
            bridge_dir = ROOT / "phase3" / "symbolic_bridge (45Q)"
            if str(bridge_dir) not in sys.path:
                sys.path.insert(0, str(bridge_dir))
            from math_solver import solve_equation

            result = solve_equation(expression)
            payload = result.to_dict() if hasattr(result, "to_dict") else str(result)
            verified = str(getattr(result, "verdict", "")).lower().endswith("verified")
        except Exception as exc:
            payload = {"error": str(exc)}
            verified = False
        return {
            "agent": "math",
            "task": task_description,
            "result": payload,
            "verified": verified,
            "tool_used": "phase3/symbolic_bridge (45Q)/math_solver.py",
            "time_taken": time.perf_counter() - start,
        }


class ResearchSpecialist:
    def __init__(self, memory_router=None) -> None:
        self.memory_router = memory_router

    def run(self, task) -> dict:
        start = time.perf_counter()
        task_description = task.get("query", task.get("prompt", "")) if isinstance(task, dict) else str(task)
        try:
            router = self.memory_router
            if router is None:
                from memory.memory_router import MemoryRouter

                router = MemoryRouter()
            rows = router.read(task_description, n=int(task.get("n", 8)) if isinstance(task, dict) else 8, tier=task.get("tier", "episodic") if isinstance(task, dict) else "episodic")
            payload = rows
            verified = True
        except Exception as exc:
            payload = {"error": str(exc)}
            verified = False
        return {
            "agent": "research",
            "task": task_description,
            "result": payload,
            "verified": verified,
            "tool_used": "memory/memory_router.py",
            "time_taken": time.perf_counter() - start,
        }
