from __future__ import annotations

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
