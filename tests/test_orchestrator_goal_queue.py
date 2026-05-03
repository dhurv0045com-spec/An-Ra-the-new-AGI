import asyncio

from agents.orchestrator import OrchestratorAgent
from goals.goal_queue import GoalQueue


class _Agent:
    async def run(self, task):
        return {"ok": True, "text": task.get("text")}


class _FailingAgent:
    async def run(self, task):
        raise RuntimeError("boom")


def test_orchestrator_dispatch_next_goal_completes_goal(tmp_path):
    queue = GoalQueue(tmp_path / "goals.json")
    queue.push("g1", "ship fix", priority=1)
    agent = _Agent()
    orchestrator = OrchestratorAgent(agent, agent, agent, agent, goal_queue=queue)

    result = asyncio.run(orchestrator.dispatch_next_goal())

    assert result == {"ok": True, "text": "ship fix"}
    assert queue._items["g1"].status == "done"


def test_orchestrator_marks_goal_failed_on_error(tmp_path):
    queue = GoalQueue(tmp_path / "goals.json")
    queue.push("g1", "ship fix", priority=1)
    failing = _FailingAgent()
    agent = _Agent()
    orchestrator = OrchestratorAgent(failing, agent, agent, agent, goal_queue=queue)

    try:
        asyncio.run(orchestrator.dispatch_next_goal())
    except RuntimeError:
        pass

    assert queue._items["g1"].status == "queued"
    assert queue._items["g1"].last_error == "boom"

