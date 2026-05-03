import asyncio
import tempfile
from pathlib import Path

from agents.orchestrator import OrchestratorAgent
from goals.goal_queue import GoalQueue


class _Agent:
    async def run(self, task):
        return {"ok": True, "text": task.get("text")}


class _FailingAgent:
    async def run(self, task):
        raise RuntimeError("boom")


class _OkAgent:
    async def run(self, task):
        return {"success": True, "lesson": "Used subprocess correctly."}


class _SoftFailAgent:
    async def run(self, task):
        return {"success": False, "error": "connection timeout"}


def _make_queue(tmp_path: str):
    q = GoalQueue(Path(tmp_path) / "queue.json")
    q.push("g1", "write a hello world script", priority=10)
    q.push("g2", "research memory architecture", priority=20)
    q.push("g3", "implement FAISS search", priority=5)
    return q


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


def test_run_session_exists():
    import inspect

    assert hasattr(OrchestratorAgent, "run_session")
    assert inspect.iscoroutinefunction(OrchestratorAgent.run_session)


def test_run_session_no_queue():
    orch = OrchestratorAgent(None, None, None, None)
    result = asyncio.run(orch.run_session())
    assert "error" in result
    assert result["goals_attempted"] == 0


def test_run_session_completes_goals():
    with tempfile.TemporaryDirectory() as td:
        queue = _make_queue(td)
        agent = _OkAgent()
        orch = OrchestratorAgent(agent, agent, agent, agent, goal_queue=queue)
        result = asyncio.run(orch.run_session(max_goals=2))
        assert result["goals_attempted"] == 2
        assert result["goals_completed"] == 2
        assert result["goals_failed"] == 0
        assert "queue_status" in result


def test_run_session_handles_failures():
    with tempfile.TemporaryDirectory() as td:
        queue = _make_queue(td)
        agent = _SoftFailAgent()
        orch = OrchestratorAgent(agent, agent, agent, agent, goal_queue=queue)
        result = asyncio.run(orch.run_session(max_goals=1))
        assert result["goals_attempted"] == 1
        assert result["goals_failed"] == 1 or result["goals_completed"] == 0


def test_run_session_generates_successor():
    with tempfile.TemporaryDirectory() as td:
        queue = _make_queue(td)
        initial_count = len(queue._items)
        agent = _OkAgent()
        orch = OrchestratorAgent(agent, agent, agent, agent, goal_queue=queue)
        asyncio.run(orch.run_session(max_goals=1))
        assert len(queue._items) >= initial_count


def test_goal_to_task_routing():
    from goals.goal_queue import GoalItem
    import time

    orch = OrchestratorAgent(None, None, None, None)

    def make_goal(text):
        return GoalItem(goal_id="x", text=text, created_at=time.time())

    assert orch._goal_to_task(make_goal("write a python script"))["kind"] == "coder"
    assert orch._goal_to_task(make_goal("research memory systems"))["kind"] == "research"
    assert orch._goal_to_task(make_goal("recall last session data"))["kind"] == "memory"
