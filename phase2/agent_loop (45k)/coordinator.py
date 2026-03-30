"""
================================================================================
FILE: agent/intelligence/coordinator.py
PROJECT: Agent Loop — 45K
PURPOSE: Multi-agent coordination — spawn sub-agents, collect results, synthesize
================================================================================

The coordinator lets the parent agent spawn sub-agents for parallel workstreams.
Each sub-agent is a lightweight wrapper around the same core loop.
Sub-agents share memory and report back to the parent when done.
Parent synthesizes all sub-agent results into a final answer.

Design: thread-based (no separate processes) — sub-agents run concurrently
in threads, each with their own plan and tool registry instance.
Memory pool is shared via the memory_tool's in-process store.
================================================================================
"""

import time
import uuid
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SubAgentResult:
    agent_id:  str
    goal:      str
    success:   bool
    output:    str
    error:     Optional[str] = None
    duration:  float         = 0.0
    metadata:  Dict          = field(default_factory=dict)


class SubAgent(threading.Thread):
    """
    A lightweight sub-agent running in its own thread.
    Executes a single goal using the provided executor function.
    Reports result back to the coordinator via callback.
    """

    def __init__(
        self,
        agent_id:    str,
        goal:        str,
        execute_fn:  Callable[[str], Dict],   # The agent's core execute function
        on_complete: Callable[["SubAgent", SubAgentResult], None],
        timeout:     int = 300,
    ):
        super().__init__(daemon=True, name=f"SubAgent-{agent_id}")
        self.agent_id   = agent_id
        self.goal       = goal
        self.execute_fn = execute_fn
        self.on_complete = on_complete
        self.timeout    = timeout
        self.result:    Optional[SubAgentResult] = None

    def run(self) -> None:
        start = time.time()
        try:
            logger.info(f"SubAgent {self.agent_id} starting: {self.goal[:80]}")
            raw_result = self.execute_fn(self.goal)
            self.result = SubAgentResult(
                agent_id=self.agent_id,
                goal=self.goal,
                success=raw_result.get("success", False),
                output=raw_result.get("outputs", {}).get("final", str(raw_result)),
                duration=time.time() - start,
                metadata=raw_result,
            )
        except Exception as e:
            logger.error(f"SubAgent {self.agent_id} crashed: {e}")
            self.result = SubAgentResult(
                agent_id=self.agent_id,
                goal=self.goal,
                success=False,
                output="",
                error=str(e),
                duration=time.time() - start,
            )
        finally:
            self.on_complete(self, self.result)


class MultiAgentCoordinator:
    """
    Coordinates parallel sub-agents for independent workstreams.

    Usage:
        coord = MultiAgentCoordinator(execute_fn=agent.execute_goal)
        results = coord.run_parallel([
            "Research GPU A: RTX 4090 specs and price",
            "Research GPU B: RTX 3090 specs and price",
        ])
        synthesis = coord.synthesize(results, "Which GPU is better under $1000?")
    """

    def __init__(
        self,
        execute_fn:   Callable[[str], Dict],
        max_parallel: int = 5,
        timeout_secs: int = 300,
    ):
        self.execute_fn   = execute_fn
        self.max_parallel = max_parallel
        self.timeout      = timeout_secs
        self._results:    List[SubAgentResult] = []
        self._lock        = threading.Lock()
        self._active:     List[SubAgent] = []

    def run_parallel(self, goals: List[str]) -> List[SubAgentResult]:
        """
        Spawn one sub-agent per goal, run all in parallel, wait for completion.

        Args:
            goals: List of goal strings — each gets its own sub-agent.

        Returns:
            List of SubAgentResult in completion order.
        """
        if not goals:
            return []

        # Limit parallelism
        batches = [goals[i:i+self.max_parallel] for i in range(0, len(goals), self.max_parallel)]
        all_results = []

        for batch in batches:
            batch_results: List[SubAgentResult] = []
            completed = threading.Event()
            pending   = [None] * len(batch)

            def on_complete(agent: SubAgent, result: SubAgentResult, idx=None):
                with self._lock:
                    batch_results.append(result)
                    self._results.append(result)
                logger.info(
                    f"SubAgent {agent.agent_id} done: "
                    f"success={result.success}, duration={result.duration:.1f}s"
                )
                if len(batch_results) >= len(batch):
                    completed.set()

            agents = []
            for i, goal in enumerate(batch):
                aid   = str(uuid.uuid4())[:8]
                agent = SubAgent(
                    agent_id=aid,
                    goal=goal,
                    execute_fn=self.execute_fn,
                    on_complete=lambda a, r, idx=i: on_complete(a, r, idx),
                    timeout=self.timeout,
                )
                agents.append(agent)
                self._active.append(agent)

            logger.info(f"Spawning {len(agents)} sub-agents for batch")
            for a in agents:
                a.start()

            # Wait with timeout
            completed.wait(timeout=self.timeout + 30)

            # Clean up any timed-out agents
            for a in agents:
                if a.is_alive():
                    logger.warning(f"Sub-agent {a.agent_id} timed out after {self.timeout}s")
                    batch_results.append(SubAgentResult(
                        agent_id=a.agent_id,
                        goal=a.goal,
                        success=False,
                        output="",
                        error="Timed out",
                        duration=self.timeout,
                    ))

            all_results.extend(batch_results)
            self._active = [a for a in self._active if a.is_alive()]

        return all_results

    def synthesize(self, results: List[SubAgentResult], synthesis_question: str = "") -> str:
        """
        Combine sub-agent results into a coherent final answer.

        Args:
            results:             List of sub-agent results.
            synthesis_question:  Optional question to focus the synthesis.

        Returns:
            Synthesized string combining all successful results.
        """
        succeeded = [r for r in results if r.success]
        failed    = [r for r in results if not r.success]

        lines = []
        if synthesis_question:
            lines.append(f"SYNTHESIS: {synthesis_question}\n")

        lines.append(f"Sub-agents: {len(results)} total, {len(succeeded)} succeeded, {len(failed)} failed\n")

        for r in succeeded:
            lines.append(f"--- Sub-agent {r.agent_id} ({r.duration:.0f}s) ---")
            lines.append(f"Goal: {r.goal[:80]}")
            lines.append(r.output[:500])
            lines.append("")

        if failed:
            lines.append("--- Failed sub-agents ---")
            for r in failed:
                lines.append(f"  {r.agent_id}: {r.error or 'unknown error'}")

        return "\n".join(lines)

    def shutdown_all(self) -> None:
        """Signal all active sub-agents to stop (best-effort)."""
        logger.info(f"Shutting down {len(self._active)} active sub-agents")
        self._active.clear()  # Threads are daemons — they'll die with the parent

    def get_results(self) -> List[SubAgentResult]:
        return list(self._results)
