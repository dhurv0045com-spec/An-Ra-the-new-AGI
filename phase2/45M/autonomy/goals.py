"""
autonomy/goals.py — Long Horizon Goal Manager

Goals spanning days, weeks, months. Breaks into daily workstreams.
Tracks progress, adjusts plans, flags blockers, estimates completion.
Handles priority, dependencies, and archiving.
"""

import uuid, json, sqlite3, threading
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from enum import Enum


STATE_DIR = Path("state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH   = STATE_DIR / "goals.db"


class GoalStatus(str, Enum):
    ACTIVE    = "active"
    PAUSED    = "paused"
    BLOCKED   = "blocked"
    COMPLETED = "completed"
    ARCHIVED  = "archived"
    CANCELLED = "cancelled"

class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"


@dataclass
class WorkstreamStep:
    step_id:     str
    goal_id:     str
    day_offset:  int          # day 1, 2, 3... relative to goal start
    title:       str
    description: str
    status:      str = "pending"   # pending / in_progress / done / skipped
    completed_at: Optional[str] = None
    output:      Optional[str] = None


@dataclass
class Goal:
    goal_id:      str
    title:        str
    description:  str
    horizon_days: int
    priority:     str
    status:       str
    created_at:   str
    start_date:   str
    target_date:  str
    depends_on:   List[str] = field(default_factory=list)   # goal_ids
    tags:         List[str] = field(default_factory=list)
    progress_pct: float = 0.0
    last_update:  Optional[str] = None
    blocker:      Optional[str] = None
    completion_estimate: Optional[str] = None
    completed_at: Optional[str] = None
    history:      List[dict] = field(default_factory=list)


class GoalDB:
    def __init__(self, path: Path = DB_PATH):
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._init()

    def _init(self):
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS goals (
                    goal_id TEXT PRIMARY KEY,
                    data    TEXT
                );
                CREATE TABLE IF NOT EXISTS steps (
                    step_id  TEXT PRIMARY KEY,
                    goal_id  TEXT,
                    data     TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_steps_goal ON steps(goal_id);
            """)
            self._conn.commit()

    def save_goal(self, goal: Goal):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO goals VALUES (?,?)",
                (goal.goal_id, json.dumps(asdict(goal)))
            )
            self._conn.commit()

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM goals WHERE goal_id=?", (goal_id,)
            ).fetchone()
        if not row:
            return None
        d = json.loads(row[0])
        return Goal(**d)

    def list_goals(self, status: Optional[str] = None) -> List[Goal]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM goals").fetchall()
        goals = [Goal(**json.loads(r[0])) for r in rows]
        if status:
            goals = [g for g in goals if g.status == status]
        return goals

    def save_step(self, step: WorkstreamStep):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO steps VALUES (?,?,?)",
                (step.step_id, step.goal_id, json.dumps(asdict(step)))
            )
            self._conn.commit()

    def get_steps(self, goal_id: str) -> List[WorkstreamStep]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM steps WHERE goal_id=? ORDER BY data",
                (goal_id,)
            ).fetchall()
        return [WorkstreamStep(**json.loads(r[0])) for r in rows]


class GoalManager:
    """
    Manages the full lifecycle of long-horizon goals.
    """

    def __init__(self):
        self.db = GoalDB()

    def create_goal(
        self,
        title:        str,
        description:  str,
        horizon_days: int,
        priority:     str = Priority.MEDIUM,
        depends_on:   List[str] = None,
        tags:         List[str] = None,
    ) -> Goal:
        now   = datetime.utcnow()
        target = now + timedelta(days=horizon_days)
        goal  = Goal(
            goal_id      = str(uuid.uuid4()),
            title        = title,
            description  = description,
            horizon_days = horizon_days,
            priority     = priority,
            status       = GoalStatus.ACTIVE,
            created_at   = now.isoformat(),
            start_date   = now.isoformat(),
            target_date  = target.isoformat(),
            depends_on   = depends_on or [],
            tags         = tags or [],
        )
        self.db.save_goal(goal)
        steps = self._generate_workstream(goal)
        for s in steps:
            self.db.save_step(s)
        return goal

    def _generate_workstream(self, goal: Goal) -> List[WorkstreamStep]:
        """
        Decompose a goal into daily steps.
        Smart decomposition: early days = research/setup,
        middle = execution, final days = review/polish.
        """
        steps = []
        n = goal.horizon_days

        phases = []
        if n <= 3:
            phases = [("Research & Setup", 1), ("Execute", max(1, n-1)), ("Review", 1)]
        elif n <= 7:
            phases = [("Research", 2), ("Planning", 1), ("Execute", n-4), ("Polish", 1), ("Review", 1)]
        elif n <= 14:
            phases = [("Research", 3), ("Architecture", 2), ("Execute", n-8),
                      ("Testing", 2), ("Polish", 2), ("Review & Deliver", 1)]
        else:
            phases = [("Deep Research", 5), ("Architecture & Planning", 3),
                      ("Core Execution", n-14), ("Integration", 3),
                      ("Testing & Refinement", 3), ("Final Polish", 2), ("Delivery", 1)]

        day = 1
        for phase_name, length in phases:
            for d in range(length):
                if day > n:
                    break
                steps.append(WorkstreamStep(
                    step_id     = str(uuid.uuid4()),
                    goal_id     = goal.goal_id,
                    day_offset  = day,
                    title       = f"Day {day}: {phase_name}" + (f" ({d+1}/{length})" if length > 1 else ""),
                    description = f"Phase '{phase_name}' for goal: {goal.title}",
                ))
                day += 1

        return steps

    def update_progress(self, goal_id: str, step_id: str,
                        output: str = "", status: str = "done"):
        goal = self.db.get_goal(goal_id)
        if not goal:
            return None
        steps = self.db.get_steps(goal_id)
        for s in steps:
            if s.step_id == step_id:
                s.status       = status
                s.completed_at = datetime.utcnow().isoformat()
                s.output       = output
                self.db.save_step(s)
                break

        # Recalculate progress
        all_steps = self.db.get_steps(goal_id)
        done = sum(1 for s in all_steps if s.status == "done")
        goal.progress_pct = round(100 * done / len(all_steps), 1) if all_steps else 0
        goal.last_update  = datetime.utcnow().isoformat()
        goal.history.append({
            "ts":     datetime.utcnow().isoformat(),
            "event":  f"Step completed: {step_id}",
            "output": output[:200] if output else "",
        })

        # Auto-complete if all done
        if goal.progress_pct == 100:
            goal.status       = GoalStatus.COMPLETED
            goal.completed_at = datetime.utcnow().isoformat()

        self.db.save_goal(goal)
        return goal

    def set_blocker(self, goal_id: str, blocker: str):
        goal = self.db.get_goal(goal_id)
        if not goal: return
        goal.status  = GoalStatus.BLOCKED
        goal.blocker = blocker
        goal.history.append({"ts": datetime.utcnow().isoformat(), "event": f"BLOCKED: {blocker}"})
        self.db.save_goal(goal)

    def clear_blocker(self, goal_id: str):
        goal = self.db.get_goal(goal_id)
        if not goal: return
        goal.status  = GoalStatus.ACTIVE
        goal.blocker = None
        goal.history.append({"ts": datetime.utcnow().isoformat(), "event": "Blocker cleared"})
        self.db.save_goal(goal)

    def estimate_completion(self, goal_id: str) -> str:
        goal  = self.db.get_goal(goal_id)
        if not goal or goal.progress_pct == 0:
            return goal.target_date if goal else "unknown"
        steps = self.db.get_steps(goal_id)
        done_steps = [s for s in steps if s.status == "done" and s.completed_at]
        if len(done_steps) < 2:
            return goal.target_date

        # Estimate based on recent velocity
        first = datetime.fromisoformat(done_steps[0].completed_at)
        last  = datetime.fromisoformat(done_steps[-1].completed_at)
        elapsed = (last - first).total_seconds() / 3600   # hours
        rate    = len(done_steps) / max(elapsed, 0.1)     # steps per hour
        remaining = len([s for s in steps if s.status == "pending"])
        hours_left = remaining / rate
        estimate   = datetime.utcnow() + timedelta(hours=hours_left)
        goal.completion_estimate = estimate.isoformat()
        self.db.save_goal(goal)
        return estimate.isoformat()

    def daily_review(self) -> dict:
        """Called by scheduler daily. Returns progress report."""
        active = self.db.list_goals(GoalStatus.ACTIVE)
        blocked = self.db.list_goals(GoalStatus.BLOCKED)
        overdue = []
        on_track = []

        now = datetime.utcnow()
        for g in active:
            target = datetime.fromisoformat(g.target_date)
            expected_pct = 100 * (now - datetime.fromisoformat(g.start_date)).total_seconds() \
                               / max((target - datetime.fromisoformat(g.start_date)).total_seconds(), 1)
            if g.progress_pct < expected_pct - 20:
                overdue.append(g.title)
            else:
                on_track.append(g.title)

        return {
            "active":   len(active),
            "blocked":  len(blocked),
            "on_track": on_track,
            "overdue":  overdue,
            "report":   f"{len(active)} active goals, {len(overdue)} behind schedule",
        }

    def archive_completed(self):
        completed = self.db.list_goals(GoalStatus.COMPLETED)
        for g in completed:
            g.status = GoalStatus.ARCHIVED
            self.db.save_goal(g)
        return len(completed)

    def get_morning_brief(self) -> str:
        active   = self.db.list_goals(GoalStatus.ACTIVE)
        blocked  = self.db.list_goals(GoalStatus.BLOCKED)
        lines    = ["═══ GOAL STATUS ═══"]
        for g in sorted(active, key=lambda x: x.priority):
            est = self.estimate_completion(g.goal_id)
            lines.append(f"  [{g.priority.upper()}] {g.title}")
            lines.append(f"    Progress: {g.progress_pct:.0f}% | Target: {g.target_date[:10]}")
        if blocked:
            lines.append("\n  BLOCKED:")
            for g in blocked:
                lines.append(f"    ⚠ {g.title}: {g.blocker}")
        return "\n".join(lines)

    def check_dependencies(self, goal_id: str) -> bool:
        """Returns True if all dependencies are completed."""
        goal = self.db.get_goal(goal_id)
        if not goal or not goal.depends_on:
            return True
        for dep_id in goal.depends_on:
            dep = self.db.get_goal(dep_id)
            if not dep or dep.status not in (GoalStatus.COMPLETED, GoalStatus.ARCHIVED):
                return False
        return True
