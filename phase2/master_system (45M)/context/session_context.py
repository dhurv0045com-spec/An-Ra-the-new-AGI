"""
context/session_context.py — Cross-Session Context Manager

Maintains rich context across sessions:
  - What the owner was working on last session
  - Unfinished tasks and their state
  - Important things said / decided / learned
  - Session summaries for fast re-orientation
  - Automatic context injection into new sessions

This is what makes the system feel like it "remembers you"
even after a restart or long absence.
"""

import json, sqlite3, threading, uuid, hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any
from pathlib import Path


STATE_DIR = Path("state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH   = STATE_DIR / "sessions.db"


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class SessionSummary:
    session_id:  str
    started_at:  str
    ended_at:    Optional[str]
    duration_min: float
    topics:      List[str]          # main topics discussed
    decisions:   List[str]          # decisions made
    tasks_started: List[str]        # tasks begun
    tasks_completed: List[str]      # tasks finished
    tasks_pending: List[str]        # tasks left unfinished
    key_facts:   List[str]          # important facts stated
    mood:        str = "neutral"    # inferred emotional tone
    summary:     str = ""           # one-paragraph human-readable summary
    turn_count:  int = 0


@dataclass
class PendingTask:
    task_id:    str
    created_at: str
    session_id: str
    title:      str
    context:    str         # what was being worked on
    status:     str = "pending"   # pending / resumed / completed / abandoned
    priority:   int = 2           # 1=high, 2=medium, 3=low
    resumed_count: int = 0


@dataclass
class CrossSessionFact:
    """Important facts that persist across all sessions."""
    fact_id:     str
    timestamp:   str
    content:     str
    category:    str   # "preference", "decision", "goal", "relationship", "knowledge"
    confidence:  float = 1.0
    source:      str = "inferred"   # "stated" / "inferred" / "corrected"
    expires_at:  Optional[str] = None   # None = permanent


# ── Database ───────────────────────────────────────────────────────────────────

class SessionDB:
    def __init__(self, path=DB_PATH):
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._init()

    def _init(self):
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY, data TEXT
                );
                CREATE TABLE IF NOT EXISTS pending_tasks (
                    task_id TEXT PRIMARY KEY, data TEXT
                );
                CREATE TABLE IF NOT EXISTS facts (
                    fact_id TEXT PRIMARY KEY, data TEXT
                );
                CREATE TABLE IF NOT EXISTS current_session (
                    key TEXT PRIMARY KEY, value TEXT
                );
            """)
            self._conn.commit()

    def save_session(self, s: SessionSummary):
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO sessions VALUES (?,?)",
                               (s.session_id, json.dumps(asdict(s))))
            self._conn.commit()

    def get_recent_sessions(self, n: int = 5) -> List[SessionSummary]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM sessions ORDER BY session_id DESC LIMIT ?", (n,)).fetchall()
        return [SessionSummary(**json.loads(r[0])) for r in rows]

    def save_task(self, t: PendingTask):
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO pending_tasks VALUES (?,?)",
                               (t.task_id, json.dumps(asdict(t))))
            self._conn.commit()

    def get_pending_tasks(self) -> List[PendingTask]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM pending_tasks").fetchall()
        tasks = [PendingTask(**json.loads(r[0])) for r in rows]
        return [t for t in tasks if t.status == "pending"]

    def save_fact(self, f: CrossSessionFact):
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO facts VALUES (?,?)",
                               (f.fact_id, json.dumps(asdict(f))))
            self._conn.commit()

    def get_facts(self, category: Optional[str] = None) -> List[CrossSessionFact]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM facts").fetchall()
        facts = [CrossSessionFact(**json.loads(r[0])) for r in rows]
        now = datetime.utcnow().isoformat()
        # Filter expired
        facts = [f for f in facts if not f.expires_at or f.expires_at > now]
        if category:
            facts = [f for f in facts if f.category == category]
        return facts

    def set_current(self, key: str, value: Any):
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO current_session VALUES (?,?)",
                               (key, json.dumps(value)))
            self._conn.commit()

    def get_current(self, key: str) -> Any:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM current_session WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None


# ── Session Context Manager ────────────────────────────────────────────────────

class SessionContextManager:
    """
    Manages rich context across all sessions.
    Produces context injections for the model.
    Learns from each session to improve future ones.
    """

    def __init__(self):
        self.db             = SessionDB()
        self._current_id    = str(uuid.uuid4())
        self._current_start = datetime.utcnow()
        self._turns:        List[Dict] = []
        self._pending_tasks_this_session: List[str] = []

        # Initialize current session
        self.db.set_current("session_id",    self._current_id)
        self.db.set_current("started_at",    self._current_start.isoformat())
        self.db.set_current("status",        "active")

    def record_turn(self, user_input: str, system_output: str,
                    metadata: dict = None):
        """
        Record a conversation turn in the current session.
        Extracts topics, decisions, tasks, and facts automatically.
        """
        turn = {
            "ts":     datetime.utcnow().isoformat(),
            "input":  user_input[:500],
            "output": system_output[:500],
            "meta":   metadata or {},
        }
        self._turns.append(turn)

        # Auto-extract facts and tasks
        self._extract_signals(user_input, system_output)

    def _extract_signals(self, user_input: str, output: str):
        """Extract tasks, decisions, and facts from a conversation turn."""
        combined = (user_input + " " + output).lower()

        # Detect task creation
        task_triggers = ["i need to", "i should", "todo:", "task:", "remind me to",
                         "don't forget", "need to finish", "working on"]
        for trigger in task_triggers:
            if trigger in combined:
                idx   = combined.find(trigger) + len(trigger)
                title = combined[idx:idx+80].split(".")[0].strip()
                if len(title) > 5:
                    self._add_pending_task(title, user_input)
                    break

        # Detect decisions
        decision_triggers = ["decided to", "i've decided", "going with", "will use",
                             "chose to", "the decision is"]
        for trigger in decision_triggers:
            if trigger in combined:
                idx     = combined.find(trigger)
                excerpt = combined[idx:idx+100].split(".")[0]
                self._save_fact(excerpt, "decision", source="stated")
                break

        # Detect preferences
        preference_triggers = ["i prefer", "i like", "i don't like", "i hate",
                               "i always", "i never", "please always", "please never"]
        for trigger in preference_triggers:
            if trigger in combined:
                idx     = combined.find(trigger)
                excerpt = combined[idx:idx+100].split(".")[0]
                self._save_fact(excerpt, "preference", source="stated")
                break

    def _add_pending_task(self, title: str, context: str):
        task = PendingTask(
            task_id    = hashlib.sha256((title + self._current_id).encode()).hexdigest()[:12],
            created_at = datetime.utcnow().isoformat(),
            session_id = self._current_id,
            title      = title[:100],
            context    = context[:300],
        )
        self.db.save_task(task)
        self._pending_tasks_this_session.append(task.task_id)

    def _save_fact(self, content: str, category: str, source: str = "inferred"):
        fact = CrossSessionFact(
            fact_id    = hashlib.sha256(content.encode()).hexdigest()[:12],
            timestamp  = datetime.utcnow().isoformat(),
            content    = content[:300],
            category   = category,
            source     = source,
        )
        self.db.save_fact(fact)

    def complete_session(self, summary_text: str = ""):
        """
        Close the current session. Build and save a SessionSummary.
        """
        now      = datetime.utcnow()
        duration = (now - self._current_start).total_seconds() / 60

        # Extract topics from turns
        all_text = " ".join(t["input"] + " " + t["output"] for t in self._turns)
        topics   = self._extract_topics(all_text)

        # Decisions and facts from this session
        facts      = self.db.get_facts()
        decisions  = [f.content for f in facts if f.category == "decision"][-10:]

        # Pending tasks
        pending = [t.title for t in self.db.get_pending_tasks()
                   if t.session_id == self._current_id]

        summary = SessionSummary(
            session_id       = self._current_id,
            started_at       = self._current_start.isoformat(),
            ended_at         = now.isoformat(),
            duration_min     = round(duration, 1),
            topics           = topics,
            decisions        = decisions[:5],
            tasks_started    = self._pending_tasks_this_session,
            tasks_completed  = [],
            tasks_pending    = pending,
            key_facts        = [f.content for f in facts[-5:]],
            summary          = summary_text or self._auto_summarize(),
            turn_count       = len(self._turns),
        )
        self.db.save_session(summary)
        self.db.set_current("status", "completed")
        return summary

    def _extract_topics(self, text: str, n: int = 5) -> List[str]:
        """Extract main topics from conversation text."""
        from collections import Counter
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        stopwords = {"this", "that", "with", "from", "have", "been", "will",
                     "would", "could", "should", "your", "about", "what",
                     "when", "where", "which", "there", "their", "they"}
        words    = [w for w in words if w not in stopwords]
        common   = Counter(words).most_common(n * 3)
        # Group similar words — just take top N unique starts
        seen_stems = set()
        topics = []
        for word, _ in common:
            stem = word[:5]
            if stem not in seen_stems:
                seen_stems.add(stem)
                topics.append(word)
            if len(topics) >= n:
                break
        return topics

    def _auto_summarize(self) -> str:
        """Auto-generate a session summary from turns."""
        if not self._turns:
            return "Empty session."
        duration = (datetime.utcnow() - self._current_start).total_seconds() / 60
        n_turns  = len(self._turns)
        first_q  = self._turns[0]["input"][:80] if self._turns else ""
        return (f"Session lasted {duration:.0f} minutes with {n_turns} exchanges. "
                f"Started with: '{first_q}...'")

    def inject_context(self, current_query: str = "") -> str:
        """
        Build a context string to prepend to the current prompt.
        Includes: last session summary, pending tasks, relevant facts.
        """
        lines = []

        # Last session
        recent = self.db.get_recent_sessions(n=1)
        if recent:
            last = recent[0]
            last_dt = datetime.fromisoformat(last.started_at)
            age_h   = (datetime.utcnow() - last_dt).total_seconds() / 3600
            if age_h < 168:   # within last week
                lines.append(f"[Last session {age_h:.0f}h ago — {last.summary[:150]}]")
                if last.tasks_pending:
                    pending_str = ", ".join(last.tasks_pending[:3])
                    lines.append(f"[Unfinished tasks: {pending_str}]")

        # Current pending tasks
        pending = self.db.get_pending_tasks()[:3]
        if pending:
            for t in pending:
                lines.append(f"[Open task: {t.title}]")

        # Relevant facts (preferences + decisions)
        prefs = self.db.get_facts("preference")[:3]
        for f in prefs:
            lines.append(f"[Preference: {f.content[:100]}]")

        return "\n".join(lines) if lines else ""

    def morning_context(self) -> str:
        """Generate a rich morning context summary."""
        recent   = self.db.get_recent_sessions(n=3)
        pending  = self.db.get_pending_tasks()
        facts    = self.db.get_facts()[:10]
        now      = datetime.utcnow()

        lines = [
            "╔══════════════════════════════════════╗",
            f"║  Context Catchup — {now.strftime('%Y-%m-%d')}          ║",
            "╚══════════════════════════════════════╝",
        ]

        if recent:
            lines.append(f"\n  Last {len(recent)} sessions:")
            for s in recent:
                age = (now - datetime.fromisoformat(s.started_at)).total_seconds() / 3600
                lines.append(f"    {age:.0f}h ago: {s.summary[:100]}")
                if s.topics:
                    lines.append(f"    Topics: {', '.join(s.topics[:4])}")

        if pending:
            lines.append(f"\n  Open tasks ({len(pending)}):")
            for t in sorted(pending, key=lambda x: x.priority)[:5]:
                lines.append(f"    [{t.priority}] {t.title}")

        if facts:
            lines.append(f"\n  Persistent facts ({len(facts)}):")
            for f in facts[:5]:
                lines.append(f"    [{f.category}] {f.content[:80]}")

        return "\n".join(lines)

    def mark_task_done(self, task_id: str):
        tasks = self.db.get_pending_tasks()
        for t in tasks:
            if t.task_id == task_id:
                t.status = "completed"
                self.db.save_task(t)
                return True
        return False

    def add_fact(self, content: str, category: str = "knowledge",
                 expires_days: Optional[int] = None):
        expires = None
        if expires_days:
            expires = (datetime.utcnow() + timedelta(days=expires_days)).isoformat()
        self._save_fact(content, category)

    def stats(self) -> dict:
        recent  = self.db.get_recent_sessions(n=100)
        pending = self.db.get_pending_tasks()
        facts   = self.db.get_facts()
        total_min = sum(s.duration_min for s in recent)
        return {
            "total_sessions":    len(recent),
            "total_time_hours":  round(total_min / 60, 1),
            "pending_tasks":     len(pending),
            "stored_facts":      len(facts),
            "fact_categories":   {c: sum(1 for f in facts if f.category == c)
                                  for c in set(f.category for f in facts)},
        }


import re   # needed for _extract_topics
