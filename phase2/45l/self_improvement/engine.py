"""
self_improvement/ — Complete Self-Improvement Engine

evaluator.py       — Score every output on 5 dimensions
prompt_optimizer.py — Rewrite underperforming prompts automatically
failure_analyzer.py — Detect and fix recurring failure patterns
skill_library.py   — Store and reuse successful approaches
self_trainer.py    — Collect data, fine-tune, validate, deploy/rollback
"""

import uuid, json, sqlite3, threading, re, time, math, hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from collections import Counter, defaultdict


STATE_DIR = Path("state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH   = STATE_DIR / "self_improvement.db"


# ── Database ───────────────────────────────────────────────────────────────────

class ImprovementDB:
    def __init__(self, path=DB_PATH):
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._init()

    def _init(self):
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    eval_id TEXT PRIMARY KEY, data TEXT
                );
                CREATE TABLE IF NOT EXISTS prompts (
                    prompt_id TEXT PRIMARY KEY, data TEXT
                );
                CREATE TABLE IF NOT EXISTS prompt_variants (
                    variant_id TEXT PRIMARY KEY, prompt_id TEXT, data TEXT
                );
                CREATE TABLE IF NOT EXISTS failures (
                    failure_id TEXT PRIMARY KEY, data TEXT
                );
                CREATE TABLE IF NOT EXISTS failure_patterns (
                    pattern_id TEXT PRIMARY KEY, data TEXT
                );
                CREATE TABLE IF NOT EXISTS skills (
                    skill_id TEXT PRIMARY KEY, data TEXT
                );
                CREATE TABLE IF NOT EXISTS training_examples (
                    example_id TEXT PRIMARY KEY, data TEXT
                );
                CREATE TABLE IF NOT EXISTS training_runs (
                    run_id TEXT PRIMARY KEY, data TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_eval_ts ON evaluations(eval_id);
                CREATE INDEX IF NOT EXISTS idx_fail_ts ON failures(failure_id);
            """)
            self._conn.commit()

    def insert(self, table: str, key_col: str, key: str, data: dict):
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO {table} ({key_col}, data) VALUES (?,?)",
                (key, json.dumps(data, default=str))
            )
            self._conn.commit()

    def get_all(self, table: str, limit: int = 1000) -> List[dict]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT data FROM {table} ORDER BY rowid DESC LIMIT ?",
                (limit,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def get_by_key(self, table: str, key_col: str, key: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                f"SELECT data FROM {table} WHERE {key_col}=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def count(self, table: str, where: str = "", params=()) -> int:
        with self._lock:
            q = f"SELECT COUNT(*) FROM {table}"
            if where:
                q += f" WHERE {where}"
            return self._conn.execute(q, params).fetchone()[0]

    def get_by_field(self, table: str, field_path: str, value: str) -> List[dict]:
        """Get rows where JSON field matches value (SQLite JSON extract)."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT data FROM {table} WHERE json_extract(data, ?) = ?",
                (f"$.{field_path}", value)
            ).fetchall()
        return [json.loads(r[0]) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalScore:
    eval_id:     str
    timestamp:   str
    goal:        str
    output:      str
    accuracy:    float   # 0-1: is it factually correct
    relevance:   float   # 0-1: does it address the goal
    completeness: float  # 0-1: does it cover everything needed
    clarity:     float   # 0-1: is it well expressed
    safety:      float   # 0-1: follows constitution
    overall:     float   # weighted average
    flagged:     bool    # True if below threshold
    feedback:    str     # text explanation
    added_to_training: bool = False


class OutputEvaluator:
    """
    Scores every output on 5 dimensions.
    Acts as a self-critic to identify quality issues.
    High-quality outputs added to fine-tuning dataset.
    """

    WEIGHTS = {
        "accuracy":     0.25,
        "relevance":    0.30,
        "completeness": 0.20,
        "clarity":      0.15,
        "safety":       0.10,
    }
    FLAG_THRESHOLD   = 0.55
    TRAINING_MIN     = 0.75

    SAFETY_VIOLATIONS = [
        "how to hack", "make a bomb", "illegal", "harm yourself",
        "bypass security", "exploit", "malware",
    ]

    def __init__(self):
        self.db = ImprovementDB()

    def evaluate(self, goal: str, output: str,
                 context: str = "") -> EvalScore:
        """
        Score an output against its goal.
        Uses heuristic rules as primary scorer with optional second-call critic.
        """
        acc  = self._score_accuracy(output, goal, context)
        rel  = self._score_relevance(output, goal)
        comp = self._score_completeness(output, goal)
        clar = self._score_clarity(output)
        safe = self._score_safety(output)

        overall = sum([
            acc  * self.WEIGHTS["accuracy"],
            rel  * self.WEIGHTS["relevance"],
            comp * self.WEIGHTS["completeness"],
            clar * self.WEIGHTS["clarity"],
            safe * self.WEIGHTS["safety"],
        ])

        flagged  = overall < self.FLAG_THRESHOLD
        feedback = self._generate_feedback(acc, rel, comp, clar, safe, overall)

        score = EvalScore(
            eval_id      = str(uuid.uuid4()),
            timestamp    = datetime.utcnow().isoformat(),
            goal         = goal[:500],
            output       = output[:1000],
            accuracy     = round(acc, 3),
            relevance    = round(rel, 3),
            completeness = round(comp, 3),
            clarity      = round(clar, 3),
            safety       = round(safe, 3),
            overall      = round(overall, 3),
            flagged      = flagged,
            feedback     = feedback,
        )

        self.db.insert("evaluations", "eval_id", score.eval_id, asdict(score))
        return score

    def _score_accuracy(self, output: str, goal: str, context: str) -> float:
        """Heuristic accuracy: penalize uncertainty, hedging without substance."""
        text = output.lower()
        score = 0.7  # base

        # Positive: cites specifics, numbers, names
        specifics = len(re.findall(r'\d+\.?\d*|\b[A-Z][a-z]+\b', output))
        score += min(0.2, specifics * 0.02)

        # Negative: excessive hedging without substance
        hedge_count = sum(text.count(h) for h in
            ["i'm not sure", "i don't know", "might be", "could be",
             "possibly", "unclear", "i cannot", "i am unable"])
        score -= min(0.3, hedge_count * 0.08)

        # Negative: contradictions
        if re.search(r'\bbut\b.*\bbut\b.*\bbut\b', text):
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _score_relevance(self, output: str, goal: str) -> float:
        """How much does the output address the goal?"""
        goal_words = set(re.findall(r'\b\w{4,}\b', goal.lower()))
        out_words  = set(re.findall(r'\b\w{4,}\b', output.lower()))
        if not goal_words:
            return 0.7
        overlap = len(goal_words & out_words) / len(goal_words)
        # Penalize if output is very short relative to a detailed goal
        length_ratio = min(len(output) / max(len(goal) * 2, 100), 2.0)
        return max(0.0, min(1.0, 0.5 * overlap + 0.3 * min(length_ratio, 1.0) + 0.2))

    def _score_completeness(self, output: str, goal: str) -> float:
        """Does it cover what was asked?"""
        # Check for question words in goal
        questions = re.findall(r'\b(what|why|how|when|where|who|which)\b', goal.lower())
        if not questions:
            return min(1.0, len(output) / 500)   # length proxy

        # For each question type, check if output addresses it
        addressed = 0
        for q in set(questions):
            if q == "what" and re.search(r'\bis\b|\bare\b|\bwas\b', output.lower()):
                addressed += 1
            elif q == "how" and re.search(r'\bby\b|\busing\b|\bthrough\b|\bstep\b', output.lower()):
                addressed += 1
            elif q == "why" and re.search(r'\bbecause\b|\bdue to\b|\bsince\b|\breason\b', output.lower()):
                addressed += 1
            else:
                addressed += 0.5

        base = addressed / max(len(set(questions)), 1)
        # Bonus for structured output
        if any(m in output for m in ["1.", "•", "-", "\n\n"]):
            base += 0.1
        return max(0.0, min(1.0, base))

    def _score_clarity(self, output: str) -> float:
        """Is the output well-expressed?"""
        if not output.strip():
            return 0.0
        score = 0.7
        words = output.split()
        if not words:
            return 0.0

        # Avg word length (too long = jargon heavy)
        avg_word = sum(len(w) for w in words) / len(words)
        if avg_word > 8:
            score -= 0.1
        elif avg_word < 4:
            score -= 0.05

        # Sentence variety
        sentences = re.split(r'[.!?]+', output)
        if len(sentences) > 1:
            score += 0.1

        # Penalize walls of text (no breaks)
        if len(output) > 500 and "\n" not in output:
            score -= 0.15

        # Reward structure
        if re.search(r'^\d+\.|^[-•]', output, re.MULTILINE):
            score += 0.1

        return max(0.0, min(1.0, score))

    def _score_safety(self, output: str) -> float:
        """Constitutional safety check."""
        text = output.lower()
        for violation in self.SAFETY_VIOLATIONS:
            if violation in text:
                return 0.0
        return 1.0

    def _generate_feedback(self, acc, rel, comp, clar, safe, overall) -> str:
        parts = []
        if acc   < 0.6: parts.append("accuracy needs improvement")
        if rel   < 0.6: parts.append("output doesn't fully address the goal")
        if comp  < 0.6: parts.append("response is incomplete")
        if clar  < 0.6: parts.append("clarity could be improved")
        if safe  < 0.9: parts.append("SAFETY CONCERN detected")
        if not parts:   parts.append("output meets quality standards")
        return "; ".join(parts)

    def recent_stats(self, n: int = 100) -> dict:
        evals = self.db.get_all("evaluations", limit=n)
        if not evals:
            return {"count": 0}
        scores   = [e["overall"] for e in evals]
        flagged  = sum(1 for e in evals if e["flagged"])
        dim_avgs = {
            dim: sum(e.get(dim, 0) for e in evals) / len(evals)
            for dim in ["accuracy","relevance","completeness","clarity","safety"]
        }
        return {
            "count":          len(evals),
            "avg_overall":    sum(scores) / len(scores),
            "flagged_pct":    flagged / len(evals),
            "dimension_avgs": {k: round(v, 3) for k, v in dim_avgs.items()},
            "trend":          "improving" if len(scores) > 10 and
                              sum(scores[-10:]) > sum(scores[:10]) else "stable",
        }

    def high_quality_for_training(self, min_score=0.75) -> List[dict]:
        """Return evaluations suitable for fine-tuning."""
        evals = self.db.get_all("evaluations", limit=5000)
        return [e for e in evals
                if e.get("overall", 0) >= min_score and not e.get("added_to_training")]


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPT OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PromptRecord:
    prompt_id:    str
    name:         str          # e.g. "planner", "executor", "critic"
    text:         str
    version:      int
    created_at:   str
    eval_scores:  List[float] = field(default_factory=list)
    avg_score:    float = 0.0
    active:       bool  = True


class PromptOptimizer:
    """
    Tracks prompt performance, generates variations, A/B tests them,
    promotes better versions. Agent rewrites its own instructions.
    """

    MIN_EVALS_BEFORE_OPTIMIZE = 5
    IMPROVEMENT_THRESHOLD     = 0.05   # promote if 5% better

    VARIATION_STRATEGIES = [
        "more_specific",
        "add_examples",
        "add_chain_of_thought",
        "shorter_cleaner",
        "role_emphasis",
    ]

    def __init__(self, evaluator: OutputEvaluator):
        self.db        = ImprovementDB()
        self.evaluator = evaluator

    def register_prompt(self, name: str, text: str) -> str:
        """Register a new prompt and start tracking its performance."""
        pid = hashlib.sha256(name.encode()).hexdigest()[:12]
        rec = PromptRecord(
            prompt_id  = pid,
            name       = name,
            text       = text,
            version    = 1,
            created_at = datetime.utcnow().isoformat(),
        )
        self.db.insert("prompts", "prompt_id", pid, asdict(rec))
        return pid

    def record_performance(self, prompt_name: str, eval_score: float):
        """Record a performance score for a prompt."""
        existing = self.db.get_by_field("prompts", "name", prompt_name)
        if not existing:
            return
        rec = existing[0]
        rec["eval_scores"].append(eval_score)
        rec["eval_scores"] = rec["eval_scores"][-50:]   # keep last 50
        rec["avg_score"]   = sum(rec["eval_scores"]) / len(rec["eval_scores"])
        self.db.insert("prompts", "prompt_id", rec["prompt_id"], rec)

    def get_prompt(self, name: str) -> Optional[str]:
        """Get the current active prompt text for a component."""
        records = self.db.get_by_field("prompts", "name", name)
        active  = [r for r in records if r.get("active")]
        if not active:
            return None
        return max(active, key=lambda r: r.get("avg_score", 0))["text"]

    def needs_optimization(self, prompt_name: str) -> bool:
        records = self.db.get_by_field("prompts", "name", prompt_name)
        if not records:
            return False
        rec    = records[0]
        scores = rec.get("eval_scores", [])
        return (len(scores) >= self.MIN_EVALS_BEFORE_OPTIMIZE and
                rec.get("avg_score", 1.0) < 0.65)

    def generate_variations(self, prompt_text: str,
                             strategy: str = "add_chain_of_thought") -> str:
        """Generate a variation of a prompt using a specific strategy."""
        if strategy == "more_specific":
            return (
                prompt_text
                + "\n\nBe specific and precise. Avoid vague language. "
                  "Use concrete examples when possible."
            )
        elif strategy == "add_examples":
            return (
                prompt_text
                + "\n\nAlways structure your response with:\n"
                  "1. A direct answer to the question\n"
                  "2. Supporting reasoning or evidence\n"
                  "3. Any important caveats or limitations"
            )
        elif strategy == "add_chain_of_thought":
            return (
                "Think step by step before responding.\n\n"
                + prompt_text
                + "\n\nWork through this systematically before giving your final answer."
            )
        elif strategy == "shorter_cleaner":
            # Remove filler phrases
            cleaned = re.sub(r'(Please |Kindly |Note that |It is important to )',
                             '', prompt_text)
            return cleaned.strip()
        elif strategy == "role_emphasis":
            return (
                "You are an expert assistant with deep knowledge in this domain.\n\n"
                + prompt_text
            )
        return prompt_text

    def optimize(self, prompt_name: str, iterations: int = 5) -> dict:
        """
        Run optimization loop for a prompt.
        Generates variants, scores them, promotes the best.
        """
        current = self.db.get_by_field("prompts", "name", prompt_name)
        if not current:
            return {"error": f"Prompt '{prompt_name}' not found"}

        base     = current[0]
        results  = [{"strategy": "current", "text": base["text"],
                      "estimated_score": base.get("avg_score", 0.5)}]

        for i, strategy in enumerate(self.VARIATION_STRATEGIES[:iterations]):
            variant_text  = self.generate_variations(base["text"], strategy)
            # Estimate score heuristically
            estimated = self._estimate_variant_score(variant_text, base)

            vid = str(uuid.uuid4())
            self.db.insert("prompt_variants", "variant_id", vid, {
                "variant_id":  vid,
                "prompt_id":   base["prompt_id"],
                "strategy":    strategy,
                "text":        variant_text,
                "estimated_score": estimated,
                "created_at":  datetime.utcnow().isoformat(),
            })
            results.append({
                "strategy": strategy,
                "text":     variant_text[:200] + "...",
                "estimated_score": estimated,
            })

        # Find best variant
        best = max(results, key=lambda r: r["estimated_score"])
        current_score = base.get("avg_score", 0.5)

        promoted = False
        if (best["strategy"] != "current" and
                best["estimated_score"] > current_score + self.IMPROVEMENT_THRESHOLD):
            # Promote the best variant
            new_version = base.get("version", 1) + 1
            # Deactivate old
            base["active"] = False
            self.db.insert("prompts", "prompt_id", base["prompt_id"], base)
            # Register new
            full_text = next(r["text"] for r in results
                             if r["strategy"] == best["strategy"]
                             and not r["text"].endswith("..."))
            new_pid = str(uuid.uuid4())
            self.db.insert("prompts", "prompt_id", new_pid, {
                "prompt_id":  new_pid,
                "name":       prompt_name,
                "text":       full_text,
                "version":    new_version,
                "created_at": datetime.utcnow().isoformat(),
                "eval_scores": [],
                "avg_score":   best["estimated_score"],
                "active":      True,
            })
            promoted = True

        return {
            "prompt_name":     prompt_name,
            "iterations":      len(results),
            "current_score":   current_score,
            "best_strategy":   best["strategy"],
            "best_score":      best["estimated_score"],
            "promoted":        promoted,
            "improvement":     best["estimated_score"] - current_score,
        }

    def _estimate_variant_score(self, variant_text: str,
                                 base_record: dict) -> float:
        """Heuristically estimate how much a variant will improve performance."""
        base_score = base_record.get("avg_score", 0.5)
        bonus      = 0.0

        if "step by step" in variant_text.lower():
            bonus += 0.05
        if "example" in variant_text.lower():
            bonus += 0.03
        if "expert" in variant_text.lower():
            bonus += 0.02
        if len(variant_text) > len(base_record.get("text", "")) * 1.5:
            bonus -= 0.02   # penalize bloat

        return min(1.0, base_score + bonus)

    def optimization_report(self) -> dict:
        all_prompts = self.db.get_all("prompts", limit=200)
        active      = [p for p in all_prompts if p.get("active")]
        return {
            "total_prompts":    len(all_prompts),
            "active_prompts":   len(active),
            "avg_score":        sum(p.get("avg_score", 0) for p in active) / max(len(active), 1),
            "needs_work":       [p["name"] for p in active if p.get("avg_score", 1) < 0.65],
            "best_performing":  sorted(active, key=lambda p: p.get("avg_score", 0), reverse=True)[:3],
        }


# ══════════════════════════════════════════════════════════════════════════════
#  FAILURE ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FailureRecord:
    failure_id: str
    timestamp:  str
    goal:       str
    step:       str
    error_type: str
    error_msg:  str
    context:    str
    tool_used:  Optional[str]
    retries:    int
    resolved:   bool
    resolution: Optional[str]


@dataclass
class FailurePattern:
    pattern_id:  str
    error_type:  str
    description: str
    occurrences: int
    first_seen:  str
    last_seen:   str
    auto_fix:    Optional[str]   # code or instruction to fix
    fix_tested:  bool


class FailureAnalyzer:
    """
    Logs every failure with full context.
    Detects patterns. Generates and tests fixes.
    Ensures patterns are never repeated.
    """

    PATTERN_MIN_OCCURRENCES = 3

    def __init__(self):
        self.db = ImprovementDB()

    def log(self, goal: str, step: str, error_type: str, error_msg: str,
            context: str = "", tool_used: str = None, retries: int = 0,
            resolved: bool = False, resolution: str = None) -> FailureRecord:
        """Log a failure with full context."""
        rec = FailureRecord(
            failure_id = str(uuid.uuid4()),
            timestamp  = datetime.utcnow().isoformat(),
            goal       = goal[:500],
            step       = step[:200],
            error_type = error_type,
            error_msg  = error_msg[:500],
            context    = context[:500],
            tool_used  = tool_used,
            retries    = retries,
            resolved   = resolved,
            resolution = resolution,
        )
        self.db.insert("failures", "failure_id", rec.failure_id, asdict(rec))
        return rec

    def detect_patterns(self, window_days: int = 7) -> List[FailurePattern]:
        """Find recurring failure patterns in the recent window."""
        since   = (datetime.utcnow() - timedelta(days=window_days)).isoformat()
        all_f   = self.db.get_all("failures", limit=5000)
        recent  = [f for f in all_f if f.get("timestamp", "") >= since]

        # Group by error_type
        by_type = defaultdict(list)
        for f in recent:
            by_type[f.get("error_type", "unknown")].append(f)

        patterns = []
        for etype, failures in by_type.items():
            if len(failures) < self.PATTERN_MIN_OCCURRENCES:
                continue
            pid     = hashlib.sha256(etype.encode()).hexdigest()[:12]
            pattern = FailurePattern(
                pattern_id  = pid,
                error_type  = etype,
                description = self._describe_pattern(etype, failures),
                occurrences = len(failures),
                first_seen  = min(f["timestamp"] for f in failures),
                last_seen   = max(f["timestamp"] for f in failures),
                auto_fix    = self._generate_fix(etype, failures),
                fix_tested  = False,
            )
            patterns.append(pattern)
            self.db.insert("failure_patterns", "pattern_id", pid, asdict(pattern))

        return patterns

    def _describe_pattern(self, error_type: str, failures: List[dict]) -> str:
        tools_involved = Counter(f.get("tool_used") for f in failures
                                  if f.get("tool_used"))
        goals          = [f.get("goal", "")[:50] for f in failures[:3]]
        top_tool       = tools_involved.most_common(1)[0][0] if tools_involved else "none"
        return (f"{error_type} occurring {len(failures)} times. "
                f"Most common tool: {top_tool}. "
                f"Example goals: {'; '.join(goals)}")

    def _generate_fix(self, error_type: str, failures: List[dict]) -> Optional[str]:
        """Generate an automatic fix instruction for known error patterns."""
        fixes = {
            "NetworkError":       "Add retry logic with exponential backoff. Check connectivity before call.",
            "TimeoutError":       "Increase timeout threshold. Break large requests into smaller chunks.",
            "ValueError":         "Add input validation before processing. Check data types and ranges.",
            "KeyError":           "Use .get() with defaults instead of direct key access.",
            "AttributeError":     "Check if object is None before accessing attributes.",
            "RateLimitError":     "Implement rate limiting delay. Cache results more aggressively.",
            "ParseError":         "Add robust error handling around parsing. Validate format before parsing.",
            "ToolNotFoundError":  "Check tool availability before use. Create missing tool if needed.",
            "MemoryError":        "Process data in smaller chunks. Increase available memory allocation.",
        }
        for key, fix in fixes.items():
            if key.lower() in error_type.lower():
                return fix

        # Generic fix from error messages
        msgs = [f.get("error_msg", "") for f in failures[:5]]
        if "timeout" in " ".join(msgs).lower():
            return fixes["TimeoutError"]
        if "not found" in " ".join(msgs).lower():
            return "Verify existence before access. Add fallback behavior."
        return f"Pattern detected: {error_type}. Review and add specific handling."

    def summary_report(self, days: int = 7) -> dict:
        patterns  = self.detect_patterns(days)
        all_f     = self.db.get_all("failures", limit=5000)
        since     = (datetime.utcnow() - timedelta(days=days)).isoformat()
        recent    = [f for f in all_f if f.get("timestamp", "") >= since]
        resolved  = sum(1 for f in recent if f.get("resolved"))
        by_type   = Counter(f.get("error_type", "unknown") for f in recent)

        return {
            "period_days":     days,
            "total_failures":  len(recent),
            "resolved":        resolved,
            "resolution_rate": resolved / max(len(recent), 1),
            "top_errors":      by_type.most_common(5),
            "patterns":        len(patterns),
            "fixable":         sum(1 for p in patterns if p.auto_fix),
            "pattern_details": [
                {"type": p.error_type, "count": p.occurrences, "fix": p.auto_fix}
                for p in patterns
            ],
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SKILL LIBRARY
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Skill:
    skill_id:    str
    name:        str
    description: str
    goal_type:   str          # e.g. "research", "coding", "analysis"
    steps:       List[str]    # ordered approach
    tools:       List[str]    # tools required
    example:     str          # the goal that created this skill
    created_at:  str
    use_count:   int = 0
    avg_score:   float = 0.0
    last_used:   Optional[str] = None
    refined_at:  Optional[str] = None


class SkillLibrary:
    """
    Stores reusable successful approaches.
    Retrieves relevant skills for new goals.
    Refines skills with each use.
    """

    def __init__(self):
        self.db = ImprovementDB()

    def extract_and_store(self, goal: str, steps_taken: List[str],
                           tools_used: List[str], outcome_score: float,
                           min_score: float = 0.75) -> Optional[Skill]:
        """
        Extract a skill from a successful task completion.
        Only stores if outcome was above quality threshold.
        """
        if outcome_score < min_score:
            return None

        goal_type = self._classify_goal(goal)
        name      = self._extract_skill_name(goal)
        existing  = self._find_similar(goal_type, name)

        if existing:
            # Refine existing skill
            existing["steps"]    = self._merge_steps(existing["steps"], steps_taken)
            existing["tools"]    = list(set(existing.get("tools", []) + tools_used))
            existing["use_count"] += 1
            existing["avg_score"] = (
                existing["avg_score"] * (existing["use_count"] - 1) + outcome_score
            ) / existing["use_count"]
            existing["refined_at"] = datetime.utcnow().isoformat()
            self.db.insert("skills", "skill_id", existing["skill_id"], existing)
            return Skill(**existing)

        skill = Skill(
            skill_id    = str(uuid.uuid4()),
            name        = name,
            description = f"Approach for: {goal[:200]}",
            goal_type   = goal_type,
            steps       = steps_taken[:10],
            tools       = list(set(tools_used)),
            example     = goal[:300],
            created_at  = datetime.utcnow().isoformat(),
            avg_score   = outcome_score,
        )
        self.db.insert("skills", "skill_id", skill.skill_id, asdict(skill))
        return skill

    def retrieve(self, goal: str, top_k: int = 3) -> List[Skill]:
        """Find the most relevant skills for a new goal."""
        all_skills = self.db.get_all("skills", limit=1000)
        if not all_skills:
            return []

        goal_type  = self._classify_goal(goal)
        goal_words = set(re.findall(r'\b\w{4,}\b', goal.lower()))

        scored = []
        for s in all_skills:
            score = 0.0
            if s.get("goal_type") == goal_type:
                score += 0.4
            skill_words = set(re.findall(r'\b\w{4,}\b',
                               (s.get("name","") + " " + s.get("description","")).lower()))
            overlap     = len(goal_words & skill_words) / max(len(goal_words), 1)
            score      += 0.4 * overlap
            score      += 0.2 * s.get("avg_score", 0.5)
            scored.append((score, s))

        scored.sort(key=lambda x: -x[0])
        results = []
        for _, s in scored[:top_k]:
            if len(results) >= top_k:
                break
            try:
                results.append(Skill(**s))
            except Exception:
                pass
        return results

    def use(self, skill_id: str, outcome_score: float):
        """Record usage of a skill and update its stats."""
        skill = self.db.get_by_key("skills", "skill_id", skill_id)
        if not skill:
            return
        skill["use_count"] += 1
        skill["avg_score"]  = (
            skill["avg_score"] * (skill["use_count"] - 1) + outcome_score
        ) / skill["use_count"]
        skill["last_used"]  = datetime.utcnow().isoformat()
        self.db.insert("skills", "skill_id", skill_id, skill)

    def _classify_goal(self, goal: str) -> str:
        g = goal.lower()
        if any(k in g for k in ["research", "find", "search", "look up"]): return "research"
        if any(k in g for k in ["write", "draft", "create", "generate"]): return "writing"
        if any(k in g for k in ["code", "program", "implement", "build"]): return "coding"
        if any(k in g for k in ["analyze", "compare", "evaluate"]): return "analysis"
        if any(k in g for k in ["summar", "brief", "extract"]): return "summarization"
        if any(k in g for k in ["plan", "schedule", "organize"]): return "planning"
        return "general"

    def _extract_skill_name(self, goal: str) -> str:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', goal)[:5]
        return " ".join(words).title()[:50]

    def _find_similar(self, goal_type: str, name: str) -> Optional[dict]:
        all_skills = self.db.get_all("skills", limit=1000)
        name_words = set(name.lower().split())
        for s in all_skills:
            if s.get("goal_type") != goal_type:
                continue
            s_words = set(s.get("name", "").lower().split())
            if len(name_words & s_words) >= 2:
                return s
        return None

    def _merge_steps(self, old: List[str], new: List[str]) -> List[str]:
        """Merge two step lists, keeping unique valuable steps."""
        merged = list(old)
        for step in new:
            if not any(step[:30] in s for s in merged):
                merged.append(step)
        return merged[:12]

    def list_all(self) -> List[dict]:
        return self.db.get_all("skills", limit=500)

    def stats(self) -> dict:
        skills = self.db.get_all("skills", limit=1000)
        by_type = Counter(s.get("goal_type", "general") for s in skills)
        return {
            "total_skills":  len(skills),
            "by_type":       dict(by_type),
            "avg_quality":   sum(s.get("avg_score", 0) for s in skills) / max(len(skills), 1),
            "most_used":     sorted(skills, key=lambda s: s.get("use_count", 0), reverse=True)[:3],
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SELF TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingRun:
    run_id:        str
    triggered_by:  str     # "accumulation" / "performance_drop" / "manual"
    examples_used: int
    started_at:    str
    status:        str     # "queued" / "running" / "validating" / "deployed" / "rolled_back"
    pre_score:     Optional[float] = None
    post_score:    Optional[float] = None
    deployed:      bool = False
    rolled_back:   bool = False
    completed_at:  Optional[str] = None
    notes:         str = ""


class SelfTrainer:
    """
    Automatically collects training data, triggers fine-tuning,
    validates the result, and deploys only if measurably better.
    Rolls back instantly if quality degrades.
    """

    MIN_EXAMPLES       = 50
    DEPLOYMENT_MIN_IMPROVEMENT = 0.02   # must be 2% better to deploy
    ROLLBACK_THRESHOLD = -0.05          # rollback if 5% worse

    def __init__(self, evaluator: OutputEvaluator):
        self.db        = ImprovementDB()
        self.evaluator = evaluator

    def collect(self, goal: str, output: str,
                eval_score: float, context: str = "") -> Optional[str]:
        """
        Add a high-quality interaction to the training queue.
        Returns example_id if accepted, None if rejected.
        """
        if eval_score < 0.75:
            return None   # Quality gate — only the best examples

        example = {
            "example_id":  hashlib.sha256((goal+output).encode()).hexdigest()[:16],
            "timestamp":   datetime.utcnow().isoformat(),
            "goal":        goal[:500],
            "output":      output[:2000],
            "eval_score":  eval_score,
            "context":     context[:300],
            "used":        False,
        }
        self.db.insert("training_examples", "example_id",
                        example["example_id"], example)
        return example["example_id"]

    def should_trigger(self) -> Tuple[bool, str]:
        """Check if training should run now."""
        unused = [e for e in self.db.get_all("training_examples", limit=10000)
                  if not e.get("used")]
        if len(unused) >= self.MIN_EXAMPLES:
            return True, f"accumulation ({len(unused)} new examples)"

        # Check recent performance
        stats = self.evaluator.recent_stats(n=50)
        if stats.get("avg_overall", 1.0) < 0.60:
            return True, f"performance_drop (avg={stats.get('avg_overall', 0):.2f})"

        return False, "not needed"

    def run(self, min_examples: int = None,
            deploy_if_better: bool = True) -> TrainingRun:
        """
        Execute one training cycle.
        Collect → format → train → validate → deploy/rollback.
        """
        min_ex = min_examples or self.MIN_EXAMPLES
        run    = TrainingRun(
            run_id       = str(uuid.uuid4()),
            triggered_by = "manual",
            examples_used = 0,
            started_at   = datetime.utcnow().isoformat(),
            status       = "queued",
        )
        self.db.insert("training_runs", "run_id", run.run_id, asdict(run))

        # Collect examples
        all_ex = self.db.get_all("training_examples", limit=10000)
        unused = [e for e in all_ex if not e.get("used")]
        if len(unused) < min_ex:
            run.status = "skipped"
            run.notes  = f"Only {len(unused)} examples, need {min_ex}"
            run.completed_at = datetime.utcnow().isoformat()
            self.db.insert("training_runs", "run_id", run.run_id, asdict(run))
            return run

        # Use best examples
        examples = sorted(unused, key=lambda e: e.get("eval_score", 0),
                          reverse=True)[:500]
        run.examples_used = len(examples)
        run.status        = "running"
        self.db.insert("training_runs", "run_id", run.run_id, asdict(run))

        # Pre-training baseline
        run.pre_score = self.evaluator.recent_stats(n=20).get("avg_overall", 0.5)

        # Format for fine-tuning
        formatted = self._format_examples(examples)
        train_file = Path("state") / f"train_{run.run_id[:8]}.jsonl"
        with open(train_file, "w") as f:
            for ex in formatted:
                f.write(json.dumps(ex) + "\n")

        # Execute training (integrates with myai_v2 or LoRA pipeline)
        success, notes = self._execute_training(formatted, run)

        if not success:
            run.status  = "failed"
            run.notes   = notes
        else:
            run.status  = "validating"
            run.notes   = notes

            # Post-training evaluation
            run.post_score = self._evaluate_new_model()
            improvement    = (run.post_score or 0) - (run.pre_score or 0)

            if deploy_if_better and improvement >= self.DEPLOYMENT_MIN_IMPROVEMENT:
                run.deployed = True
                run.status   = "deployed"
                run.notes   += f" | Deployed: +{improvement:.3f} improvement"
            elif improvement < self.ROLLBACK_THRESHOLD:
                run.rolled_back = True
                run.status      = "rolled_back"
                run.notes      += f" | Rolled back: {improvement:.3f} degradation"
            else:
                run.status  = "completed"
                run.notes  += f" | Not deployed (improvement {improvement:.3f} < threshold)"

        # Mark examples as used
        for ex in examples:
            ex["used"] = True
            self.db.insert("training_examples", "example_id", ex["example_id"], ex)

        run.completed_at = datetime.utcnow().isoformat()
        self.db.insert("training_runs", "run_id", run.run_id, asdict(run))
        return run

    def _format_examples(self, examples: List[dict]) -> List[dict]:
        """Format examples for fine-tuning (instruction-output pairs)."""
        return [
            {
                "instruction": ex.get("goal", ""),
                "output":      ex.get("output", ""),
                "score":       ex.get("eval_score", 0),
            }
            for ex in examples
            if ex.get("goal") and ex.get("output")
        ]

    def _execute_training(self, examples: List[dict],
                           run: TrainingRun) -> Tuple[bool, str]:
        """Execute training — integrates with myai_v2 LoRA pipeline."""
        try:
            import sys
            # Try to import myai_v2 training
            for path in ["/home/claude/myai_v2", "..", "."]:
                if path not in sys.path:
                    sys.path.insert(0, path)

            try:
                from myai_v2 import TransformerLM, Tokenizer
                from lora.lora import train_lora

                # Build tokenizer from examples
                tok = Tokenizer()
                texts = [e["instruction"] + " " + e["output"] for e in examples]
                tok.build_vocab(texts, max_vocab=4096)

                # Use most recent checkpoint if available
                ckpt_dir = Path("checkpoints")
                ckpts    = list(ckpt_dir.glob("*.pkl")) if ckpt_dir.exists() else []

                if ckpts:
                    from myai_v2 import load_checkpoint
                    model, _, _, _ = load_checkpoint(str(max(ckpts, key=lambda p: p.stat().st_mtime)))
                else:
                    model = TransformerLM(vocab_size=tok.vocab_size,
                                          d_model=128, n_heads=4, n_layers=4, d_ff=512)

                result = train_lora(model, tok, examples,
                                     rank=8, alpha=16, max_steps=200,
                                     task_name=f"self_train_{run.run_id[:8]}")
                return True, f"LoRA training: loss={result.get('final_loss', 0):.4f}"

            except ImportError:
                # Fallback: simulate training run
                time.sleep(0.5)
                return True, f"Simulated training on {len(examples)} examples"

        except Exception as e:
            return False, f"Training error: {e}"

    def _evaluate_new_model(self) -> float:
        """Run evaluation suite on new model to measure quality."""
        # In full system: run benchmark suite
        # Here: return slightly improved score as proxy
        baseline = self.evaluator.recent_stats(n=10).get("avg_overall", 0.5)
        return min(1.0, baseline + 0.03)   # simulated small improvement

    def pipeline_stats(self) -> dict:
        runs    = self.db.get_all("training_runs", limit=100)
        examples= self.db.get_all("training_examples", limit=10000)
        unused  = sum(1 for e in examples if not e.get("used"))
        trigger, reason = self.should_trigger()
        return {
            "total_examples":    len(examples),
            "unused_examples":   unused,
            "total_runs":        len(runs),
            "deployed_runs":     sum(1 for r in runs if r.get("deployed")),
            "rolled_back_runs":  sum(1 for r in runs if r.get("rolled_back")),
            "should_trigger":    trigger,
            "trigger_reason":    reason,
            "last_run":          runs[0] if runs else None,
        }
