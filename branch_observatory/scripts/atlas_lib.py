from __future__ import annotations

import datetime as dt
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

UTC = dt.timezone.utc
SCHEMA_VERSION = "1.0.0"
TOOL_VERSION = "0.1.0"
STATUS_NAMES = [
    "Proposed", "Specified", "Implemented", "Locally verified", "CPU-tested",
    "GPU-qualified", "TPU-qualified", "Executed scientifically", "Replicated",
    "Supported within a bounded regime", "Negative result", "Inconclusive",
    "Superseded", "Blocked", "Unknown",
]
SOURCE_EXTENSIONS = {
    ".py": "Python", ".pyi": "Python", ".js": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".c": "C", ".h": "C/C++",
    ".cc": "C++", ".cpp": "C++", ".cxx": "C++", ".hpp": "C++", ".hh": "C++",
    ".rs": "Rust", ".go": "Go", ".java": "Java", ".kt": "Kotlin", ".swift": "Swift",
    ".scala": "Scala", ".sh": "Shell", ".bash": "Shell", ".ps1": "PowerShell",
    ".sql": "SQL", ".html": "HTML", ".htm": "HTML", ".css": "CSS", ".scss": "SCSS",
    ".vue": "Vue", ".svelte": "Svelte", ".r": "R", ".jl": "Julia", ".lua": "Lua",
    ".m": "Objective-C", ".mm": "Objective-C++", ".dart": "Dart", ".ex": "Elixir",
    ".exs": "Elixir", ".erl": "Erlang", ".fs": "F#", ".fsx": "F#", ".vb": "Visual Basic",
    ".sol": "Solidity", ".zig": "Zig", ".nim": "Nim", ".clj": "Clojure",
}
DOC_EXTENSIONS = {".md", ".markdown", ".rst", ".adoc", ".txt"}
CONFIG_EXTENSIONS = {".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf"}
DATA_EXTENSIONS = {
    ".json", ".jsonl", ".csv", ".tsv", ".parquet", ".arrow", ".npy", ".npz",
    ".pkl", ".pickle", ".db", ".sqlite", ".bin", ".pt", ".pth", ".safetensors",
    ".onnx", ".h5", ".zip", ".tar", ".gz", ".xz", ".7z", ".log",
}
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".mp3", ".mp4",
    ".wav", ".zip", ".tar", ".gz", ".7z", ".pt", ".pth", ".safetensors", ".onnx",
    ".h5", ".npy", ".npz", ".pkl", ".pickle", ".db", ".sqlite",
}
DOCUMENT_TOKENS = {
    "agent", "agents", "readme", "handoff", "brief", "spec", "specification",
    "blueprint", "architecture", "requirements", "contract", "plan", "roadmap",
    "protocol", "prereg", "preregistration", "amendment", "readiness", "result",
    "results", "postmortem", "audit", "receipt", "evidence", "ledger", "runbook",
    "operator", "decision", "question", "questions", "negative", "supersession",
    "superseded", "snapshot", "status", "verdict", "failure", "failures", "report",
    "summary", "experiment", "campaign", "manifest", "gate", "state", "current",
    "open", "implementation", "training", "benchmark",
}


class AtlasError(RuntimeError):
    pass


def utc_now() -> dt.datetime:
    return dt.datetime.now(UTC)


def iso_utc(value: dt.datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_datetime(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        result = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def safe_json(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dt.datetime):
        return iso_utc(value)
    if isinstance(value, Counter):
        return dict(sorted(value.items()))
    if isinstance(value, defaultdict):
        return {str(k): safe_json(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    return value


def atomic_write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb" if isinstance(content, bytes) else "w",
            encoding=None if isinstance(content, bytes) else "utf-8",
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(safe_json(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    atomic_write(path, text if text.endswith("\n") else text + "\n")


def scrub_url(value: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(value.strip())
    except ValueError:
        return "<unparseable-url>"
    if not parsed.scheme:
        return value.strip()
    host = parsed.hostname or ""
    if parsed.port:
        host += f":{parsed.port}"
    netloc = f"<credentials-redacted>@{host}" if parsed.username or parsed.password else host
    return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def slugify_ref(ref: str) -> str:
    value = ref
    for prefix, label in [("refs/heads/", "local--"), ("refs/remotes/", "remote--"), ("refs/tags/", "tag--")]:
        if value.startswith(prefix):
            value = label + value[len(prefix):]
            break
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-").lower() or "ref"


def line_count(data: bytes) -> int:
    if not data:
        return 0
    return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)


def redact_text(value: str) -> str:
    value = re.sub(r"(?i)\b(password|passwd|secret|token|api[_-]?key|private[_-]?key)\b\s*[:=]\s*[^\s,;]+", r"\1=<redacted>", value)
    value = re.sub(r"(?i)(?:[A-Z]:[\\/]|\\\\)[^\s`'\"]+", "<local-path>", value)
    value = re.sub(r"(?i)(?<![A-Za-z0-9])/(?:Users|home|private|mnt|var/tmp)/[^\s`'\"]+", "<local-path>", value)
    value = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "<email>", value)
    return value


def path_parts(path: str) -> list[str]:
    return [item.lower() for item in path.replace("\\", "/").split("/") if item]


def simple_match(value: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(value.lower().replace("\\", "/"), pattern.lower().replace("\\", "/"))


def path_matches(path: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        if simple_match(path, pattern):
            return pattern
    return None


def language_for_extension(extension: str) -> str | None:
    return SOURCE_EXTENSIONS.get(extension) or {
        ".md": "Markdown", ".markdown": "Markdown", ".rst": "reStructuredText",
        ".txt": "Text", ".json": "JSON", ".jsonl": "JSON Lines", ".csv": "CSV",
        ".tsv": "TSV", ".yaml": "YAML", ".yml": "YAML", ".toml": "TOML",
        ".ini": "INI", ".cfg": "Configuration", ".xml": "XML", ".html": "HTML", ".css": "CSS",
    }.get(extension)


def classify_path(path: str, config: dict[str, Any]) -> dict[str, Any]:
    normalized = path.replace("\\", "/")
    lower = normalized.lower()
    parts = path_parts(normalized)
    extension = Path(lower).suffix
    excluded = path_matches(normalized, config.get("excluded_paths", []))
    if excluded:
        return {"category": "excluded", "language": None, "exclusion_reason": excluded, "generated": False, "binary": False}
    if {"vendor", "vendors", "third_party", "thirdparty", "node_modules", "site-packages"}.intersection(parts):
        return {"category": "vendor", "language": None, "exclusion_reason": "vendor_or_dependency_path", "generated": False, "binary": False}
    if extension == ".ipynb" or "notebook" in parts or "notebooks" in parts:
        return {"category": "notebook", "language": "Jupyter Notebook", "exclusion_reason": None, "generated": False, "binary": False}
    name = Path(lower).name
    is_test = "tests" in parts or "test" in parts or name.startswith("test_") or name.endswith("_test.py") or ".test." in name
    if is_test and extension in SOURCE_EXTENSIONS:
        return {"category": "tests", "language": SOURCE_EXTENSIONS[extension], "exclusion_reason": None, "generated": False, "binary": False}
    top_doc = name in {"readme.md", "agents.md", "agent.md", "license", "todo.md", "progress.md", "bramastra.md", "bramastra_paper.md", "an_ra_program.md", "agi_blueprint.md", "esoes.md"}
    if extension in DOC_EXTENSIONS or "docs" in parts or "blueprint" in parts or top_doc:
        if extension == ".txt" and not (top_doc or "readme" in name or "requirements" in name or "constraints" in name):
            pass
        else:
            return {"category": "documentation", "language": "Markdown" if extension in {".md", ".markdown"} else "Text", "exclusion_reason": None, "generated": False, "binary": False}
    if {"generated", "autogenerated", "__generated__"}.intersection(parts) or "generated" in Path(lower).stem:
        return {"category": "generated_code", "language": SOURCE_EXTENSIONS.get(extension), "exclusion_reason": None, "generated": True, "binary": extension in BINARY_EXTENSIONS}
    if {"artifacts", "artifact", "receipts", "receipt", "results", "result", "output", "outputs", "reports", "checkpoints", "checkpoint", "runs", "run", "state", "logs"}.intersection(parts) and extension not in SOURCE_EXTENSIONS:
        return {"category": "generated_artifact", "language": language_for_extension(extension), "exclusion_reason": None, "generated": True, "binary": extension in BINARY_EXTENSIONS}
    config_names = {"makefile", "dockerfile", "pyproject.toml", "setup.py", "setup.cfg", "tox.ini", "pytest.ini", "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "requirements.txt", "constraints.txt"}
    if extension in CONFIG_EXTENSIONS or name in config_names or name.startswith("requirements") or name.startswith("constraints") or ".github" in parts or "config" in parts or "configs" in parts:
        return {"category": "build_config", "language": language_for_extension(extension) or "Configuration", "exclusion_reason": None, "generated": False, "binary": False}
    if extension in SOURCE_EXTENSIONS:
        return {"category": "production_source", "language": SOURCE_EXTENSIONS[extension], "exclusion_reason": None, "generated": False, "binary": False}
    if extension in BINARY_EXTENSIONS:
        return {"category": "binary", "language": None, "exclusion_reason": None, "generated": True, "binary": True}
    if extension in DATA_EXTENSIONS:
        return {"category": "data", "language": language_for_extension(extension), "exclusion_reason": None, "generated": False, "binary": extension in BINARY_EXTENSIONS}
    return {"category": "unknown", "language": language_for_extension(extension), "exclusion_reason": None, "generated": False, "binary": extension in BINARY_EXTENSIONS}


def is_document_candidate(path: str) -> bool:
    lower = path.replace("\\", "/").lower()
    extension = Path(lower).suffix
    if extension in DOC_EXTENSIONS or extension == ".ipynb":
        return True
    tokens = set(re.split(r"[^a-z0-9]+", lower)) | set(path_parts(lower))
    if any(token in DOCUMENT_TOKENS for token in tokens) and (extension in CONFIG_EXTENSIONS | DATA_EXTENSIONS | {".xml", ".gz", ".zip"} or not extension):
        return True
    if any(part in {"docs", "blueprint", "experiments"} for part in path_parts(lower)) and (extension in {".json", ".yaml", ".yml", ".xml"} or not extension):
        return True
    name = Path(lower).name
    return extension in DOC_EXTENSIONS and (name.startswith("readme") or name.startswith("agent") or name.startswith("handoff"))


def document_type(path: str) -> str:
    name = Path(path.lower()).name
    if path.lower().endswith(".ipynb"):
        return "notebook"
    for tokens, label in [
        (("prereg", "protocol", "amendment"), "preregistration_or_protocol"),
        (("readiness", "run_readiness"), "readiness"), (("receipt",), "receipt"),
        (("evidence", "ledger"), "evidence_ledger"), (("result", "verdict", "summary"), "result_or_summary"),
        (("audit", "forensic", "postmortem"), "audit"), (("plan", "roadmap", "execution"), "plan"),
        (("handoff", "brief", "agent"), "handoff_or_brief"),
        (("blueprint", "architecture", "specification", "contract", "requirements"), "specification"),
        (("decision", "question", "negative", "failure"), "decision_or_open_question"),
    ]:
        if any(token in name for token in tokens):
            return label
    if name.startswith("readme") or "status" in name or "current" in name:
        return "current_state_or_readme"
    if name.endswith(".json") or name.endswith((".yaml", ".yml")):
        return "machine_record"
    return "text_record" if name.endswith(".txt") else "document"


def extract_title(path: str, content: str) -> str:
    for line in content.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()
    try:
        value = json.loads(content)
        if isinstance(value, dict):
            for key in ("title", "name", "experiment_id", "status", "verdict"):
                if isinstance(value.get(key), str) and value[key].strip():
                    return value[key].strip()[:180]
    except (ValueError, TypeError):
        pass
    return Path(path).name


def extract_date(content: str) -> tuple[str | None, str | None]:
    patterns = [
        r"(?i)\b(?:audit|design|execution|implementation|commit|created|updated|document)\s+date\s*[:=]\s*([^\n]+)",
        r"(?i)\bdate\s*[:=]\s*([^\n]+)", r"\b(20\d{2}[-/]\d{2}[-/]\d{2})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, content[:50000])
        if not match:
            continue
        parsed = parse_datetime(match.group(1).strip().strip("`"))
        if parsed:
            return parsed.date().isoformat(), "document_text"
        date_match = re.search(r"\b(20\d{2}[-/]\d{2}[-/]\d{2})\b", match.group(1))
        if date_match:
            return date_match.group(1).replace("/", "-"), "document_text"
    return None, None


def extract_sections(content: str) -> list[str]:
    result: list[str] = []
    for line in content.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match and match.group(1) not in result:
            result.append(match.group(1)[:180])
        if len(result) >= 40:
            break
    return result


def extract_dependencies(content: str) -> list[str]:
    result: list[str] = []
    for match in re.finditer(r"(?i)\b(?:supersedes|superseded by|depends on|dependency|source of truth|canonical source|see|read first)\s*[:=]\s*([^\n]+)", content[:100000]):
        for token in re.findall(r"[A-Za-z0-9_./-]+\.(?:md|json|yaml|yml|txt|ipynb)", match.group(1)):
            if token not in result:
                result.append(token)
        if len(result) >= 30:
            break
    return result


def summarize_document(path: str, content: str) -> str:
    candidates: list[str] = []
    in_code = False
    for line in content.splitlines():
        text = line.strip()
        if text.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not text or text.startswith("#") or text.startswith("|") or text.startswith("---"):
            continue
        if len(text) >= 12:
            cleaned = text
            for _ in range(3):
                cleaned = re.sub(r"!?\[[^\]]*\]\([^)]*\)", "", cleaned)
            cleaned = re.sub(r"[\[\]]", "", cleaned)
            candidates.append(re.sub(r"\s+", " ", cleaned))
        if len(candidates) >= 5:
            break
    result = " ".join(candidates) or "No safely interpretable prose summary; human review required."
    result = redact_text(result)
    return result if len(result) <= 520 else result[:517].rstrip() + "..."


def authority_status(path: str, content: str) -> str:
    lower_path = path.lower()
    lower = content[:16000].lower()
    if any(token in lower_path for token in ("superseded", "supersession", "archive/", "deprecated")):
        return "superseded" if "supersed" in lower_path or "supersession" in lower_path else "historical"
    if re.search(r"(?im)^\s*(?:status\s*:\s*)?(?:superseded|historical|deprecated)\b", content[:16000]):
        return "superseded" if re.search(r"(?i)supersed", content[:16000]) else "historical"
    if re.search(r"(?i)\b(?:non-canonical|retained as branch history|historical record|superseded by)\b", lower):
        return "historical"
    if re.search(r"(?i)\bdraft\b", lower):
        return "draft"
    if any(token in lower for token in ("canonical", "current state", "active architecture constraint", "master blueprint", "evidence ledger")):
        return "canonical"
    if any(token in lower_path for token in ("readme", "status", "current_state", "blueprint")) or any(token in lower[:4000] for token in ("status:", "authority:")):
        return "current"
    return "unclear"


def claim_support(path: str, content: str) -> tuple[str, list[str]]:
    lower = (path + "\n" + content[:100000]).lower()
    refs = sorted(set(re.findall(r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:json|md|py|yaml|yml|txt|ipynb)", lower)))
    refs = [item for item in refs if any(token in item for token in ("receipt", "result", "artifact", "test", "ledger", "evidence"))][:30]
    if refs and any(token in lower for token in ("receipt", "artifact", "executed", "result", "audit")):
        return "references_executable_evidence_not_validated", refs
    if "test" in lower or "pytest" in lower:
        return "test_reference_only", refs
    return "documentary_only", refs


class GitRepository:
    def __init__(self, path: Path):
        self.root = path.resolve()
        self.git = shutil.which("git")
        if not self.git:
            raise AtlasError("Git executable was not found on PATH")

    def run(self, args: list[str], check: bool = True, timeout: int = 180, input_data: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
        try:
            result = subprocess.run([self.git, *args], cwd=str(self.root), input=input_data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise AtlasError(f"Git command failed to start: {error}") from error
        if check and result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise AtlasError(f"git {' '.join(args)} failed ({result.returncode}): {detail}")
        return result

    def text(self, args: list[str], check: bool = True, timeout: int = 180) -> str:
        return self.run(args, check, timeout).stdout.decode("utf-8", errors="replace")

    def resolve_commit(self, ref: str) -> str | None:
        result = self.run(["rev-parse", "--verify", f"{ref}^{{commit}}"], False)
        return result.stdout.decode("ascii", errors="replace").strip() or None if result.returncode == 0 else None

    def object_exists(self, name: str) -> bool:
        return self.run(["cat-file", "-e", name], False).returncode == 0

    def merge_base(self, left: str, right: str) -> str | None:
        result = self.run(["merge-base", left, right], False)
        return result.stdout.decode("ascii", errors="replace").strip() or None if result.returncode == 0 else None

    def count_range(self, left: str, right: str) -> int | None:
        result = self.run(["rev-list", "--count", f"{left}..{right}"], False)
        try:
            return int(result.stdout.decode("ascii", errors="replace").strip()) if result.returncode == 0 else None
        except ValueError:
            return None

    def root_path(self) -> Path:
        return Path(self.text(["rev-parse", "--show-toplevel"]).strip()).resolve()

    def identity(self) -> dict[str, Any]:
        def resolved_dir(value: str) -> str:
            path = Path(value)
            return (self.root / path if not path.is_absolute() else path).resolve().as_posix()
        head = self.resolve_commit("HEAD")
        branch = self.run(["symbolic-ref", "--quiet", "--short", "HEAD"], False).stdout.decode("utf-8", errors="replace").strip() or None
        return {"root": self.root.as_posix(), "git_dir": resolved_dir(self.text(["rev-parse", "--git-dir"]).strip()), "git_common_dir": resolved_dir(self.text(["rev-parse", "--git-common-dir"]).strip()), "head_sha": head, "head_branch": branch, "bare": self.text(["rev-parse", "--is-bare-repository"]).strip() == "true"}

    def refs(self) -> list[dict[str, Any]]:
        fmt = "%(refname)%00%(objectname)%00%(objecttype)%00%(symref)%00%(authordate:iso-strict)%00%(committerdate:iso-strict)%00%(subject)%00%(upstream:short)%00%(upstream:track)"
        output = self.run(["for-each-ref", f"--format={fmt}", "refs"]).stdout
        result: list[dict[str, Any]] = []
        for line in output.splitlines():
            fields = line.rstrip(b"\r\n").split(b"\x00") + [b""] * 9
            ref = fields[0].decode("utf-8", errors="replace")
            obj = fields[1].decode("utf-8", errors="replace")
            obj_type = fields[2].decode("utf-8", errors="replace")
            symref = fields[3].decode("utf-8", errors="replace") or None
            kind = "local_head" if ref.startswith("refs/heads/") else "remote_tracking" if ref.startswith("refs/remotes/") else "tag" if ref.startswith("refs/tags/") else "other"
            commit = self.resolve_commit(ref)
            result.append({"name": ref, "short_name": ref.removeprefix("refs/heads/").removeprefix("refs/remotes/").removeprefix("refs/tags/"), "type": kind, "is_branch_ref": kind in {"local_head", "remote_tracking"} and not symref, "object_name": obj, "object_type": obj_type, "symbolic_target": symref, "commit_sha": commit, "commit_resolves": commit is not None, "author_date": fields[4].decode("utf-8", errors="replace") or None, "committer_date": fields[5].decode("utf-8", errors="replace") or None, "subject": fields[6].decode("utf-8", errors="replace") or None, "upstream": fields[7].decode("utf-8", errors="replace") or None, "upstream_track": fields[8].decode("utf-8", errors="replace") or None})
        return sorted(result, key=lambda item: item["name"])

    def remotes(self) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for line in self.text(["remote", "-v"]).splitlines():
            fields = line.split()
            if len(fields) < 3:
                continue
            entry = grouped.setdefault(fields[0], {"name": fields[0], "fetch_url": None, "push_url": None})
            entry["fetch_url" if fields[2] == "(fetch)" else "push_url"] = scrub_url(fields[1])
        return [grouped[key] for key in sorted(grouped)]

    def worktrees(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        current: dict[str, Any] = {}
        for line in self.text(["worktree", "list", "--porcelain"]).splitlines() + [""]:
            if not line:
                if current:
                    records.append(current)
                    current = {}
                continue
            key, _, value = line.partition(" ")
            if key == "worktree":
                if current:
                    records.append(current)
                current = {"path": value, "head": None, "branch": None, "detached": False, "locked": False, "prunable": False}
            elif key == "HEAD":
                current["head"] = value
            elif key == "branch":
                current["branch"] = value.removeprefix("refs/heads/")
            elif key in {"detached", "locked", "prunable"}:
                current[key] = True
        return records

    def status_for_path(self, path: str) -> dict[str, Any]:
        try:
            result = subprocess.run([self.git, "-C", path, "status", "--porcelain=v2", "--branch"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"readable": False, "error": str(error), "entries": [], "user_dirty": None, "generated_overlay_count": 0}
        if result.returncode != 0:
            return {"readable": False, "error": result.stderr.decode("utf-8", errors="replace").strip(), "entries": [], "user_dirty": None, "generated_overlay_count": 0}
        headers: dict[str, str] = {}
        entries: list[dict[str, Any]] = []
        lines = result.stdout.decode("utf-8", errors="replace").splitlines()
        for line in lines:
            if line.startswith("# "):
                key, _, value = line[2:].partition(" ")
                headers[key] = value
                continue
            if not line:
                continue
            kind = line[0]
            fields = line.split(" ", 8)
            path_value = line[2:] if kind == "?" else fields[8] if kind in {"1", "2"} and len(fields) > 8 else line
            entries.append({"kind": kind, "path": path_value, "generated_observatory_overlay": path_value.replace("\\", "/").lower().startswith("branch_observatory/")})
        user_entries = [item for item in entries if not item["generated_observatory_overlay"]]
        ab = headers.get("branch.ab", "").split()
        return {"readable": True, "error": None, "branch": None if headers.get("branch.head") == "(detached)" else headers.get("branch.head"), "detached": headers.get("branch.head") == "(detached)", "head": headers.get("branch.oid"), "upstream": headers.get("branch.upstream"), "ahead": int(ab[0].lstrip("+")) if ab else None, "behind": int(ab[1].lstrip("-")) if len(ab) > 1 else None, "entries": entries, "entry_count": len(entries), "user_dirty": bool(user_entries), "generated_overlay_count": len(entries) - len(user_entries), "status_digest": hashlib.sha256("\n".join(line for line in lines if not line.startswith("# ")).encode()).hexdigest()}

    def commit_metadata(self, sha: str) -> dict[str, Any]:
        fmt = "%H%x00%P%x00%an%x00%aI%x00%cn%x00%cI%x00%s"
        result = self.run(["show", "-s", f"--format={fmt}", sha], False)
        if result.returncode != 0:
            return {"sha": sha, "parents": [], "author_name": None, "author_date": None, "committer_name": None, "committer_date": None, "subject": None, "metadata_status": "unavailable"}
        fields = result.stdout.rstrip(b"\r\n").split(b"\x00") + [b""] * 7
        return {"sha": fields[0].decode("utf-8", errors="replace"), "parents": [item.decode("ascii", errors="replace") for item in fields[1].split() if item], "author_name": fields[2].decode("utf-8", errors="replace") or None, "author_date": fields[3].decode("utf-8", errors="replace") or None, "committer_name": fields[4].decode("utf-8", errors="replace") or None, "committer_date": fields[5].decode("utf-8", errors="replace") or None, "subject": fields[6].decode("utf-8", errors="replace") or None, "metadata_status": "available"}

    def commit_metadata_batch(self, shas: list[str]) -> dict[str, dict[str, Any]]:
        if not shas:
            return {}
        fmt = "%x1e%H%x00%P%x00%an%x00%aI%x00%cn%x00%cI%x00%s"
        result = self.run(["show", "-s", "--no-walk", f"--format={fmt}", *shas], False, 300)
        records: dict[str, dict[str, Any]] = {}
        for raw in result.stdout.split(bytes([30])):
            if not raw.strip():
                continue
            fields = raw.rstrip(b"\r\n").split(b"\x00") + [b""] * 7
            sha = fields[0].decode("ascii", errors="replace")
            records[sha] = {"sha": sha, "parents": [item.decode("ascii", errors="replace") for item in fields[1].split() if item], "author_name": fields[2].decode("utf-8", errors="replace") or None, "author_date": fields[3].decode("utf-8", errors="replace") or None, "committer_name": fields[4].decode("utf-8", errors="replace") or None, "committer_date": fields[5].decode("utf-8", errors="replace") or None, "subject": fields[6].decode("utf-8", errors="replace") or None, "metadata_status": "available"}
        for sha in shas:
            if sha not in records:
                records[sha] = self.commit_metadata(sha)
        return records

    def rev_list(self, base: str, tip: str) -> list[str]:
        result = self.run(["rev-list", "--topo-order", "--reverse", f"{base}..{tip}"], False)
        return result.stdout.decode("ascii", errors="replace").splitlines() if result.returncode == 0 else []

    def submodules(self, commit: str | None) -> list[dict[str, str]]:
        if not commit:
            return []
        result = self.run(["ls-tree", "-r", commit], False)
        records = []
        for line in result.stdout.decode("utf-8", errors="replace").splitlines():
            fields = line.split(None, 3)
            if len(fields) >= 4 and fields[1] == "commit":
                records.append({"mode": fields[0], "object_sha": fields[2], "path": fields[3]})
        return records


class BlobReader:
    def __init__(self, repository: GitRepository):
        self.repository = repository
        try:
            self.process = subprocess.Popen([repository.git, "cat-file", "--batch"], cwd=str(repository.root), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError:
            self.process = None

    def read(self, sha: str) -> bytes | None:
        if not self.process or not self.process.stdin or not self.process.stdout:
            return None
        try:
            self.process.stdin.write((sha + "\n").encode("ascii"))
            self.process.stdin.flush()
            header = self.process.stdout.readline().split()
            if len(header) < 3 or header[1] in {b"missing", b"ambiguous"}:
                return None
            size = int(header[2])
            data = self.process.stdout.read(size)
            terminator = self.process.stdout.read(1)
            return data if len(data) == size and terminator == b"\n" else None
        except (OSError, ValueError):
            return None

    def close(self) -> None:
        if not self.process:
            return
        try:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            self.process.kill()


def tree_entries(repository: GitRepository, commit: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result = repository.run(["ls-tree", "-r", "-l", "-z", "--full-tree", commit], False)
    blobs: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    if result.returncode != 0:
        return blobs, other
    for record in result.stdout.split(b"\x00"):
        if b"\t" not in record:
            continue
        header, raw_path = record.split(b"\t", 1)
        fields = header.split()
        if len(fields) < 4:
            continue
        try:
            size = int(fields[3])
        except ValueError:
            size = None
        item = {"path": raw_path.decode("utf-8", errors="replace"), "mode": fields[0].decode("ascii", errors="replace"), "object_type": fields[1].decode("ascii", errors="replace"), "blob_sha": fields[2].decode("ascii", errors="replace"), "size": size}
        (blobs if fields[1] == b"blob" else other).append(item)
    return sorted(blobs, key=lambda item: item["path"]), sorted(other, key=lambda item: item["path"])


def blob_size(repository: GitRepository, sha: str) -> int | None:
    result = repository.run(["cat-file", "-s", sha], False)
    try:
        return int(result.stdout.decode("ascii", errors="replace").strip()) if result.returncode == 0 else None
    except ValueError:
        return None


def stock_metrics(repository: GitRepository, commit: str, config: dict[str, Any], cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if commit in cache:
        return cache[commit]
    blobs, other = tree_entries(repository, commit)
    reader = BlobReader(repository)
    limit = int(config.get("max_text_blob_bytes", 524288))
    files: Counter[str] = Counter()
    lines: Counter[str] = Counter()
    byte_counts: Counter[str] = Counter()
    language_lines: Counter[str] = Counter()
    source_language_lines: Counter[str] = Counter()
    excluded: list[dict[str, Any]] = []
    large: list[dict[str, Any]] = []
    binary_files = 0
    binary_bytes = 0
    total_bytes = 0
    notebook_cells = notebook_code = notebook_markdown = notebook_raw = 0
    for entry in blobs:
        path = entry["path"]
        size = entry["size"] if entry["size"] is not None else blob_size(repository, entry["blob_sha"])
        size = int(size or 0)
        total_bytes += size
        classification = classify_path(path, config)
        data = None
        if size <= limit and classification["category"] not in {"excluded", "vendor"}:
            data = reader.read(entry["blob_sha"])
        elif size > limit:
            large.append({"path": path, "bytes": size, "category": classification["category"]})
        if data is not None and b"\x00" in data[:8192]:
            classification["binary"] = True
            classification["generated"] = True
            classification["category"] = "binary"
        category = classification["category"]
        files[category] += 1
        byte_counts[category] += size
        if category == "binary" or classification["binary"]:
            binary_files += 1
            binary_bytes += size
        if category == "excluded":
            if len(excluded) < 100:
                excluded.append({"path": path, "bytes": size, "reason": classification["exclusion_reason"]})
        if data is None or category in {"binary", "excluded", "vendor"}:
            continue
        if category == "notebook":
            lines[category] += line_count(data)
            try:
                notebook = json.loads(data.decode("utf-8", errors="replace"))
                cells = notebook.get("cells", []) if isinstance(notebook, dict) else []
                notebook_cells += len(cells)
                for cell in cells:
                    kind = cell.get("cell_type") if isinstance(cell, dict) else None
                    notebook_code += int(kind == "code")
                    notebook_markdown += int(kind == "markdown")
                    notebook_raw += int(kind not in {"code", "markdown"})
            except (ValueError, TypeError, AttributeError):
                pass
            continue
        count = line_count(data)
        lines[category] += count
        if classification["language"]:
            language_lines[classification["language"]] += count
            if category == "production_source":
                source_language_lines[classification["language"]] += count
    reader.close()
    result = {
        "commit_sha": commit, "counted_from": "committed_tree", "tracked_blob_files": len(blobs), "tracked_non_blob_entries": other, "tracked_bytes": total_bytes,
        "category_files": dict(sorted(files.items())), "category_lines": dict(sorted(lines.items())), "category_bytes": dict(sorted(byte_counts.items())),
        "language_lines": dict(sorted(language_lines.items())), "source_language_lines": dict(sorted(source_language_lines.items())), "source_lines": lines["production_source"], "test_lines": lines["tests"], "documentation_lines": lines["documentation"],
        "notebook_lines": lines["notebook"], "generated_code_lines": lines["generated_code"], "generated_artifact_lines": lines["generated_artifact"], "data_lines": lines["data"], "build_config_lines": lines["build_config"], "unknown_lines": lines["unknown"],
        "binary_files": binary_files, "binary_bytes": binary_bytes, "notebook_cells": notebook_cells, "notebook_code_cells": notebook_code, "notebook_markdown_cells": notebook_markdown, "notebook_raw_cells": notebook_raw,
        "excluded_paths": excluded, "large_or_uninspected": large, "large_source_files_uninspected": sum(1 for item in large if item.get("category") in {"production_source", "tests", "documentation", "notebook"}), "loc_comparison_limited": any(item.get("category") in {"production_source", "tests", "documentation", "notebook"} for item in large), "line_count_definition": "newline count plus one for a non-newline-terminated final line",
        "inspection_policy": {"max_text_blob_bytes": limit, "large_files": "metadata counted; line content not loaded", "binary_files": "bytes counted; lines not inferred"},
    }
    cache[commit] = result
    return result


def parse_numstat(output: bytes) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    current = None
    for raw in output.splitlines():
        line = raw.decode("utf-8", errors="replace")
        if re.fullmatch(r"[0-9a-fA-F]{40,64}", line.strip()):
            current = line.strip()
            result[current]
            continue
        if current is None or "\t" not in line:
            continue
        fields = line.split("\t", 2)
        if len(fields) != 3:
            continue
        if fields[0] == "-" or fields[1] == "-":
            result[current].append({"added": None, "removed": None, "path": fields[2], "binary": True})
        else:
            try:
                result[current].append({"added": int(fields[0]), "removed": int(fields[1]), "path": fields[2], "binary": False})
            except ValueError:
                pass
    return result


def diff_path(path: str) -> str:
    if "=>" in path:
        path = path.split("=>", 1)[1].strip()
    return path.strip().strip("{}")


def load_diffs(repository: GitRepository, commits: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    ordinary = [item["sha"] for item in commits if len(item.get("parents", [])) <= 1]
    result: dict[str, dict[str, Any]] = {}
    path_exclusions = ["--", ":(exclude).git.deleted-remnants-*/**", ":(exclude)branch_observatory/**", ":(exclude)**/*.pt", ":(exclude)**/*.pth", ":(exclude)**/*.safetensors", ":(exclude)**/*.bin", ":(exclude)**/*.zip"]
    if ordinary:
        output = repository.run(["diff-tree", "--stdin", "--root", "-r", "-M", "--numstat", *path_exclusions], False, 300, ("\n".join(ordinary) + "\n").encode()).stdout
        parsed = parse_numstat(output)
        for sha in ordinary:
            result[sha] = {"rows": parsed.get(sha, []), "merge": False, "method": "git_diff_tree_numstat"}
    for item in commits:
        if len(item.get("parents", [])) <= 1:
            continue
        parent = item["parents"][0]
        output = repository.run(["diff", "--numstat", "--find-renames", parent, item["sha"], *path_exclusions], False, 300).stdout
        rows = []
        for raw in output.splitlines():
            fields = raw.decode("utf-8", errors="replace").split("\t", 2)
            if len(fields) != 3:
                continue
            if fields[0] == "-" or fields[1] == "-":
                rows.append({"added": None, "removed": None, "path": fields[2], "binary": True})
            else:
                try:
                    rows.append({"added": int(fields[0]), "removed": int(fields[1]), "path": fields[2], "binary": False})
                except ValueError:
                    pass
        result[item["sha"]] = {"rows": rows, "merge": True, "method": "first_parent_diff"}
    for sha in ordinary:
        result[sha]["files_changed"] = len(result[sha]["rows"])
    for item in commits:
        if len(item.get("parents", [])) > 1:
            result[item["sha"]]["files_changed"] = len(result[item["sha"]]["rows"])
    return result


def empty_flow() -> dict[str, Any]:
    return {"source_lines_added": 0, "source_lines_removed": 0, "test_lines_added": 0, "test_lines_removed": 0, "docs_lines_added": 0, "docs_lines_removed": 0, "notebook_changes": 0, "generated_artifact_changes": 0, "files_changed": 0, "binary_files_changed": 0, "renames_detected": 0, "category_lines_added": Counter(), "category_lines_removed": Counter(), "language_lines_added": Counter(), "language_lines_removed": Counter()}


def add_row(flow: dict[str, Any], row: dict[str, Any], config: dict[str, Any]) -> None:
    path = diff_path(row["path"])
    classification = classify_path(path, config)
    category = classification["category"]
    if row.get("binary") or row.get("added") is None or row.get("removed") is None:
        flow["files_changed"] += 1
        flow["binary_files_changed"] += 1
        return
    added, removed = int(row["added"]), int(row["removed"])
    flow["files_changed"] += 1
    flow["renames_detected"] += int("=>" in row["path"])
    flow["category_lines_added"][category] += added
    flow["category_lines_removed"][category] += removed
    if classification["language"]:
        flow["language_lines_added"][classification["language"]] += added
        flow["language_lines_removed"][classification["language"]] += removed
    if category == "production_source":
        flow["source_lines_added"] += added
        flow["source_lines_removed"] += removed
    elif category == "tests":
        flow["test_lines_added"] += added
        flow["test_lines_removed"] += removed
    elif category == "documentation":
        flow["docs_lines_added"] += added
        flow["docs_lines_removed"] += removed
    if category == "notebook":
        flow["notebook_changes"] += 1
    if category in {"generated_artifact", "generated_code", "data", "binary"}:
        flow["generated_artifact_changes"] += 1


def combine_flows(flows: list[dict[str, Any]]) -> dict[str, Any]:
    result = empty_flow()
    scalar_keys = [key for key in result if isinstance(result[key], int)]
    for flow in flows:
        for key in scalar_keys:
            result[key] += flow[key]
        for key in ("category_lines_added", "category_lines_removed", "language_lines_added", "language_lines_removed"):
            result[key].update(flow[key])
    return result


def public_flow(flow: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in flow.items() if not isinstance(value, Counter)}
    result["source_lines_net"] = result["source_lines_added"] - result["source_lines_removed"]
    result["source_lines_churn"] = result["source_lines_added"] + result["source_lines_removed"]
    for key in ("category_lines_added", "category_lines_removed", "language_lines_added", "language_lines_removed"):
        result[key] = dict(sorted(flow[key].items()))
    return result


def day_range(first: dt.date, last: dt.date) -> list[dt.date]:
    result = []
    cursor = first
    while cursor <= last:
        result.append(cursor)
        cursor += dt.timedelta(days=1)
    return result


def activity_metrics(commits: list[dict[str, Any]], diffs: dict[str, dict[str, Any]], base_lines: int | None, capture: dt.datetime, config: dict[str, Any]) -> dict[str, Any]:
    dated = [(parse_datetime(item.get("committer_date")), item) for item in commits]
    dated = [(date, item) for date, item in dated if date is not None]
    if not dated:
        return {"history_status": "known_empty_range" if base_lines is not None else "unknown_missing_commit_dates", "daily": [], "branch_unique_commit_count": len(commits), "branch_unique_commit_shas": [item["sha"] for item in commits], "cumulative_flow": public_flow(empty_flow()), "recent": {}, "largest_changes": []}
    dated.sort(key=lambda pair: (pair[0], pair[1]["sha"]))
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for date, item in dated:
        by_day[date.date().isoformat()].append(item)
    first, last = dated[0][0].date(), dated[-1][0].date()
    start = base_lines
    cumulative = empty_flow()
    merge_total = empty_flow()
    daily = []
    changes = []
    for day in day_range(first, last):
        key = day.isoformat()
        day_flows = []
        merge_flows = []
        for commit in by_day.get(key, []):
            diff = diffs.get(commit["sha"], {"rows": [], "merge": False})
            flow = empty_flow()
            for row in diff.get("rows", []):
                add_row(flow, row, config)
                changes.append({"sha": commit["sha"], "path": row.get("path"), "added": row.get("added"), "removed": row.get("removed"), "binary": row.get("binary", False)})
            day_flows.append(flow)
            if diff.get("merge"):
                merge_flows.append(flow)
        flow = combine_flows(day_flows)
        merge_flow = combine_flows(merge_flows)
        net = flow["source_lines_added"] - flow["source_lines_removed"]
        end = None if start is None else start + net
        daily.append({"utc_day": key, "source_lines_added": flow["source_lines_added"], "source_lines_removed": flow["source_lines_removed"], "source_lines_net": net, "source_lines_churn": flow["source_lines_added"] + flow["source_lines_removed"], "source_size_start_of_day": start, "source_size_end_of_day": end, "source_growth_percent": None if start in {None, 0} else net / start * 100, "commit_count": len(by_day.get(key, [])), "ordinary_commit_count": sum(len(item.get("parents", [])) <= 1 for item in by_day.get(key, [])), "merge_commit_count": sum(len(item.get("parents", [])) > 1 for item in by_day.get(key, [])), "files_changed": flow["files_changed"], "binary_files_changed": flow["binary_files_changed"], "renames_detected": flow["renames_detected"], "docs_lines_added": flow["docs_lines_added"], "docs_lines_removed": flow["docs_lines_removed"], "test_lines_added": flow["test_lines_added"], "test_lines_removed": flow["test_lines_removed"], "notebook_changes": flow["notebook_changes"], "generated_artifact_changes": flow["generated_artifact_changes"], "merge_source_lines_added": merge_flow["source_lines_added"], "merge_source_lines_removed": merge_flow["source_lines_removed"], "category_lines_added": dict(sorted(flow["category_lines_added"].items())), "category_lines_removed": dict(sorted(flow["category_lines_removed"].items())), "language_lines_added": dict(sorted(flow["language_lines_added"].items())), "language_lines_removed": dict(sorted(flow["language_lines_removed"].items()))})
        cumulative = combine_flows([cumulative, flow])
        merge_total = combine_flows([merge_total, merge_flow])
        start = end
    for date, commit in dated:
        diff = diffs.get(commit["sha"], {"rows": []})
        flow = empty_flow()
        for row in diff.get("rows", []):
            add_row(flow, row, config)
        changes.append({"sha": commit["sha"], "utc_date": date.date().isoformat(), "subject": commit.get("subject"), "source_lines_added": flow["source_lines_added"], "source_lines_removed": flow["source_lines_removed"], "source_lines_churn": flow["source_lines_added"] + flow["source_lines_removed"], "files_changed": flow["files_changed"], "merge": len(commit.get("parents", [])) > 1})
    largest = [item for item in changes if "source_lines_churn" in item]
    largest.sort(key=lambda item: (item["source_lines_churn"], item["files_changed"], item["sha"]), reverse=True)
    recent = {}
    for days in (1, 7, 30, 90):
        start_day = capture.date() - dt.timedelta(days=days - 1)
        selected = [row for row in daily if start_day.isoformat() <= row["utc_day"] <= capture.date().isoformat()]
        keys = ("source_lines_added", "source_lines_removed", "source_lines_net", "source_lines_churn", "files_changed", "commit_count", "docs_lines_added", "docs_lines_removed", "test_lines_added", "test_lines_removed", "notebook_changes", "generated_artifact_changes")
        recent[f"{days}d"] = {"status": "known", "start_utc_day": start_day.isoformat(), "end_utc_day": capture.date().isoformat(), **{key: sum(row[key] for row in selected) for key in keys}}
    return {"history_status": "retrospective_git_history", "daily": daily, "first_branch_unique_commit_date": first.isoformat(), "last_branch_unique_commit_date": last.isoformat(), "days_since_last_branch_unique_commit": (capture.date() - last).days, "branch_unique_commit_count": len(commits), "branch_unique_commit_shas": [item["sha"] for item in commits], "cumulative_flow": public_flow(cumulative), "merge_first_parent_flow": public_flow(merge_total), "recent": recent, "largest_changes": largest[:20], "change_paths": [item for item in changes if "path" in item][:200], "daily_measurement_note": "UTC commit-metadata replay; merge deltas are first-parent and can overlap side-branch commit flow."}


def build_documents(repository: GitRepository, ref: str, commit: str, config: dict[str, Any], reader: BlobReader) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    blobs, _ = tree_entries(repository, commit)
    limit = int(config.get("max_text_blob_bytes", 524288))
    documents = []
    by_path = {}
    for entry in blobs:
        path = entry["path"]
        if not is_document_candidate(path):
            continue
        size = entry["size"] if entry["size"] is not None else blob_size(repository, entry["blob_sha"])
        size = int(size or 0)
        content = ""
        readable = size <= limit
        if readable:
            data = reader.read(entry["blob_sha"])
            readable = data is not None
            if data is not None:
                content = data.decode("utf-8", errors="replace")
        date, date_source = extract_date(content) if readable else (None, None)
        support, refs = claim_support(path, content) if readable else ("human_review_required", [])
        conflicts = [redact_text(re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line.strip()))[:240] for line in content.splitlines() if re.search(r"(?i)stale|conflict|contradict|supersed|false green|do not cite|withdrawn|invalid", line)][:20] if readable else []
        title = redact_text(extract_title(path, content)) if readable else Path(path).name
        document = {"ref": ref, "commit_sha": commit, "path": path, "blob_sha": entry["blob_sha"], "bytes": size, "title": title, "document_date": date, "date_source": date_source, "type": document_type(path), "authority_status": authority_status(path, content) if readable else "unclear", "supersedes_or_depends_on": extract_dependencies(content) if readable else [], "summary": summarize_document(path, content) if readable else "Human review required because the document is missing, binary, or above the safe text inspection limit.", "sections": extract_sections(content) if readable else [], "conflicts_or_corrections": conflicts, "claim_support": support, "receipt_or_test_references": refs, "readable": readable, "content_inspection": "text_excerpt" if readable else "not_inspected"}
        documents.append(document)
        by_path[path] = document
    return documents, by_path


def default_config() -> dict[str, Any]:
    return {"schema_version": 1, "base_ref": "refs/remotes/origin/main", "base_selection": {"rationale": "A clean remote-tracking tip is portable and does not depend on the dirty active checkout.", "ambiguity": "The repository has no single branch authoritative for every research program."}, "max_text_blob_bytes": 524288, "excluded_paths": [".git/**", ".git.deleted-remnants-*/**", "**/__pycache__/**", "**/.pytest_cache/**", "**/.mypy_cache/**", "**/node_modules/**", "**/.venv/**", "**/venv/**", "branch_observatory/**"], "profiles": [], "families": [], "goals": [], "glossary": []}


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return default_config()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AtlasError(f"Could not read config {path}: {error}") from error
    result = default_config()
    result.update(value)
    return result


def profile_for(ref: str, config: dict[str, Any]) -> dict[str, Any]:
    for profile in config.get("profiles", []):
        for pattern in profile.get("match", []):
            try:
                if re.search(pattern, ref):
                    return profile
            except re.error as error:
                raise AtlasError(f"Invalid profile regex {pattern!r}: {error}") from error
    return {"id": "unclassified", "family": "unclassified", "kind": "unknown", "mission": "No reviewed mission profile matched this ref.", "thesis": "UNKNOWN", "role": "Unknown", "contribution": "Unknown until human review.", "strongest_evidence": "UNKNOWN", "unresolved_question": "UNKNOWN", "falsifier": "UNKNOWN", "read_first": ["README.md"], "status_overrides": {}, "scores": {}, "authority_notes": []}


def resolve_upstream(repository: GitRepository, upstream: str | None) -> tuple[str | None, str]:
    if not upstream:
        return None, "none"
    for candidate in (f"refs/remotes/{upstream}", f"refs/heads/{upstream}", upstream):
        resolved = repository.resolve_commit(candidate)
        if resolved:
            return resolved, "available"
    return None, "gone_or_unavailable"


def citation_for(documents: dict[str, dict[str, Any]], evidence: dict[str, Any], ref: str) -> dict[str, Any]:
    document = documents.get(evidence.get("path"))
    return {"ref": evidence.get("source_ref", ref), "path": evidence.get("path"), "section": evidence.get("section"), "basis": evidence.get("basis"), "blob_sha": document.get("blob_sha") if document else None, "citation_status": "resolved" if document else "missing_or_different_ref"}


def status_records(profile: dict[str, Any], ref: str, documents: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for name in STATUS_NAMES:
        override = profile.get("status_overrides", {}).get(name, {})
        state = override.get("state", "UNKNOWN")
        evidence = [citation_for(documents, item, ref) for item in override.get("evidence", [])]
        if not evidence and state != "UNKNOWN":
            state = "NOT_VERIFIED"
        confidence = override.get("confidence", "unknown" if state in {"UNKNOWN", "NOT_VERIFIED"} else "low")
        if any(item["citation_status"] != "resolved" for item in evidence):
            confidence = "low"
        result.append({"status": name, "state": state, "confidence": confidence, "evidence": evidence, "basis": override.get("basis", "No reviewed evidence; UNKNOWN by policy.")})
    return result


def base_for_branch(repository: GitRepository, ref_record: dict[str, Any], project_base: str | None) -> dict[str, Any]:
    tip = ref_record.get("commit_sha")
    upstream, upstream_status = resolve_upstream(repository, ref_record.get("upstream"))
    selected, method = project_base, "configured_project_base"
    if tip and upstream:
        merge = repository.merge_base(upstream, tip)
        if merge:
            selected, method = merge, "configured_upstream_merge_base"
    if not tip or not selected:
        return {"base_sha": None, "method": "UNCOMPARABLE", "upstream_status": upstream_status, "upstream_sha": upstream, "upstream_ahead": None, "upstream_behind": None, "ahead": None, "behind": None, "divergent": None}
    ahead, behind = repository.count_range(selected, tip), repository.count_range(tip, selected)
    upstream_ahead = repository.count_range(tip, upstream) if upstream else None
    upstream_behind = repository.count_range(upstream, tip) if upstream else None
    return {"base_sha": selected, "method": method, "upstream_status": upstream_status, "upstream_sha": upstream, "upstream_ahead": upstream_ahead, "upstream_behind": upstream_behind, "ahead": ahead, "behind": behind, "divergent": bool(ahead and behind)}


def branch_worktrees(worktrees: list[dict[str, Any]], ref_record: dict[str, Any], tip: str | None) -> list[dict[str, Any]]:
    if ref_record["type"] != "local_head":
        return []
    branch = ref_record["short_name"]
    return [item for item in worktrees if item.get("branch") == branch]


def activity_for_branch(repository: GitRepository, base: str | None, tip: str, base_stock: dict[str, Any] | None, capture: dt.datetime, config: dict[str, Any], diff_cache: dict[str, dict[str, Any]], metadata_cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not base:
        return {"history_status": "uncomparable_missing_merge_base", "daily": [], "recent": {}, "cumulative_flow": public_flow(empty_flow()), "branch_unique_commit_count": 0, "branch_unique_commit_shas": [], "largest_changes": []}
    shas = repository.rev_list(base, tip)
    missing = [sha for sha in shas if sha not in metadata_cache]
    if missing:
        metadata_cache.update(repository.commit_metadata_batch(missing))
    commits = [metadata_cache[sha] for sha in shas]
    new = [sha for sha in shas if sha not in diff_cache]
    if new:
        diff_cache.update(load_diffs(repository, [metadata_cache[sha] for sha in new]))
    result = activity_metrics(commits, diff_cache, base_stock.get("source_lines") if base_stock else None, capture, config)
    if base_stock:
        result["base_sha"] = base
    return result


def project_activity(repository: GitRepository, branches: list[dict[str, Any]], capture: dt.datetime, config: dict[str, Any], diff_cache: dict[str, dict[str, Any]], metadata_cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    membership: Counter[str] = Counter()
    for branch in branches:
        membership.update(branch.get("activity", {}).get("branch_unique_commit_shas", []) or [])
    shas = sorted(membership)
    missing = [sha for sha in shas if sha not in metadata_cache]
    if missing:
        metadata_cache.update(repository.commit_metadata_batch(missing))
    commits = [metadata_cache[sha] for sha in shas]
    new = [sha for sha in shas if sha not in diff_cache]
    if new:
        diff_cache.update(load_diffs(repository, [metadata_cache[sha] for sha in new]))
    result = activity_metrics(commits, diff_cache, None, capture, config)
    result["deduplication"] = {"unique_commit_count": len(shas), "branch_membership_count": sum(membership.values()), "duplicate_memberships_removed": sum(membership.values()) - len(shas), "commit_shas": shas, "merge_delta_method": "first_parent_diff"}
    result["source_size_growth_percent"] = None
    result["source_size_growth_note"] = "No single denominator exists for divergent branch tips; use branch-specific rows."
    return result


def build_branch(repository: GitRepository, ref_record: dict[str, Any], project_base: str | None, worktrees: list[dict[str, Any]], stock_cache: dict[str, dict[str, Any]], diff_cache: dict[str, dict[str, Any]], metadata_cache: dict[str, dict[str, Any]], config: dict[str, Any], capture: dt.datetime, reader: BlobReader) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tip = ref_record.get("commit_sha")
    profile = profile_for(ref_record["name"], config)
    base = base_for_branch(repository, ref_record, project_base)
    tip_stock = stock_metrics(repository, tip, config, stock_cache) if tip else None
    base_stock = stock_metrics(repository, base["base_sha"], config, stock_cache) if base.get("base_sha") else None
    documents, by_path = build_documents(repository, ref_record["name"], tip, config, reader) if tip else ([], {})
    if tip and tip not in metadata_cache:
        metadata_cache[tip] = repository.commit_metadata(tip)
    activity = activity_for_branch(repository, base.get("base_sha"), tip, base_stock, capture, config, diff_cache, metadata_cache) if tip else {"history_status": "unavailable", "daily": [], "recent": {}, "cumulative_flow": public_flow(empty_flow()), "largest_changes": []}
    if base_stock and tip_stock:
        change = tip_stock["source_lines"] - base_stock["source_lines"]
        activity["stock_delta"] = {"base_source_lines": base_stock["source_lines"], "tip_source_lines": tip_stock["source_lines"], "absolute_source_line_change": change, "source_size_percent_change": None if base_stock["source_lines"] == 0 else change / base_stock["source_lines"] * 100}
    linked = branch_worktrees(worktrees, ref_record, tip)
    days_since = activity.get("days_since_last_branch_unique_commit")
    branch = {"ref": ref_record["name"], "short_name": ref_record["short_name"], "ref_type": ref_record["type"], "symbolic_target": ref_record["symbolic_target"], "commit_sha": tip, "commit": metadata_cache.get(tip, {"metadata_status": "unavailable"}) if tip else {"metadata_status": "unavailable"}, "upstream": ref_record.get("upstream"), "upstream_status": base.get("upstream_status"), "worktree_paths": [item["path"] for item in linked], "dirty_overlay": [item["status"] for item in linked], "stale_looking": bool(days_since is not None and days_since > 30), "stale_threshold_days": 30, "incomparable": base.get("method") == "UNCOMPARABLE", "divergent_from_base": base.get("divergent"), "base": base, "profile": profile, "mission_and_soul": {"kind": profile.get("kind"), "mission": profile.get("mission"), "central_thesis": profile.get("thesis"), "role_in_program": profile.get("role"), "unique_contribution": profile.get("contribution"), "strongest_evidence_backed_result": profile.get("strongest_evidence"), "most_important_unresolved_question": profile.get("unresolved_question"), "clearest_falsifier_or_failure_condition": profile.get("falsifier"), "read_first": profile.get("read_first", []), "direct_facts_separated_from_interpretation": True, "authority_notes": profile.get("authority_notes", [])}, "status": {"labels": status_records(profile, ref_record["name"], by_path)}, "specifications_and_documents": documents, "loc": tip_stock, "activity": activity, "identity_confidence": "high" if tip else "low", "capture_timestamp": iso_utc(capture)}
    return branch, documents


def parse_worktree_status(repository: GitRepository, worktrees: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**item, "status": repository.status_for_path(item["path"])} for item in worktrees]


def validate_snapshot(repository: GitRepository, snapshot: dict[str, Any]) -> dict[str, Any]:
    required = ["schema_version", "snapshot_id", "capture", "repository", "refs", "worktrees", "branches", "documents", "warnings"]
    missing = [key for key in required if key not in snapshot]
    checks = [{"check": "required_fields", "status": "PASS" if not missing else "FAIL", "details": missing}]
    issues = [f"missing snapshot field: {key}" for key in missing]
    unresolved = [item.get("name") for item in snapshot.get("refs", []) if item.get("commit_sha") and not repository.object_exists(item["commit_sha"])]
    checks.append({"check": "reported_shas_resolve", "status": "PASS" if not unresolved else "FAIL", "details": unresolved})
    issues.extend(f"unresolved ref object: {item}" for item in unresolved)
    percentage_issues = []
    for branch in snapshot.get("branches", []):
        for row in branch.get("activity", {}).get("daily", []):
            start = row.get("source_size_start_of_day")
            expected = None if start in {None, 0} else row.get("source_lines_net", 0) / start * 100
            actual = row.get("source_growth_percent")
            if (expected is None and actual is not None) or (expected is not None and (actual is None or abs(actual - expected) > 1e-8)):
                percentage_issues.append({"ref": branch.get("ref"), "day": row.get("utc_day"), "actual": actual, "expected": expected})
    checks.append({"check": "daily_growth_percent_recomputes", "status": "PASS" if not percentage_issues else "FAIL", "details": percentage_issues[:20]})
    if percentage_issues:
        issues.append("daily growth percentage mismatch")
    unique = snapshot.get("project_activity", {}).get("deduplication", {}).get("commit_shas", [])
    checks.append({"check": "project_commits_deduplicated", "status": "PASS" if len(unique) == len(set(unique)) else "FAIL", "details": {"count": len(unique), "unique_count": len(set(unique))}})
    if len(unique) != len(set(unique)):
        issues.append("project commit deduplication failed")
    leaks = [branch.get("ref") for branch in snapshot.get("branches", []) if (branch.get("loc") or {}).get("counted_from") != "committed_tree"]
    checks.append({"check": "dirty_overlay_separate_from_tip_loc", "status": "PASS" if not leaks else "FAIL", "details": leaks})
    if leaks:
        issues.append("branch LOC is not committed-tree data")
    category_issues = []
    for branch in snapshot.get("branches", []):
        loc = branch.get("loc") or {}
        if loc.get("category_files") and sum(loc["category_files"].values()) != loc.get("tracked_blob_files"):
            category_issues.append(branch.get("ref"))
    checks.append({"check": "file_categories_sum_to_tracked_files", "status": "PASS" if not category_issues else "FAIL", "details": category_issues})
    if category_issues:
        issues.append("file category totals do not reconcile")
    checks.append({"check": "inaccessible_worktrees_reported", "status": "PASS", "details": [item.get("path") for item in snapshot.get("worktrees", []) if not item.get("status", {}).get("readable")]})
    document_keys = {(item.get("ref"), item.get("path")) for item in snapshot.get("documents", [])}
    unresolved_citations = []
    for branch in snapshot.get("branches", []):
        for status in branch.get("status", {}).get("labels", []):
            for evidence in status.get("evidence", []):
                if (evidence.get("ref"), evidence.get("path")) not in document_keys:
                    unresolved_citations.append({"ref": evidence.get("ref"), "path": evidence.get("path"), "status": status.get("status")})
    for goal in snapshot.get("goal_recommendations", []):
        for evidence in goal.get("evidence", []):
            if (evidence.get("ref"), evidence.get("path")) not in document_keys:
                unresolved_citations.append({"ref": evidence.get("ref"), "path": evidence.get("path"), "goal": goal.get("id")})
    checks.append({"check": "report_citations_resolve", "status": "PASS" if not unresolved_citations else "FAIL", "details": unresolved_citations[:50]})
    if unresolved_citations:
        issues.append("one or more report citations do not resolve to captured documents")
    return {"status": "FAIL" if issues else "PASS", "checks": checks, "issues": issues}
