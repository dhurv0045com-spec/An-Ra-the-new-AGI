"""Universal Experiment Registry and Historical AN-RA Research Atlas (V6-V8)."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from anra_v5.elendril.codec import digest

from .contracts import (
    CausalContrastSpec,
    ExperimentRegistration,
    FalsifierSpec,
    HypothesisSpec,
)


def _deserialize_registration(raw: Mapping[str, Any]) -> ExperimentRegistration:
    hypotheses = tuple(
        HypothesisSpec(
            hypothesis_id=str(h["hypothesis_id"]),
            statement=str(h["statement"]),
            mechanism_target=str(h["mechanism_target"]),
            falsifiers=tuple(
                FalsifierSpec(
                    falsifier_id=str(f["falsifier_id"]),
                    description=str(f["description"]),
                    metric_name=str(f["metric_name"]),
                    comparator=str(f["comparator"]),
                    threshold=float(f["threshold"]),
                )
                for f in h.get("falsifiers", ())
            ),
            expected_signatures=tuple(str(s) for s in h.get("expected_signatures", ())),
        )
        for h in raw.get("hypotheses", ())
    )
    contrasts = tuple(
        CausalContrastSpec(
            contrast_id=str(c["contrast_id"]),
            factor_name=str(c["factor_name"]),
            control_arm=str(c["control_arm"]),
            treatment_arm=str(c["treatment_arm"]),
            held_constant=tuple(str(k) for k in c.get("held_constant", ())),
            expected_mechanism_signature=str(c["expected_mechanism_signature"]),
        )
        for c in raw.get("contrasts", ())
    )
    return ExperimentRegistration(
        experiment_id=str(raw["experiment_id"]),
        program=str(raw["program"]),
        architecture_family=str(raw["architecture_family"]),
        parameter_count=int(raw["parameter_count"]),
        dataset_identity=str(raw["dataset_identity"]),
        source_commit=str(raw["source_commit"]),
        hypotheses=hypotheses,
        contrasts=contrasts,
        budget=dict(raw.get("budget", {})),
        dependencies=tuple(str(d) for d in raw.get("dependencies", ())),
        status=str(raw.get("status", "REGISTERED")),
        schema=str(raw.get("schema", "anra-planthir-registration-v1")),
    )


class PlanthirRegistry:
    """Universal content-addressed registry for hypotheses, contrasts, and experiment execution."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}

    def register(
        self,
        registration: ExperimentRegistration,
        *,
        allow_idempotent: bool = True,
    ) -> str:
        reg_sha = registration.sha256()
        existing = self._entries.get(registration.experiment_id)
        if existing is not None:
            if allow_idempotent and existing["registration_sha256"] == reg_sha:
                return reg_sha
            raise ValueError(
                f"conflicting registration for experiment_id {registration.experiment_id!r}"
            )
        self._entries[registration.experiment_id] = {
            "registration_sha256": reg_sha,
            "registration": registration.canonical_dict(),
            "status": registration.status,
            "result_ref": None,
            "status_events": [{"status": registration.status, "result_ref": None, "sequence": 0}],
        }
        return reg_sha

    def update_status(
        self,
        experiment_id: str,
        status: str,
        *,
        result_ref: str | None = None,
    ) -> None:
        if experiment_id not in self._entries:
            raise KeyError(f"unknown experiment_id: {experiment_id!r}")
        entry = self._entries[experiment_id]
        # Keep frozen `registration` immutable so registration_sha256 never drifts across status events.
        entry["status"] = status
        if result_ref is not None:
            entry["result_ref"] = result_ref
        events = entry.setdefault("status_events", [])
        events.append(
            {
                "status": str(status),
                "result_ref": result_ref if result_ref is not None else entry.get("result_ref"),
                "sequence": len(events),
            }
        )

    def get(self, experiment_id: str) -> dict[str, Any]:
        if experiment_id not in self._entries:
            raise KeyError(f"unknown experiment_id: {experiment_id!r}")
        entry = self._entries[experiment_id]
        return {
            "registration_sha256": entry["registration_sha256"],
            "registration": _deserialize_registration(entry["registration"]),
            "status": entry["status"],
            "result_ref": entry["result_ref"],
            "status_events": list(entry.get("status_events", ())),
        }

    def list_registrations(
        self,
        *,
        program: str | None = None,
        architecture_family: str | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for exp_id in sorted(self._entries):
            entry = self._entries[exp_id]
            reg_dict = entry["registration"]
            if program is not None and reg_dict.get("program") != program:
                continue
            if architecture_family is not None and reg_dict.get("architecture_family") != architecture_family:
                continue
            out.append(dict(entry))
        return out

    def export_dict(self) -> dict[str, Any]:
        payload = {
            "schema": "anra-planthir-registry-v1",
            "count": len(self._entries),
            "entries": {k: self._entries[k] for k in sorted(self._entries)},
        }
        payload["registry_sha256"] = digest(payload)
        return payload

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.export_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: Path | str) -> "PlanthirRegistry":
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        if doc.get("schema") != "anra-planthir-registry-v1":
            raise ValueError("unsupported PlanthirRegistry schema")
        recorded_reg_sha = doc.get("registry_sha256")
        if recorded_reg_sha is not None:
            unsigned = {k: v for k, v in doc.items() if k != "registry_sha256"}
            expected_reg_sha = digest(unsigned)
            if recorded_reg_sha != expected_reg_sha:
                raise ValueError(
                    f"PlanthirRegistry digest mismatch: stored={recorded_reg_sha} != expected={expected_reg_sha}"
                )
        reg = cls()
        for exp_id, entry in doc.get("entries", {}).items():
            parsed = _deserialize_registration(entry["registration"])
            computed_sha = parsed.sha256()
            stored_sha = entry.get("registration_sha256")
            if stored_sha is not None and stored_sha != computed_sha:
                raise ValueError(
                    f"entry {exp_id!r} registration_sha256 mismatch: stored={stored_sha} != computed={computed_sha}"
                )
            reg._entries[exp_id] = {
                "registration_sha256": computed_sha,
                "registration": parsed.canonical_dict(),
                "status": entry.get("status", parsed.status),
                "result_ref": entry.get("result_ref"),
                "status_events": list(entry.get("status_events", ())),
            }
        return reg


def _mean(values: Sequence[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _summarize_model_cohorts(models_dict: Mapping[str, Any]) -> dict[str, Any]:
    conf_raw_accs: list[float] = []
    conf_gram_accs: list[float] = []
    conf_q_recalls: list[float] = []
    hop_gram: dict[str, list[float]] = {f"hop-{h}": [] for h in (1, 2, 3, 4)}
    alias_gram: list[float] = []
    param_counts: list[int] = []
    by_method: dict[str, list[float]] = {}

    for model_id, mdata in models_dict.items():
        if isinstance(mdata.get("parameters"), int):
            param_counts.append(int(mdata["parameters"]))
        cohorts = mdata.get("cohorts", {})
        conf = cohorts.get("confirmation", {})
        raw_acc = conf.get("raw", {}).get("accuracy")
        gram_acc = conf.get("grammar", {}).get("accuracy")
        qr = conf.get("grammar", {}).get("query_recall")
        if raw_acc is not None:
            conf_raw_accs.append(float(raw_acc))
        if gram_acc is not None:
            conf_gram_accs.append(float(gram_acc))
        if qr is not None:
            conf_q_recalls.append(float(qr))
        method = str(mdata.get("job", {}).get("method", model_id.split("-")[1] if "-" in model_id else "unknown"))
        if gram_acc is not None:
            by_method.setdefault(method, []).append(float(gram_acc))

        for hkey in hop_gram:
            val = cohorts.get(hkey, {}).get("grammar", {}).get("accuracy")
            if val is not None:
                hop_gram[hkey].append(float(val))
        al = cohorts.get("robustness-alias", {}).get("grammar", {}).get("accuracy")
        if al is not None:
            alias_gram.append(float(al))

    return {
        "model_count": len(models_dict),
        "parameter_count": param_counts[0] if len(set(param_counts)) == 1 else (param_counts or None),
        "mean_confirmation_raw_accuracy": _mean(conf_raw_accs),
        "mean_confirmation_grammar_accuracy": _mean(conf_gram_accs),
        "mean_confirmation_query_recall": _mean(conf_q_recalls),
        "mean_hop_grammar_accuracy": {k: _mean(v) for k, v in hop_gram.items()},
        "mean_alias_grammar_accuracy": _mean(alias_gram),
        "mean_confirmation_grammar_by_method": {k: _mean(v) for k, v in sorted(by_method.items())},
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_historical_atlas(repo_root: Path | str) -> dict[str, Any]:
    """Scan curated Experiment Result artifacts across V6-V8 into a quantitative historical synthesis."""

    root = Path(repo_root)
    exp_root = root / "Experiment Result"
    generations: list[dict[str, Any]] = []
    total_archived_runs = 0

    # 1. V6
    v6_status = _read_json(exp_root / "V6" / "STATUS.json")
    if v6_status is not None:
        v6_runs = int(v6_status.get("training_records_complete", 0))
        total_archived_runs += v6_runs
        generations.append({
            "generation": "V6",
            "status": str(v6_status.get("campaign", "Elendril V6")),
            "runs_counted": v6_runs,
            "supplemental_runs": int(v6_status.get("supplemental_count", 0)),
            "primary_panel_counts": v6_status.get("primary_counts", {}),
            "key_finding": "Only 2/15 primary diagnostic panels passed; calibration, retention, alias OOD, and 3-hop panels failed.",
        })

    # 2. V7
    v7_status = _read_json(exp_root / "V7" / "STATUS.json")
    v7_metrics = _read_json(exp_root / "V7" / "c9981309a94847a4a97d8115c56d75c8" / "METRICS.json")
    if v7_metrics is not None:
        jobs = v7_metrics.get("jobs", [])
        accs = [
            j["confirmation_correct"] / j["episodes"]
            for j in jobs
            if j.get("episodes", 0) > 0 and "confirmation_correct" in j
        ]
        q_recalls = [
            j["needed_queried"] / j["query_needed"]
            for j in jobs
            if j.get("query_needed", 0) > 0
        ]
        total_archived_runs += len(jobs)
        generations.append({
            "generation": "V7",
            "profile": "S3 (~2.3M)" if v7_status else "S3",
            "runs_counted": len(jobs),
            "mean_confirmation_accuracy": _mean(accs),
            "max_confirmation_accuracy": max(accs) if accs else None,
            "mean_query_recall": _mean(q_recalls),
            "cause_status": v7_metrics.get("scientific_conclusion"),
            "key_finding": "High query recall co-occurred with incomplete end-to-end confirmation solving; cause unresolved.",
        })

    # 3. V7.1
    v71_metrics = _read_json(exp_root / "V7.1" / "20c80f66e76e4b589aed2053afbddffc" / "METRICS.json")
    if v71_metrics is not None:
        rows = v71_metrics.get("rows", [])
        unique_jobs = sorted({str(r.get("job_id")) for r in rows if r.get("job_id")})
        conf_rows = [r for r in rows if r.get("cohort") == "confirmation"]
        raw_accs = [r["raw"]["accuracy"] for r in conf_rows if "raw" in r]
        gram_accs = [r["grammar"]["accuracy"] for r in conf_rows if "grammar" in r]
        q_recalls = [
            r["grammar"]["query_recall"]
            for r in conf_rows
            if r.get("grammar", {}).get("query_recall") is not None
        ]
        hop_acc = {}
        for h in (1, 2, 3, 4):
            h_rows = [r["grammar"]["accuracy"] for r in rows if r.get("cohort") == f"hop-{h}" and "grammar" in r]
            hop_acc[f"hop-{h}"] = _mean(h_rows)
        alias_rows = [r["grammar"]["accuracy"] for r in rows if r.get("cohort") == "robustness-alias" and "grammar" in r]
        total_archived_runs += len(unique_jobs)
        generations.append({
            "generation": "V7.1",
            "profile": "S3",
            "runs_counted": len(unique_jobs),
            "mean_confirmation_raw_accuracy": _mean(raw_accs),
            "mean_confirmation_grammar_accuracy": _mean(gram_accs),
            "mean_confirmation_query_recall": _mean(q_recalls),
            "mean_hop_grammar_accuracy": hop_acc,
            "mean_alias_grammar_accuracy": _mean(alias_rows),
            "cause_status": v71_metrics.get("cause"),
            "key_finding": "Expanded S3 mechanistic suite across uniform, blocked, and blocked-intermediate arms; surface alias brittleness and cause remained unresolved.",
        })

    # 4. V7.15
    v715_metrics = _read_json(exp_root / "V7.15" / "3981e34a2ffe4b4b9528b4b4d588883b" / "METRICS.json")
    if v715_metrics is not None:
        summary = _summarize_model_cohorts(v715_metrics.get("models", {}))
        conf_paired = v715_metrics.get("confirmation_uniform_vs_blocked", {})
        total_archived_runs += summary["model_count"]
        generations.append({
            "generation": "V7.15",
            "profile": "S8 (~7.97M)",
            "runs_counted": summary["model_count"],
            "summary": summary,
            "paired_uniform_vs_blocked": conf_paired,
            "key_finding": "At S8, blocked order outperformed uniform across all 3 seeds (right_only 32/15/52 vs left_only 2/2/2), while alias robustness lagged badly.",
        })

    # 5. V7.16
    v716_metrics = _read_json(exp_root / "V7.16" / "51f3aeb7cb604c6d8c80f6ae4cd9f3ff" / "METRICS.json")
    if v716_metrics is not None:
        summary = _summarize_model_cohorts(v716_metrics.get("models", {}))
        conf_paired = v716_metrics.get("confirmation_uniform_vs_blocked", {})
        total_archived_runs += summary["model_count"]
        generations.append({
            "generation": "V7.16",
            "profile": str(v716_metrics.get("profile", "S20")),
            "runs_counted": summary["model_count"],
            "summary": summary,
            "paired_uniform_vs_blocked": conf_paired,
            "key_finding": "At S20 (~26M), seed 7 flipped toward uniform (left_only=29 vs right_only=11) whereas seeds 19/41 favored blocked (31/30 vs 4/1); initial aggregator suffered epoch-shadowing bug.",
        })

    # 6. V7.17
    v717_metrics = _read_json(exp_root / "V7.17" / "f443d3f4141b4163bbc130d46ebd7395" / "METRICS.json")
    if v717_metrics is not None:
        summary = _summarize_model_cohorts(v717_metrics.get("models", {}))
        total_archived_runs += summary["model_count"]
        generations.append({
            "generation": "V7.17",
            "profile": "S20 init-vs-order-vs-low-lr",
            "runs_counted": summary["model_count"],
            "summary": summary,
            "cause_status": v717_metrics.get("cause"),
            "interpretation": v717_metrics.get("interpretation"),
            "key_finding": "Init/order crossing showed order effects depend on init; low-LR improved raw standard episodes yet reduced alias correctness (55 -> 40/336); internal mechanism left unresolved.",
        })

    # 7. V8
    v8_template = _read_json(exp_root / "V8" / "TEMPLATE_METRICS.json")
    generations.append({
        "generation": "V8",
        "status": str(v8_template.get("status", "NOT_RUN")) if v8_template else "NOT_RUN",
        "runs_counted": 0,
        "designed_arms": ["B0 (byte)", "J (joint)", "F (factorized)", "P (pointer)", "F-cap (bottleneck)", "U (recurrent)"],
        "key_finding": "Pre-run protocol fixed evaluator consumed-tool leak via Planthir audit (enforce_tool_consumption=True); dual-T4 study awaiting Kaggle execution.",
    })

    instrument_defects_historical = [
        {
            "id": "HIST-DEFECT-V716-SHADOW",
            "program": "V7.16",
            "location": "tools/elendril_v716_report.py",
            "description": "Comparison `ep > final_epoch` compared int against tuple `(int, dict)`, silently caught by `except TypeError: pass` and keeping epoch 1 as final.",
            "planthir_control": "instrument.validate_metric_aggregator + run_fault_injection_suite fault #5",
        },
        {
            "id": "HIST-DEFECT-V8-UNCONSUMED-TOOL",
            "program": "V8",
            "location": "anra_v5/elendril/v8/evaluation.py",
            "description": "run_episode set queried/retrieved booleans without clearing query_avail/mem_avail, leaving consumed tools marked available across subsequent prompt renders.",
            "planthir_control": "enforce_tool_consumption=True + instrument.validate_environment_kernel + fault #1",
        },
        {
            "id": "HIST-DEFECT-PRE-V8-METRIC-ONLY",
            "program": "V6-V7.17",
            "location": "Evaluators & reporters prior to Planthir",
            "description": "Logged outcome accuracy and query recall without internal representation geometry, Stage 1-4 causal process funnels, or counterfactual linear probes.",
            "planthir_control": "observatory.ProcessFunnelAnalyzer + observatory.InternalTelemetryCollector + causal.*",
        },
    ]

    unresolved_program_questions = [
        "Why does near-100% query recall (V7-V7.17) still yield <50% confirmation accuracy on novel surface aliases and 3-hop/4-hop chains?",
        "Does factorized typed heads + pointer attention (V8 F/P/U) overcome the readout-vs-representation bottleneck at S3/S20?",
        "Do recurrent step dynamics (V8 U, T=1..3) improve multi-hop variable chaining without overfitting surface token statistics?",
    ]

    return {
        "schema": "anra-planthir-historical-atlas-v1",
        "repo_root": str(root),
        "generations_indexed": len(generations),
        "total_archived_runs_indexed": total_archived_runs,
        "generations": generations,
        "historical_instrument_defects": instrument_defects_historical,
        "unresolved_scientific_questions": unresolved_program_questions,
    }


__all__ = ["PlanthirRegistry", "build_historical_atlas"]
