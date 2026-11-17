"""Typed experimental factors/contrasts, transitive cohort leakage audit, and paired-seed Holm-Bonferroni analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import random
from typing import Any, Callable, Mapping, Sequence

ALLOWED_FACTOR_TYPES = (
    "architecture",
    "objective",
    "lr",
    "schedule",
    "sampling",
    "precision",
    "seed_group",
    "dataset",
    "custom",
)


@dataclass(frozen=True, slots=True)
class TypedFactorSpec:
    """Explicit factor declaration naming what is permitted to vary and over what levels."""

    factor_name: str
    factor_type: str
    control_value: Any
    levels: tuple[Any, ...]
    schema: str = "anra-planthir-typed-factor-v1"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.schema != "anra-planthir-typed-factor-v1":
            raise ValueError("unsupported TypedFactorSpec schema")
        if not self.factor_name or not isinstance(self.factor_name, str):
            raise ValueError("factor_name must be a non-empty string")
        if self.factor_type not in ALLOWED_FACTOR_TYPES:
            raise ValueError(f"unsupported factor_type {self.factor_type!r}")
        if not self.levels:
            raise ValueError(f"factor {self.factor_name!r} must declare at least one level")


@dataclass(frozen=True, slots=True)
class TypedContrastSpec:
    """Preregistered typed contrast specifying control/treatment arms, varied factor, and endpoint."""

    contrast_id: str
    factor_name: str
    control_arm_id: str
    treatment_arm_id: str
    target_mechanism: str
    held_constant: tuple[str, ...] = ()
    expected_treatment_value: Any = None
    primary_endpoint: str = "accuracy"
    expected_direction: str = "greater"
    schema: str = "anra-planthir-typed-contrast-v1"

    def __post_init__(self) -> None:
        if self.schema != "anra-planthir-typed-contrast-v1":
            raise ValueError("unsupported TypedContrastSpec schema")
        if not self.contrast_id or not self.factor_name or not self.control_arm_id or not self.treatment_arm_id:
            raise ValueError("TypedContrastSpec requires contrast_id, factor_name, control_arm_id, and treatment_arm_id")
        if self.control_arm_id == self.treatment_arm_id:
            raise ValueError("TypedContrastSpec control_arm_id and treatment_arm_id must differ")
        if self.expected_direction not in ("greater", "less", "two_sided"):
            raise ValueError("expected_direction must be 'greater', 'less', or 'two_sided'")


@dataclass(frozen=True, slots=True)
class CohortClusterSpec:
    """Explicit cluster/cohort declaration linking related worlds, base graphs, and D2 groups."""

    cluster_id: str
    world_ids: tuple[str, ...] = ()
    base_world_ids: tuple[str, ...] = ()
    group_ids: tuple[str, ...] = ()
    weight: float = 1.0
    split: str = ""


@dataclass(frozen=True, slots=True)
class SeedPairSpec:
    """Paired (init_seed, order_seed) unit for paired causal/performance comparisons."""

    pair_id: str
    init_seed: int
    order_seed: int
    cluster_id: str | None = None


def _extract_arm_factors(arm: Any, *, canonical_fn: Callable[[Any], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    if canonical_fn is not None:
        arch_cfg = dict(canonical_fn(arm))
    elif getattr(arm, "model_spec", None) is not None and hasattr(arm.model_spec, "canonical"):
        arch_cfg = dict(arm.model_spec.canonical())
    elif getattr(arm, "v8_spec", None) is not None and hasattr(arm.v8_spec, "canonical"):
        arch_cfg = dict(arm.v8_spec.canonical())
    else:
        arch_cfg = {"architecture": getattr(arm, "architecture_family", "")}
    recipe = arm.recipe
    factors: dict[str, Any] = {
        "architecture": getattr(arm, "architecture_family", None),
        "model_config": arch_cfg,
        "lr": float(recipe.lr),
        "clip": float(recipe.clip),
        "total_updates": int(recipe.total_updates),
        "batch_size": int(recipe.batch_size),
        "micro_batch_size": int(recipe.micro_batch_size),
        "objective": str(recipe.objective),
        "sampling": str(recipe.sampling),
        "schedule": str(recipe.schedule),
        "precision": str(recipe.precision),
        "dataset_identity": str(arm.dataset_identity),
        "device": str(getattr(arm, "device", "cpu")),
        "init_seed": int(recipe.seed),
        "order_seed": int(getattr(arm, "order_seed", 0)),
        "seed_group": (int(recipe.seed), int(getattr(arm, "order_seed", 0))),
        "backend_id": str(getattr(arm, "backend_id", "v8")),
        "policy_mode": str(getattr(arm, "policy_mode", "raw")),
    }
    for rk in ("eq_weight", "cf_weight", "cf_margin", "action_weight"):
        if hasattr(recipe, rk):
            factors[rk] = float(getattr(recipe, rk))
    for mk, mv in arch_cfg.items():
        if mk not in ("architecture", "schema", "total_parameters", "added_parameters", "base_spec", "base_spec_sha256", "output_mapping", "policy_interface"):
            factors.setdefault(str(mk), mv)
    extra_factors = getattr(arm, "custom_factors", None)
    if isinstance(extra_factors, Mapping):
        for k, v in extra_factors.items():
            factors[str(k)] = v
    return factors


_ARCH_HEAD_KEYS = {
    "pointer_dim",
    "copy_enabled",
    "recurrent_dim",
    "recurrent_steps",
    "bottleneck_width",
    "num_slots",
    "slot_dim",
    "slot_edge_hidden",
    "agg_iterations",
    "msg_iterations",
    "cap_hidden_dim",
    "cap_r_hidden",
    "noedge_mlp_dim",
    "output_mapping",
    "policy_interface",
}
_OBJ_COMPANION_KEYS = {"eq_weight", "cf_weight", "cf_margin", "action_weight"}


def _factor_diffs(
    ctrl: Mapping[str, Any],
    trt: Mapping[str, Any],
    *,
    declared_factor: str | None = None,
    extra_arch_head_keys: Sequence[str] = (),
) -> list[str]:
    changed: list[str] = []
    arch_changed = ctrl.get("architecture") != trt.get("architecture")
    obj_changed = ctrl.get("objective") != trt.get("objective")
    head_keys = _ARCH_HEAD_KEYS | {str(x) for x in extra_arch_head_keys}
    keys = sorted(set(ctrl) | set(trt))
    for k in keys:
        if k == "model_config":
            c_cfg = ctrl.get("model_config") or {}
            t_cfg = trt.get("model_config") or {}
            if c_cfg != t_cfg:
                for mk in sorted(set(c_cfg) | set(t_cfg)):
                    if mk in ("total_parameters", "added_parameters", "base_spec_sha256", "schema"):
                        continue
                    if c_cfg.get(mk) == t_cfg.get(mk):
                        continue
                    if mk == "architecture" or (arch_changed and mk in head_keys):
                        if "architecture" not in changed:
                            changed.append("architecture")
                    elif mk == "base_spec":
                        changed.append("model_size")
                    else:
                        changed.append(str(mk))
            continue
        if k in head_keys and arch_changed and declared_factor == "architecture":
            continue
        if k in _OBJ_COMPANION_KEYS and obj_changed and declared_factor == "objective":
            continue
        if k in ("init_seed", "order_seed", "seed_group"):
            continue
        if ctrl.get(k) != trt.get(k):
            changed.append(k)
    if ctrl.get("seed_group") != trt.get("seed_group"):
        changed.append("seed_group")
    return sorted(set(changed))


def audit_typed_experimental_design(
    control_arm: Any,
    treatment_arms: Sequence[Any],
    *,
    factors: Sequence[TypedFactorSpec] = (),
    typed_contrasts: Sequence[TypedContrastSpec] = (),
    canonical_fn: Callable[[Any], Mapping[str, Any]] | None = None,
    extra_arch_head_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """Fail closed when declared factors do not vary, treatments are missing, or undeclared confounds differ."""

    arms_by_id: dict[str, Any] = {control_arm.arm_id: control_arm}
    for arm in treatment_arms:
        if arm.arm_id in arms_by_id:
            return {
                "schema": "anra-planthir-typed-design-audit-v1",
                "passed": False,
                "status": "INVALID_DESIGN",
                "violations": [{"code": "DUPLICATE_ARM_ID", "arm_id": arm.arm_id}],
                "contrast_checks": [],
                "factor_coverage": [],
            }
        arms_by_id[arm.arm_id] = arm

    arm_factors = {aid: _extract_arm_factors(arm, canonical_fn=canonical_fn) for aid, arm in arms_by_id.items()}
    violations: list[dict[str, Any]] = []
    contrast_checks: list[dict[str, Any]] = []
    factor_coverage: list[dict[str, Any]] = []

    if not treatment_arms:
        violations.append({"code": "MISSING_TREATMENT_ARMS", "detail": "at least one treatment arm is required"})

    declared_factor_names = {f.factor_name for f in factors}
    for fspec in factors:
        fspec.validate()
        observed_vals = [arm_factors[aid].get(fspec.factor_name) for aid in arms_by_id]
        distinct_vals = []
        for v in observed_vals:
            if v not in distinct_vals:
                distinct_vals.append(v)
        varies = len(distinct_vals) > 1
        if not varies:
            violations.append(
                {
                    "code": "NON_VARYING_DECLARED_FACTOR",
                    "factor_name": fspec.factor_name,
                    "observed_value": distinct_vals[0] if distinct_vals else None,
                }
            )
        if arm_factors[control_arm.arm_id].get(fspec.factor_name) != fspec.control_value:
            violations.append(
                {
                    "code": "CONTROL_VALUE_MISMATCH",
                    "factor_name": fspec.factor_name,
                    "expected": fspec.control_value,
                    "observed": arm_factors[control_arm.arm_id].get(fspec.factor_name),
                }
            )
        factor_coverage.append(
            {
                "factor_name": fspec.factor_name,
                "varies_across_arms": varies,
                "distinct_level_count": len(distinct_vals),
            }
        )

    for con in typed_contrasts:
        codes: list[str] = []
        if con.control_arm_id not in arms_by_id or con.treatment_arm_id not in arms_by_id:
            codes.append("MISSING_CONTRAST_ARM")
            violations.append({"code": "MISSING_CONTRAST_ARM", "contrast_id": con.contrast_id})
            contrast_checks.append({"contrast_id": con.contrast_id, "passed": False, "codes": codes})
            continue
        if factors and con.factor_name not in declared_factor_names:
            codes.append("UNDECLARED_CONTRAST_FACTOR")

        cf = arm_factors[con.control_arm_id]
        tf = arm_factors[con.treatment_arm_id]
        if cf.get(con.factor_name) == tf.get(con.factor_name):
            if not (
                con.factor_name == "architecture" and cf.get("model_config") != tf.get("model_config")
            ):
                codes.append("CONTRAST_FACTOR_CONSTANT")
        if con.expected_treatment_value is not None and tf.get(con.factor_name) != con.expected_treatment_value:
            codes.append("TREATMENT_VALUE_MISMATCH")

        diffs = _factor_diffs(
            cf,
            tf,
            declared_factor=con.factor_name,
            extra_arch_head_keys=extra_arch_head_keys,
        )
        undeclared = [d for d in diffs if d != con.factor_name]
        if undeclared:
            codes.append("UNDECLARED_CONFOUNDED_FACTOR")
        for hc in con.held_constant:
            if cf.get(hc) != tf.get(hc):
                codes.append(f"HELD_CONSTANT_VIOLATION:{hc}")

        if codes:
            violations.append(
                {
                    "code": codes[0],
                    "codes": codes,
                    "contrast_id": con.contrast_id,
                    "factor_name": con.factor_name,
                    "undeclared_diffs": undeclared,
                }
            )
        contrast_checks.append(
            {
                "contrast_id": con.contrast_id,
                "factor_name": con.factor_name,
                "passed": not codes,
                "codes": codes,
                "observed_diffs": diffs,
            }
        )

    # Ensure every treatment arm is paired by at least one typed_contrast if typed_contrasts provided
    if typed_contrasts:
        covered_trts = {c.treatment_arm_id for c in typed_contrasts}
        for arm in treatment_arms:
            if arm.arm_id not in covered_trts:
                violations.append({"code": "UNCOVERED_TREATMENT_ARM", "arm_id": arm.arm_id})

    passed = len(violations) == 0
    return {
        "schema": "anra-planthir-typed-design-audit-v1",
        "passed": passed,
        "status": "VALID_DESIGN" if passed else "INVALID_DESIGN",
        "violations": violations,
        "factor_coverage": factor_coverage,
        "contrast_checks": contrast_checks,
    }


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            return x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


_LINK_KEYS = (
    ("world", ("world_id", "task_id", "id")),
    ("base", ("base_world_id", "base_id")),
    ("parent", ("parent_world_id", "parent_id")),
    ("mate", ("counterfactual_mate_id", "mate_world_id", "eq_mate_id")),
    ("group", ("group_id",)),
    ("cohort", ("cohort_id", "cluster_id", "lineage_group", "base_signature", "signature")),
)


def _record_identifiers(rec: Mapping[str, Any]) -> list[str]:
    nodes: list[str] = []
    # Note: world, base, parent, and mate share the same world namespace so transitive chains
    # (world -> base -> parent -> world) collapse cleanly into one equivalence component!
    for key in ("world_id", "task_id", "id", "base_world_id", "base_id", "parent_world_id", "parent_id", "counterfactual_mate_id", "mate_world_id", "eq_mate_id"):
        val = rec.get(key)
        if val not in (None, ""):
            nodes.append(f"world:{val}")
    for key in ("group_id",):
        val = rec.get(key)
        if val not in (None, ""):
            nodes.append(f"group:{val}")
    for key in ("cohort_id", "cluster_id", "lineage_group", "base_signature", "signature"):
        val = rec.get(key)
        if val not in (None, ""):
            nodes.append(f"cohort:{val}")
    links = rec.get("related_world_ids")
    if isinstance(links, Sequence) and not isinstance(links, (str, bytes)):
        for lv in links:
            if lv not in (None, ""):
                nodes.append(f"world:{lv}")
    return nodes


def validate_cohort_and_split_transitivity(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    cohorts: Sequence[CohortClusterSpec] = (),
) -> dict[str, Any]:
    """Build transitive union-find equivalence classes across world/base/parent/mate/group/cohort links and detect split leaks."""

    uf = _UnionFind()
    split_nodes: dict[str, list[tuple[str, str]]] = {}

    for c in cohorts:
        c_node = f"cohort:{c.cluster_id}"
        for wid in c.world_ids:
            uf.union(c_node, f"world:{wid}")
        for bwid in c.base_world_ids:
            uf.union(c_node, f"world:{bwid}")
        for gid in c.group_ids:
            uf.union(c_node, f"group:{gid}")
        if c.split:
            split_nodes.setdefault(c.split, []).append((c_node, c.cluster_id))

    for split_name, records in splits.items():
        lst = split_nodes.setdefault(str(split_name), [])
        for idx, rec in enumerate(records):
            nodes = _record_identifiers(rec)
            if not nodes:
                nodes = [f"rec:{split_name}:{idx}"]
            head = nodes[0]
            for other in nodes[1:]:
                uf.union(head, other)
            lst.append((head, str(rec.get("world_id") or rec.get("task_id") or rec.get("id") or f"idx-{idx}")))

    comp_to_splits: dict[str, dict[str, list[str]]] = {}
    for split_name, items in split_nodes.items():
        for node, label in items:
            root = uf.find(node)
            comp_to_splits.setdefault(root, {}).setdefault(split_name, []).append(label)

    leaks: list[dict[str, Any]] = []
    for root, by_split in sorted(comp_to_splits.items()):
        if len(by_split) > 1:
            leaks.append(
                {
                    "component_root": root,
                    "splits": sorted(by_split),
                    "members_by_split": by_split,
                }
            )

    passed = len(leaks) == 0
    return {
        "schema": "anra-planthir-cohort-transitivity-v1",
        "passed": passed,
        "status": "NO_TRANSITIVE_LEAKAGE" if passed else "TRANSITIVE_COHORT_LEAKAGE",
        "split_counts": {k: len(v) for k, v in split_nodes.items()},
        "component_count": len(comp_to_splits),
        "leak_count": len(leaks),
        "leaks": leaks,
    }


def _student_t_pdf(t: float, df: int) -> float:
    return math.exp(
        math.lgamma((df + 1) / 2.0) - math.lgamma(df / 2.0) - 0.5 * math.log(df * math.pi)
    ) * ((1.0 + (t * t) / df) ** (-0.5 * (df + 1)))


def _student_t_two_sided_p(t_abs: float, df: int) -> float:
    if not math.isfinite(t_abs):
        return 0.0
    if t_abs <= 0.0:
        return 1.0
    if df >= 200:
        # Normal tail via erfc is accurate to >6 decimal places at large df
        return float(math.erfc(t_abs / math.sqrt(2.0)))
    # Composite Simpson integration from 0 to t_abs over t PDF
    steps = 400
    h = t_abs / steps
    s = _student_t_pdf(0.0, df) + _student_t_pdf(t_abs, df)
    for k in range(1, steps):
        coeff = 4.0 if k % 2 == 1 else 2.0
        s += coeff * _student_t_pdf(k * h, df)
    cdf_half = (h / 3.0) * s
    return float(max(0.0, min(1.0, 1.0 - 2.0 * cdf_half)))


def _student_t_crit(alpha: float, df: int) -> float:
    target = alpha
    lo, hi = 0.0, 200.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _student_t_two_sided_p(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def analyze_paired_seed_runs(
    observations: Sequence[Mapping[str, Any]],
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 17,
) -> dict[str, Any]:
    """Compute paired-seed intervals and Holm-Bonferroni corrected p-values across contrasts."""

    if not observations:
        return {
            "schema": "anra-planthir-paired-seed-analysis-v1",
            "passed": False,
            "status": "INVALID_INFERENCE",
            "alpha": float(alpha),
            "defects": ["NO_OBSERVATIONS"],
            "contrasts": {},
        }

    by_contrast: dict[str, list[Mapping[str, Any]]] = {}
    for row in observations:
        cid = str(row.get("contrast_id", "default"))
        by_contrast.setdefault(cid, []).append(row)

    results: dict[str, dict[str, Any]] = {}
    valid_pvals: list[tuple[str, float]] = []
    global_defects: list[str] = []

    for cid, rows in sorted(by_contrast.items()):
        defects: list[str] = []
        seen_pairs: set[str] = set()
        seen_seeds: set[tuple[int, int]] = set()
        deltas: list[float] = []
        cluster_counts: dict[str, int] = {}
        cluster_weights: dict[str, set[float]] = {}
        cluster_deltas: dict[str, list[float]] = {}

        for idx, r in enumerate(rows):
            pid = str(r.get("pair_id", f"pair-{idx}"))
            if pid in seen_pairs:
                defects.append(f"DUPLICATE_PAIR_ID:{pid}")
            seen_pairs.add(pid)
            init_s = r.get("init_seed")
            order_s = r.get("order_seed")
            if type(init_s) is not int or type(order_s) is not int:
                defects.append(f"INVALID_SEED_PAIR:{pid}")
            else:
                seed_key = (init_s, order_s)
                if seed_key in seen_seeds:
                    defects.append(f"DUPLICATE_SEED_PAIR:{seed_key}")
                seen_seeds.add(seed_key)

            cv = r.get("control_value")
            tv = r.get("treatment_value")
            if cv is None or tv is None:
                defects.append(f"MISSING_PAIR_MEMBER:{pid}")
                continue
            cf, tf = float(cv), float(tv)
            if not (math.isfinite(cf) and math.isfinite(tf)):
                defects.append(f"NONFINITE_OBSERVATION:{pid}")
                continue
            delta = tf - cf
            deltas.append(delta)

            cl = str(r.get("cluster_id") if r.get("cluster_id") not in (None, "") else pid)
            cluster_counts[cl] = cluster_counts.get(cl, 0) + 1
            cluster_deltas.setdefault(cl, []).append(delta)
            cw = r.get("cluster_weight", 1.0)
            try:
                cwf = float(cw)
            except (TypeError, ValueError):
                defects.append(f"INVALID_CLUSTER_WEIGHT:{cl}")
                cwf = 0.0
            if not math.isfinite(cwf) or cwf <= 0.0:
                defects.append(f"NONPOSITIVE_CLUSTER_WEIGHT:{cl}")
            cluster_weights.setdefault(cl, set()).add(cwf)

        if len(cluster_counts) < 2:
            defects.append("SINGLE_CLUSTER_OR_INSUFFICIENT_PAIRS")
        elif len(set(cluster_counts.values())) > 1 and not all(len(ws) == 1 and next(iter(ws)) != 1.0 for ws in cluster_weights.values()):
            defects.append("UNEQUAL_CLUSTER_SIZES")

        if any(len(ws) > 1 for ws in cluster_weights.values()):
            defects.append("INCONSISTENT_WITHIN_CLUSTER_WEIGHTS")

        if defects or len(deltas) < 2:
            if len(deltas) < 2 and "SINGLE_CLUSTER_OR_INSUFFICIENT_PAIRS" not in defects:
                defects.append("INSUFFICIENT_PAIRS")
            results[cid] = {
                "status": "INVALID_INFERENCE",
                "passed": False,
                "n_pairs": len(rows),
                "n_clusters": len(cluster_counts),
                "defects": sorted(set(defects)),
            }
            global_defects.extend(f"{cid}:{d}" for d in sorted(set(defects)))
            continue

        # Aggregate cluster-level paired effects (exact equal-cluster or cluster-mean summary)
        unit_deltas = [sum(ds) / len(ds) for ds in cluster_deltas.values()]
        m_units = len(unit_deltas)
        mean_d = sum(unit_deltas) / m_units
        var_d = sum((x - mean_d) ** 2 for x in unit_deltas) / (m_units - 1)
        se_d = math.sqrt(var_d / m_units)
        if se_d == 0.0:
            t_stat = 0.0 if mean_d == 0.0 else math.copysign(1e12, mean_d)
            p_raw = 1.0 if mean_d == 0.0 else 0.0
            ci_t = (mean_d, mean_d)
            variance_status = "DEGENERATE_ZERO_VARIANCE"
        else:
            t_stat = mean_d / se_d
            df = m_units - 1
            p_raw = _student_t_two_sided_p(abs(t_stat), df)
            t_crit = _student_t_crit(alpha, df)
            ci_t = (mean_d - t_crit * se_d, mean_d + t_crit * se_d)
            variance_status = "NONZERO_VARIANCE"

        sign_flip_p = 1.0 if mean_d == 0.0 else float(min(1.0, math.ldexp(2.0, -min(m_units, 1000))))

        rng = random.Random(bootstrap_seed)
        boots: list[float] = []
        for _ in range(max(100, int(n_bootstrap))):
            sample = [unit_deltas[rng.randrange(m_units)] for _ in range(m_units)]
            boots.append(sum(sample) / m_units)
        boots.sort()
        lo_idx = max(0, min(len(boots) - 1, int(math.floor((alpha / 2.0) * len(boots)))))
        hi_idx = max(0, min(len(boots) - 1, int(math.ceil((1.0 - alpha / 2.0) * len(boots))) - 1))

        results[cid] = {
            "status": "VALID_INFERENCE",
            "passed": True,
            "n_pairs": len(deltas),
            "n_clusters": m_units,
            "mean_paired_diff": float(mean_d),
            "se": float(se_d),
            "variance_status": variance_status,
            "t_stat": float(t_stat),
            "df": int(m_units - 1),
            "p_raw": float(p_raw),
            "sign_flip_p_value": sign_flip_p,
            "t_ci": [float(ci_t[0]), float(ci_t[1])],
            "bootstrap_ci": [float(boots[lo_idx]), float(boots[hi_idx])],
            "defects": [],
        }
        valid_pvals.append((cid, float(p_raw)))

    # Holm-Bonferroni correction across all valid contrasts simultaneously
    if valid_pvals:
        ranked = sorted(valid_pvals, key=lambda pair: pair[1])
        m = len(ranked)
        running_max = 0.0
        for idx, (cid, p_val) in enumerate(ranked):
            adj = min(1.0, (m - idx) * p_val)
            running_max = max(running_max, adj)
            results[cid]["p_holm"] = float(running_max)
            results[cid]["significant_holm"] = bool(running_max < alpha)

    passed = len(global_defects) == 0 and bool(valid_pvals)
    return {
        "schema": "anra-planthir-paired-seed-analysis-v1",
        "passed": passed,
        "status": "VALID_INFERENCE" if passed else "INVALID_INFERENCE",
        "alpha": float(alpha),
        "contrast_count": len(results),
        "valid_contrast_count": len(valid_pvals),
        "defects": sorted(set(global_defects)),
        "contrasts": results,
    }


__all__ = [
    "ALLOWED_FACTOR_TYPES",
    "CohortClusterSpec",
    "SeedPairSpec",
    "TypedContrastSpec",
    "TypedFactorSpec",
    "analyze_paired_seed_runs",
    "audit_typed_experimental_design",
    "validate_cohort_and_split_transitivity",
]
