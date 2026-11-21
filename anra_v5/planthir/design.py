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
        per_seed_effects: list[dict[str, Any]] = []
        family_ses: list[float] = []
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
            fam_se = r.get("family_se")
            fam_se_val = float(fam_se) if fam_se is not None and math.isfinite(float(fam_se)) else None
            if fam_se_val is not None:
                family_ses.append(fam_se_val)
            per_seed_effects.append(
                {
                    "pair_id": pid,
                    "init_seed": init_s,
                    "order_seed": order_s,
                    "cluster_id": cl,
                    "control_value": cf,
                    "treatment_value": tf,
                    "delta": float(delta),
                    "family_se": fam_se_val,
                }
            )
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
        non_zero = [d for d in unit_deltas if d != 0.0]
        if not non_zero:
            sign_flip_p = 1.0
        elif all(d > 0.0 for d in non_zero) or all(d < 0.0 for d in non_zero):
            sign_flip_p = float(min(1.0, math.ldexp(2.0, -min(len(non_zero), 1000))))
        elif len(non_zero) <= 16:
            abs_obs = abs(sum(non_zero))
            extreme = 0
            n_nz = len(non_zero)
            for mask in range(1 << n_nz):
                signed_sum = sum(non_zero[k] if (mask >> k) & 1 else -non_zero[k] for k in range(n_nz))
                if abs(signed_sum) >= abs_obs - 1e-12:
                    extreme += 1
            sign_flip_p = float(extreme / (1 << n_nz))
        else:
            z_eq = sum(1.0 if d > 0.0 else -1.0 for d in non_zero) / math.sqrt(len(non_zero))
            sign_flip_p = float(math.erfc(abs(z_eq) / math.sqrt(2.0)))

        if se_d == 0.0:
            t_stat: float | None = None
            p_raw = float(sign_flip_p)
            ci_t = (mean_d, mean_d)
            variance_status = "DEGENERATE_ZERO_VARIANCE"
        else:
            t_stat = float(mean_d / se_d)
            df = m_units - 1
            p_raw = _student_t_two_sided_p(abs(t_stat), df)
            t_crit = _student_t_crit(alpha, df)
            ci_t = (mean_d - t_crit * se_d, mean_d + t_crit * se_d)
            variance_status = "NONZERO_VARIANCE"

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
            "independent_unit": "seed_pair_cluster",
            "n_pairs": len(deltas),
            "n_clusters": m_units,
            "mean_paired_diff": float(mean_d),
            "se": float(se_d),
            "variance_status": variance_status,
            "t_stat": t_stat,
            "df": int(m_units - 1),
            "p_raw": float(p_raw),
            "sign_flip_p_value": sign_flip_p,
            "positive_count": sum(1 for d in unit_deltas if d > 0.0),
            "negative_count": sum(1 for d in unit_deltas if d < 0.0),
            "tie_count": sum(1 for d in unit_deltas if d == 0.0),
            "per_seed_effects": per_seed_effects,
            "family_uncertainty_conditional_on_seeds": (
                {"mean_family_se": float(sum(family_ses) / len(family_ses)), "reported_pairs": len(family_ses)}
                if family_ses
                else None
            ),
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


def bootstrap_paired_families(
    control_scores: Mapping[str, float] | Sequence[float],
    treatment_scores: Mapping[str, float] | Sequence[float],
    *,
    resamples: int = 2000,
    seed: int = 1801,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Stratified paired-family bootstrap with exact sign-flip checks and zero-variance flagging."""
    if isinstance(control_scores, Mapping) and isinstance(treatment_scores, Mapping):
        if set(control_scores) != set(treatment_scores):
            raise ValueError("control_scores and treatment_scores mappings must contain identical family keys")
        keys = sorted(str(k) for k in control_scores)
        if not keys:
            raise ValueError("bootstrap_paired_families requires at least one paired family")
        ctrl_vals = [float(control_scores[k]) for k in keys]
        trt_vals = [float(treatment_scores[k]) for k in keys]
    elif not isinstance(control_scores, Mapping) and not isinstance(treatment_scores, Mapping):
        ctrl_vals = [float(v) for v in control_scores]
        trt_vals = [float(v) for v in treatment_scores]
        if not ctrl_vals or len(ctrl_vals) != len(trt_vals):
            raise ValueError("control_scores and treatment_scores sequences must have equal non-zero length")
    else:
        raise TypeError("control_scores and treatment_scores must both be mappings or both be sequences")

    deltas = [t - c for c, t in zip(ctrl_vals, trt_vals)]
    n = len(deltas)
    mean_d = float(sum(deltas) / n)
    ctrl_mean = float(sum(ctrl_vals) / n)
    trt_mean = float(sum(trt_vals) / n)
    pos_cnt = sum(1 for d in deltas if d > 0.0)
    neg_cnt = sum(1 for d in deltas if d < 0.0)
    tie_cnt = sum(1 for d in deltas if d == 0.0)
    zero_var = all(d == deltas[0] for d in deltas)
    variance_status = "DEGENERATE_ZERO_VARIANCE" if zero_var else "NONZERO_VARIANCE"

    non_zero = [d for d in deltas if d != 0.0]
    n_nz = len(non_zero)
    if n_nz == 0:
        sign_flip_p = 1.0
    elif n_nz <= 16:
        obs_abs = abs(sum(non_zero))
        extreme = 0
        total_signs = 1 << n_nz
        for mask in range(total_signs):
            s_val = sum(non_zero[j] if ((mask >> j) & 1) else -non_zero[j] for j in range(n_nz))
            if abs(s_val) >= obs_abs - 1e-12:
                extreme += 1
        sign_flip_p = float(extreme / total_signs)
    else:
        denom = math.sqrt(sum(d * d for d in non_zero))
        z_eq = abs(sum(non_zero)) / denom if denom > 0.0 else 0.0
        sign_flip_p = float(math.erfc(z_eq / math.sqrt(2.0)))

    rng = random.Random(int(seed))
    n_boot = max(100, int(resamples))
    boots: list[float] = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo_idx = max(0, min(n_boot - 1, int(math.floor((alpha / 2.0) * n_boot))))
    hi_idx = max(0, min(n_boot - 1, int(math.ceil((1.0 - alpha / 2.0) * n_boot)) - 1))
    le_zero = sum(1 for b in boots if b <= 0.0) / n_boot
    ge_zero = sum(1 for b in boots if b >= 0.0) / n_boot
    boot_p = float(min(1.0, 2.0 * min(le_zero, ge_zero)))

    return {
        "schema": "anra-planthir-family-bootstrap-v1",
        "n_families": n,
        "resamples": n_boot,
        "alpha": float(alpha),
        "control_mean": ctrl_mean,
        "treatment_mean": trt_mean,
        "mean_delta": mean_d,
        "positive_count": pos_cnt,
        "negative_count": neg_cnt,
        "tie_count": tie_cnt,
        "variance_status": variance_status,
        "sign_flip_p_value": sign_flip_p,
        "ci_low": float(boots[lo_idx]),
        "ci_high": float(boots[hi_idx]),
        "bootstrap_p_two_sided": boot_p,
    }


@dataclass(frozen=True, slots=True)
class PairedSeedObservation:
    """Typed matched-seed control vs treatment comparison row accepted by `analyze_paired_seed_runs`."""

    contrast_id: str
    pair_id: str
    init_seed: int
    order_seed: int
    control_arm_id: str
    treatment_arm_id: str
    control_value: float
    treatment_value: float
    family_se: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "contrast_id": str(self.contrast_id),
            "pair_id": str(self.pair_id),
            "init_seed": int(self.init_seed),
            "order_seed": int(self.order_seed),
            "control_arm_id": str(self.control_arm_id),
            "treatment_arm_id": str(self.treatment_arm_id),
            "control_value": float(self.control_value),
            "treatment_value": float(self.treatment_value),
        }
        if self.family_se is not None:
            out["family_se"] = float(self.family_se)
        return out


def _ep_first_and_final_action(rec: Mapping[str, Any]) -> tuple[tuple[str, Any] | None, tuple[str, Any] | None]:
    steps = rec.get("steps")
    if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes)) and len(steps) > 0:
        s0, s_end = steps[0], steps[-1]
        if isinstance(s0, Mapping) and isinstance(s_end, Mapping):
            f = (str(s0.get("executed_op", s0.get("operation", ""))), s0.get("executed_arg", s0.get("argument")))
            l = (str(s_end.get("executed_op", s_end.get("operation", ""))), s_end.get("executed_arg", s_end.get("argument")))
            return f, l
    acts = rec.get("actions")
    if isinstance(acts, Sequence) and not isinstance(acts, (str, bytes)) and len(acts) > 0:
        return (acts[0][0], acts[0][1]), (acts[-1][0], acts[-1][1])
    if "executed_op" in rec or "operation" in rec:
        act = (str(rec.get("executed_op", rec.get("operation", ""))), rec.get("executed_arg", rec.get("argument")))
        return act, act
    return None, None


def compare_paired_episode_records(
    control_records: Sequence[Mapping[str, Any]],
    treatment_records: Sequence[Mapping[str, Any]],
    *,
    id_key: str = "world_id",
    base_key: str = "base_world_id",
    family_key: str = "family",
    view_key: str = "form",
) -> dict[str, Any]:
    """Compare paired control vs treatment episode traces on identical world IDs and surface opposing solves."""

    ctrl_by_id: dict[str, Mapping[str, Any]] = {}
    trt_by_id: dict[str, Mapping[str, Any]] = {}
    for idx, r in enumerate(control_records):
        eid = str(r.get(id_key, "")).strip()
        if not eid:
            raise ValueError(f"control record at index {idx} missing {id_key!r}")
        if eid in ctrl_by_id:
            raise ValueError(f"duplicate {id_key!r} in control_records: {eid!r}")
        ctrl_by_id[eid] = r
    for idx, r in enumerate(treatment_records):
        eid = str(r.get(id_key, "")).strip()
        if not eid:
            raise ValueError(f"treatment record at index {idx} missing {id_key!r}")
        if eid in trt_by_id:
            raise ValueError(f"duplicate {id_key!r} in treatment_records: {eid!r}")
        trt_by_id[eid] = r

    if set(ctrl_by_id) != set(trt_by_id):
        miss_c = sorted(set(trt_by_id) - set(ctrl_by_id))
        miss_t = sorted(set(ctrl_by_id) - set(trt_by_id))
        raise ValueError(f"episode ID sets differ (missing_in_control={miss_c[:5]}, missing_in_treatment={miss_t[:5]})")
    if not ctrl_by_id:
        raise ValueError("compare_paired_episode_records requires non-empty paired records")

    both = 0
    trt_only = 0
    ctrl_only = 0
    neither = 0
    first_op_disagree = 0
    action_disagree = 0

    fam_stats: dict[str, dict[str, Any]] = {}
    view_stats: dict[str, dict[str, Any]] = {}
    trt_only_firstop_ids: list[str] = []
    trt_only_any_ids: list[str] = []
    ctrl_only_firstop_ids: list[str] = []
    ctrl_only_any_ids: list[str] = []
    distinct_bases: set[str] = set()

    for eid in sorted(ctrl_by_id):
        c = ctrl_by_id[eid]
        t = trt_by_id[eid]
        c_ok = bool(c.get("correct", c.get("raw_exact_correct", False)))
        t_ok = bool(t.get("correct", t.get("raw_exact_correct", False)))
        fam = str(c.get(family_key) or c.get("family_id") or c.get("category") or "default")
        view = str(c.get(view_key) or c.get("view") or c.get("cell_key") or "canonical")
        distinct_bases.add(str(c.get(base_key) or c.get("family_id") or eid))

        cf, cl = _ep_first_and_final_action(c)
        tf, tl = _ep_first_and_final_action(t)
        op_diff = (cf is not None and tf is not None and cf[0] != tf[0])
        act_diff = (cf != tf) or (cl != tl)
        if op_diff:
            first_op_disagree += 1
        if act_diff:
            action_disagree += 1

        if c_ok and t_ok:
            both += 1
            bucket = "both_solved"
        elif t_ok and not c_ok:
            trt_only += 1
            bucket = "treatment_only_solved"
            trt_only_any_ids.append(eid)
            if op_diff:
                trt_only_firstop_ids.append(eid)
        elif c_ok and not t_ok:
            ctrl_only += 1
            bucket = "control_only_solved"
            ctrl_only_any_ids.append(eid)
            if op_diff:
                ctrl_only_firstop_ids.append(eid)
        else:
            neither += 1
            bucket = "neither_solved"

        for grp_map, grp_key in ((fam_stats, fam), (view_stats, view)):
            entry = grp_map.setdefault(
                grp_key,
                {
                    "both_solved": 0,
                    "treatment_only_solved": 0,
                    "control_only_solved": 0,
                    "neither_solved": 0,
                    "episodes": 0,
                },
            )
            entry[bucket] += 1
            entry["episodes"] += 1

    def _finalize_groups(grp_map: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for k, v in sorted(grp_map.items()):
            n_g = max(1, v["episodes"])
            c_acc = (v["both_solved"] + v["control_only_solved"]) / n_g
            t_acc = (v["both_solved"] + v["treatment_only_solved"]) / n_g
            out[k] = {
                **v,
                "control_accuracy": float(c_acc),
                "treatment_accuracy": float(t_acc),
                "delta": float(t_acc - c_acc),
            }
        return out

    n_ep = len(ctrl_by_id)
    ctrl_acc = (both + ctrl_only) / n_ep
    trt_acc = (both + trt_only) / n_ep
    rule = "first_sorted_episode_id_with_opposing_correctness_and_first_op_divergence_fallback_any_opposing"
    return {
        "schema": "anra-planthir-paired-episode-comparison-v1",
        "episode_count": n_ep,
        "distinct_base_world_count": len(distinct_bases),
        "control_accuracy": float(ctrl_acc),
        "treatment_accuracy": float(trt_acc),
        "net_accuracy_delta": float(trt_acc - ctrl_acc),
        "both_solved": both,
        "treatment_only_solved": trt_only,
        "control_only_solved": ctrl_only,
        "neither_solved": neither,
        "first_operation_disagreement_count": first_op_disagree,
        "action_disagreement_count": action_disagree,
        "by_family": _finalize_groups(fam_stats),
        "by_view": _finalize_groups(view_stats),
        "contrasting_trace_ids": {
            "selection_rule": rule,
            "treatment_only_example_id": trt_only_firstop_ids[0] if trt_only_firstop_ids else (trt_only_any_ids[0] if trt_only_any_ids else None),
            "control_only_example_id": ctrl_only_firstop_ids[0] if ctrl_only_firstop_ids else (ctrl_only_any_ids[0] if ctrl_only_any_ids else None),
        },
    }


def build_paired_seed_observations(
    arm_records: Sequence[Mapping[str, Any]],
    *,
    contrast_id: str = "PRIMARY",
    control_arm: str,
    treatment_arm: str,
    metric_key: str = "value",
    init_seed_key: str = "init_seed",
    order_seed_key: str = "order_seed",
    arm_key: str = "architecture",
) -> dict[str, Any]:
    """Pair arm-level job rows into typed `PairedSeedObservation` objects, reporting missing arms cleanly without discarding completed rows."""

    grouped: dict[tuple[int, int], dict[str, Mapping[str, Any]]] = {}
    for idx, rec in enumerate(arm_records):
        if arm_key not in rec or init_seed_key not in rec or metric_key not in rec:
            raise ValueError(f"arm record at index {idx} missing required key ({arm_key!r}, {init_seed_key!r}, or {metric_key!r})")
        init_s = int(rec[init_seed_key])
        ord_s = int(rec.get(order_seed_key, init_s))
        arm_val = str(rec[arm_key])
        bucket = grouped.setdefault((init_s, ord_s), {})
        if arm_val in bucket:
            raise ValueError(f"duplicate record for arm {arm_val!r} at seed ({init_s}, {ord_s})")
        bucket[arm_val] = rec

    observations: list[PairedSeedObservation] = []
    missing_pairs: list[dict[str, Any]] = []
    for (init_s, ord_s) in sorted(grouped):
        arms = grouped[(init_s, ord_s)]
        has_ctrl = control_arm in arms
        has_trt = treatment_arm in arms
        if has_ctrl and has_trt:
            c_rec = arms[control_arm]
            t_rec = arms[treatment_arm]
            c_se = c_rec.get("family_se")
            t_se = t_rec.get("family_se")
            comb_se = (math.sqrt(float(c_se) ** 2 + float(t_se) ** 2) if (c_se is not None and t_se is not None) else (float(c_se) if c_se is not None else (float(t_se) if t_se is not None else None)))
            observations.append(
                PairedSeedObservation(
                    contrast_id=str(contrast_id),
                    pair_id=f"{init_s}:{ord_s}",
                    init_seed=init_s,
                    order_seed=ord_s,
                    control_arm_id=str(c_rec.get("arm_id", control_arm)),
                    treatment_arm_id=str(t_rec.get("arm_id", treatment_arm)),
                    control_value=float(c_rec[metric_key]),
                    treatment_value=float(t_rec[metric_key]),
                    family_se=comb_se,
                )
            )
        else:
            missing = [a for a, present in ((control_arm, has_ctrl), (treatment_arm, has_trt)) if not present]
            present = sorted(arms)
            missing_pairs.append(
                {
                    "pair_id": f"{init_s}:{ord_s}",
                    "init_seed": init_s,
                    "order_seed": ord_s,
                    "missing_arms": missing,
                    "present_arms": present,
                    "explanation": (
                        f"Seed pair (init_seed={init_s}, order_seed={ord_s}) is missing arm(s) {missing}; "
                        f"present arm(s): {present}."
                    ),
                }
            )

    if len(observations) >= 2:
        inference = analyze_paired_seed_runs([o.to_dict() for o in observations])
    else:
        inference = {
            "schema": "anra-planthir-paired-seed-analysis-v1",
            "passed": False,
            "status": "INSUFFICIENT_COMPLETE_PAIRS",
            "defects": [f"fewer than 2 complete matched seed pairs ({len(observations)} found)"],
            "contrasts": {},
        }

    complete = not missing_pairs and len(observations) >= 2 and bool(inference.get("passed", False))
    return {
        "schema": "anra-planthir-paired-seed-bundle-v1",
        "status": "COMPLETE_PAIRS" if complete else "INCOMPLETE_PAIRS_REPORTED",
        "contrast_id": str(contrast_id),
        "control_arm": str(control_arm),
        "treatment_arm": str(treatment_arm),
        "complete_pair_count": len(observations),
        "observations": [o.to_dict() for o in observations],
        "missing_pairs": missing_pairs,
        "preserved_arm_records": [dict(r) for r in arm_records],
        "inference": inference,
    }


def run_paired_comparison_preflight(
    *,
    control_arm: str = "P",
    treatment_arm: str = "R",
    seeds: Sequence[int] = (7, 19, 41),
) -> dict[str, Any]:
    """Validate the paired comparison & dropped-arm reporting pipeline on tiny synthetic fixtures before GPU spend."""

    seed_list = [int(s) for s in seeds]
    if len(seed_list) < 2:
        raise ValueError("run_paired_comparison_preflight requires at least 2 seeds")

    complete_rows: list[dict[str, Any]] = []
    for idx, s in enumerate(seed_list):
        complete_rows.append({"architecture": control_arm, "init_seed": s, "order_seed": s, "value": 0.60 + 0.02 * idx})
        complete_rows.append({"architecture": treatment_arm, "init_seed": s, "order_seed": s, "value": 0.62 + 0.03 * (idx % 2)})

    complete_fx = build_paired_seed_observations(
        complete_rows,
        contrast_id="PREFLIGHT_PRIMARY",
        control_arm=control_arm,
        treatment_arm=treatment_arm,
    )
    dropped_rows = complete_rows[:-1]  # drop final seed's treatment row while keeping control + earlier traces
    dropped_fx = build_paired_seed_observations(
        dropped_rows,
        contrast_id="PREFLIGHT_PRIMARY",
        control_arm=control_arm,
        treatment_arm=treatment_arm,
    )

    last_s = seed_list[-1]
    complete_ok = (
        complete_fx["status"] == "COMPLETE_PAIRS"
        and complete_fx["complete_pair_count"] == len(seed_list)
        and complete_fx["missing_pairs"] == []
        and bool(complete_fx["inference"].get("passed", False))
    )
    dropped_ok = (
        dropped_fx["status"] == "INCOMPLETE_PAIRS_REPORTED"
        and dropped_fx["complete_pair_count"] == len(seed_list) - 1
        and len(dropped_fx["missing_pairs"]) == 1
        and dropped_fx["missing_pairs"][0]["missing_arms"] == [treatment_arm]
        and str(last_s) in dropped_fx["missing_pairs"][0]["explanation"]
        and len(dropped_fx["preserved_arm_records"]) == len(dropped_rows)
        and (len(seed_list) - 1 < 2 or bool(dropped_fx["inference"].get("passed", False)))
    )
    passed = bool(complete_ok and dropped_ok)
    return {
        "schema": "anra-planthir-paired-preflight-v1",
        "passed": passed,
        "status": "PREFLIGHT_PASSED" if passed else "PREFLIGHT_FAILED",
        "complete_fixture": complete_fx,
        "dropped_arm_fixture": dropped_fx,
    }


def exact_mcnemar_test(
    baseline_correct: Sequence[bool],
    treatment_correct: Sequence[bool],
) -> dict[str, Any]:
    """Compute an exact two-sided McNemar binomial test on paired binary episode outcomes."""

    b_seq = [bool(x) for x in baseline_correct]
    t_seq = [bool(x) for x in treatment_correct]
    if not b_seq or len(b_seq) != len(t_seq):
        raise ValueError("baseline_correct and treatment_correct must be non-empty and equal length")

    both_correct = 0
    both_wrong = 0
    b_only = 0
    t_only = 0
    for b_ok, t_ok in zip(b_seq, t_seq):
        if b_ok and t_ok:
            both_correct += 1
        elif b_ok and not t_ok:
            b_only += 1
        elif not b_ok and t_ok:
            t_only += 1
        else:
            both_wrong += 1

    n = len(b_seq)
    discordant = b_only + t_only
    if discordant == 0:
        p_value = 1.0
    else:
        k = min(b_only, t_only)
        tail = sum(math.comb(discordant, i) for i in range(k + 1)) / float(1 << discordant)
        p_value = min(1.0, float(2.0 * tail))

    return {
        "schema": "anra-planthir-mcnemar-v1",
        "n": n,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "b_baseline_only": b_only,
        "c_treatment_only": t_only,
        "discordant": discordant,
        "accuracy_delta": float(t_only - b_only) / float(n),
        "p_value": p_value,
        "significant_at_005": p_value < 0.05,
    }


def holm_bonferroni_adjust(
    p_values: Mapping[str, float] | Sequence[float],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Apply step-down Holm-Bonferroni multiple-comparison adjustment to named or indexed p-values."""

    alpha_f = float(alpha)
    if not math.isfinite(alpha_f) or not (0.0 < alpha_f < 1.0):
        raise ValueError("alpha must satisfy 0 < alpha < 1")

    if isinstance(p_values, Mapping):
        items = [(str(k), float(v)) for k, v in p_values.items()]
    else:
        items = [(str(idx), float(v)) for idx, v in enumerate(p_values)]
    if not items:
        raise ValueError("p_values must be non-empty")
    for k, v in items:
        if not math.isfinite(v) or not (0.0 <= v <= 1.0):
            raise ValueError(f"invalid p-value for {k!r}: {v!r}")

    m = len(items)
    ordered = sorted(items, key=lambda kv: (kv[1], kv[0]))
    adjusted_map: dict[str, float] = {}
    running_max = 0.0
    for rank, (key, raw_p) in enumerate(ordered):
        scaled = min(1.0, (m - rank) * raw_p)
        running_max = max(running_max, scaled)
        adjusted_map[key] = running_max

    rejected_map = {k: adjusted_map[k] < alpha_f for k, _ in items}
    return {
        "schema": "anra-planthir-holm-bonferroni-v1",
        "alpha": alpha_f,
        "m": m,
        "adjusted_p_values": adjusted_map,
        "adjusted_list": [adjusted_map[k] for k, _ in items],
        "rejected": rejected_map,
        "rejected_keys": [k for k, _ in items if rejected_map[k]],
    }


__all__ = [
    "ALLOWED_FACTOR_TYPES",
    "CohortClusterSpec",
    "PairedSeedObservation",
    "SeedPairSpec",
    "TypedContrastSpec",
    "TypedFactorSpec",
    "analyze_paired_seed_runs",
    "audit_typed_experimental_design",
    "bootstrap_paired_families",
    "build_paired_seed_observations",
    "compare_paired_episode_records",
    "exact_mcnemar_test",
    "holm_bonferroni_adjust",
    "run_paired_comparison_preflight",
    "validate_cohort_and_split_transitivity",
]
