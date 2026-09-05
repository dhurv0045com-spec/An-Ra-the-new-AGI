"""Notebook torture test: prove the Colab/Kaggle launchers cannot fail trivially.

Run:  python tests/test_citadel_notebooks.py   (exit 0 = all pass)
For every notebooks/citadel_*.ipynb:
  1. valid JSON + nbformat 4 + TPU accelerator metadata,
  2. EVERY code cell compiles (SyntaxError hunt — shell/magic lines stripped),
  3. cross-cell name flow: no Load of a name never Stored earlier in notebook
     order (NameError hunt; builtins + documented kernel names allowed),
  4. every bare-word key subscripted or printed from a receipt dict exists in
     the producing code's schema (KeyError hunt against hand-verified sets
     below; each set cites its producer).

What this does NOT prove (stated honestly): device-side numerics, XLA compile
success, torch-xla API presence on the future runtime, or timing. Those can
only be proven on the TPU.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CITADEL_ROOT = HERE.parents[1]
NB_DIR = CITADEL_ROOT / "notebooks"

BUILTINS = set(dir(__builtins__)) | {"__import__", "get_ipython"}

# Hand-verified against producer code (module: keys). Update with the producer.
SCHEMAS = {
    "env": {"schema", "probe_utc", "git_sha", "platform", "accelerator_requested",
            "accelerator_detected", "pjrt_device_env", "xla_device_count",
            "python_version", "torch_version", "torch_xla_version", "xla_runtime",
            "xla_device_type", "xla_devices", "tpu_present", "tpu_generation",
            "host_cpu", "host_ram_gb", "local_disk_free_bytes",
            "kaggle_session_limits", "sys_argv0", "probe_pass"},
    "t0": {"schema", "citadel_sha", "cymek_runtime_sha", "environment", "model",
           "initial_parameter_sha256", "final_parameter_sha256",
           "input_token_sha256", "loss", "supervised_tokens", "grad_norm_pre_clip",
           "learning_rate", "device_count", "wall_seconds", "tokens_processed",
           "tokens_per_second", "checkpoint_sha256", "reload_identical",
           "certification"},
    "model": {"spec", "parameter_count"},
    "train": {"schema", "citadel_sha", "cymek_runtime_sha", "environment", "model",
              "data", "training", "eval", "diagnostics", "heuristic_nulls",
              "strongest_heuristic_null", "gate_rules",
              "pre_reload_prediction_sha256", "post_reload_prediction_sha256",
              "reload_identical", "checkpoint", "device_count", "wall_seconds",
              "status", "interpretation"},
    "train_training": {"ladder", "endpoint_updates", "rung_evals", "batch_rows",
                       "sequence_length", "optimizer", "learning_rate", "seed",
                       "tokens_consumed_capacity", "tokens_real",
                       "tokens_supervised", "first_loss", "last_loss",
                       "first_update_seconds", "train_wall_seconds",
                       "steady_tokens_per_second"},
    "train_eval": {"untrained_dev", "untrained_dev_ce", "untrained_test",
                   "untrained_test_ce", "untrained_train_sample", "trained_test",
                   "trained_test_ce", "trained_train_sample", "reload_test",
                   "reload_test_ce"},
    "summary": {"correct", "total", "accuracy", "wilson_lcb", "wilson_ucb"},
    "canary_data": {"schema", "generator_version", "generator_code_sha256",
                    "counts", "seeds", "ranges", "op_distribution", "split_sha256",
                    "overlap", "generalization_slices",
                    "slice_test_duplicates_dropped", "scored_slice_counts",
                    "receipt_sha256"},
    "canary_build": {"schema", "generator_version", "splits",
                     "split_overlap_rows", "generalization", "receipt_sha256"},
    "null": {"name", "accuracy"},
    "cal": {"schema", "citadel_sha", "cymek_runtime_sha", "environment",
            "candidates", "selected", "selected_tokens_per_second"},
    "cal_selected": {"batch", "length"},
    "manifest": {"schema", "generator_version", "splits", "audit_sample",
                 "leakage", "transfer_overlap_expected", "total_bytes",
                 "max_row_chars"},
    "arm": {"schema", "arm", "config", "citadel_sha", "cymek_runtime_sha",
            "environment", "batch", "sequence_length", "model", "data",
            "training", "untrained", "trained", "intermediates", "diagnostics",
            "heuristic_nulls", "strongest_heuristic_null", "gate_rules",
            "pre_reload_prediction_sha256", "post_reload_prediction_sha256",
            "reload_identical", "checkpoint", "device_count", "wall_seconds",
            "status"},
    "arm_training": {"updates", "ledgers", "first_loss", "last_loss",
                     "capacity_tokens", "answer_supervised_tokens",
                     "whole_supervised_tokens", "grad_norm_mean", "grad_norm_max",
                     "first_update_includes_compile", "train_wall_seconds"},
    "arm_diag": {"stop_histogram", "samples", "memorization_flag"},
    "arm_ckpt": {"path", "sha256"},
    "xsummary": {"labels", "reasons"},
    "bundle": {"files", "checkpoints", "checkpoint_bytes",
               "checkpoints_bundled", "zip", "zip_bytes"},
    "session": {"schema", "citadel_sha", "cymek_runtime_sha", "shape",
                "calibrated_rate", "budgets_scaled", "budgets", "arms",
                "labels", "bundle"},
    "curves": {"schema", "arms"},
    "arm_curves": {"status", "train", "dev", "test", "untrained_test",
                   "first_train_lift_tier", "first_test_lift_tier"},
    "decision": {"ready_for_50m_training", "blocking_reasons"},
    "verify": {"schema", "files", "status"},
}

# Every bare-word key touched per notebook -> schema group(s) it must belong to.
# Built by reading each notebook cell and the producer code it calls.
NOTEBOOK_KEYS = {
    "citadel_colab_tpu.ipynb": {
        "env": {"platform", "accelerator_detected", "xla_device_count",
                "torch_version", "torch_xla_version", "probe_pass"},
        "t0": {"citadel_sha", "cymek_runtime_sha", "certification", "loss",
               "tokens_per_second", "reload_identical"},
        "train": {"status", "gate_rules", "eval", "training",
                  "strongest_heuristic_null",
                  "pre_reload_prediction_sha256",
                  "post_reload_prediction_sha256", "reload_identical"},
        "train_eval": {"untrained_test", "trained_test"},
        "train_training": {"endpoint_updates"},
        "canary_data": {"counts", "overlap", "scored_slice_counts"},
    },
    "citadel_kaggle_tpu.ipynb": {
        "t0": {"citadel_sha", "cymek_runtime_sha", "certification", "loss",
               "tokens_per_second", "reload_identical"},
        "train": {"training", "eval", "interpretation"},
        "canary_build": {"splits", "split_overlap_rows"},
    },
    "citadel_colab_t1.ipynb": {
        "canary_data": {"counts", "overlap", "scored_slice_counts"},
        "train": {"status", "gate_rules", "eval", "training",
                  "strongest_heuristic_null",
                  "pre_reload_prediction_sha256",
                  "post_reload_prediction_sha256", "reload_identical"},
        "train_eval": {"untrained_test", "trained_test"},
        "train_training": {"endpoint_updates"},
    "null": {"name", "accuracy"},
    },
    "citadel_colab_t1b.ipynb": {
        "train": {"status", "eval", "diagnostics", "interpretation",
                  "gate_rules", "pre_reload_prediction_sha256",
                  "post_reload_prediction_sha256", "reload_identical",
                  "checkpoint", "training"},
        "train_eval": {"trained_test", "trained_train_sample"},
        "train_training": {"endpoint_updates"},
        "summary": {"accuracy"},
        "arm_diag": {"memorization_flag"},
        "arm_ckpt": {"path"},
    },
    "citadel_colab_t1c.ipynb": {
        "cal": {"selected", "selected_tokens_per_second"},
        "cal_selected": {"batch", "length"},
        "manifest": {"total_bytes", "leakage"},
        "arm": {"status"},
        "train_eval": {"trained_test"},
        "summary": {"accuracy"},
        "xsummary": {"labels", "reasons"},
        "bundle": {"zip_bytes", "checkpoints_bundled", "checkpoints"},
        "session": {"arms", "labels"},
    },
    "citadel_colab_t1d.ipynb": {
        "cal": {"selected", "selected_tokens_per_second"},
        "manifest": {"total_bytes", "leakage", "max_row_chars"},
        "session": {"arms", "labels"},
        "xsummary": {"labels", "reasons"},
        "curves": {"arms"},
        "arm_curves": {"first_train_lift_tier", "first_test_lift_tier"},
        "decision": {"ready_for_50m_training", "blocking_reasons"},
        "verify": {"status"},
        "bundle": {"zip_bytes", "checkpoints_bundled", "checkpoints"},
    },
}


def _cells(nb_path: Path):
    doc = json.loads(nb_path.read_text(encoding="utf-8"))
    assert doc.get("nbformat") == 4, f"{nb_path.name}: not nbformat 4"
    accel = ((doc.get("metadata") or {}).get("accelerator") or "").upper()
    assert accel == "TPU", f"{nb_path.name}: accelerator metadata is {accel!r}, want TPU"
    return [c for c in doc["cells"] if c.get("cell_type") == "code"]


def _cell_names(src: str, nb_name: str, idx: int):
    body = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith(("!", "%")))
    try:
        tree = ast.parse(body)
    except SyntaxError as exc:
        raise AssertionError(f"{nb_name} cell {idx}: SyntaxError: {exc}")
    stored, loaded = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                stored.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
        elif isinstance(node, ast.Import):
            stored.update((a.asname or a.name.split(".")[0]) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            stored.update((a.asname or a.name) for a in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stored.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            stored.add(node.name)  # `except ... as e` binds without a Name node
    return stored, loaded


def _touched_keys(src: str):
    """Bare-word keys a cell reads from receipt dicts: d['key'] subscripts
    (excluding os.environ lookups) plus the print-selection tuples
    `for k in ('a', 'b')`. Command argv lists and misc literals are ignored."""
    body = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith(("!", "%")))
    tree = ast.parse(body)
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            val = node.slice
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                target = node.value
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) \
                        and target.value.id == "os":
                    continue  # os.environ[...] lookups, not receipt keys
                if "/" not in val.value and "." not in val.value:
                    keys.add(val.value)
        elif isinstance(node, ast.Tuple):
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str) \
                        and elt.value.isidentifier():
                    keys.add(elt.value)
    return keys


def test_notebook_cells_compile_and_flow() -> None:
    nbs = sorted(NB_DIR.glob("citadel_*.ipynb"))
    assert len(nbs) >= 5, f"expected >=5 citadel notebooks, found {len(nbs)}"
    for nb in nbs:
        ns: set[str] = set()
        for idx, cell in enumerate(_cells(nb)):
            src = "".join(cell["source"])
            stored, loaded = _cell_names(src, nb.name, idx)
            # A cell may use what it defines itself; prior cells accumulate in ns.
            undefined = loaded - ns - BUILTINS - stored
            assert not undefined, f"{nb.name} cell {idx}: names used before defined: {sorted(undefined)}"
            ns |= stored


def test_notebook_receipt_keys_exist() -> None:
    for nb_name, groups in NOTEBOOK_KEYS.items():
        nb = NB_DIR / nb_name
        assert nb.is_file(), f"missing notebook {nb_name}"
        for group, keys in groups.items():
            assert group in SCHEMAS, f"unknown schema group {group}"
            missing = set(keys) - SCHEMAS[group]
            assert not missing, f"{nb_name}: keys {sorted(missing)} not in {group} schema"
        # every key the notebook touches in these groups must be covered above
        touched: set[str] = set()
        for cell in _cells(nb):
            touched |= _touched_keys("".join(cell["source"]))
        covered = set().union(*groups.values())
        # identifiers that are not dict keys (paths, literals, api names)
        noise = {k for k in touched - covered}
        unexplained = {k for k in noise if k not in _KNOWN_NONKEYS}
        assert not unexplained, f"{nb_name}: touched identifiers not in schema sets: {sorted(unexplained)}"


_KNOWN_NONKEYS = {
    # paths / filenames / urls / misc literals appearing in cells
    "An", "Ra", "colab", "kaggle", "json", "pt", "zip", "md", "py",
    "utf", "ascii", "cpu", "tpu", "CITADEL", "TPU", "PASS", "FAIL", "YES", "NO",
    "true", "false", "null",
    # T1B arm tags + loop labels (tuple literals, not receipt keys)
    "A1k", "A2p5k", "A5k", "A10k", "A20k", "final",
}


def main() -> int:
    tests = [test_notebook_cells_compile_and_flow, test_notebook_receipt_keys_exist]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}", flush=True)
    total = len(tests)
    print(f"{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
