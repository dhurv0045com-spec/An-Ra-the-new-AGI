"""Regression tests for §0 Palantiri independent replay release blocker."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from anra_v5.planthir.evidence import (
    BUNDLE_POINTER_NAME,
    FULL_CHECKPOINT_ZIP_NAME,
    RESULT_ZIP_NAME,
    _write_verified_zip,
    resolve_published_generation,
)
from anra_v5.planthir.replay import independent_replay_bundle


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args: list[str], *, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.pop("PLANTHIR_BACKEND_MODULES", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "anra_v5.palantiri", *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _rewrite_result_zip_with_valid_custody(
    bundle_dir: Path,
    transform_files: dict[str, bytes | None],
) -> Path:
    """Re-pack result.zip in a bundle generation while recomputing ARCHIVE_MANIFEST, sidecar, and CURRENT.json."""
    ptr_path = bundle_dir / BUNDLE_POINTER_NAME
    ptr = resolve_published_generation(bundle_dir, verify=False)
    gen_dir = Path(ptr["generation_dir"])
    rzip = gen_dir / RESULT_ZIP_NAME
    entries: dict[str, bytes] = {}
    with zipfile.ZipFile(rzip, "r") as zf:
        for name in zf.namelist():
            if name == "ARCHIVE_MANIFEST.json":
                continue
            entries[name] = zf.read(name)
        old_manifest = json.loads(zf.read("ARCHIVE_MANIFEST.json").decode("utf-8"))

    for key, val in transform_files.items():
        if val is None:
            entries.pop(key, None)
        else:
            entries[key] = val

    meta = {k: v for k, v in old_manifest.items() if k not in ("schema", "files", "manifest_sha256")}
    new_manifest = _write_verified_zip(rzip, entries, meta=meta)
    rzip_sha = _sha256_file(rzip)
    (gen_dir / f"{RESULT_ZIP_NAME}.sha256").write_text(f"{rzip_sha}  {RESULT_ZIP_NAME}\n", encoding="utf-8")
    ptr_doc = json.loads(ptr_path.read_text(encoding="utf-8"))
    ptr_doc["result_manifest_sha256"] = new_manifest["manifest_sha256"]
    ptr_doc["result_zip_sha256"] = rzip_sha
    ptr_path.write_text(json.dumps(ptr_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return gen_dir


def _head_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()


def _create_tiny_spec(spec_path: Path, backend: str, source_commit: str) -> None:
    p = _run_cli(["init", "--backend", backend, "--out", str(spec_path)])
    assert p.returncode == 0, p.stderr + p.stdout
    doc = json.loads(spec_path.read_text(encoding="utf-8"))
    doc["source_commit"] = source_commit
    spec_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def test_x1_s20_public_cli_replay_bundle_matches_archived_reference(tmp_path: Path) -> None:
    commit = _head_commit()
    spec_path = tmp_path / "x1_s20_spec.json"
    work_root = tmp_path / "work"
    _create_tiny_spec(spec_path, "x1_s20", commit)

    val = _run_cli(["validate", "--spec", str(spec_path), "--canary"])
    assert val.returncode == 0, val.stdout + val.stderr

    run = _run_cli(["run", "--spec", str(spec_path), "--work-root", str(work_root), "--canary", "--device", "cpu"])
    assert run.returncode == 0, run.stdout + run.stderr

    bundle_dir = work_root / "result_bundle"
    ver = _run_cli(["verify-bundle", "--bundle-dir", str(bundle_dir), "--expected-source-commit", commit])
    assert ver.returncode == 0, ver.stdout + ver.stderr

    receipt_path = tmp_path / "replay_receipt.json"
    rep = _run_cli(
        [
            "replay-bundle",
            "--bundle-dir",
            str(bundle_dir),
            "--expected-source-commit",
            commit,
            "--device",
            "cpu",
            "--output",
            str(receipt_path),
        ]
    )
    assert rep.returncode == 0, rep.stdout + rep.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["verified"] is True
    assert receipt["status"] == "REPLAY_VERIFIED"
    assert receipt["archived_reference_used"] is True
    assert receipt["prediction_agreement_status"] == "MATCHED"
    assert receipt["original_prediction_agreement"]["checked_cases"] >= 1
    assert receipt["restored_lineages"][0]["backend_id"] == "x1_s20"


def test_replay_detects_input_tamper_wrong_backend_and_missing_reference(tmp_path: Path) -> None:
    commit = _head_commit()
    spec_path = tmp_path / "x1_s20_spec.json"
    work_root = tmp_path / "work"
    _create_tiny_spec(spec_path, "x1_s20", commit)

    run = _run_cli(["run", "--spec", str(spec_path), "--work-root", str(work_root), "--canary", "--device", "cpu"])
    assert run.returncode == 0, run.stdout + run.stderr
    bundle_dir = work_root / "result_bundle"
    gen_dir = Path(resolve_published_generation(bundle_dir, verify=False)["generation_dir"])
    with zipfile.ZipFile(gen_dir / RESULT_ZIP_NAME, "r") as zf:
        orig_ref_bytes = zf.read("REPLAY_REFERENCE.json")
    orig_ref = json.loads(orig_ref_bytes.decode("utf-8"))

    # 1. Deliberate input hash mismatch -> reports offending world_id
    tampered_input = json.loads(json.dumps(orig_ref))
    first_lin = next(iter(tampered_input["lineages"].values()))
    offending_wid = first_lin["reference_predictions"][0]["world_id"]
    first_lin["reference_predictions"][0]["rendered_input_sha256"] = "0" * 64
    _rewrite_result_zip_with_valid_custody(
        bundle_dir,
        {"REPLAY_REFERENCE.json": (json.dumps(tampered_input, indent=2, sort_keys=True) + "\n").encode("utf-8")},
    )
    out_tamper = tmp_path / "tamper_receipt.json"
    with pytest.raises(ValueError, match="ORIGINAL_PREDICTION_MISMATCH"):
        independent_replay_bundle(bundle_dir, expected_source_commit=commit, output_path=out_tamper)
    tamper_doc = json.loads(out_tamper.read_text(encoding="utf-8"))
    assert tamper_doc["verified"] is False
    mismatches = tamper_doc["original_prediction_agreement"]["mismatches"]
    assert any(m["reason"] == "RENDERED_INPUT_HASH_MISMATCH" and m["world_id"] == offending_wid for m in mismatches)

    # 2. Wrong backend_id -> fails with BACKEND_MISMATCH
    wrong_backend = json.loads(json.dumps(orig_ref))
    wrong_backend["backend_id"] = "v8"
    _rewrite_result_zip_with_valid_custody(
        bundle_dir,
        {"REPLAY_REFERENCE.json": (json.dumps(wrong_backend, indent=2, sort_keys=True) + "\n").encode("utf-8")},
    )
    out_backend = tmp_path / "wrong_backend_receipt.json"
    with pytest.raises(ValueError, match="BACKEND_MISMATCH"):
        independent_replay_bundle(bundle_dir, expected_source_commit=commit, output_path=out_backend)

    # 3. Legacy non-V8 bundle stripped of REPLAY_REFERENCE.json -> fails closed with INSUFFICIENT_REPLAY_EVIDENCE
    _rewrite_result_zip_with_valid_custody(bundle_dir, {"REPLAY_REFERENCE.json": None})
    out_legacy = tmp_path / "legacy_non_v8_receipt.json"
    with pytest.raises(ValueError, match="INSUFFICIENT_REPLAY_EVIDENCE"):
        independent_replay_bundle(bundle_dir, expected_source_commit=commit, output_path=out_legacy)
    legacy_doc = json.loads(out_legacy.read_text(encoding="utf-8"))
    assert legacy_doc["status"] == "INSUFFICIENT_REPLAY_EVIDENCE"
    assert legacy_doc["verified"] is False
    assert legacy_doc["supported_recovery_path"]


def test_unregistered_plugin_backend_fails_loudly_without_v8_fallback(tmp_path: Path) -> None:
    commit = _head_commit()
    plugin_mod = tmp_path / "dummy_plugin_mod.py"
    plugin_mod.write_text(
        """
from __future__ import annotations
import torch
from anra_v5.planthir.runtime import ExperimentModule, GenericDatasetBundle

class TinyMod(torch.nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.embed = torch.nn.Embedding(32, width)
        self.head = torch.nn.Linear(width, 2)
    def forward(self, x):
        h = self.embed(x % 32).mean(dim=1)
        return {"logits": self.head(h), "hidden": h}

def register_planthir_backends():
    def model_factory(architecture, model_config, seed):
        torch.manual_seed(int(seed))
        return TinyMod(width=int(model_config.get("width", 8)))

    def dataset_factory(dataset_config, cpu_canary):
        n = int(dataset_config.get("num_samples", 8))
        samples = [{"x": [i % 16, (i + 1) % 16], "y": i % 2} for i in range(n)]
        train_worlds = [
            {"world_id": f"tr_{i}", "sig": f"tr_sig_{i}", "public_prompt": f"p {i}", "label": i % 2, "hops": 1}
            for i in range(n)
        ]
        eval_worlds = [
            {"world_id": f"ev_{i}", "sig": f"ev_sig_{i}", "public_prompt": f"p {i + 100}", "label": i % 2, "hops": 1}
            for i in range(4)
        ]
        return {
            "train_data": GenericDatasetBundle(
                items=tuple(samples),
                dataset_name="dummy_plugin_ds",
                data_identity="dummy_id_v1",
                manifest={"split": "canary", "license": "MIT"},
            ),
            "train_worlds": train_worlds,
            "eval_worlds": eval_worlds,
        }

    def loss_fn(model, batch, recipe):
        dev = next(model.parameters()).device
        x = torch.tensor([b["x"] for b in batch], dtype=torch.long, device=dev)
        y = torch.tensor([b["y"] for b in batch], dtype=torch.long, device=dev)
        return torch.nn.functional.cross_entropy(model(x)["logits"], y)

    def enc(obs):
        val = int(obs.rendered_text.splitlines()[0].split()[-1])
        return torch.tensor([[val % 16, (val + 1) % 16]], dtype=torch.long)

    ExperimentModule(
        backend_id="dummy_unreg_plugin",
        architectures=("ctrl", "treat"),
        objectives=("ce",),
        model_factory=model_factory,
        dataset_factory=dataset_factory,
        loss_fn=loss_fn,
        input_encoder=enc,
        action_labels=("A", "B"),
        default_model_config={"width": 8},
        treatment_architecture="treat",
        treatment_model_config={"width": 8},
    ).register()
""",
        encoding="utf-8",
    )

    extra_env = {
        "PYTHONPATH": str(tmp_path) + os.pathsep + str(REPO_ROOT),
        "PLANTHIR_BACKEND_MODULES": "dummy_plugin_mod",
    }
    spec_path = tmp_path / "dummy_spec.json"
    work_root = tmp_path / "dummy_work"
    p_init = _run_cli(["init", "--backend", "dummy_unreg_plugin", "--out", str(spec_path)], env_extra=extra_env)
    assert p_init.returncode == 0, p_init.stdout + p_init.stderr
    doc = json.loads(spec_path.read_text(encoding="utf-8"))
    doc["source_commit"] = commit
    spec_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    p_run = _run_cli(
        ["run", "--spec", str(spec_path), "--work-root", str(work_root), "--canary", "--device", "cpu"],
        env_extra=extra_env,
    )
    assert p_run.returncode == 0, p_run.stdout + p_run.stderr
    bundle_dir = work_root / "result_bundle"

    # With plugin registered -> replay succeeds and matches
    p_ok = _run_cli(
        ["replay-bundle", "--bundle-dir", str(bundle_dir), "--expected-source-commit", commit, "--device", "cpu"],
        env_extra=extra_env,
    )
    assert p_ok.returncode == 0, p_ok.stdout + p_ok.stderr
    ok_doc = json.loads(p_ok.stdout)
    assert ok_doc["verified"] is True
    assert ok_doc["prediction_agreement_status"] == "MATCHED"

    # Without PLANTHIR_BACKEND_MODULES -> must fail loudly with MISSING_BACKEND_PLUGIN
    p_fail = _run_cli(
        ["replay-bundle", "--bundle-dir", str(bundle_dir), "--expected-source-commit", commit, "--device", "cpu"]
    )
    assert p_fail.returncode != 0
    assert "MISSING_BACKEND_PLUGIN:dummy_unreg_plugin" in (p_fail.stdout + p_fail.stderr)


def test_legacy_v8_bundle_without_replay_reference_preserves_compatibility(tmp_path: Path) -> None:
    commit = _head_commit()
    spec_path = tmp_path / "v8_spec.json"
    work_root = tmp_path / "v8_work"
    _create_tiny_spec(spec_path, "v8", commit)

    run = _run_cli(["run", "--spec", str(spec_path), "--work-root", str(work_root), "--canary", "--device", "cpu"])
    assert run.returncode == 0, run.stdout + run.stderr
    bundle_dir = work_root / "result_bundle"
    _rewrite_result_zip_with_valid_custody(bundle_dir, {"REPLAY_REFERENCE.json": None})

    receipt = independent_replay_bundle(bundle_dir, expected_source_commit=commit)
    assert receipt["verified"] is True
    assert receipt["legacy_v8_compatibility"] is True
    assert receipt["supported_recovery_path"]
