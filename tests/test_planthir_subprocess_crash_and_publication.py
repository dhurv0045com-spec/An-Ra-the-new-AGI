"""Stage 2 verification: real subprocess os._exit crash recovery & atomic CURRENT.json publication."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from anra_v5.elendril.v8.data import D1_TRAIN_SEED, build_d1_training_dataset
from anra_v5.planthir import (
    CartridgeArmSpec,
    CartridgeRunner,
    ExperimentCartridge,
)
from anra_v5.planthir.adapters.x1_rd2_adapter import (
    X1RD2EnvironmentKernel,
    X1RD2Recipe,
    X1RD2Spec,
    prepare_x1_rd2_training_data,
)
from anra_v5.planthir.causal import _probe_split_keys
from anra_v5.planthir.evidence import (
    publish_bundle_generation,
    quarantine_partial_generations,
    resolve_published_generation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMIT = "00b2128b34755d332414ffae8eb78dd3f430e04f"


def _tiny_x1_cartridge() -> ExperimentCartridge:
    data, train_worlds = prepare_x1_rd2_training_data(num_groups=4)
    _, forbidden = _probe_split_keys(train_worlds)
    pool = build_d1_training_dataset(seed=D1_TRAIN_SEED + 31, quota_per_family=16).worlds
    disjoint = [w for w in pool if int(w["renderer_template"]) == 0 and not (_probe_split_keys([w])[1] & forbidden)]
    eval_w = tuple(disjoint[:4])
    mh_w = tuple(disjoint[4:6])
    probe_w = tuple(disjoint[6:10])
    spec = X1RD2Spec(architecture="X1-P", hidden_dim=16, num_layers=1, num_heads=2, pointer_dim=8, max_sequence_length=512)
    recipe = X1RD2Recipe(seed=7, total_updates=2, batch_size=8, micro_batch_size=4, lr=1e-3, objective="pair_eq")
    arm = CartridgeArmSpec(
        arm_id="X1P",
        architecture_family="X1-P",
        is_control=True,
        v8_spec=None,
        recipe=recipe,
        dataset_identity=data.data_identity,
        milestone_updates=(1, 2),
        backend_id="x1_rd2",
        model_spec=spec,
    )
    return ExperimentCartridge(
        campaign_id="SUBPROC-CRASH",
        program="X1-RD2",
        source_commit=COMMIT,
        kernel=X1RD2EnvironmentKernel(),
        control_arm=arm,
        treatment_arms=(),
        train_data=data,
        eval_worlds=eval_w,
        multihop_worlds=mh_w,
        probe_train_worlds=probe_w,
        train_worlds=train_worlds,
        backend_id="x1_rd2",
    )


def _run_cartridge_subprocess(
    work_dir: Path,
    *,
    resume: bool = False,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    code = (
        "import json, torch\n"
        "from pathlib import Path\n"
        "from tests.test_planthir_subprocess_crash_and_publication import _tiny_x1_cartridge\n"
        "from anra_v5.planthir import CartridgeRunner\n"
        "torch.set_num_threads(1)\n"
        f"out = CartridgeRunner().run(_tiny_x1_cartridge(), work_root=Path({str(work_dir)!r}), "
        f"checkpoint_interval=1, resume={resume!r})\n"
        "print(json.dumps({'sha': out['arms']['X1P']['checkpoint_sha256'], "
        "'gen': out['bundle_receipt']['generation_id'], "
        "'status': out['bundle_receipt']['status']}))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "fault_point,is_ckpt_component",
    [
        ("after_component:model.bin", True),
        ("after_journal_append", False),
        ("after_checkpoint_commit", False),
        ("publication_stage", False),
        ("publication_pointer", False),
    ],
)
def test_real_subprocess_crash_and_recovery_parity(tmp_path: Path, fault_point: str, is_ckpt_component: bool) -> None:
    ref_dir = tmp_path / "reference"
    ref = _run_cartridge_subprocess(ref_dir, resume=False)
    assert ref.returncode == 0, ref.stderr
    ref_info = json.loads(ref.stdout.strip().splitlines()[-1])

    work_dir = tmp_path / "crashed"
    once_file = tmp_path / f"{fault_point.replace(':', '_')}.once"
    env = {
        "PLANTHIR_FAULT_INJECTION_POINT": fault_point,
        "PLANTHIR_FAULT_ONCE_FILE": str(once_file),
        "PLANTHIR_FAULT_EXIT_CODE": "99",
    }
    if is_ckpt_component:
        env["PLANTHIR_CHECKPOINT_INJECT_CRASH"] = fault_point

    child1 = _run_cartridge_subprocess(work_dir, resume=False, extra_env=env)
    assert child1.returncode == 99, f"expected hard os._exit(99), got rc={child1.returncode}, stderr={child1.stderr}"
    assert once_file.is_file()

    if fault_point == "publication_stage":
        # Partial bundle staging exists while CURRENT.json still resolves prior committed PARTIAL generation cleanly.
        assert any(p.name.startswith(".staging-") for p in (work_dir / "result_bundle").iterdir())
        prior = resolve_published_generation(work_dir / "result_bundle", expected_source_commit=COMMIT, verify=True)
        assert prior["status"] == "PARTIAL"
    elif fault_point == "publication_pointer":
        # Committed COMPLETE generation dir exists under generations/, yet CURRENT.json still resolves prior gen-0001 cleanly.
        assert (work_dir / "result_bundle" / "generations" / "gen-0002").is_dir()
        prior = resolve_published_generation(work_dir / "result_bundle", expected_source_commit=COMMIT, verify=True)
        assert prior["current_generation"] == "gen-0001"
        assert prior["status"] == "PARTIAL"

    # Resume in a fresh child process (even keeping the same once-guarded fault env) completes cleanly and matches reference.
    child2 = _run_cartridge_subprocess(work_dir, resume=True, extra_env=env)
    assert child2.returncode == 0, child2.stderr
    res_info = json.loads(child2.stdout.strip().splitlines()[-1])
    assert res_info["sha"] == ref_info["sha"]
    assert res_info["status"] == "COMPLETE"

    resolved = resolve_published_generation(work_dir / "result_bundle", expected_source_commit=COMMIT, verify=True)
    assert resolved["status"] == "COMPLETE"
    assert resolved["current_generation"] == res_info["gen"]
    if fault_point == "publication_stage":
        qdir = work_dir / "result_bundle" / ".quarantine"
        assert qdir.is_dir() and any(p.name.startswith(".staging-") for p in qdir.iterdir())


def test_corrupt_journal_line_rejected_on_resume(tmp_path: Path) -> None:
    work_dir = tmp_path / "journal_corrupt"
    once_file = tmp_path / "commit.once"
    env = {
        "PLANTHIR_FAULT_INJECTION_POINT": "after_checkpoint_commit",
        "PLANTHIR_FAULT_ONCE_FILE": str(once_file),
        "PLANTHIR_FAULT_EXIT_CODE": "99",
    }
    child1 = _run_cartridge_subprocess(work_dir, resume=False, extra_env=env)
    assert child1.returncode == 99

    journal = work_dir / "updates" / "X1P.jsonl"
    assert journal.is_file()
    # Overwrite committed journal record with malformed JSON to confirm fail-closed journal parsing
    journal.write_text('{"update": 1, "loss": "NOT_JSON_COMPLETE"\n', encoding="utf-8")

    child2 = _run_cartridge_subprocess(work_dir, resume=True)
    assert child2.returncode != 0
    assert "ValueError" in child2.stderr or "JSONDecodeError" in child2.stderr


def test_concurrent_reader_isolation_during_failed_second_generation_publication(tmp_path: Path) -> None:
    cart = _tiny_x1_cartridge()
    work_dir = tmp_path / "reader_isolation"
    out1 = CartridgeRunner().run(cart, work_root=work_dir, checkpoint_interval=1)
    bdir = work_dir / "result_bundle"
    gen1 = resolve_published_generation(bdir, expected_source_commit=COMMIT, verify=True)
    assert gen1["current_generation"] == out1["bundle_receipt"]["generation_id"]

    # Hold open an archive from gen-0001 just like a concurrent reader process while gen-0002 fails verification
    gen1_zip = Path(gen1["generation_dir"]) / "result.zip"
    with zipfile.ZipFile(gen1_zip, "r") as reader_zf:
        before_status = json.loads(reader_zf.read("FINAL_STATUS.json").decode("utf-8"))
        assert before_status["status"] == "COMPLETE"

        def fail_verify(*args, **kwargs):
            raise ValueError("simulated failed verification during gen-0002")

        with pytest.raises(ValueError, match="failed verification"):
            publish_bundle_generation(
                bdir,
                campaign_id=cart.campaign_id,
                source_commit=COMMIT,
                status="COMPLETE",
                experiment_bundles={},
                verify_fn=fail_verify,
            )

        # Gen-0001 pointer & files remain completely intact and valid for both open and fresh readers
        gen_after = resolve_published_generation(bdir, expected_source_commit=COMMIT, verify=True)
        assert gen_after["current_generation"] == gen1["current_generation"]
        assert gen_after["result_manifest_sha256"] == gen1["result_manifest_sha256"]
        assert (bdir / ".quarantine").is_dir()

    # Manually plant an interrupted staging directory and confirm quarantine cleans it without altering CURRENT.json
    leftover = bdir / ".staging-gen-9999"
    leftover.mkdir()
    (leftover / "PARTIAL_GARBAGE").write_text("corrupt", encoding="utf-8")
    quarantined = quarantine_partial_generations(bdir)
    assert quarantined == [".staging-gen-9999"]
    assert not leftover.exists()
    gen_final = resolve_published_generation(bdir, expected_source_commit=COMMIT, verify=True)
    assert gen_final["current_generation"] == gen1["current_generation"]
