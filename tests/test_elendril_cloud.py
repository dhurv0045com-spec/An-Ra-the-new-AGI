import ast
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import pytest

from tools import build_elendril_cloud_notebooks as builder
from tools import elendril_cloud as cloud


def test_notebooks_validate_and_run_all_default_cells_without_side_effects(monkeypatch, tmp_path):
    nbformat = pytest.importorskip("nbformat")

    def forbidden(*args, **kwargs):
        raise AssertionError("default notebook attempted a subprocess or persistent write")

    for platform, filename in builder.FILES.items():
        notebook = json.loads((Path("notebooks") / filename).read_bytes())
        nbformat.validate(nbformat.from_dict(notebook))
        source = notebook["metadata"]["elendril"]["source"]
        assert notebook == builder.build(platform, source)
        assert subprocess.check_output(["git", "cat-file", "-t", source], text=True).strip() == "commit"
        subprocess.run(["git", "cat-file", "-e", f"{source}:tools/elendril_cloud.py"], check=True)
        state = {}
        with monkeypatch.context() as blocked:
            blocked.setattr(subprocess, "run", forbidden)
            blocked.setattr(Path, "mkdir", forbidden)
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    script = ast.parse("".join(cell["source"]), filename=filename)
                    exec(compile(script, filename, "exec"), state)
        assert state["RUN_CANARIES"] is False and state["RUN_CAMPAIGN"] is False
        assert state["STATUS"]["status"] == "DISABLED"


def test_distinct_t4_lanes_and_forecast_matrix_are_frozen():
    gpus = cloud.gpu_rows("0, Tesla T4, GPU-aa00, 15109, 12000, 580\n"
                          "1, Tesla T4, GPU-bb11, 15109, 11000, 580")
    cloud.admit_devices(gpus, 2)
    assert cloud.device_env(gpus[1])["CUDA_VISIBLE_DEVICES"] == "GPU-bb11"
    with pytest.raises(ValueError, match="duplicate"):
        cloud.gpu_rows("0, Tesla T4, GPU-aa00, 15109, 12000, 580", "0,0")
    with pytest.raises(ValueError, match="exactly"):
        cloud.admit_devices(gpus, 1)
    assert len(cloud.PLANS["colab"].arms()) == 16
    assert len(cloud.PLANS["kaggle"].arms()) == 45


def test_transaction_readback_corruption_and_restore_without_training(tmp_path):
    checkpoint = tmp_path / "local"
    checkpoint.mkdir()
    components = []
    for name in (*cloud.RESUME_COMPONENTS, "training_state.json"):
        data = name.encode()
        (checkpoint / name).write_bytes(data)
        components.append({"name": name, "sha256": sha256(data).hexdigest(), "byte_size": len(data)})
    manifest = cloud.blob({"components": components, "state": {"global_update": 1, "lineage_id": "mock"}})
    (checkpoint / "manifest.json").write_bytes(manifest)
    digest = sha256(manifest).hexdigest()
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(cloud.blob({"final_checkpoint_sha256": digest, "lineage": "mock",
                                   "evaluations": [{"update": 1}]}))
    campaign = "a"*64
    destination = tmp_path / "destination"
    cloud.publish_boundary(destination, campaign, receipt, checkpoint)
    boundary = cloud.committed_boundary(destination, campaign)
    assert boundary is not None
    cloud.materialize(boundary, tmp_path / "restored")
    assert (tmp_path / "restored" / "mock" / "LATEST").read_text().strip() == digest
    (boundary[2] / "model.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="component integrity"):
        cloud.committed_boundary(destination, campaign)
    bad = tmp_path / "bad"
    (checkpoint / "optimizer.bin").write_bytes(b"bad")
    with pytest.raises(ValueError, match="component integrity"):
        cloud.publish_boundary(bad, campaign, receipt, checkpoint)
    assert not (bad / "COMMITTED.json").exists()
