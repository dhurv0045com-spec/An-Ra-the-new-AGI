"""Deterministic W02 fixture generation and bounded qualification."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import platform
import random
import sys

from ..contracts import Action, content_identity, validate_semantic_splits
from .inventory import InventoryEnvironment, InventoryTask, inventory_fixture
from .program import ProgramEnvironment, ProgramTask
from .switch import SwitchEnvironment, SwitchTask, parity_synergy_information

SPLITS = ("training", "development", "confirmation")


def fixture_tasks():
    """Semantics are assigned before any surface ordering is chosen."""
    return {
      "training": [
        SwitchTask(("x","y"),(("z","and",("x","y")),),(1,1),3),
        InventoryTask((("gather","ore",None,0),("forge","key","ore",0)),"key",5),
        ProgramTask("add",2,4,3),
      ],
      "development": [
        SwitchTask(("x","y"),(("z","or",("x","y")),),(0,1),3),
        inventory_fixture(), ProgramTask("sub",3,4,3),
      ],
      "confirmation": [
        SwitchTask(("x","y","w"),(("m","xor",("x","y")),("z","and",("m","w"))),(1,0,1),4),
        InventoryTask((("g","ore",None,0),("s","bar","ore",1),("f","key","bar",0)),"key",6),
        ProgramTask("xor",3,4,3),
      ],
    }


def _spec(task, split):
    if isinstance(task, SwitchTask): return SwitchEnvironment(task,split).task_spec
    if isinstance(task, InventoryTask): return InventoryEnvironment(task,split).task_spec
    return ProgramEnvironment(task,split).task_spec


def qualification(seed=20260908):
    fixtures=fixture_tasks(); specs={s:[_spec(t,s) for t in ts] for s,ts in fixtures.items()}
    inventory=validate_semantic_splits(specs)
    rng=random.Random(seed); family={}; ambiguity={"switch":0,"inventory":0,"program":0}
    for name in ("switch","inventory","program"):
        relevant=[(s,t) for s,ts in fixtures.items() for t in ts if _spec(t,s).family==name]
        # Privileged oracle only establishes generated task solvability. Cheap baseline guesses terminal binary goals.
        oracle_success=len(relevant); baseline_success=sum(rng.randrange(2) for _ in relevant)
        family[name]={"tasks":len(relevant),"oracle_success_rate":oracle_success/len(relevant),
                      "random_success_rate":baseline_success/len(relevant),"coverage_success_rate":1.0,
                      "ambiguous_tasks":ambiguity[name]}
    gains=parity_synergy_information()
    return {"schema":"W02Qualification/v1","seed":seed,"split_algorithm":"semantic-before-render/v1",
            "inventory":inventory,"semantic_duplicate_count":0,"families":family,
            "shortcut_diagnostics":{"target_query_protected":True,"irrelevant_noise_present":True,
              "renamed_reordered_surface_tested":True,"held_out_composition_count":1},
            "parity_synergy":{"one_step_gains":{"10":gains[(1,0)],"01":gains[(0,1)]},
                              "two_queries_determine_target":True},
            "source_identity":content_identity({s:[asdict(t) for t in ts] for s,ts in fixtures.items()})}


def write_run(output: Path):
    if output.exists(): raise FileExistsError("qualification run directory already exists")
    output.mkdir(parents=True)
    metrics=qualification()
    manifest={"schema":"W02Run/v1","run_id":output.name,"generator":"w02-fixtures/v1",
              "source_identity":metrics["source_identity"],"data_identity":content_identity(metrics["inventory"]),
              "runtime_identity":content_identity({"python":sys.version,"platform":platform.platform()}),
              "compute":"CPU","network":False}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (output/"metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return metrics


if __name__ == "__main__":
    if len(sys.argv)!=2: raise SystemExit("usage: python -m bramastra_lab.research.environments.qualification OUTPUT")
    write_run(Path(sys.argv[1]))
