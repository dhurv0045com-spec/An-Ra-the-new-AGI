"""Replay the frozen Y4 byte-to-graph interface on paired native and re-rendered worlds."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import torch

from anra_v5.elendril.representation import SpecialToken, UTF8
from experiments.avendi_y4.models import (
    AvendiY4ByteModel, Y4TypedCompositionModel, decode_y4_byte_continuations,
    encode_y4_typed_batch, get_y4_byte_spec, parse_bytes_with_model,
)
from experiments.avendi_y4.plan import STREAM_SEEDS
from experiments.avendi_y4.tasks import (
    ENTITY_64, allocate_y4_disjoint_pools, build_functional_graph_families,
    parse_public_byte_observation,
)

SEEDS = (7, 19, 41, 73, 101)
CPU = torch.device("cpu")
ANS = UTF8.special(SpecialToken.ANSWER)


def load_model(archive, job_id, model):
    root = f"ckpts/y4-{job_id}/"
    object_id = archive.read(root + "LATEST").decode().strip()
    receipt = json.loads(archive.read(f"jobs/{job_id}/receipt.json"))
    root += f"objects/{object_id}/"
    manifest = archive.read(root + "manifest.json")
    assert sha256(manifest).hexdigest() == receipt["result"]["train"]["checkpoint_sha256"] == object_id
    metadata = json.loads(manifest)
    assert metadata["state"]["global_update"] == receipt["completed_updates"] == 1024
    weights = archive.read(root + "model.bin")
    component = next(c for c in metadata["components"] if c["name"] == "model.bin")
    assert sha256(weights).hexdigest() == component["sha256"] and len(weights) == component["byte_size"]
    model.load_state_dict(torch.load(BytesIO(weights), map_location="cpu", weights_only=True), strict=True)
    model.eval()
    return object_id


def make_families(n):
    # Exclude the endpoint/key quadruples furnished to the byte model and the C train/confirm generators.
    historical = allocate_y4_disjoint_pools(
        {"train": 1024, "dev": 256, "confirm": 256, "transfer": 256, "refresh": 1024}, STREAM_SEEDS["corpus"]
    )["pools"]
    used = {frozenset(roles) for split in historical.values() for roles in split}
    for size, seed in [(1024, STREAM_SEEDS["corpus"] + 400), (256, STREAM_SEEDS["confirm"] + 410)]:
        used.update(frozenset(roles) for roles in allocate_y4_disjoint_pools({"x": size}, seed)["pools"]["x"])
    roles = []
    for block in allocate_y4_disjoint_pools({"fresh": n * 4 + 1024}, 261126)["pools"]["fresh"]:
        key = frozenset(block)
        if key not in used:
            roles.append(block)
            used.add(key)
        if len(roles) == n * 4:
            break
    assert len(roles) == n * 4
    fs = build_functional_graph_families("bridge_261126", n * 4, 4261126, role_blocks=roles)
    return fs


def prefix(cell):
    ids = UTF8.frame(cell["public_prompt"], "").ids
    return list(ids[:ids.index(ANS) + 1])


def raw_batch(model, cells):
    ids = [prefix(c) for c in cells]
    assert len({len(row) for row in ids}) == 1
    batch = torch.tensor(ids)
    outputs = [[] for _ in ids]
    for _ in range(4):
        nxt = model.forward_tokens(batch)[0][:, -1].argmax(-1)
        for row, token in zip(outputs, nxt.tolist()):
            if not row or row[-1] != UTF8.eos_id:
                row.append(token)
        if all(row[-1] == UTF8.eos_id for row in outputs):
            break
        batch = torch.cat((batch, nxt[:, None]), dim=1)
    return outputs


def symbolic(graph):
    edges = dict(graph["rows"])
    node = graph["query"]
    for _ in range(graph["depth"]):
        node = edges.get(node)
        if node is None:
            break
    return node


def usable(graph):
    rows = graph["rows"]
    return (graph["query"] in ENTITY_64 and bool(rows) and len(dict(rows)) == len(rows)
            and all(k in ENTITY_64 and v in ENTITY_64 for k, v in rows))


def learned_endpoints(model, graphs, depth):
    result = [None] * len(graphs)
    indices = [i for i, g in enumerate(graphs) if usable(g)]
    if indices:
        # Encoder expects a scoring label; the dummy value never reaches forward_trace.
        rows = [dict(graphs[i], gold=ENTITY_64[0]) for i in indices]
        keys, values, queries, mask, _ = encode_y4_typed_batch(rows, CPU)
        prediction = model.forward_trace(keys, values, queries, mask, steps=depth)["final_logits"].argmax(-1)
        for i, value in zip(indices, prediction.tolist()):
            result[i] = ENTITY_64[value]
    return result


def compact_family_masks(result):
    """Each byte records the eight case outcomes, least significant bit first."""
    keys = list(next(iter(result["seeds"].values()))["results"]["native"]["family_masks"])
    result["family_ids"] = keys
    result["family_mask_encoding"] = "One hex byte per family_id. Bit i is success on cell i."
    result["environment"] = {"python": sys.version.split()[0], "torch": str(torch.__version__),
                             "device": "cpu", "threads": torch.get_num_threads()}
    for record in result["seeds"].values():
        for panel in record["results"].values():
            masks = panel.pop("family_masks")
            assert list(masks) == keys
            panel["case_masks_hex"] = {
                metric: bytes(masks[k][metric] for k in keys).hex() for metric in masks[keys[0]]
            }
    return result


def historical_parity(z, model, seed):
    pool = allocate_y4_disjoint_pools({"native": 256}, STREAM_SEEDS["dev"] + 50)["pools"]["native"]
    fs = build_functional_graph_families("dev_native", 4, STREAM_SEEDS["dev"], role_blocks=pool[:4])
    job = f"A-pointer-s{seed}"
    cases = json.loads(z.read(f"jobs/{job}/receipt.json"))["result"]["eval_dev"]["native"]["cases"]
    with np.load(BytesIO(z.read(f"jobs/{job}/dev_native_first_logits.npz")), allow_pickle=False) as arr:
        logits = dict(zip(arr["ids"].tolist(), arr["logits"]))
    err = 0.0
    exact = top = 0
    for f in fs:
        for cell in f["cells"]:
            cid = cell["world_id"]
            pred = decode_y4_byte_continuations(model, prefix(cell), cell["gold"], cell["rival"])
            scores = pred["first_logits_260"]
            err = max(err, float(np.abs(scores - logits[cid]).max()))
            top += int(scores.argmax() == logits[cid].argmax())
            exact += int(pred["exact_eos"] == cases[cid][0])
    return {"cases": 32, "first_argmax_equal": top, "exact_flag_equal": exact, "max_abs_logit_diff": err}


@torch.inference_mode()
def run(z, n):
    families = make_families(n)
    result = {"family_seed": 4261126, "role_pool_seed": 261126, "families_per_depth": n,
              "seeds": {}, "panels": ["native", "brackets"]}
    for seed in SEEDS:
        byte = AvendiY4ByteModel(get_y4_byte_spec("S3", cpu_canary=False), seed=seed, use_aux=True)
        executors = {arm: Y4TypedCompositionModel("categorical", 128, seed=seed)
                     for arm in ("true_step", "true_step_iit")}
        sources = {"pointer": load_model(z, f"A-pointer-s{seed}", byte)}
        for arm, model in executors.items():
            sources[arm] = load_model(z, f"C-categorical_{arm}-s{seed}", model)
        record = {"checkpoints": sources, "historical_replay": historical_parity(z, byte, seed), "results": {}}
        for panel in result["panels"]:
            depth_stats = defaultdict(Counter)
            family_masks = {}
            examples = {}
            for family in families:
                cells = []
                for cell in family["cells"]:
                    c = dict(cell)
                    if panel == "brackets":
                        body = " ".join(f"[{k}=>{v}]" for k, v in c["rows"])
                        c["public_prompt"] = f"ROWS: {body} || STEP={c['depth']} Q:{c['query']}"
                    cells.append(c)
                depth = family["depth"]
                oracle = [parse_public_byte_observation(prefix(c)) for c in cells]
                assert all(symbolic(g) == c["gold"] for g, c in zip(oracle, cells))
                parsed = [parse_bytes_with_model(byte, dict(c, gold="")) for c in cells]
                raw = raw_batch(byte, cells)
                if depth_stats[depth]["families"] == 0:
                    for c, ids in zip(cells, raw):
                        ref = decode_y4_byte_continuations(byte, prefix(c), c["gold"], c["rival"])
                        assert ids == ref["generated_ids"], "Batched raw decoder differs from reference"
                endpoints = {"oracle_oracle": [symbolic(g) for g in oracle],
                             "learned_oracle": [symbolic(g) if usable(g) else None for g in parsed]}
                for arm, model in executors.items():
                    endpoints["oracle_" + arm] = learned_endpoints(model, oracle, depth)
                    endpoints["learned_" + arm] = learned_endpoints(model, parsed, depth)
                masks = defaultdict(int)
                counts = depth_stats[depth]
                counts["families"] += 1
                for i, (c, g, o, ids) in enumerate(zip(cells, parsed, oracle, raw)):
                    counts["cases"] += 1
                    events = {k: v[i] == c["gold"] for k, v in endpoints.items()}
                    events.update(parse_exact=g["exact_parse"], query_exact=g["query"] == o["query"],
                                  rows_exact=g["rows"] == o["rows"], parse_usable=usable(g),
                                  raw=ids == [*UTF8.encode(c["gold"]), UTF8.eos_id])
                    for k, passed in events.items():
                        counts[k] += int(passed)
                        masks[k] |= int(passed) << i
                    if not events["learned_true_step"] or not events["raw"]:
                        label = ("invalid_parse" if not usable(g) else "parsed_path_error" if not events["learned_oracle"]
                                 else "executor_error" if not events["learned_true_step"] else "direct_decoder_only")
                        if label not in examples:
                            examples[label] = {"id": c["world_id"], "prompt": c["public_prompt"], "expected": c["gold"],
                                               "parsed_rows": g["rows"], "parsed_query": g["query"], "raw_ids": ids,
                                               "raw_text": UTF8.decode_lossy(ids, strip_special=True)[0],
                                               "bridge_endpoint": endpoints["learned_true_step"][i]}
                for k, mask in masks.items():
                    counts[k + "_complete_families"] += int(mask == 255)
                family_masks[family["family_id"]] = dict(masks)
            record["results"][panel] = {"depths": dict(depth_stats), "examples": examples, "family_masks": family_masks}
        result["seeds"][str(seed)] = record
        print("completed seed", seed, record["historical_replay"], flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--families-per-depth", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    with ZipFile(args.archive) as archive:
        evidence = run(archive, args.families_per_depth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compact_family_masks(evidence), separators=(",", ":")) + "\n", encoding="utf-8")
