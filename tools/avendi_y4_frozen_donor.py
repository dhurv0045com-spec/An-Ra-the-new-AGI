"""Replay frozen Y4 categorical donor interventions on generated graph pairs.

Run: python -m tools.avendi_y4_frozen_donor FULL.zip output.json
"""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import torch
import torch.nn.functional as F

from experiments.avendi_y4.models import ENT_TO_IDX, Y4TypedCompositionModel, encode_y4_typed_batch
from experiments.avendi_y4.tasks import build_state_swap_pairs, solve_recipient_suffix
from tools.avendi_y4_frozen_bridge import load_model

SEEDS = (7, 19, 41, 73, 101)
FIXTURE_SEED = 98271
PAIRS_PER_DEPTH = 256


def encode(pairs, rows, query, gold):
    cells = [{"rows": p[rows], "query": p[query], "gold": p[gold]} for p in pairs]
    return encode_y4_typed_batch(cells, torch.device("cpu"))


def prepare_pairs(depth):
    pairs = build_state_swap_pairs(PAIRS_PER_DEPTH, FIXTURE_SEED, depth=depth, patch_step=1)
    for pair in pairs:
        pair["original_donor_rows"] = pair["donor_rows"]
        old = pair["wrong_donor_state"]
        new = pair["donor_step"]
        assert solve_recipient_suffix(pair["original_donor_rows"], pair["donor_query"], 1) == old
        # Rename the intermediate throughout the donor graph, preserving its entire chain.
        pair["donor_rows"] = [(new if k == old else k, new if v == old else v)
                              for k, v in pair["original_donor_rows"]]
        assert solve_recipient_suffix(pair["donor_rows"], pair["donor_query"], 1) == new
        assert solve_recipient_suffix(pair["donor_rows"], pair["donor_query"], depth) == pair["donor_own_answer"]
        assert solve_recipient_suffix(pair["original_donor_rows"], pair["donor_query"], depth) == pair["donor_own_answer"]
        assert solve_recipient_suffix(pair["recipient_rows"], pair["query"], depth) == pair["recipient_answer"]
        assert solve_recipient_suffix(pair["recipient_rows"], new, depth - 1) == pair["donor_on_recipient_answer"]
        assert solve_recipient_suffix(pair["rewired_rows"], new, depth - 1) == pair["rewired_donor_on_recipient_answer"]
        assert solve_recipient_suffix(pair["wrong_donor_rows"], old, depth - 1) == pair["wrong_donor_actual_answer"]
        assert pair["wrong_donor_actual_answer"] != pair["donor_on_recipient_answer"]
    return pairs


def run(model, depth):
    pairs = prepare_pairs(depth)
    dr, dv, dq, dm, _ = encode(pairs, "donor_rows", "donor_query", "donor_own_answer")
    odr, odv, odq, odm, _ = encode(pairs, "original_donor_rows", "donor_query", "donor_own_answer")
    rr, rv, rq, rm, rg = encode(pairs, "recipient_rows", "query", "recipient_answer")
    wr, wv, wq, wm, wg = encode(pairs, "rewired_rows", "query", "rewired_donor_on_recipient_answer")
    xr, xv, xq, xm, xg = encode(pairs, "wrong_donor_rows", "query", "wrong_donor_actual_answer")
    target = torch.tensor([ENT_TO_IDX[p["donor_on_recipient_answer"]] for p in pairs])
    state_target = torch.tensor([ENT_TO_IDX[p["donor_step"]] for p in pairs])
    wrong_state_target = torch.tensor([ENT_TO_IDX[p["wrong_donor_state"]] for p in pairs])

    base = model.forward_trace(rr, rv, rq, rm, steps=depth)
    donor = model.forward_trace(dr, dv, dq, dm, steps=1)["states"][0]
    wrong_donor = model.forward_trace(odr, odv, odq, odm, steps=1)["states"][0]
    oracle = F.one_hot(state_target, num_classes=64).float()

    def predict(r, v, q, mask, state):
        return model.forward_trace(r, v, q, mask, steps=depth, patch_at=1,
                                   patch_tensor=state)["final_logits"].argmax(-1)

    baseline = base["final_logits"].argmax(-1)
    identity = predict(rr, rv, rq, rm, base["states"][0])
    learned = predict(rr, rv, rq, rm, donor)
    oracle_prediction = predict(rr, rv, rq, rm, oracle)
    rewired = predict(wr, wv, wq, wm, donor)
    oracle_rewired = predict(wr, wv, wq, wm, oracle)
    wrong_prediction = predict(xr, xv, xq, xm, wrong_donor)

    def count(value):
        return int(value.sum())

    return {
        "n": len(pairs),
        "unpatched_correct": count(baseline == rg),
        "identity_correct": count(identity == rg),
        "identity_same_prediction": count(identity == baseline),
        "donor_state_argmax_correct": count(donor.argmax(-1) == state_target),
        "mean_donor_target_probability": float(donor.gather(1, state_target[:, None]).mean()),
        "learned_donor_correct": count(learned == target),
        "oracle_donor_correct": count(oracle_prediction == target),
        "learned_rewire_correct": count(rewired == wg),
        "oracle_rewire_correct": count(oracle_rewired == wg),
        "learned_both_suffixes_correct": count((learned == target) & (rewired == wg)),
        "oracle_both_suffixes_correct": count((oracle_prediction == target) & (oracle_rewired == wg)),
        "learned_matches_oracle_prediction": count(learned == oracle_prediction),
        "wrong_donor_state_argmax_correct": count(wrong_donor.argmax(-1) == wrong_state_target),
        "wrong_donor_actual_suffix_correct": count(wrong_prediction == xg),
        "wrong_donor_intended_target": count(wrong_prediction == target),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = {}
    with ZipFile(args.archive) as archive, torch.inference_mode():
        for arm in ("true_step", "true_step_iit"):
            for seed in SEEDS:
                job = f"C-categorical_{arm}-s{seed}"
                model = Y4TypedCompositionModel("categorical", 128, seed=seed)
                checkpoint = load_model(archive, job, model)
                results[f"{arm}-s{seed}"] = {
                    "checkpoint_manifest_sha256": checkpoint,
                    "depths": {str(depth): run(model, depth) for depth in (2, 3, 4)},
                }
    report = {
        "schema": "y4-frozen-donor-replay-v2",
        "fixture_seed": FIXTURE_SEED,
        "pairs_per_depth": PAIRS_PER_DEPTH,
        "patch_step": 1,
        "arms": results,
        "fixture_correction": "The original donor first hop yields wrong_donor_state, not donor_step. Rename that intermediate in all donor rows; full donor answer is preserved. Original donor state is used for the wrong-donor control.",
        "interpretation": "Public parsed graph IDs and supplied depth. Donor is actual soft first-hop state; oracle is one-hot truth. Generated exploratory data, not preregistered confirmation; no byte parser or EOS evaluation.",
        "checkpoint_integrity": "Shared loader checks LATEST, receipt binding, manifest hash, update, model size and hash, then safely loads weights only. Archive source provenance not independently established.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
