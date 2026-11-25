"""Factor the Y4 renderer shift into three public punctuation edits."""
import argparse
from itertools import product
import json
from pathlib import Path
from zipfile import ZipFile

import torch

from experiments.avendi_y4.models import AvendiY4ByteModel, get_y4_byte_spec, parse_bytes_with_model
from tools.avendi_y4_frozen_bridge import SEEDS, load_model, make_families


@torch.inference_mode()
def run(archive):
    # This nested sample is exploratory. Select the first four families per depth in the bridge.
    families = make_families(32)[:16]
    scores = {}
    for seed in SEEDS:
        model = AvendiY4ByteModel(get_y4_byte_spec("S3", cpu_canary=False), seed=seed, use_aux=True)
        load_model(archive, f"A-pointer-s{seed}", model)
        by_render = {}
        for fat_arrow, wrap_brackets, remove_semicolons in product((False, True), repeat=3):
            label = f"arrow={int(fat_arrow)},brackets={int(wrap_brackets)},space_only={int(remove_semicolons)}"
            arrow = "=>" if fat_arrow else "->"
            separator = " " if remove_semicolons else " ; "
            exact = [0] * 4
            complete = [0] * 4
            for family in families:
                depth = family["depth"]
                hits = []
                for cell in family["cells"]:
                    edges = [key + arrow + value for key, value in cell["rows"]]
                    if wrap_brackets:
                        edges = [f"[{edge}]" for edge in edges]
                    prompt = f"ROWS: {separator.join(edges)} || STEP={depth} Q:{cell['query']}"
                    parsed = parse_bytes_with_model(model, {"public_prompt": prompt, "gold": ""})
                    hits.append(parsed["exact_parse"])
                exact[depth - 1] += sum(hits)
                complete[depth - 1] += int(all(hits))
            by_render[label] = {"exact_by_depth": exact, "complete_families_by_depth": complete}
        scores[str(seed)] = by_render
    return {"cases_per_seed_render_depth": 32, "families_per_seed_render_depth": 4, "scores": scores}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with ZipFile(args.archive) as archive:
        result = run(archive)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
