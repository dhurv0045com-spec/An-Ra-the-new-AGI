"""Render the committed correctness evidence; does not run a model."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, default=Path("artifacts/elendril/foundation-v5"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite an existing plot")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    learning = json.loads((args.receipts / "learning-canary.json").read_bytes())
    controls = json.loads((args.receipts / "action-controls.json").read_bytes())["splits"]["validation"]["controls"]
    names = ("fixed", "always-query", "no-query", "random", "recency")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), constrained_layout=True)
    axes[0].plot([r["update"] for r in learning["updates"]], [r["loss"] for r in learning["updates"]],
                 marker="o", color="#275d8a")
    axes[0].set(xlabel="Optimizer update", ylabel="Answer + EOS cross-entropy",
                title="Tiny overfit control (4 prompts)\n4/4 training QUERY+EOS; no holdout")
    axes[0].set_ylim(bottom=0)
    for axis, metric, title, ylabel in (
        (axes[1], "correct", "Engineered validation controls", "Correct episodes / 10"),
        (axes[2], "requests", "Same 10 validation worlds", "Charged evidence requests"),
    ):
        values = [controls[name]["totals"]["all"][metric] for name in names]
        bars = axis.bar(names, values, color=["#456f5d", "#789d86", "#c08c4b", "#a1a1a1", "#777777"])
        axis.bar_label(bars)
        axis.set(title=title, ylabel=ylabel, ylim=(0, 11))
        axis.tick_params(axis="x", labelrotation=35)
    fig.suptitle("ELENDRIL V5 development mechanics — no learned action-triage or transfer claim", fontsize=12)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
