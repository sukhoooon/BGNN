"""Plot new-run summary statistics, not reconstructed historical paper panels."""
import argparse
import json
from pathlib import Path

METRIC_LABELS = {"mae": "MAE", "sigma_y": r"$\sigma_y$", "sigma_w": r"$\sigma_w$",
                 "lpas": "LPAS", "ece": "ECE", "duc": "DUC"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output exists; choose a new file")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    summary = json.loads(args.summary.read_text())
    if summary.get("schema_version") != 2:
        raise ValueError("Expected a schema-2 summary; keep historical plots separate")
    names = list(summary["mean"])
    fig, axes = plt.subplots(1, len(names), figsize=(2.3 * len(names), 3), squeeze=False)
    for ax, name in zip(axes[0], names):
        ax.bar([0], [summary["mean"][name]], yerr=[summary["std_ddof0"][name]],
               width=0.6, capsize=4, color="#417c87")
        ax.set_title(METRIC_LABELS.get(name, name.replace("_", " ")), fontsize=10)
        ax.set_xticks([])
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"New-run summary (n={summary['n_runs']}; LPAS: {summary['alignment_convention']}; "
                 f"ECE: {summary['ece_convention']})", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
