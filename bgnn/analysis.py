"""Aggregate complete run sets. No test-based coefficient or run filtering."""
import argparse
import json
from pathlib import Path

import numpy as np

from .metrics import latent_posterior_alignment_score, density_uncertainty_criterion, regression_ece
from .runtime import require_complete_runs, write_json


def summarize(root, convention="paper", duc=None, *, alignment_convention="paper"):
    if duc not in (None, "paper"):
        raise ValueError("DUC convention must be paper or omitted")
    experiment = json.loads((root / "experiment.json").read_text())
    ids, metrics = [], []
    for folder in sorted(root.glob("run_*")):
        marker = folder / "completed.json"
        if not marker.exists():
            raise ValueError(f"Incomplete run: {folder.name}")
        ids.append(json.loads(marker.read_text())["run_id"])
        with np.load(folder / "predictions_test.npz", allow_pickle=False) as a:
            if ("schema_version" not in a or a["schema_version"].item() != 2
                    or "posterior_sampling" not in a or a["posterior_sampling"].item() != "independent"):
                raise ValueError(f"Expected schema 2 independent-sampling predictions in {folder.name}; keep old archives separate")
            values = {"mae": float(np.mean(np.abs(a["y"] - a["y_mean"]))),
                      "sigma_y": float(a["sigma_y"].mean()),
                      "sigma_w": float(a["sigma_w"].mean()),
                      "lpas": latent_posterior_alignment_score(a["z"], a["sigma_w"], alignment_convention),
                      "ece": regression_ece(a["y"], a["y_mean"], a["sigma_y"], convention)}
            if duc:
                values["duc"] = density_uncertainty_criterion(a["z"], a["sigma_y"], duc)
            if not all(np.isfinite(v) for v in values.values()):
                raise ValueError(f"Non-finite metrics in {folder.name}")
            metrics.append(values)
    require_complete_runs(ids, experiment["config"]["n_runs"])
    return {"n_runs": len(ids), "run_ids": ids, "convention": convention, "duc_convention": duc,
            "schema_version": 2, "posterior_sampling": "independent",
            "ece_convention": convention, "alignment_convention": alignment_convention,
            "historical_paper_reproduction": False, "per_run": metrics,
            "mean": {k: float(np.mean([r[k] for r in metrics])) for k in metrics[0]},
            "std_ddof0": {k: float(np.std([r[k] for r in metrics], ddof=0)) for k in metrics[0]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--convention", choices=("source", "paper"), default="paper",
                        help="ECE confidence-grid convention only (default: paper)")
    parser.add_argument("--alignment-convention", choices=("source", "paper"), default="paper",
                        help="LPAS aggregation (default: paper; source is an archive diagnostic)")
    parser.add_argument("--duc", choices=("paper",),
                        help="Include DUC using negative KDE log density; omitted by default")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output exists; choose a new file")
    write_json(args.output, summarize(args.experiment, args.convention, args.duc,
                                      alignment_convention=args.alignment_convention))


if __name__ == "__main__":
    main()
