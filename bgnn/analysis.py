"""Aggregate test-set metrics across all configured runs."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import __version__
from .metrics import ECE_LEVELS, latent_posterior_alignment_score, density_uncertainty_criterion, regression_ece
from .runtime import require_complete_runs, write_json


def summarize(root, *, duc=False):
    if not isinstance(duc, bool):
        raise ValueError("duc must be a boolean")
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
                raise ValueError(f"Expected schema 2 independent-sampling predictions in {folder.name}")
            values = {"mae": float(np.mean(np.abs(a["y"] - a["y_mean"]))),
                      "sigma_y": float(a["sigma_y"].mean()),
                      "sigma_w": float(a["sigma_w"].mean()),
                      "lpas": latent_posterior_alignment_score(a["z"], a["sigma_w"]),
                      "ece": regression_ece(a["y"], a["y_mean"], a["sigma_y"])}
            if duc:
                values["duc"] = density_uncertainty_criterion(a["z"], a["sigma_y"])
            if not all(np.isfinite(v) for v in values.values()):
                raise ValueError(f"Non-finite metrics in {folder.name}")
            metrics.append(values)
    require_complete_runs(ids, experiment["config"]["n_runs"])
    return {"n_runs": len(ids), "run_ids": ids, "package_version": __version__,
            "schema_version": 2, "posterior_sampling": "independent",
            "metric_definitions": {"ece_levels": ECE_LEVELS.tolist(),
                                   "lpas_aggregation": "mean_of_normalized_molecules",
                                   "duc": {"density": "negative_kde_log_density", "bandwidth": 0.2,
                                           "mi_random_state": 0} if duc else None},
            "per_run": metrics,
            "mean": {k: float(np.mean([r[k] for r in metrics])) for k in metrics[0]},
            "std_ddof0": {k: float(np.std([r[k] for r in metrics], ddof=0)) for k in metrics[0]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--duc", action="store_true",
                        help="Include DUC using negative KDE log density; omitted by default")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output exists; choose a new file")
    write_json(args.output, summarize(args.experiment, duc=args.duc))


if __name__ == "__main__":
    main()
