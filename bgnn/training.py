"""Train seeded GNN/BGNN runs and save checkpoints, predictions and metadata."""
import argparse
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .artifacts import save_model
from .data import load_dataset, paper_batch_size, split_indices, subset
from .features import PROPERTIES
from .models import BGNN_unary
from .runtime import run_ids, seed_everything, sha256, write_json


def validate_config(config):
    if config["profile"] != "independent_sampling":
        raise ValueError("Expected profile: independent_sampling")
    list(run_ids(config["n_runs"]))
    if config["property"] not in PROPERTIES:
        raise ValueError("Unsupported property")
    if config["fit"]["n_rand"] < 2 or config["fit"]["n_epoch"] < 1:
        raise ValueError("Need >=2 MC draws and >=1 epoch")
    if config["fit"]["resume"] or not config["fit"]["scaling"]:
        raise ValueError("Resume and unscaled training are not supported by this driver")
    if config["fit"]["z_regularize_opt"] not in (None, "L1", "L2", "anti_alignment", "agl"):
        raise ValueError("Unknown regularization")
    if config["fit"]["z_regularize_opt"] is None and config["fit"]["scale_loss"] != 1:
        raise ValueError("Baseline training requires scale_loss=1")


def train_one(graphs, config, run_id, destination, device):
    seed = config["seed_start"] + run_id - 1
    indices = split_indices(len(graphs), seed, **config["split"])
    datasets = subset(graphs, indices)
    write_json(destination / "split.json", indices)
    seed_everything(seed)
    fit = deepcopy(config["fit"])
    if fit["n_batch"] is None:
        fit["n_batch"] = paper_batch_size(len(graphs))
    common = dict(prop=config["property"], conv_model="GIN", num_node_features=37,
                  num_edge_features=7, device=device, **config["architecture"])
    if fit["warmstart"]:
        gnn = BGNN_unary(**common, bayes=False, dropout=config["gnn_dropout"])
        pretrain = dict(fit, warmstart=False, z_regularize_opt=None, scale_loss=1000)
        gnn_logs = gnn.fit(datasets, **pretrain)
        save_model(gnn, destination / "checkpoint_GNN.pt")
        np.savez_compressed(destination / "gnn_log.npz", **gnn_logs)
        del gnn
    model = BGNN_unary(**common, bayes=True, dropout=config["bgnn_dropout"])
    logs = model.fit(datasets, checkpointdir=str(destination / "checkpoint_BGNN.pt"), **fit)
    save_model(model, destination / "checkpoint_BGNN.pt")
    np.savez_compressed(destination / "log.npz", **logs)
    for split in ("train", "validate", "test"):
        predictions = np.asarray(model.predict(datasets[split])).reshape(len(datasets[split]), -1)
        np.savez_compressed(destination / f"predictions_{split}.npz",
                            schema_version=2, posterior_sampling=model.posterior_sampling,
                            row_ids=indices[split], y=np.array([float(g.y) for g in datasets[split]]),
                            y_mean=predictions.mean(axis=1), sigma_y=predictions.std(axis=1, ddof=0),
                            z=model.z.detach().cpu().numpy(),
                            mu_w=model.W_mu.detach().cpu().numpy().ravel(),
                            sigma_w=torch.log(1 + torch.exp(model.W_rho)).detach().cpu().numpy().ravel())
    write_json(destination / "completed.json", {"run_id": run_id, "seed": seed, "fit": fit,
                                               "schema_version": 2,
                                               "posterior_sampling": model.posterior_sampling})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    validate_config(config)
    if args.dry_run:
        print(json.dumps({"run_ids": list(run_ids(config["n_runs"])), "config": config}, indent=2))
        return
    if args.data is None or args.output is None:
        parser.error("--data and --output are required for training")
    if args.output.exists():
        parser.error("Output already exists; choose a new directory to protect previous runs")
    graphs = load_dataset(args.data)
    # Check the smallest split before creating any output.
    split_indices(len(graphs), config["seed_start"], **config["split"])
    args.output.mkdir(parents=True)
    packages = {name: importlib.metadata.version(name) for name in
                ("torch", "torch-geometric", "rdkit", "numpy", "scikit-learn")}
    write_json(args.output / "experiment.json", {"config": config, "data_sha256": sha256(args.data),
                                               "versions": packages, "device": args.device,
                                               "schema_version": 2, "posterior_sampling": "independent",
                                               "package_version": __version__})
    for run_id in run_ids(config["n_runs"]):
        destination = args.output / f"run_{run_id:03d}"
        destination.mkdir()
        train_one(graphs, config, run_id, destination, args.device)


if __name__ == "__main__":
    main()
