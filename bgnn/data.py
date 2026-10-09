"""Portable tensor-only graphs and explicit, recorded experiment splits."""
import csv
import math
import random
from pathlib import Path

import torch

from .features import UnaryMolData, featurize


def validate_dataset(graphs):
    if len(graphs) < 6:
        raise ValueError("At least six molecules are required")
    for i, graph in enumerate(graphs):
        if graph.x.ndim != 2 or graph.x.shape[1] != 37 or not len(graph.x):
            raise ValueError(f"Graph {i}: expected nonempty 37-column node features")
        if graph.edge_index.shape[0] != 2 or graph.edge_attr.shape != (graph.edge_index.shape[1], 7):
            raise ValueError(f"Graph {i}: malformed edges")
        if not torch.isfinite(graph.x).all() or not math.isfinite(float(graph.y)):
            raise ValueError(f"Graph {i}: non-finite value")
    return graphs


def save_dataset(graphs, path):
    validate_dataset(graphs)
    records = []
    for g in graphs:
        records.append({"smiles": str(g.smiles), "x": g.x.cpu(),
                        "y": float(g.y), "edge_index": g.edge_index.cpu(),
                        "edge_attr": g.edge_attr.cpu(), "charge": int(g.charge)})
    torch.save({"schema_version": 1, "graphs": records}, path)


def load_dataset(path):
    if Path(path).suffix == ".csv":
        with open(path, newline="") as stream:
            rows = list(csv.DictReader(stream))
        graphs = [featurize(row["smiles"], float(row["target"])) for row in rows]
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported dataset schema")
        graphs = [UnaryMolData(**row) for row in payload["graphs"]]
    return validate_dataset(graphs)


def split_indices(size, seed, test_fraction, validation_fraction, train_fraction=1.0):
    """Validation is a fraction of the non-test pool; subset train only.

    This is a new explicit protocol, NOT a recovered historical paper split.
    Test and validation stay fixed when train_fraction changes.
    """
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must lie strictly between zero and one")
    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must lie in (0, 1]")
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    n_test = math.ceil(size * test_fraction)
    n_val = math.ceil((size - n_test) * validation_fraction)
    available = indices[n_test + n_val:]
    n_train = math.floor(len(available) * train_fraction)
    if n_train < 2 or n_test < 1 or n_val < 1:
        raise ValueError("Split is too small; use more data or a larger train_fraction")
    return {"train": available[:n_train], "validate": indices[n_test:n_test+n_val],
            "test": indices[:n_test]}


def subset(graphs, indices):
    return {name: [graphs[i] for i in ids] for name, ids in indices.items()}


def paper_batch_size(size):
    """Supplementary Table 3 policy, using the full input dataset size N."""
    if size <= 0:
        raise ValueError("Dataset size must be positive")
    for bound, batch in [(10_000, 64), (100_000, 256)]:
        if size < bound:
            return batch
    return 1024
