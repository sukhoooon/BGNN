"""Optional retained-calculation checks against an author's local source snapshot.

No original imports or top-level experiment cells are executed.
Set BGNN_ORIGINAL_SOURCE to the original script_2 directory to enable.
Both models use the corrected sampler in memory; this is not historical equivalence.
"""
import ast
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from bgnn import models
from bgnn.data import load_dataset, split_indices, subset
from bgnn.runtime import seed_everything


@pytest.mark.skipif(not os.environ.get("BGNN_ORIGINAL_SOURCE"), reason="Author's source not provided")
@pytest.mark.parametrize("regularization", [None, "L1", "L2", "anti_alignment", "agl"])
def test_retained_training_matches_original_with_same_corrected_sampler(regularization):
    path = Path(os.environ["BGNN_ORIGINAL_SOURCE"]) / "architecture.py"
    tree = ast.parse(path.read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in ("Scaler", "BGNN", "BGNN_unary")]
    namespace = dict(vars(models))
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)
    namespace["BGNN"].sampling = models.BGNN.sampling
    torch.set_num_threads(1)
    graphs = load_dataset(Path(__file__).resolve().parents[1] / "data/example_molecules.csv")
    data = subset(graphs, split_indices(len(graphs), 1, .2, .1))
    outputs = []
    for cls in (namespace["BGNN_unary"], models.BGNN_unary):
        seed_everything(73)
        net = cls("logP", "GIN", 37, 7, bayes=True, dropout=0, device="cpu")
        option = "align" if cls is namespace["BGNN_unary"] and regularization == "anti_alignment" else regularization
        logs = net.fit(data, n_epoch=2, n_batch=8, n_rand=10, warmstart=False,
                       scale_loss=1, z_regularize_opt=option,
                       loss_thres_rel=float('-inf'), error_thres_rel=0,
                       loss_patience=60, loss_tolerance=0)
        outputs.append((net, logs))
    for key in outputs[0][0].state_dict():
        torch.testing.assert_close(outputs[0][0].state_dict()[key], outputs[1][0].state_dict()[key], rtol=0, atol=0)
    for key in outputs[0][1]:
        np.testing.assert_array_equal(outputs[0][1][key], outputs[1][1][key])
