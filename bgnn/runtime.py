"""Explicit randomness and experiment bookkeeping; no import-time seeding."""
import hashlib
import json
import random
from pathlib import Path


def cprint(*values, **kwargs):
    print(*values)


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_ids(n_runs=30):
    if type(n_runs) is not int or n_runs < 1:
        raise ValueError("n_runs must be a positive integer")
    return range(1, n_runs + 1)


def require_complete_runs(ids, n_runs=30):
    ids = list(ids)
    expected = set(run_ids(n_runs))
    if len(ids) != n_runs or set(ids) != expected:
        raise ValueError(f"Expected runs 1..{n_runs}; received {sorted(ids)}")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
