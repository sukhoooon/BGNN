# Testing

Install the test dependencies with `python -m pip install -e '.[test]'`.

## Without training

```bash
python -m compileall -q bgnn tests
python -m pytest -m "not training" -q
```

These checks cover:

- Six baseline configurations, 30-run counting, batch thresholds, and stopping rules.
- Molecular features, tensor dataset round trips, seeded splits, and single-molecule scaling.
- Independent posterior sample shapes, gradients, covariance, and predictive variance.
- Checkpoint serialization, inference, and schema/sampling compatibility.
- LPAS, ECE, DUC, complete-run aggregation, summary metadata, and plotting.
- CLI dry runs, output protection, and experiment metadata with model fitting replaced by a test stub.

GitHub Actions runs these checks on Python 3.12 with CPU PyTorch. Numerical checks may perform forward or backward operations, but the `not training` selection excludes model fitting and optimizer updates.

Settings-only checks require NumPy and can also be run using:

```bash
python -m unittest discover -s tests -p test_settings.py -v
```

## Optional training integration test

```bash
python -m pytest -m training -q
```

This test performs a one-epoch GNN/BGNN run with synthetic targets, then checks saved predictions and checkpoint inference. It is separate from the six 30-run configurations. Running `python -m pytest -q` includes this test as well as the non-training checks.
