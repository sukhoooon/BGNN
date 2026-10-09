# BGNN: Latent-Posterior Alignment

Research code accompanying **Hidden Axis of Uncertainty: Latent-Posterior Alignment in Graph Neural Networks with Bayesian Output Layers** ([arXiv:2608.20758v1](https://arxiv.org/abs/2608.20758)).

This repository provides a molecular GIN with a Bayesian output layer, a seeded training workflow, and uncertainty/latent-alignment analysis for six scalar targets: `bp`, `mp`, `logP`, `logPvap`, `H_bonds` and `G_bonds`.

**Scope:** this is a cleaned and modified implementation, not an exact archive of the code that generated the manuscript figures. Final research datasets, pretrained models, historical results, complete experiment grids and exact manuscript figure panels are not included yet. The example has synthetic targets. See [reproduction status](docs/reproduction.md) and [implementation differences](docs/numerical_conventions.md).

## Repository contents

```text
.
|-- README.md
|-- CITATION.cff
|-- pyproject.toml
|-- .gitignore
|-- .github/
|   `-- workflows/
|       `-- tests.yml             # No-training automated checks
|-- bgnn/
|   |-- __init__.py
|   |-- features.py              # Molecular graph features
|   |-- data.py                  # Input, recorded splits and batch policy
|   |-- models.py                # GIN/BGNN, objectives and training loop
|   |-- training.py              # Seeded runs, warm start and predictions
|   |-- artifacts.py             # Checkpoint serialization
|   |-- metrics.py               # LPAS, ECE and optional DUC
|   |-- analysis.py              # Complete-run aggregation
|   |-- plotting.py              # Summary graphs, not manuscript panels
|   |-- legacy.py                # Trusted local graph-pickle converter
|   `-- runtime.py               # Seeds, hashes, JSON and run validation
|-- configs/
|   |-- baseline_bp.json
|   |-- baseline_mp.json
|   |-- baseline_logP.json
|   |-- baseline_logPvap.json
|   |-- baseline_H_bonds.json
|   |-- baseline_G_bonds.json
|   |-- smoke.json               # Two one-epoch synthetic trials
|   `-- arxiv_reference.json     # Reference only, not executable
|-- data/
|   |-- README.md
|   `-- example_molecules.csv
|-- docs/
|   |-- reproduction.md          # Workflow, outputs and remaining scope
|   |-- numerical_conventions.md # Changed and retained calculations
|   |-- terminology.md           # Manuscript and historical names
|   |-- code_mapping.md          # Mapping of 50 original source files
|   |-- source_hashes.json       # Original-source SHA256 provenance
|   `-- validation.md            # Checks performed and limitations
`-- tests/
    |-- test_settings.py         # No-training settings checks
    |-- test_metrics.py          # Analysis tests with synthetic arrays
    |-- test_core.py             # Includes a tiny training test
    `-- test_source_equivalence.py # Optional original-source comparison
```

`data/local/` and `outputs/` are created during use and excluded from Git. Research data, checkpoints, caches and local environments are not included.

## Installation

Use Python 3.12 in an isolated environment. From the repository root on macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. Dependencies are pinned in `pyproject.toml`. CPU is the default; CUDA requires a compatible PyTorch installation and GPU driver. GPU execution has not been verified for this package.

Torch 2.7.0, RDKit 2024.9.6 and scikit-learn 1.6.1 match manuscript-reported versions. Other pins, including PyG 2.6.1, are packaging choices rather than a recovered historical environment. Installation and training times depend on hardware and input size; no representative timing benchmark is provided.

## Check without training

```bash
python -m unittest discover -s tests -p test_settings.py -v
python -m pytest tests/test_core.py tests/test_metrics.py --deselect tests/test_core.py::test_training_checkpoint_predictions -q
```

The first command requires only NumPy. Neither command trains a model: the second explicitly excludes the training test and checks graph input, scaling, posterior sampling and analysis. GitHub Actions runs the same checks and a syntax check. The full `python -m pytest -q` suite includes a tiny synthetic training test and is optional. See [validation](docs/validation.md) for checks actually performed locally.

## Prepare data

Provide a CSV with `smiles,target` columns, or a tensor-only graph file prepared with this package. Targets must already have the intended units and property transformations; the loader does not infer units, apply logarithms or assemble formation energies. See [data instructions](data/README.md).

Place private inputs under `data/local/`, for example `data/local/logP.csv`. The bundled `data/example_molecules.csv` is a software demonstration only.

## Optional synthetic demonstration

**These commands train small models.** The smoke configuration uses 20 molecules, synthetic targets, two runs and one epoch for each GNN/BGNN stage. It is not a paper experiment.

```bash
python -m bgnn.training --config configs/smoke.json --data data/example_molecules.csv --output outputs/smoke --acknowledge-source-discrepancies
python -m bgnn.analysis outputs/smoke --output outputs/smoke_summary.json
python -m bgnn.plotting outputs/smoke_summary.json --output outputs/smoke_summary.png
```

Expected products are checkpoints, logs, prediction arrays, a summary JSON and a summary PNG. Numeric outputs are not promised to be identical across platforms. Existing output directories/files are rejected; choose new names for another run.

## Train a baseline

Inspect a configuration without starting training:

```bash
python -m bgnn.training --config configs/baseline_logP.json --dry-run
```

After preparing real data and reading the [implementation notes](docs/numerical_conventions.md), explicitly launch a baseline:

```bash
python -m bgnn.training --config configs/baseline_logP.json --data data/local/logP.csv --output outputs/logP_baseline --device cpu --acknowledge-source-discrepancies
```

Use `--device cuda` only on a configured CUDA machine. Each baseline requests **30 runs**, seeds 1 through 30, maximum 500 epochs per training stage, GNN/BGNN dropout 0.3, loss patience 60 and 100 Monte Carlo draws. Automatic batch size uses the full input count: 64 below 10,000, 256 below 100,000, otherwise 1024. These experiments can be expensive; installation and CI never start them.

The new split reserves 20% for testing and 10% of the remaining pool for validation (approximately 72/8/20 overall). This is an explicit new protocol, not a recovered historical split. Each run first trains its own GNN warm start and then its BGNN. No automatic resume is provided. The six configs are baseline examples, not the complete manuscript experiment grid.

## Analyze and plot

```bash
python -m bgnn.analysis outputs/logP_baseline --output outputs/logP_summary.json
python -m bgnn.plotting outputs/logP_summary.json --output outputs/logP_summary.png
```

Analysis requires all configured run IDs exactly once and never filters runs by test metrics. It reports MAE, mean predictive standard deviation, mean posterior standard deviation, LPAS and ECE, with across-run standard deviation (`ddof=0`).

LPAS normalizes each molecule before averaging. ECE defaults to confidence levels 0.1 through 0.9 and 0.99. Add `--duc paper` to analysis to calculate DUC from negative KDE log density, without a second logarithm. DUC is optional because KDE can be expensive for large datasets.

`--convention source` selects the old ECE grid including 0.95; `--alignment-convention source` selects the old aggregate-first LPAS diagnostic. Neither converts historical checkpoints. New checkpoints/predictions use schema 2 and independent posterior sampling; incompatible older artifacts are rejected.

The likelihood, priors, regularization coefficients and all-split target scaler are retained from the original implementation. In particular, the scaler still uses validation/test targets; this workflow is not a leakage-free benchmark or verified historical reproduction. See [reproduction status](docs/reproduction.md) for remaining limitations, including large-dataset memory use.

## Citation and terms

Please cite the [paper](https://arxiv.org/abs/2608.20758); metadata are in [CITATION.cff](CITATION.cff). Code package version: `0.1.0`. No software DOI or GitHub release is claimed by these files.

No software license is included at this stage. Public availability alone does not supply a software license, and the manuscript's arXiv license is not a license for this code. Research datasets are not redistributed here.
