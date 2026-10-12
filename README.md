# BGNN: Latent-Posterior Alignment

Research code accompanying **Hidden Axis of Uncertainty: Latent-Posterior Alignment in Graph Neural Networks with Bayesian Output Layers** ([paper](https://arxiv.org/abs/2608.20758)).

The package provides a molecular GIN with a Bayesian output layer, repeated training, predictive uncertainty estimation, Latent-Posterior Alignment Score (LPAS), and Density Uncertainty Criterion (DUC). Six target configurations are included: `bp`, `mp`, `logP`, `logPvap`, `H_bonds`, and `G_bonds`.

## Installation

Use Python 3.12 in an isolated environment. From the repository root on macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell. Dependencies are pinned in `pyproject.toml`. CPU is the default; CUDA requires a compatible PyTorch installation and GPU driver.

## Data

Provide a CSV with `smiles,target` columns or a graph file written by `bgnn.data.save_dataset`. Place local inputs under `data/local/`. Targets must have the intended units and transformations before loading; see [data preparation](data/README.md).

Research datasets and pretrained models are not bundled. `data/example_molecules.csv` contains 20 molecules with synthetic targets for the quick check below.

## Train

Inspect settings without training:

```bash
python -m bgnn.training --config configs/baseline_logP.json --dry-run
```

Train using prepared data:

```bash
python -m bgnn.training --config configs/baseline_logP.json --data data/local/logP.csv --output outputs/logP_baseline --device cpu
```

Use `--device cuda` for GPU training. Each baseline runs 30 seeds, with up to 500 epochs per GNN/BGNN stage, dropout 0.3, and 100 Monte Carlo draws. Every run records its split indices, checkpoints, logs, and predictions. Choose a new output directory for each experiment.

## Analyze and plot

```bash
python -m bgnn.analysis outputs/logP_baseline --output outputs/logP_summary.json
python -m bgnn.plotting outputs/logP_summary.json --output outputs/logP_summary.png
```

The summary includes MAE, mean predictive standard deviation, mean posterior standard deviation, LPAS, and ECE across all configured runs. Add `--duc` to the analysis command to include DUC. The plot displays metric means with across-run standard deviations.

## Quick check

The smoke configuration trains two one-epoch GNN/BGNN runs using the bundled synthetic targets:

```bash
python -m bgnn.training --config configs/smoke.json --data data/example_molecules.csv --output outputs/smoke
python -m bgnn.analysis outputs/smoke --output outputs/smoke_summary.json
python -m bgnn.plotting outputs/smoke_summary.json --output outputs/smoke_summary.png
```

For checks without training:

```bash
python -m pytest -m "not training" -q
```

## Repository structure

```text
bgnn/
  models.py       GIN/BGNN and training loop
  training.py     Seeded experiments and predictions
  features.py     Molecular graph features
  data.py         Data input and splitting
  artifacts.py    Checkpoint storage and loading
  metrics.py      LPAS, ECE, and DUC
  analysis.py     Across-run aggregation
  plotting.py     Summary plots
  runtime.py      Seeds and experiment bookkeeping
  __init__.py     Package metadata
configs/          Six baseline configurations and smoke configuration
data/             Input specification and synthetic test data
docs/             Usage, implementation, and testing
tests/            Unit and integration tests
```

Local data, outputs, and checkpoints are excluded from Git. Detailed references:

- [Usage and output files](docs/usage.md)
- [Model, objective, scaling, and metric definitions](docs/implementation.md)
- [Test commands and coverage](docs/testing.md)

## Citation

Please cite the [paper](https://arxiv.org/abs/2608.20758) when using this code. Citation metadata are provided in [CITATION.cff](CITATION.cff).
