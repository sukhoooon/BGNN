# Usage

Run commands from the repository root after [installation](../README.md#installation).

## Prepare and train

1. Prepare a dataset as described in [data preparation](../data/README.md).
2. Select the matching `configs/baseline_<property>.json` configuration.
3. Inspect it with `python -m bgnn.training --config configs/baseline_logP.json --dry-run`.
4. Launch training with `--data`, `--output`, and optionally `--device cuda`.

```bash
python -m bgnn.training --config configs/baseline_logP.json --data data/local/logP.csv --output outputs/logP_baseline
```

Each seed defines a shuffled split: 20% test, then 10% of the remaining pool for validation (approximately 72/8/20 train/validation/test). Setting `split.train_fraction` reduces only the training subset; validation and test indices remain fixed within a seed. The six baseline configurations use seeds 1 through 30.

With `fit.warmstart=true`, each run trains a deterministic GNN followed by its BGNN. The executable path is `training.main -> train_one -> models.BGNN.fit`; prediction files are written after fitting. See [implementation](implementation.md) for the objective and configuration meanings.

Training and analysis require new output paths to protect existing results. Checkpoints support inference; the driver does not resume optimizer state. Validation, test evaluation, and prediction load whole splits into memory, so memory requirements also depend on split size, not only the training batch size.

## Saved outputs

```text
outputs/logP_baseline/
  experiment.json
  run_001/
    split.json
    checkpoint_GNN.pt
    gnn_log.npz
    checkpoint_BGNN.pt
    log.npz
    predictions_train.npz
    predictions_validate.npz
    predictions_test.npz
    completed.json
  ...
  run_030/
```

`experiment.json` records the configuration, input SHA256, device, package and dependency versions, and artifact schema. `split.json` records zero-based input row indices. GNN checkpoints and logs are written when warm start is enabled. `completed.json` is written after all outputs for a run are saved.

Each prediction file contains:

| Field | Meaning |
|---|---|
| `row_ids` | Input row indices in prediction order |
| `y`, `y_mean`, `sigma_y` | Target, predictive mean, predictive standard deviation in input-target units |
| `z` | Pooled latent vectors, shape `[molecules, latent_dimensions]` |
| `mu_w`, `sigma_w` | Bayesian output-weight posterior means and standard deviations |
| `schema_version`, `posterior_sampling` | Artifact format and posterior sampling method |

Checkpoint and prediction schema is 2. Bayesian artifacts use `posterior_sampling="independent"`; deterministic GNN checkpoints use `"deterministic"`. Load checkpoints with `bgnn.artifacts.load_model(path)`.

## Analyze and plot

```bash
python -m bgnn.analysis outputs/logP_baseline --duc --output outputs/logP_summary.json
python -m bgnn.plotting outputs/logP_summary.json --output outputs/logP_summary.png
```

Analysis reads test predictions from every configured run. Missing, duplicate, incomplete, or incompatible runs are rejected. It reports per-run metrics, means, and population standard deviations (`ddof=0`), together with the package version and metric definitions. `--duc` enables KDE-based DUC computation, which can be expensive for large datasets.

The plotter produces a bar chart for each metric with across-run standard deviation as the error bar. Analysis and plotting do not train models.

The installed console commands `bgnn-train`, `bgnn-analyze`, and `bgnn-plot` accept the same arguments as the corresponding Python modules.
