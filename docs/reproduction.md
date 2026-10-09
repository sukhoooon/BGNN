# Reproduction status

## Current scope

The new executable path is `training.main -> train_one -> models.BGNN.fit -> eval_loss -> backward -> optimizer.step`, followed by prediction and artifact writing. It replaces the orchestration in `active_learning.train_model_all` and `data_analysis_model_training.train_model_without_save` without importing those side-effectful scripts.

The public core preserves the source GIN structure, priors, training loss, regularization reductions, target scaling and scheduler. At the author's request, posterior sampling now draws independent noise for each weight and MC sample; dropout, epoch ceiling, batch policy and stopping settings follow the manuscript tables. The stopping denominator uses only training targets, independently of the retained all-split target scaler. It removes import-time seeds, plotting helpers, unsupported multi-molecule classes, server/path discovery and object-heavy checkpoint persistence. Empty-edge shapes are normalized for reliable batching. Schema-2 checkpoints store tensor state dictionaries, primitive metadata and the sampling convention with `weights_only=True` loading; old artifacts are not silently reinterpreted.

New-run protocol choices are explicit: run IDs 1..30, seeds 1..30, fixed test/validation subsets within each seed, selectable training fraction, per-run warm-start GNN trained on that split, saved indices and input hash. They are not inferred historical settings. The source's shared warm-start model may have seen different data; its original training provenance has not been recovered.

## Validation

For the latest checks actually performed, see [validation.md](validation.md). The commands below describe available tests, not a claim that all of them have been run on the current revision.

Run the local tests with `python -m pytest -q` (includes tiny training tests). For settings-only checks without training, use `python -m unittest discover -s tests -p test_settings.py`; only NumPy is required. To compare retained calculations against an original snapshot, set `BGNN_ORIGINAL_SOURCE` to its directory before running pytest. Only selected class definitions are compiled; original imports and top-level experiment cells are not executed. The optional comparison replaces the original sampler in memory with the new independent sampler, maps `align` to `anti_alignment`, and disables early termination during its two-epoch comparison. It checks the remaining numerical path, not equivalence to historical training.

The smoke workflow checks training, checkpoint saving/reloading, predictions and aggregation using synthetic targets. It is deliberately separate from the six 30-run baseline configs. No full 30-run scientific experiment has been executed as part of repository cleanup.

Local numerical tests do not establish scientific correctness. GPU behavior, full datasets, old checkpoints and exact final-figure reproduction remain unverified. Independent sampling changes both training and predictive uncertainty; old learned weights are not automatically equivalent to retraining with it.

## Remaining implementation

After author decisions: restore final datasets/splits, port the six raw-data loaders, implement explicit regularization/KL/prior/AGL condition grids, convert historical artifact dictionaries with audited keys, and port the exact main/SI figure panels. Do not recreate old test-based coefficient or run filtering silently. `plotting.py` currently produces a diagnostic summary only, not the paper's Figures 1-4 or supplementary panels.

## Saved outputs

Each invocation requires a new output directory. A completed baseline has the following layout (the same pattern repeats through `run_030`):

```text
experiment.json                  # Config, input SHA256, device, package versions
run_001/
  split.json                     # Zero-based input indices for each split
  checkpoint_GNN.pt              # Deterministic warm start, when enabled
  gnn_log.npz                    # GNN training log, when warm start is enabled
  checkpoint_BGNN.pt             # BGNN state and inference metadata
  log.npz                        # BGNN training log
  predictions_train.npz
  predictions_validate.npz
  predictions_test.npz
  completed.json                 # Written after all run outputs are saved
```

Prediction fields are `row_ids`, `y`, `y_mean`, `sigma_y`, `z`, `mu_w`, `sigma_w`, `schema_version` and `posterior_sampling`. `y_mean` and `sigma_y` are in input-target units after inverse scaling. `sigma_w` is an output-layer parameter scale, not the predictive uncertainty. See [terminology](terminology.md).

Analysis consumes only the test predictions of all completed configured runs. It writes run IDs, per-run metrics, means, across-run standard deviations and separately recorded metric conventions. DUC is opt-in. The summary plotter uses this JSON and does not retrain models. No archived run is padded or automatically removed.

## Operational limits

- `resume` is not supported by the driver. Checkpoints are inference artifacts, not full optimizer-state snapshots for resuming interrupted runs.
- Validation, test evaluation and prediction currently materialize whole splits. Large datasets may exceed CPU/GPU memory even with a small training batch size; full-size quantum-property runs are unverified.
- The retained target scaler fits train, validation and test targets. This is a known leakage risk; only the stopping-rule target span has been restricted to training data. No claim of a leakage-free benchmark is made.
- Original likelihood, warm-start prior and regularization scaling are retained by design. `scale_prior` does not alter the warm-start prior branch. See [numerical conventions](numerical_conventions.md) before designing sweeps.
- No exact historical figure reproduction or new 30-run scientific experiment is claimed. Old learned weights/results are not interchangeable with new independent-sampling outputs.
