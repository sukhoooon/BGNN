# Numerical conventions and manuscript audit

Reference: [arXiv:2608.20758v1](https://arxiv.org/html/2608.20758v1), checked 2026-10-07. Original source line numbers below refer to the hashes in `source_hashes.json`.

## Confirmed changes

The public driver uses `range(1, n_runs + 1)` with `n_runs=30`. Four inspected historical performance dictionaries contain arrays of length 29. Changing the loop cannot supply missing historical runs. The `*30` inside the original alignment penalty is a loss coefficient, NOT the experiment repetition count, and is unchanged.

Author-requested changes (2026-10-07): posterior sampling is now independent across weights and MC draws. LPAS defaults to per-molecule normalization followed by averaging, matching Methods Eq. 14. The previous aggregate-first convention remains explicitly selectable for diagnostics only. DUC uses `scarcity = -log_density`; the extra-log execution branch has been removed and requests for it fail explicitly. CLI DUC remains opt-in via `--duc paper`. ECE now defaults to the manuscript grid. Likelihood, priors and training penalty values are unchanged. Public terminology and schema-2 artifacts are documented in `terminology.md`. These changes do not establish which path produced historical figures, and archived summaries must not be silently relabelled as corrected results.

## Parameter comparison

| Item | Manuscript | Available source / artifact | Draft action |
|---|---|---|---|
| Architecture | GIN, 5 layers, 128 hidden, 64 latent | Same | Preserve |
| Repetitions | 30 | Many analysis loops `range(1,30)`; inspected arrays length 29 | New driver: 30, no archive padding |
| Epoch ceiling | 500 | `architecture.fit`: 500; `train_model_all`: 1000; another main: 900; inspected bp/logP checkpoints: 500 | Six baseline configs: 500 |
| Dropout | Table 2: 0.3 | Bayesian wrapper and stored bp/logP BGNN: 0; GNN: 0.3 | GNN and BGNN: 0.3 |
| Batch size | 64 / 256 / 1024 by dataset size | Additional 4096 branch at N >= 1e6; chosen using full dataset size, not subset size | Paper thresholds, full input dataset N; 1024 for all N >= 1e5 |
| Learning rate | 1e-3, floor 1e-6 | Same; stored checkpoints agree | Preserve |
| Scheduler | factor 0.8, patience 10 | Same | Preserve |
| Optimizer / clipping | AdamW / [-10,10] | Same; AdamW default decay 0.01 unless disabled | Preserve |
| Early stopping | 60 epochs; relative loss/error rules | 75 epochs, absolute loss tolerance 0.01; relative-loss rule commented; relative-error check is in `elif` | 60 epochs, tolerance 0, both relative rules checked independently |
| Target span / scaler | Training-domain interpretation | Both use train + validation + test targets | Stopping span now train-only; scaler still uses all splits and retains test leakage |
| Split | 80/20 | Helper default test .1; wrapper default .001; manual splits can override | New split explicitly recorded, not called historical |
| Node features | 37-dimensional claim; element table lists 10 elements | 11 elements including fluorine (atomic number 9), total 37 | Preserve code ordering; table omission needs confirmation |
| Input normalization | [0,1] | Unary features already binary; no separate fitted input scaler | No additional scaler introduced |
| MC draws | Not specified in inspected training table | 100 for training and prediction | Preserve and expose |
| ECE grid | 0.1..0.9, 0.99 | Source includes 0.95 too | Paper by default; source grid is opt-in |

### Stopping implementation details

The epoch limit is 500. Independently after every epoch, stop if current loss / initial loss < 0.001, validation MAE / unscaled training-target span < 0.001, or no lower recorded training loss occurs for 60 epochs. The manuscript does not identify the MAE split; the existing validation MAE is retained. Loss tolerance is now zero, since the table specifies improvement without an absolute minimum delta. The loss ratio is enabled from the second epoch only when the initial loss is positive; with zero or negative initial loss it is disabled rather than dividing by zero or reversing the meaning of improvement. A zero training-target span disables only the MAE ratio. The other criteria and epoch ceiling remain active. These safeguards and validation choice are explicit implementation interpretations, not extra statements from the manuscript.

The smoke config remains two one-epoch trials with batch size 8 and 10 MC draws. Warm-start lookup now selects `checkpoint_GNN.pt` in the same directory, without replacing `BGNN` in parent directory names. No training was run for these settings changes.

Final upload review (2026-10-09): scalar-target extraction now preserves a length-one sample axis. Previously, a one-molecule validation/test split became a zero-dimensional tensor and failed during target transformation. This shape-only fix does not change target values, standardization statistics or the retained all-split scaler policy. Two no-training regression tests cover this case.

## Implementation differences and historical provenance

1. **Posterior sampling (corrected in public code):** original `architecture.py:516` broadcasts a single epsilon vector across weight dimensions. Public `models.BGNN.sampling` now generates a `[latent_dim, n_rand]` tensor of independent standard normals and separate bias noise. For fixed z, the resulting variance is `sum(z_i^2 sigma_i^2) + sigma_b^2`, matching the manuscript's independent-weight formula. The likelihood itself is unchanged; this sampler correction alone is not a claim of full ELBO equivalence or historical result reproduction.
2. **Prior under warm start:** `set_priors:317` uses pretrained output means and standard deviation 0.5; cold start uses zero mean and `scale_prior`. Main-text discussion of N(0,1), warm-start wording and prior-sweep execution therefore require reconciliation. `scale_prior` has no effect in the retained warm-start branch.
3. **Likelihood / objective:** `eval_log_like:481` fits a Gaussian to MC predictive mean/std and evaluates the targets under it; that is not generally the expectation of per-weight conditional log likelihood described in the ELBO formula. KL is divided by number of minibatches and likelihood averaged per minibatch. Do not relabel its scaling without validating it.
4. **Regularization coefficients:** `eval_z_regularize:489` uses sum(abs(z))/10, sum(z^2)/10 and 30 times alignment. `scale_loss` multiplies the ELBO, not the penalty. Relative to an unscaled ELBO, the code's coefficient is `1/(10*scale_loss)` for L1/squared L2 and `30/scale_loss` for alignment. Batch reductions and gradient clipping prevent treating this as an exact optimization equivalence. Table 5 lambdas cannot be copied directly into `scale_loss`; its L2 norm also differs from squared L2 in source.
5. **Latent aggregation:** `6_z_alignment_5.py:1072`, the latest QM counterpart and `1_fig_drawing.py` normalize a sum of absolute activations, whereas Methods Eq. 14 normalizes each molecule then averages. This is a plotting/analysis distinction: training-time AGL already normalizes each molecule separately. The AGL sign is consistent with maximizing LPAS up to an additive constant. Rechecking the extractor against [PyG 2.6.1](https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/nn/models/basic_gnn.html) shows that the configured GIN (out_channels=64, jk=None) has no extra activation/projection after its final convolution. Pooling that hook output should equal model.z; hook location itself is not an established discrepancy. Historical library versions and artifact provenance remain unverified.
6. **DUC:** `density_estimate.kde_density:790` returns log density, and the recomputation callers pass it directly to `cal_udi:839`, which applies another negative log. The transformed scarcity differs from the manuscript. However, the inspected general and QM density archives had 6,589,728 and 2,784,000 KDE values respectively, all positive and finite (ranges 35.56215-43.90982 and 36.59106-43.77200); no NaN failure was observed. These values alone do not prove the archived density convention or the final figures' execution path. On positive log density, the two scarcity variables are invertibly related, so exact MI is invariant, although finite-sample kNN estimates can differ. Do not infer invalid historical DUC conclusions from the transform alone. The public function now defaults to the corrected paper calculation and rejects the extra-log convention; summary DUC remains opt-in for computational cost. See [MI invariance](https://arxiv.org/abs/1907.13625).
7. **Selection:** latest AGL selection checks uncertainty/alignment/DUC calculated from test predictions; regularization drawing also filters runs by uncertainty. The manuscript specifies validation-only tuning and 30-run summaries. The new driver never does those filters, so its summaries are not asserted to match those archived panels.
8. **Data provenance:** current raw-table counts differ from the manuscript and mp data is missing. Finding a checkpoint does not establish that it generated the final figures. See `data/README.md`.

Prediction standard deviation uses NumPy `ddof=0`; source training likelihood uses Torch sample standard deviation. This distinction is retained. The observation-noise model is not redesigned. Existing numerical warnings remain documented rather than hidden.

## What is and is not reproduced

Retained training calculations can be compared on a small fixed split with `tests/test_source_equivalence.py`, which applies the corrected sampler to both implementations in memory and disables early termination during its fixed two-epoch comparison. Passing that test establishes agreement of the remaining numerical path, not equivalence to historical shared-noise training, correctness of the statistical model, provenance of the manuscript figures, or equivalence across GPUs. This repository is a source-code release, not a complete reproducibility package; see [remaining scope](reproduction.md).
