# Manuscript terminology and source mapping

Reference: [arXiv:2608.20758v1](https://arxiv.org/html/2608.20758v1). The table distinguishes renaming from numerical changes. Original files and archived result dictionaries are not rewritten.

| Original code / earlier draft | Manuscript term | Public code / result | Treatment |
|---|---|---|---|
| `AAI`, `aai_all`, `cal_alignment`, `alignment_complement` | Latent-Posterior Alignment Score (LPAS) | `latent_posterior_alignment_score`, `lpas` | Return `1 - dot(z_tilde, sigma_w_tilde)`, not the old AAI. Normalize each molecule before averaging. |
| `LAA`, `laa` in analysis sections and filenames | Latent-Posterior Alignment (LPA) | LPA in documentation | A mechanism, not a second numeric score. Old filenames remain only in provenance tables. |
| `cal_udi`, `UDI`, plot label `UDC`; caller variable `mu` | Density Uncertainty Criterion (DUC) | `density_uncertainty_criterion`, `duc` | The MI return value is DUC; Spearman, Kendall and c-index outputs are not renamed DUC. Compute MI from negative KDE log density. |
| Analysis archive `w_rho`, `w_importance`; draft `weight_std`, `posterior_std` | Posterior standard deviation, sigma_w | `sigma_w` | The inspected archive keys already contain softplus(W_rho), not raw rho. |
| Archive `w_mu`; draft `weight_mean` | Posterior mean, mu_w | `mu_w` in predictions | Naming only. |
| Result `sigma`, `uncertainty`; draft `std`, `predictive_std` | Predictive standard deviation, sigma_y | `sigma_y` | Naming only; prediction ddof=0 is retained. |
| Draft prediction `mean` | Predicted mean | `y_mean` | Naming only, distinct from weight posterior mean. |
| `z_pool_test`, `z_pooled_all`; draft `latent` | Pooled latent vector, z | `z` in predictions | Store actual readout vectors, not normalized or absolute values. |
| `z_importance`, training `z_norm` | Normalized latent magnitude, z_tilde | `z_tilde` | Analysis follows Eq. 14; training normalization is unchanged. |
| `w_importance / sum(w_importance)`, `sigma_w_norm` | Normalized posterior scale, sigma_w_tilde | `sigma_w_tilde` | Eq. 15; normalization does not turn standard deviation into variance. |
| Training option `align` | Anti-alignment regularization | `anti_alignment` | Same penalty value and sign; maximizes the old complement and lowers LPAS. |
| Training option `agl` | Alignment-Guided Learning (AGL) | `agl` | Same penalty value and sign; minimizes the complement and raises LPAS. |
| `density_all` in KDE recomputation | Log latent density; scarcity s(z) | `log_density`, `scarcity` | KDE scores are log p(z); scarcity is their negative. |
| `zzww_all` in figure helpers | Relative uncertainty contribution, C_i | Historical panel port still pending | Do not rename all intermediate arrays C_i: raw squared contributions, normalized fractions and percentages differ. |

## Definitions and direction

For a single model, `z_tilde = mean_n(abs(z_n) / sum_i(abs(z_ni)))`, `sigma_w_tilde = sigma_w / sum(sigma_w)`, and `lpas = 1 - dot(z_tilde, sigma_w_tilde)`. A larger LPAS corresponds to more use of lower-posterior-scale directions. The `source` aggregation diagnostic also returns LPAS, not AAI; it differs only in aggregation order.

The former `cal_udi` returns four values. Only its fourth value, mutual information, corresponds to manuscript DUC. Its caller sometimes names that value `mu`, which must not be confused with a predictive or posterior mean.

Raw trainable parameters `W_rho` and `b_rho` retain their names and state-dict keys. They are unconstrained parameters: `sigma_w = log(1 + exp(W_rho))`. Renaming these raw tensors to `sigma_w` would be mathematically wrong. `W_mu`/`b_mu` are also retained inside the model for state-dict clarity; exported weight means use `mu_w`.

## Artifact compatibility

New prediction files and checkpoints use `schema_version=2` and explicit `posterior_sampling` metadata. Prediction NPZ fields are `row_ids`, `y`, `y_mean`, `sigma_y`, `z`, `mu_w`, `sigma_w`, plus schema and sampling metadata. Summary metrics use `mae`, `sigma_y`, `sigma_w`, `lpas`, `ece`, and optionally `duc`; plots display MAE, sigma_y, sigma_w, LPAS, ECE and DUC.

The profile is `independent_sampling`, not `source_compatibility`. Old schema-1 checkpoints and prediction archives are rejected by the new public loader/summary path rather than silently relabelled. Renaming an old key, taking `1 - archived_AAI`, or changing a metadata field cannot reproduce new independent-sampling training or the corrected per-molecule aggregation. Use a separate output directory for new runs.

The likelihood and regularization coefficients are intentionally unchanged at the author's request. This terminology update is not a claim that every training setting now matches the manuscript.
