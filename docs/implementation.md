# Implementation

## Architecture and features

The backbone is a five-layer GIN with 128 hidden channels and 64 output channels, followed by global mean pooling to obtain the molecular latent vector `z`. Both deterministic and Bayesian configurations use dropout 0.3. The Bayesian output is `y = z @ W + b`.

Graphs have 37 node features and 7 edge features. Element channels are ordered H, B, C, N, O, F, P, S, Cl, Br, I; the remaining node features encode formal charge, hybridization, hydrogen count, aromaticity, valence, degree, ring membership, and a reaction-center indicator. Edge features encode bond type, conjugation, rotatability, and acyclic membership. The full order is defined in `bgnn/features.py`; unary inputs leave the reaction-center indicator zero. Bonds are represented in both directions.

## Target scaling and splits

Scalar targets are standardized as `(y - mean) / (std + 1e-7)`. The scaler is fitted to the combined train, validation, and test targets, using Torch sample standard deviation. Predictions are inverse-transformed to input-target units. The early-stopping target span is computed from training targets only.

The split reserves 20% for testing and 10% of the remainder for validation. Counts are rounded upward for test and validation. Every run saves the exact indices and seed.

## Posterior and prior

Output weights and bias have independent Gaussian posteriors. Their standard deviations are `log(1 + exp(rho))`, where `W_rho` and `b_rho` are unconstrained trainable parameters. Sampling draws independent noise for every weight, bias, and Monte Carlo draw. For fixed `z`, the predictive variance in standardized target space is:

```text
Var(y | z) = sum_i(z_i^2 * sigma_w_i^2) + sigma_b^2
```

With warm start, posterior weight and bias means are initialized from that run's deterministic GNN; prior means use those pretrained values and prior standard deviations are 0.5. Without warm start, priors have zero mean and standard deviation `fit.scale_prior`. Consequently, `scale_prior` applies only when `warmstart=false`.

## Training objective

For each minibatch, the implementation draws `fit.n_rand` posterior samples (100 by default), computes the predictive mean `mu` and sample standard deviation `s`, and evaluates the targets under `Normal(mu, max(s, 0.001))`. `log_like` is the mean of these target log probabilities. This is a predictive-moment Gaussian likelihood calculation.

`log_post` and `log_prior` sum parameter log probabilities averaged over Monte Carlo draws, each divided by the number of training minibatches. The minimized loss is:

```text
loss = scale_loss * (scale_kl * (log_post - log_prior) - log_like) + penalty
```

Baseline BGNN configurations use `scale_loss=1`, `scale_kl=1`, and no penalty. Deterministic GNN pretraining uses `1000 * MSE`.

Regularization options in `fit.z_regularize_opt` are:

| Option | Minibatch penalty |
|---|---|
| `null` | 0 |
| `L1` | `sum(abs(z)) / 10` |
| `L2` | `sum(z**2) / 10` |
| `anti_alignment` | `-30 * mean_n(dot(z_tilde_n, sigma_w_tilde))` |
| `agl` | `30 * mean_n(dot(z_tilde_n, sigma_w_tilde))` |

Training uses `z_tilde_n = abs(z_n) / (sum(abs(z_n)) + 1e-8)` and `sigma_w_tilde = sigma_w / (sum(sigma_w) + 1e-8)`. AGL denotes Alignment-Guided Learning. `scale_loss` multiplies the likelihood/KL term, not the penalty.

## Optimization and stopping

- AdamW with learning rate 0.001, default weight decay 0.01, and elementwise gradient clipping to `[-10, 10]`. `fit.no_decay=true` sets weight decay to zero.
- ReduceLROnPlateau monitors validation MAE, with factor 0.8, patience 10, and minimum learning rate `1e-6`.
- Automatic batch size uses the full input count: 64 for `N < 10,000`, 256 for `N < 100,000`, and 1024 otherwise.
- Each stage runs up to 500 epochs. It stops when current training loss / initial loss is below 0.001, validation MAE / training-target span is below 0.001, or training loss has not improved for 60 epochs. Improvement tolerance is zero.
- The loss-ratio criterion starts at the second epoch and requires positive initial loss. The MAE-ratio criterion requires a positive training-target span.

## Analysis metrics

| Name | Definition |
|---|---|
| `mae` | Mean absolute error in input-target units |
| `sigma_y` | Mean predictive standard deviation, computed over MC predictions with `ddof=0` |
| `sigma_w` | Mean output-weight posterior standard deviation |
| `lpas` | Latent-Posterior Alignment Score, defined below |
| `ece` | Integral of absolute empirical coverage error over the confidence grid |
| `duc` | Density Uncertainty Criterion, defined below |

LPAS uses equal weight for every molecule:

```text
z_tilde = mean_n(abs(z_n) / sum_i(abs(z_ni)))
sigma_w_tilde = sigma_w / sum_i(sigma_w_i)
LPAS = 1 - dot(z_tilde, sigma_w_tilde)
```

Larger LPAS indicates more latent magnitude in directions with smaller posterior standard deviation. Zero latent rows or a zero total posterior scale do not define this normalization and are rejected.

ECE evaluates Gaussian interval coverage at confidence levels `0.1, 0.2, ..., 0.9, 0.99` and integrates the absolute difference between coverage and confidence using the trapezoidal rule. Interval radii use predictive standard deviations floored at `1e-12`.

DUC estimates mutual information between latent scarcity and predictive standard deviation. Gaussian KDE with bandwidth 0.2 is fitted and evaluated on the test latent vectors. Scarcity is the negative of the returned log density: `scarcity = -log_density`. Mutual information uses `sklearn.feature_selection.mutual_info_regression` with `random_state=0` and its default three neighbors. DUC requires at least four samples.

Training likelihood uses sample standard deviation (`ddof=1`); exported predictive standard deviations and across-run error bars use population standard deviation (`ddof=0`). Analysis aggregates every configured run without metric-based filtering.
