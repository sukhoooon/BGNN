"""Calibration, latent-posterior alignment and density-uncertainty metrics."""
import numpy as np
from scipy.stats import norm
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import KernelDensity


ECE_LEVELS = np.r_[np.linspace(0.1, 0.9, 9), 0.99]


def regression_ece(y, y_mean, sigma_y):
    y, y_mean, sigma_y = [np.asarray(a).reshape(-1) for a in (y, y_mean, sigma_y)]
    if not (y.shape == y_mean.shape == sigma_y.shape) or not len(y):
        raise ValueError("Empty or mismatched predictions")
    grid = ECE_LEVELS
    radius = np.maximum(sigma_y, 1e-12)[:, None] * norm.ppf((1 + grid) / 2)
    coverage = (np.abs(y[:, None] - y_mean[:, None]) <= radius).mean(axis=0)
    return float(np.trapezoid(np.abs(coverage - grid), grid))


def latent_posterior_alignment_score(z, sigma_w):
    """Return LPAS with equal weight for each normalized molecular latent vector."""
    z_abs, sigma_w = np.abs(np.asarray(z)), np.asarray(sigma_w).reshape(-1)
    if (z_abs.ndim != 2 or z_abs.shape[1] != len(sigma_w) or np.any(sigma_w < 0)
            or not np.isfinite(z_abs).all() or not np.isfinite(sigma_w).all()):
        raise ValueError("Invalid latent vectors or posterior scales")
    if sigma_w.sum() <= 0 or z_abs.sum() <= 0:
        raise ValueError("Alignment is undefined for zero vectors")
    row_sum = z_abs.sum(axis=1, keepdims=True)
    if np.any(row_sum == 0):
        raise ValueError("LPAS is undefined for zero latent rows")
    z_tilde = (z_abs / row_sum).mean(axis=0)
    sigma_w_tilde = sigma_w / sigma_w.sum()
    return float(1 - np.dot(z_tilde, sigma_w_tilde))


def density_uncertainty_criterion(z, sigma_y, *, bandwidth=0.2):
    """Estimate mutual information between negative KDE log density and uncertainty."""
    z, sigma_y = np.asarray(z), np.asarray(sigma_y).reshape(-1)
    if (z.ndim != 2 or len(z) != len(sigma_y) or len(z) < 4
            or not np.isfinite(z).all() or not np.isfinite(sigma_y).all()):
        raise ValueError("DUC requires at least four finite, matching latent vectors and uncertainties")
    log_density = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(z).score_samples(z)
    scarcity = -log_density
    return float(mutual_info_regression(scarcity[:, None], sigma_y, random_state=0)[0])
