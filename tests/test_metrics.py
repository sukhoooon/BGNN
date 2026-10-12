"""Analysis regressions, runnable without the Torch training dependencies."""
import json

import numpy as np
import pytest
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

from bgnn import metrics
from bgnn.analysis import main, summarize


def test_alignment_defaults_to_equal_molecule_weight():
    z = np.array([[100., 0.], [0., 1.]])
    sigma = np.array([1., 2.])
    assert metrics.latent_posterior_alignment_score(z, sigma) == pytest.approx(0.5)
    scaled = z * np.array([[0.01], [100.]])
    assert metrics.latent_posterior_alignment_score(scaled, sigma) == pytest.approx(0.5)


def test_lpas_direction():
    sigma_w = [1., 9.]
    assert metrics.latent_posterior_alignment_score([[1., 0.]], sigma_w) == pytest.approx(.9)
    assert metrics.latent_posterior_alignment_score([[0., 1.]], sigma_w) == pytest.approx(.1)


@pytest.mark.parametrize("z,sigma", [
    ([[0., 0.], [1., 2.]], [1., 2.]),
    ([[1., 2.]], [0., 0.]),
    ([[np.nan, 2.]], [1., 2.]),
    ([[1., 2.]], [1., np.inf]),
])
def test_alignment_rejects_undefined_inputs(z, sigma):
    with pytest.raises(ValueError):
        metrics.latent_posterior_alignment_score(z, sigma)


@pytest.mark.parametrize("bandwidth", [0.01, 10.])
def test_duc_passes_negative_log_density_to_mi(monkeypatch, bandwidth):
    z = np.arange(12, dtype=float).reshape(6, 2)
    uncertainty = np.arange(6, dtype=float)
    log_density = KernelDensity(bandwidth=bandwidth).fit(z).score_samples(z)
    assert np.all(log_density > 0) if bandwidth == 0.01 else np.all(log_density < 0)
    estimate_mi = metrics.mutual_info_regression
    expected = estimate_mi(-log_density[:, None], uncertainty, random_state=0)[0]

    def checked_mi(x, y, **kwargs):
        np.testing.assert_allclose(x[:, 0], -log_density)
        np.testing.assert_array_equal(y, uncertainty)
        return estimate_mi(x, y, **kwargs)

    monkeypatch.setattr(metrics, "mutual_info_regression", checked_mi)
    assert metrics.density_uncertainty_criterion(z, uncertainty, bandwidth=bandwidth) == pytest.approx(expected)


def test_lpas_matches_per_molecule_formula():
    z = np.array([[2., -1., 7.], [0.5, 1.5, -0.5]])
    sigma = np.array([.2, .5, .3])
    normalized = [np.abs(row) / sum(np.abs(row)) for row in z]
    expected = 1 - np.mean([sum(row * sigma / sum(sigma)) for row in normalized])
    assert metrics.latent_posterior_alignment_score(z, sigma) == pytest.approx(expected)


def test_ece_matches_confidence_grid_integral():
    levels = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9, .99])
    y, mean, sigma = np.array([0., .4, 1.5, 3.]), np.zeros(4), np.ones(4)
    errors = [abs(np.mean(abs(y - mean) <= sigma * norm.ppf((1 + p) / 2)) - p)
              for p in levels]
    expected = sum((errors[i] + errors[i + 1]) * (levels[i + 1] - levels[i]) / 2
                   for i in range(len(levels) - 1))
    np.testing.assert_allclose(metrics.ECE_LEVELS, levels)
    assert metrics.regression_ece(y, mean, sigma) == pytest.approx(expected)


@pytest.mark.parametrize("z,sigma", [
    (np.ones((3, 2)), np.ones(3)),
    (np.ones((6, 2)), np.ones(5)),
    (np.full((6, 2), np.nan), np.ones(6)),
])
def test_duc_rejects_invalid_inputs(z, sigma):
    with pytest.raises(ValueError, match="DUC requires"):
        metrics.density_uncertainty_criterion(z, sigma)


@pytest.fixture
def experiment(tmp_path):
    (tmp_path / "experiment.json").write_text(json.dumps({"config": {"n_runs": 1}}))
    run = tmp_path / "run_001"
    run.mkdir()
    (run / "completed.json").write_text(json.dumps({"run_id": 1}))
    np.savez(run / "predictions_test.npz", schema_version=2, posterior_sampling="independent",
             y=np.arange(6), y_mean=np.arange(6), sigma_y=np.arange(1, 7) / 10, sigma_w=[1., 2.],
             z=np.array([[100., 0.], [0., 1.]] * 3))
    return tmp_path


def test_summary_metrics_and_definitions(experiment):
    result = summarize(experiment)
    assert result["mean"]["lpas"] == pytest.approx(0.5)
    definitions = result["metric_definitions"]
    assert definitions["lpas_aggregation"] == "mean_of_normalized_molecules"
    np.testing.assert_allclose(definitions["ece_levels"], metrics.ECE_LEVELS)
    assert definitions["duc"] is None
    assert result["package_version"]
    assert result["posterior_sampling"] == "independent"
    assert {"mae", "sigma_y", "sigma_w", "lpas", "ece"} == set(result["mean"])
    assert "duc" not in result["mean"]
    assert all(value == 0 for value in result["std_ddof0"].values())
    with pytest.raises(ValueError, match="boolean"):
        summarize(experiment, duc="invalid")


def test_cli_metrics_and_metadata(experiment):
    output = experiment / "summary.json"
    main([str(experiment), "--duc", "--output", str(output)])
    result = json.loads(output.read_text())
    assert result["metric_definitions"]["duc"]["density"] == "negative_kde_log_density"
    assert result["mean"]["lpas"] == pytest.approx(0.5)
    assert np.isfinite(result["mean"]["duc"])
    with pytest.raises(SystemExit):
        main([str(experiment), "--duc", "--output", str(output)])


def test_summary_rejects_old_prediction_schema(experiment):
    np.savez(experiment / "run_001/predictions_test.npz", mean=[1.], std=[.1])
    with pytest.raises(ValueError, match="schema 2"):
        summarize(experiment)


def test_summary_requires_complete_runs(experiment):
    (experiment / "experiment.json").write_text(json.dumps({"config": {"n_runs": 2}}))
    with pytest.raises(ValueError, match="Expected runs"):
        summarize(experiment)
    (experiment / "run_001/completed.json").unlink()
    with pytest.raises(ValueError, match="Incomplete run"):
        summarize(experiment)


@pytest.mark.parametrize("sampling,scale,error", [
    ("shared", .1, "schema 2"),
    ("independent", np.nan, "Non-finite"),
])
def test_summary_rejects_incompatible_or_nonfinite_predictions(experiment, sampling, scale, error):
    np.savez(experiment / "run_001/predictions_test.npz", schema_version=2,
             posterior_sampling=sampling, y=np.arange(6), y_mean=np.arange(6),
             sigma_y=np.full(6, scale), sigma_w=[1., 2.], z=np.ones((6, 2)))
    with pytest.raises(ValueError, match=error):
        summarize(experiment)


def test_summary_plot_without_training(experiment):
    from bgnn.plotting import main as plot
    from matplotlib.image import imread

    summary, output = experiment / "summary.json", experiment / "summary.png"
    main([str(experiment), "--duc", "--output", str(summary)])
    plot([str(summary), "--output", str(output)])
    pixels = imread(output)
    assert pixels.shape[0] > 100 and pixels.shape[1] > 100
    assert np.std(pixels[:, :, :3]) > .01
    with pytest.raises(SystemExit):
        plot([str(summary), "--output", str(output)])
