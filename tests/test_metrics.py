"""Analysis regressions, runnable without the Torch training dependencies."""
import json

import numpy as np
import pytest
from sklearn.neighbors import KernelDensity

from bgnn import metrics
from bgnn.analysis import main, summarize


def test_alignment_defaults_to_equal_molecule_weight():
    z = np.array([[100., 0.], [0., 1.]])
    sigma = np.array([1., 2.])
    assert metrics.latent_posterior_alignment_score(z, sigma) == pytest.approx(0.5)
    assert metrics.latent_posterior_alignment_score(z, sigma, "source") == pytest.approx(1 - 102 / 303)
    scaled = z * np.array([[0.01], [100.]])
    assert metrics.latent_posterior_alignment_score(scaled, sigma) == pytest.approx(0.5)


def test_lpas_direction_is_not_aai():
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


def test_duc_rejects_removed_extra_log_convention():
    with pytest.raises(ValueError, match="Only paper DUC"):
        metrics.density_uncertainty_criterion(np.ones((6, 2)), np.arange(6), "source")


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


def test_summary_defaults_and_explicit_archive_diagnostic(experiment):
    result = summarize(experiment)
    assert result["mean"]["lpas"] == pytest.approx(0.5)
    assert result["alignment_convention"] == "paper"
    assert result["ece_convention"] == "paper"
    assert result["duc_convention"] is None
    assert result["posterior_sampling"] == "independent"
    assert {"mae", "sigma_y", "sigma_w", "lpas", "ece"} == set(result["mean"])
    assert "duc" not in result["mean"]
    old = summarize(experiment, alignment_convention="source")
    assert old["mean"]["lpas"] == pytest.approx(1 - 102 / 303)
    assert old["mean"]["ece"] == result["mean"]["ece"]
    with pytest.raises(ValueError, match="DUC convention"):
        summarize(experiment, duc="source")


def test_cli_corrected_metrics_and_metadata(experiment):
    output = experiment / "summary.json"
    main([str(experiment), "--duc", "paper", "--output", str(output)])
    result = json.loads(output.read_text())
    assert result["alignment_convention"] == result["duc_convention"] == "paper"
    assert result["mean"]["lpas"] == pytest.approx(0.5)
    assert np.isfinite(result["mean"]["duc"])
    with pytest.raises(SystemExit):
        main([str(experiment), "--duc", "source", "--output", str(output)])


def test_summary_rejects_old_prediction_schema(experiment):
    np.savez(experiment / "run_001/predictions_test.npz", mean=[1.], std=[.1])
    with pytest.raises(ValueError, match="schema 2"):
        summarize(experiment)
