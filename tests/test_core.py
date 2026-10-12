import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

from bgnn import training
from bgnn.artifacts import load_model, save_model
from bgnn.data import load_dataset, save_dataset, split_indices
from bgnn.features import featurize
from bgnn.models import BGNN_unary, Scaler
from bgnn.runtime import require_complete_runs, run_ids
from bgnn.training import train_one, validate_config

ROOT = Path(__file__).resolve().parents[1]


def test_thirty_runs_and_completeness():
    assert list(run_ids()) == list(range(1, 31))
    require_complete_runs(range(1, 31))
    for ids in [range(1, 30), [1] * 30, list(range(1, 30)) + [31]]:
        with pytest.raises(ValueError):
            require_complete_runs(ids)


def test_all_public_baselines_use_thirty_runs():
    configs = list((ROOT / "configs").glob("baseline_*.json"))
    assert len(configs) == 6
    for path in configs:
        config = json.loads(path.read_text())
        validate_config(config)
        assert config["n_runs"] == 30
        assert config["profile"] == "independent_sampling"


def test_unsupported_profile_and_regularization_are_rejected():
    config = json.loads((ROOT / "configs/smoke.json").read_text())
    config["profile"] = "unsupported"
    with pytest.raises(ValueError, match="independent_sampling"):
        validate_config(config)
    config["profile"] = "independent_sampling"
    config["fit"]["z_regularize_opt"] = "align"
    with pytest.raises(ValueError, match="regularization"):
        validate_config(config)
    config["fit"]["z_regularize_opt"] = "anti_alignment"
    validate_config(config)


def test_split_is_disjoint_and_seeded():
    a = split_indices(100, 1, .2, .1)
    b = split_indices(100, 1, .2, .1, .5)
    assert a == split_indices(100, 1, .2, .1)
    assert a["test"] == b["test"] and a["validate"] == b["validate"]
    assert len(a["train"]) == 72 and len(a["test"]) == 20
    assert not (set(a["train"]) & set(a["test"]))
    assert set(b["train"]) <= set(a["train"])


def test_features_and_edgeless_batch(tmp_path):
    graphs = load_dataset(ROOT / "data/example_molecules.csv")
    assert graphs[0].x.shape == (1, 37)
    assert graphs[0].edge_index.shape == (2, 0)
    assert graphs[0].edge_attr.shape == (0, 7)
    assert featurize("F", 1).x[0, 5] == 1
    assert graphs[1].edge_attr.shape[1] == 7
    batch = next(iter(DataLoader(graphs, batch_size=20)))
    assert batch.num_graphs == 20
    path = tmp_path / "data.pt"
    save_dataset(graphs, path)
    restored = load_dataset(path)
    torch.testing.assert_close(restored[1].x, graphs[1].x)


def test_scaler_handles_single_molecule_splits_without_training():
    graphs = load_dataset(ROOT / "data/example_molecules.csv")[:6]
    scaler = Scaler("logP", "standardization", device="cpu")
    scaler.fit(graphs)
    sample = [featurize("C", 1.)]
    assert scaler.extract_y(sample).shape == (1,)
    transformed = scaler.transform_dataset(sample)
    assert len(transformed) == 1
    restored = scaler.inverse_transform(transformed[0].y)
    torch.testing.assert_close(restored, torch.tensor(1.))


def test_smallest_dataset_scaling_without_training():
    graphs = load_dataset(ROOT / "data/example_molecules.csv")[:6]
    indices = split_indices(6, 1, .2, .1)
    splits = {name: [graphs[i] for i in ids] for name, ids in indices.items()}
    model = BGNN_unary("logP", "GIN", 37, 7, bayes=True, device="cpu")
    train, validation, test = model.scale_dataset(splits)
    assert (len(train), len(validation), len(test)) == (3, 1, 2)
    assert all(torch.isfinite(g.y) for split in (train, validation, test) for g in split)


@pytest.mark.parametrize("n_rand", [1, 100])
def test_independent_sampling_shapes_and_gradients(n_rand):
    model = BGNN_unary("logP", "GIN", 37, 7, bayes=True, device="cpu", dropout=0)
    model.n_rand = n_rand
    torch.manual_seed(7)
    weights, bias = model.sampling()
    assert weights.shape == (64, n_rand) and bias.shape == (n_rand,)
    noise = (weights - model.W_mu) / torch.log(1 + torch.exp(model.W_rho))
    assert not torch.allclose(noise[0], noise[1])
    (weights.square().mean() + bias.square().mean()).backward()
    for parameter in (model.W_mu, model.W_rho, model.b_mu, model.b_rho):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_independent_sampling_covariance_and_predictive_variance():
    model = BGNN_unary("logP", "GIN", 37, 7, bayes=True, device="cpu", dropout=0,
                       fingerprint_channels=3).double()
    model.n_rand = 100000
    with torch.no_grad():
        model.W_mu.copy_(torch.tensor([[1.], [-2.], [3.]]))
        model.W_rho.copy_(torch.log(torch.expm1(torch.tensor([[.3], [.7], [1.2]]))))
        model.b_rho.fill_(np.log(np.expm1(.4)))
    torch.manual_seed(19)
    weights, bias = model.sampling()
    sigma_w = torch.log(1 + torch.exp(model.W_rho))
    sigma_b = torch.log(1 + torch.exp(model.b_rho))
    noise = torch.cat(((weights - model.W_mu) / sigma_w,
                       ((bias - model.b_mu) / sigma_b).unsqueeze(0)))
    torch.testing.assert_close(torch.cov(noise), torch.eye(4, dtype=torch.float64), atol=.02, rtol=0)
    z = torch.tensor([1., -2., .5], dtype=torch.float64)
    predictions = z @ weights + bias
    expected_variance = ((z * sigma_w[:, 0]) ** 2).sum() + sigma_b.square().sum()
    torch.testing.assert_close(predictions.var(), expected_variance, rtol=.025, atol=0)
    torch.testing.assert_close(predictions.mean(), z @ model.W_mu[:, 0] + model.b_mu[0], atol=.03, rtol=0)


@pytest.mark.training
def test_training_checkpoint_predictions(tmp_path):
    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/smoke.json").read_text())
    train_one(load_dataset(ROOT / "data/example_molecules.csv"), config, 1, tmp_path, "cpu")
    loaded = load_model(tmp_path / "checkpoint_BGNN.pt")
    assert loaded.fingerprint_channels == 64
    assert not load_model(tmp_path / "checkpoint_GNN.pt").bayes
    result = np.load(tmp_path / "predictions_test.npz", allow_pickle=False)
    assert result["schema_version"].item() == 2
    assert result["posterior_sampling"].item() == "independent"
    assert result["z"].shape == (4, 64)
    assert np.isfinite(result["y_mean"]).all()
    assert np.isfinite(result["sigma_y"]).all()
    torch.manual_seed(42)
    first = loaded.predict(load_dataset(ROOT / "data/example_molecules.csv")[:4])
    reloaded = load_model(tmp_path / "checkpoint_BGNN.pt")
    torch.manual_seed(42)
    second = reloaded.predict(load_dataset(ROOT / "data/example_molecules.csv")[:4])
    np.testing.assert_array_equal(first, second)
    payload = torch.load(tmp_path / "checkpoint_BGNN.pt", weights_only=True)
    payload["schema_version"] = 1
    torch.save(payload, tmp_path / "old_checkpoint.pt")
    with pytest.raises(ValueError, match="schema 2"):
        load_model(tmp_path / "old_checkpoint.pt")
    payload["schema_version"] = 2
    payload["posterior_sampling"] = "shared"
    torch.save(payload, tmp_path / "wrong_sampling.pt")
    with pytest.raises(ValueError, match="sampling convention"):
        load_model(tmp_path / "wrong_sampling.pt")
    assert (tmp_path / "completed.json").exists()


def test_training_cli_metadata_without_fitting(tmp_path, monkeypatch, capsys):
    calls = []

    def record_run(graphs, config, run_id, destination, device):
        calls.append((run_id, destination.name, device))

    monkeypatch.setattr(training, "train_one", record_run)
    config = ROOT / "configs/smoke.json"
    training.main(["--config", str(config), "--dry-run"])
    assert json.loads(capsys.readouterr().out)["run_ids"] == [1, 2]
    assert calls == []
    output = tmp_path / "experiment"
    args = ["--config", str(config), "--data", str(ROOT / "data/example_molecules.csv"),
            "--output", str(output)]
    training.main(args)
    assert calls == [(1, "run_001", "cpu"), (2, "run_002", "cpu")]
    metadata = json.loads((output / "experiment.json").read_text())
    assert metadata["package_version"]
    assert len(metadata["data_sha256"]) == 64
    assert metadata["posterior_sampling"] == "independent"
    with pytest.raises(SystemExit):
        training.main(args)
    assert len(calls) == 2


@pytest.mark.parametrize("bayes", [False, True])
def test_checkpoint_roundtrip_without_training(tmp_path, bayes):
    model = BGNN_unary("logP", "GIN", 37, 7, bayes=bayes, device="cpu")
    model.n_rand = 10
    model.scaler = Scaler("logP", "standardization", device="cpu")
    graphs = load_dataset(ROOT / "data/example_molecules.csv")
    model.scaler.fit(graphs)
    path = tmp_path / "checkpoint.pt"
    save_model(model, path)
    loaded = load_model(path)
    assert loaded.bayes == bayes
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key])
    torch.manual_seed(42)
    expected = model.predict(graphs[:4])
    torch.manual_seed(42)
    np.testing.assert_array_equal(loaded.predict(graphs[:4]), expected)
    payload = torch.load(path, weights_only=True)
    payload["schema_version"] = 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="schema 2"):
        load_model(path)
    payload["schema_version"] = 2
    payload["posterior_sampling"] = "unsupported"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="sampling convention"):
        load_model(path)
