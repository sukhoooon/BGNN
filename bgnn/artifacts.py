"""Save state dictionaries and plain metadata instead of Python model objects."""
import torch

from .models import BGNN_unary, Scaler


def save_model(model, path):
    keys = ("prop", "conv_model", "num_node_features", "num_edge_features",
            "num_layers", "hidden_channels", "fingerprint_channels", "dropout", "bayes")
    config = {key: getattr(model, key) for key in keys}
    scaler = {key: torch.as_tensor(getattr(model.scaler, key)).cpu()
              for key in ("min", "max", "mean", "std")}
    torch.save({"schema_version": 2,
                "posterior_sampling": model.posterior_sampling if model.bayes else "deterministic",
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": config, "scaler": scaler, "n_rand": model.n_rand}, path)


def load_model(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 2:
        raise ValueError("Expected schema 2; old checkpoints must not be silently reinterpreted with independent sampling")
    expected_sampling = "independent" if payload["config"]["bayes"] else "deterministic"
    if payload.get("posterior_sampling") != expected_sampling:
        raise ValueError("Checkpoint sampling convention does not match this model")
    model = BGNN_unary(**payload["config"], device="cpu")
    model.load_state_dict(payload["model"])
    model.scaler = Scaler(model.prop, "standardization", device="cpu")
    for key, value in payload["scaler"].items():
        setattr(model.scaler, key, value)
    model.n_rand = payload["n_rand"]
    return model.eval()
