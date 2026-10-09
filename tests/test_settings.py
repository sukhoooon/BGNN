"""No-training checks: python -m unittest discover -s tests -p test_settings.py.

Compile only the inspected methods to avoid importing Torch or executing models.
These checks do not replace integration tests in the pinned environment.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_function(module, name):
    tree = ast.parse((ROOT / "bgnn" / module).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"np": np, "cprint": lambda *args, **kwargs: None}
    exec(compile(ast.Module(body=[node], type_ignores=[]), module, "exec"), namespace)
    return namespace[name]


class PaperSettingsTests(unittest.TestCase):
    def setUp(self):
        self.stop = load_function("models.py", "satisfy_termination_criteria")
        self.model = SimpleNamespace(loss_thres_rel=.001, error_thres_rel=.001,
                                     loss_patience=60, loss_tolerance=0.,
                                     etype="mae", y_span=10.)

    def stopped(self, losses, error=1.):
        return self.stop(self.model, {"loss": losses, "mae_validate": [error]})

    def test_six_baseline_configs(self):
        paths = list((ROOT / "configs").glob("baseline_*.json"))
        self.assertEqual(len(paths), 6)
        for path in paths:
            with self.subTest(config=path.name):
                config = json.loads(path.read_text())
                fit = config["fit"]
                self.assertEqual(config["n_runs"], 30)
                self.assertEqual(config["gnn_dropout"], .3)
                self.assertEqual(config["bgnn_dropout"], .3)
                self.assertEqual(fit["n_epoch"], 500)
                self.assertEqual(fit["loss_patience"], 60)
                self.assertEqual(fit["loss_tolerance"], 0.)
                self.assertEqual(fit["loss_thres_rel"], .001)
                self.assertEqual(fit["error_thres_rel"], .001)
                self.assertIsNone(fit["n_batch"])

    def test_smoke_stays_small(self):
        config = json.loads((ROOT / "configs/smoke.json").read_text())
        self.assertEqual((config["n_runs"], config["fit"]["n_epoch"]), (2, 1))

    def test_fit_and_model_defaults(self):
        tree = ast.parse((ROOT / "bgnn/models.py").read_text())
        model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BGNN")
        fit = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == "fit")
        defaults = dict(zip([a.arg for a in fit.args.args][-len(fit.args.defaults):], fit.args.defaults))
        for name, expected in {"n_epoch": 500, "loss_patience": 60, "loss_tolerance": 0.}.items():
            self.assertEqual(ast.literal_eval(defaults[name]), expected)
        unary = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BGNN_unary")
        init = next(n for n in unary.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        defaults = dict(zip([a.arg for a in init.args.args][-len(init.args.defaults):], init.args.defaults))
        self.assertEqual(ast.literal_eval(defaults["dropout"]), .3)

    def test_batch_boundaries(self):
        batch_size = load_function("data.py", "paper_batch_size")
        for size, expected in [(1, 64), (9999, 64), (10000, 256), (99999, 256),
                               (100000, 1024), (1000000, 1024), (10000000, 1024)]:
            self.assertEqual(batch_size(size), expected)
        for size in [0, -1]:
            with self.assertRaises(ValueError):
                batch_size(size)

    def test_loss_ratio_and_strict_threshold(self):
        self.assertTrue(self.stopped([100., .09]))
        self.assertFalse(self.stopped([100., .1]))

    def test_nonpositive_initial_loss_does_not_use_ratio(self):
        self.assertFalse(self.stopped([0., -1.]))
        self.assertFalse(self.stopped([-10., 1.]))

    def test_error_ratio_and_strict_threshold(self):
        self.assertTrue(self.stopped([100.], error=.009))
        self.assertFalse(self.stopped([100.], error=.01))

    def test_error_rule_still_applies_after_patience_window(self):
        self.assertTrue(self.stopped(list(range(100, 39, -1)), error=.009))

    def test_plateau_requires_sixty_epochs_without_improvement(self):
        self.assertFalse(self.stopped([100.] * 60))
        self.assertTrue(self.stopped([100.] * 61))
        self.assertFalse(self.stopped([100.] * 60 + [99.999]))

    def test_zero_span_does_not_divide_by_zero(self):
        self.model.y_span = 0.
        self.assertFalse(self.stopped([100.], error=0.))
        self.assertTrue(self.stopped([100.] * 61, error=0.))

    def test_span_uses_training_targets_only(self):
        get_span = load_function("models.py", "get_y_span")
        datasets = {"train": [SimpleNamespace(y=2.), SimpleNamespace(y=5.)],
                    "validate": [SimpleNamespace(y=-100.)], "test": [SimpleNamespace(y=100.)]}
        self.assertEqual(get_span(None, datasets), 3.)

    def test_warmstart_keeps_parent_directory(self):
        tree = ast.parse((ROOT / "bgnn/models.py").read_text())
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == "load_weights")
        expr = compile(ast.Expression(call.args[0]), "models.py", "eval")
        checkpoint = Path("BGNN_paperwork/BGNN_public/outputs/run_001/checkpoint_BGNN.pt")
        self.assertEqual(eval(expr, {"Path": Path, "checkpointdir": str(checkpoint)}),
                         checkpoint.with_name("checkpoint_GNN.pt"))

    def test_ece_default_is_paper(self):
        for module, name in [("metrics.py", "regression_ece"), ("analysis.py", "summarize")]:
            fn = load_function(module, name)
            self.assertEqual(fn.__defaults__[0], "paper")
        tree = ast.parse((ROOT / "bgnn/analysis.py").read_text())
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and n.args
                    and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--convention")
        default = next(k.value for k in call.keywords if k.arg == "default")
        self.assertEqual(ast.literal_eval(default), "paper")


if __name__ == "__main__":
    unittest.main()
