# Validation record

## Final upload review: 2026-10-09

- Reproduced a failure for single-molecule validation/test splits, then fixed scalar-target extraction to preserve the sample axis. Both regression tests pass; no objective or scaler-fit policy was changed.
- **35 passed, 1 deselected** in a fresh Python 3.12 CPU environment using the pinned Torch, PyG, RDKit, NumPy, pandas, SciPy, scikit-learn, Matplotlib and pytest versions from `pyproject.toml`.
- Command: `python -B -m pytest -p no:cacheprovider tests/test_core.py tests/test_metrics.py tests/test_settings.py --deselect tests/test_core.py::test_training_checkpoint_predictions -q`.
- The model-training test was explicitly deselected, and the optional original-source training comparisons were not collected. No optimizer-based model training was performed. The tests cover feature construction, input round trips, scaling, independent posterior draws/gradients, analytical variance, metric calculations/CLI, settings and split bookkeeping.
- One joblib warning reported physical-core detection falling back to logical cores; tests passed without suppressing it.
- A wheel built successfully from an isolated copy of the source and installed into the temporary environment. `pip check` found no broken requirements. Build outputs and the environment are outside the public folder.
- All four installed console commands returned `--help` successfully from outside the source directory. All six baseline configurations passed the installed training command's `--dry-run`, with run IDs 1 through 30 and the 500-epoch setting, without training.
- Citation and workflow YAML parsed successfully; citation/package versions and all seven author entries agree. CI explicitly excludes the model-training test.

This supersedes the earlier settings-only environment limitation below. It is still not a full training/GPU run or a scientific replication.

## Upload preparation: 2026-10-09

These checks validate source packaging and selected logic, not scientific reproduction. No model training, prediction, remote CI, GitHub push or release was performed during upload preparation.

- **13 settings tests passed** using `python -B -m unittest discover -s tests -p test_settings.py -v`.
- Parsed all 15 Python files and all eight configuration JSON files successfully.
- Parsed `pyproject.toml`, checked its version against `bgnn.__version__`, verified the README path and resolved all four console entry points to functions in the source.
- All 11 relative Markdown links in the public documents resolved to existing local files at the time of checking.
- Reviewed the 36-file upload tree: no generated model files, research datasets, caches, environment folders, OS metadata or parent manuscript/review files were included.
- A limited scan found no user-home/volume absolute paths or the checked private-key/cloud-token patterns. This is not a comprehensive security certification.

The settings tests compile selected methods without importing Torch or instantiating models. They cover the six baseline configs, 30 runs, epoch/dropout/patience defaults, batch boundaries, stopping thresholds and guards, train-only target span, warm-start path preservation and ECE defaults. They do not validate the full integrated training workflow.

The current settings-only environment uses Python 3.12 and the bundled NumPy 2.3.5, rather than the package's pinned NumPy 2.2.6. The full pinned environment and analysis integration tests were not rerun during upload preparation. GitHub Actions is configured for syntax, settings and analysis tests without model training; its remote result is not claimed here.

The external original-source directory was unavailable during this check, so its SHA256 comparison could not be repeated. The last successful comparison on 2026-10-07 found all 50 original source files unchanged. Their reference hashes remain in `source_hashes.json`. Upload preparation made no changes to original research files.

## Earlier numerical checks: 2026-10-07

Before the final epoch/stopping/dropout-setting changes, the full suite reported **26 passed** with 55 warnings in a pinned CPU environment. It included independent-noise/covariance checks, LPAS/DUC tests, checkpoint checks and five two-epoch synthetic comparisons with the corrected sampler applied to both implementations. A two-run, one-epoch synthetic smoke workflow also completed.

Those historical results are not a full-suite result for the current revision. The optional comparison test has since been adjusted to disable early termination during its fixed two-epoch comparison and has not been rerun. Historical shared-noise equivalence, full-size/GPU behavior, 30-run research experiments and final manuscript panels remain unverified.

## Repeating checks

From the repository root after installing the dependencies:

```bash
python -m unittest discover -s tests -p test_settings.py -v
python -m pytest tests/test_core.py tests/test_metrics.py --deselect tests/test_core.py::test_training_checkpoint_predictions -q
```

Both commands avoid model training. The full `python -m pytest -q` suite includes tiny training tests. Optional original-source comparisons require an explicitly supplied `BGNN_ORIGINAL_SOURCE` directory; original imports and top-level experiment cells are not executed. Do not infer historical paper reproduction from software-test success.
