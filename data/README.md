# Data preparation

## Included data

`example_molecules.csv` contains 20 molecules with synthetic targets for software tests and the smoke workflow. Research datasets and pretrained models are not bundled. The baseline configurations read local data and do not download datasets.

Store prepared inputs under `data/local/`, which is excluded from Git. A configuration's `property` selects the modeled target; it does not transform the supplied values.

## CSV input

Required columns are `smiles` and `target`, with one molecule and one finite scalar target per row. Use valid RDKit-readable SMILES. The bundled CSV provides a complete format example.

At least six molecules are required. A reduced training fraction may require more samples to keep all splits nonempty. The loader does not perform deduplication, outlier filtering, unit conversion, logarithms, or energy assembly.

| Property | Required prepared target |
|---|---|
| `bp` | Boiling point in the chosen temperature unit |
| `mp` | Melting point in the chosen temperature unit |
| `logP` | Log partition coefficient |
| `logPvap` | Log vapor pressure, with log base and pressure unit recorded |
| `H_bonds` | Prepared enthalpy-based bond-energy target |
| `G_bonds` | Prepared Gibbs-energy-based bond-energy target |

For `H_bonds` and `G_bonds`, perform the dataset's molecular/atomic energy assembly before loading; raw molecular energies are not converted by this package. Record preprocessing and final target units alongside each research dataset.

## Tensor graph input

`bgnn.data.save_dataset(graphs, path)` writes a schema-1 tensor graph file. The loader uses `torch.load(..., weights_only=True)`.

Each graph contains `smiles`, scalar `y`, 37-column node features `x`, a `[2, n_edges]` `edge_index`, 7-column `edge_attr`, and `charge`. Feature order must match `bgnn/features.py`. Empty-edge graphs use shapes `[2, 0]` and `[0, 7]`.

Dataset schema 1 is distinct from checkpoint/prediction schema 2.

## Research input records

For each dataset, retain the source URL/version, redistribution conditions, target definition and units, transformations, exclusions, final row count, and file SHA256. Preserve row ordering because saved split indices refer to input order.

Training automatically records the input SHA256 and per-run train/validation/test indices. See [usage](../docs/usage.md) for saved outputs and [implementation](../docs/implementation.md) for target scaling.
