# Data preparation

## Included and external data

`example_molecules.csv` contains 20 SMILES strings with **synthetic targets** for software testing. These are not measured molecular properties or manuscript results.

Final research datasets, pretrained models and historical prediction archives are not distributed in this repository. The six baseline configurations do not download data. Dataset versions, final curation and distribution instructions will be added separately. Do not assume that another copy of a named public dataset is identical to the paper's input.

Store local inputs in `data/local/`, which is excluded from Git:

```text
data/
|-- README.md
|-- example_molecules.csv
`-- local/                 # Create locally; not uploaded
    |-- bp.csv
    |-- mp.csv
    |-- logP.csv
    |-- logPvap.csv
    |-- H_bonds.pt
    `-- G_bonds.pt
```

These names illustrate the layout; the files do not ship with this code.

## CSV input

Required columns are `smiles` and `target`. The bundled example illustrates the format:

```csv
smiles,target
C,1.0
CC,2.0
O,1.5
```

Each row represents one molecule and one finite numeric target. This three-row excerpt only illustrates syntax; the loader requires at least six molecules, and small training fractions may require more to obtain nonempty splits. Use valid RDKit-readable SMILES. No deduplication, outlier filtering or unit conversion is performed by the CSV loader.

The configuration's `property` selects the target being modeled, not an automatic transformation:

| Configuration property | Target meaning to establish before loading |
|---|---|
| `bp` | Boiling-point value and temperature unit |
| `mp` | Melting-point value and temperature unit |
| `logP` | Already transformed partition-coefficient target |
| `logPvap` | Already transformed vapor-pressure target; document log base and original pressure unit |
| `H_bonds` | Prepared energy target under the original energy-assembly convention |
| `G_bonds` | Prepared energy target under the original energy-assembly convention |

No extra logarithm is applied to a target named `logP` or `logPvap`. The historical `H_bonds`/`G_bonds` labels must not be treated as proof that raw formation energies can be used unchanged. The raw quantum-property energy transformation has not yet been ported or certified against the final paper data. Document all preprocessing and final target units with each dataset.

## Tensor graph input

Files written by `bgnn.data.save_dataset` contain schema-1 tensor-only graphs and are loaded with `torch.load(..., weights_only=True)`. Each graph stores SMILES, a scalar target, 37-column node features, a `[2, n_edges]` edge index, 7-column edge features and a charge field. Feature ordering is defined by `bgnn/features.py`; do not supply a different feature layout with the same column count.

Schema 1 here is the **dataset** schema, independent of schema 2 for **model checkpoints/predictions**. Generic model checkpoints are not accepted as training datasets.

## Trusted legacy graph conversion

Only convert a local pickle whose source you trust. Pickle loading can execute arbitrary code; this converter is not a security sandbox.

```bash
mkdir -p data/local
python -m bgnn.legacy /path/to/trusted/dataset_preprocessed.pkl data/local/logP.pt --trust-local-pickle
```

The converter preserves existing graph tensors and targets, with empty-edge shape normalization. It does not recover preprocessing, certify final dataset versions, convert arbitrary model/result pickles, or change target units. Parent output directories must exist; existing output files are not overwritten.

## Provenance for research inputs

When adding final data or external download instructions, record the source URL/version, redistribution conditions, property definition, units, transformations, exclusions, final row count and SHA256. Preserve row ordering: saved split indices refer to the input order.

The training workflow records the input-file SHA256 and per-run train/validation/test indices. It does not establish that these are the historical manuscript splits. Historical results must retain their actual run counts; missing runs must not be supplied by copying other results.
