# Source-to-public code mapping

All 50 top-level source files are accounted for below. `Pending` means **not implemented**, even when a planned destination is named. No excluded original file was modified or deleted. Source line numbers refer to `source_hashes.json`.

The package consolidates small shared responsibilities. Only `models.py`, `features.py`, `runtime.py`, `data.py`, `training.py`, `artifacts.py`, `legacy.py`, `metrics.py`, `analysis.py`, `plotting.py` and package initialization currently exist under `bgnn/`. `configs/` uses JSON so standard-library configuration parsing suffices. Planned destinations in the table are not existing files.

| Original | Status | Current / planned destination | Retained or reviewed content |
|---|---|---|---|
| `1_fig_drawing.py` | Pending historical figure port | planned bgnn/figures/paradox.py | 47-281 posterior/performance/loss; 282-401 latent; 403-780 panels. Resolve LPAS convention and archive selection first. |
| `1_paradox_laa.py` | Reference; panels pending | planned postprocess.py; figures/supplementary.py | 252-454 bias/contributions, 645-730 metrics, 1072-1307 t-SNE, 1308-1359 outputs. Deduplicate against final drawing. |
| `1_paradox_laa_qm.py` | Reference; panels pending | planned postprocess.py; figures/paradox.py | 31-129 saved models, 130-616 QM statistics, 617-702 alignment, 704-1103 panels. |
| `2_fig_1_z_regularization_drawing.py` | Pending | planned figures/interventions.py | 67-305 final panels, 307-483 helper calculations. Review uncertainty-based run selection at 101+, drop legacy 651+. |
| `2_fig_1_z_regularization_qm.py` | Pending | planned figures/interventions.py | 37-320 inputs/panels, 323-533 latent; make selected conditions and retained runs explicit. |
| `2_fig_2_kl_drawing.py` | Pending | planned figures/interventions.py | 31-90 inputs, 107-201 performance, 208-439 latent panels. |
| `2_fig_2_kl_drawing_cal_qm.py` | Pending | planned postprocess.py; figures/interventions.py | 24-173 aggregation and 174-510 panels; remove 687+ ignore diagnostics. |
| `2_fig_3_prior_drawing.py` | Pending | planned figures/interventions.py | 31-206 performance/posterior, 207-411 latent panels. |
| `2_fig_3_prior_drawing_cal_qm.py` | Pending | planned postprocess.py; figures/interventions.py | 24-310 metrics/panels, 351-601 latent. Exclude 311-350 older posterior-mean panel. |
| `2_z_regularize.py` | Excluded older version | None | Highest numbered non-autosave version wins; originals remain untouched. Older QM alignment file also has a syntax error. |
| `2_z_regularize_2.py` | Final-version reference; pending | planned postprocess.py; regularization config | 25-205 aggregation, 311-438 metrics. Preserve L1/L2/anti-align naming; eliminate mid-cell overrides. |
| `2_z_regularize_check.py` | Excluded older version | None | Highest numbered non-autosave version wins; originals remain untouched. Older QM alignment file also has a syntax error. |
| `2_z_regularize_qm.py` | Excluded older version | None | Highest numbered non-autosave version wins; originals remain untouched. Older QM alignment file also has a syntax error. |
| `2_z_regularize_qm_2.py` | Excluded older version | None | Highest numbered non-autosave version wins; originals remain untouched. Older QM alignment file also has a syntax error. |
| `2_z_regularize_qm_3.py` | Final QM reference; pending | planned postprocess.py; regularization config | 26-204 aggregation, 449-579 metrics; QM grid includes .001. Do not merge grids blindly. |
| `3_fig_drawing.py` | Pending final AGL panels | planned figures/agl.py | 32-168 metric summary, 169-239 latent, 240-299 maps, 301-584 helpers. Reconcile result_2 versus later analysis paths. |
| `3_fig_drawing_qm.py` | Pending final QM panels | planned figures/agl.py | 41-298 normalization and 305-472 summary; deduplicate spider code without silently changing metric choice. |
| `3_kl_test.py` | Pending | planned postprocess.py; KL configs | 24-143 saved metrics, 162-191 performance, 369-464 alignment; confirm data-regime labels. |
| `4_prior_test.py` | Pending | planned postprocess.py; prior configs | 24-170 prior results, 345-422 alignment, 423-518 metrics; distinguish prior standard deviation from variance. |
| `6_spider_plot.py` | Supplementary reference only | planned figures/supplementary.py if used | Retain radar panels only if present in final manuscript; review filtering and MI versus Spearman definitions. |
| `6_spider_plot_qm.py` | Supplementary reference only | planned figures/supplementary.py if used | Retain radar panels only if present in final manuscript; review filtering and MI versus Spearman definitions. |
| `6_z_alignment_4.py` | Excluded older version | None | Highest numbered non-autosave version wins; originals remain untouched. Older QM alignment file also has a syntax error. |
| `6_z_alignment_5.py` | Metric reference; panels pending | bgnn/metrics.py; planned figures/agl.py | KDE/DUC 779-874: remove the extra log. Alignment 1072-1100: default to manuscript per-molecule normalization; source aggregation is an explicit diagnostic. Test-metric coefficient selection 414+ is not migrated. |
| `6_z_alignment_5_autosave.py` | Excluded older version | None | Highest numbered non-autosave version wins; originals remain untouched. Older QM alignment file also has a syntax error. |
| `6_z_alignment_qm.py` | Excluded older version | None | Highest numbered non-autosave version wins; originals remain untouched. Older QM alignment file also has a syntax error. |
| `6_z_alignment_qm_2.py` | QM metric reference; panels pending | bgnn/metrics.py; planned figures/agl.py | QM counterpart 708-803 and later alignment; keep QM condition/key differences for future artifact conversion. |
| `__init__.py` | Replaced | bgnn/__init__.py | New package metadata; original empty file not copied. |
| `active_learning.py` | Rewritten orchestration | bgnn/training.py; bgnn/data.py | train_model_all 150-249; replace global-path and active-learning workflow with seeded runs and recorded splits. Historical split helper 82-116 is reference, not claimed equivalent. |
| `active_learning_post_functions.py` | Metric ported | bgnn/metrics.py | regression_ece 241-272; paper confidence grid is the default, source grid remains opt-in. |
| `architecture.py` | Adapted | bgnn/models.py; bgnn/artifacts.py | Scaler 77-124, numerical BGNN methods 127-941, unary 944-978. Replace shared noise at 516+ with independent weight/bias draws; retain likelihood and penalty values. Rename `align` to `anti_alignment`. New schema-2 state-dict IO records sampling. |
| `cal_latent_functions.py` | Investigated; old hook path pending | bgnn/training.py (new readout latent); planned legacy latent adapter | cal_latent 133-179, QM hook 263-300, save_latent 359-402. Draft saves actual model.z, not a claim of equivalence to archived last-conv hooks. |
| `cal_latent_run.py` | Excluded duplicate/old workflow | Replacement roles in training.py, analysis.py; pending exact panels | Do not copy duplicate training, external analysis imports or interactive execution cells. |
| `chemistry.py` | Pending source curation | planned bgnn/chemistry.py | Sanitization 1087-1138 and atom-energy helpers 943-986. Generic CSV uses RDKit without new canonical reordering; raw-source energy processing not yet ported. |
| `chemprice.py` | Excluded unrelated or infrastructure | None | Outside six-property unary path; exclude pricing/API credentials, Ray server setup and unused architectures. |
| `cutils.py` | Excluded unrelated or infrastructure | None | Outside six-property unary path; exclude pricing/API credentials, Ray server setup and unused architectures. |
| `data_analysis.py` | Excluded duplicate/old workflow | Replacement roles in training.py, analysis.py; pending exact panels | Do not copy duplicate training, external analysis imports or interactive execution cells. |
| `data_analysis_functions.py` | Excluded duplicate/old workflow | Replacement roles in training.py, analysis.py; pending exact panels | Do not copy duplicate training, external analysis imports or interactive execution cells. |
| `data_analysis_model_training.py` | Partly ported | bgnn/training.py; bgnn/data.py; bgnn/artifacts.py | Training wrapper 137-192, prediction mean/std 195-208, full-N batch rule 397-408. Remove plotting coupling; pretrain on each new split, not shared legacy checkpoint. |
| `datamanager.py` | Partly replaced; raw loaders pending | bgnn/data.py; planned chemistry.py | Prepared graph loading 136+, full-data batch inputs; new recorded split replaces historical split. bp 611-647, mp 650-666, logP 684-714, logPvap 735-777, G 467-525, H 549-608 require final input provenance. |
| `drawer.py` | Figure support pending | planned figures/style.py; figures/supplementary.py | Retain final styles, parity/amine/MW panels and necessary molecule depictions; exclude server fonts and GUI side effects. |
| `feature.py` | Extracted | bgnn/features.py | NODE/EDGE_FEATURES 35-72, bond query 127-131, mol_to_features 160-203, UnaryMolData 237-256; unary featurizing simplified. Preserve ordering; normalize edgeless shapes. |
| `figure_config.py` | Figure support pending | planned figures/style.py; figures/supplementary.py | Retain final styles, parity/amine/MW panels and necessary molecule depictions; exclude server fonts and GUI side effects. |
| `layer.py` | Excluded unrelated or infrastructure | None | Outside six-property unary path; exclude pricing/API credentials, Ray server setup and unused architectures. |
| `modeling.py` | Excluded duplicate/old workflow | Replacement roles in training.py, analysis.py; pending exact panels | Do not copy duplicate training, external analysis imports or interactive execution cells. |
| `pKa.py` | Excluded unrelated or infrastructure | None | Outside six-property unary path; exclude pricing/API credentials, Ray server setup and unused architectures. |
| `plotter.py` | Figure support pending | planned figures/style.py; figures/supplementary.py | Retain final styles, parity/amine/MW panels and necessary molecule depictions; exclude server fonts and GUI side effects. |
| `price.py` | Excluded unrelated or infrastructure | None | Outside six-property unary path; exclude pricing/API credentials, Ray server setup and unused architectures. |
| `putils.py` | Excluded unrelated or infrastructure | None | Outside six-property unary path; exclude pricing/API credentials, Ray server setup and unused architectures. |
| `utils.py` | Replaced selectively | bgnn/runtime.py | Explicit seed/logging/checks only; no root scanning, import-time style, work schedules, destructive directory helpers or server configuration. |
| `webtool.py` | Excluded unrelated or infrastructure | None | Outside six-property unary path; exclude pricing/API credentials, Ray server setup and unused architectures. |

## New code without a direct predecessor

- `runtime.run_ids` and completeness checks: inclusive 30-run policy; no padding of archives.
- `artifacts`: tensor-only inference checkpoints instead of pickled optimizer/model objects.
- `legacy`: opt-in trusted-pickle graph conversion; no original modules imported.
- `analysis`: complete-run aggregation with convention labels; no automatic test-based filtering.
- `plotting`: diagnostic new-run summary, not an exact manuscript figure replacement.
- `tests`: repeat-count, split, feature, independent sampling/variance, metric direction, IO and retained-training checks using the same corrected sampler on both sides.
- `configs/arxiv_reference.json`: non-executable manuscript parameter record, not a validated training recipe.

See `terminology.md` for the separate identifier and metric-definition mapping (AAI/LPAS, UDI/UDC/DUC, LAA/LPA and sigma/rho distinctions).
