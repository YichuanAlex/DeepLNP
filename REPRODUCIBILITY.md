# Reproducibility and provenance record

## Frozen inputs and outputs

| Artifact | SHA-256 |
|---|---|
| Analysis-ready merged table | `49f902ff6d40de89915e2f2b3e09ff23dbc76f203f802c8d64ce3702cfc02e90` |
| Selected predictor source | `8cc7626e6f04c67be58a570aaa7a444ab81879c387bf38f2128d73f48fda6040` |
| Neural ablation runner | `69aa8336f52706540c8634e42e87f0e81d362a87ca2b43839ec268fa3161505f` |

The selected model is `A115_dual_rank_residual`. The bundled compressed OOF file contains exactly 12,692 test predictions with unique dataset indices and covers all 38 delivery screens. `data/reproducibility/split_assignments.csv` records row, group and fold assignments for the loader-defined 13,605-row corpus.

## Loader audit note

`results/selected_model/integrity.json` retains the hash `09fbb77d...` under the legacy field `loader_sha256`. The held-screen runner itself imports `deeplnp.data.unified_dataset`; it does not import `merged_dataset_loader.py`. The legacy loader was subsequently updated during the PEG/sterol structure-resolution audit, while the immutable merged table and selected predictor/runner hashes remained unchanged. The current source is included so that the audit trail is visible, but the reported A115 results are tied to the immutable table, split assignments, bundled OOF predictions and runner listed above.

## Leakage controls

- Experimental groups are assigned before endpoint preprocessing.
- Training-only statistics are used for model fitting and response scaling.
- Validation groups determine checkpoint and ablation selection.
- Held-out test labels are not used for fitting, preprocessing, thresholding or hybrid-weight selection.
- Ranking pairs and screen-centred losses are constructed within screen.
- Reported OOF rows are unique across the five held-screen folds.

## Quantitative figure provenance

`results/analysis_manifest.json` maps the manuscript figures to their machine-readable inputs. `results/manuscript_tables/` contains the corresponding values. Running `bib_revision_analysis.py` regenerates the quantitative figure set under `artifacts/figures/` and refreshed tables under `artifacts/tables/`.
