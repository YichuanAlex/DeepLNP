# Raw-data provenance

## LNP_ML / LiON

- Official repository: https://github.com/jswitten/LNP_ML
- Downloaded repository commit: `d4cb3dad295bbb78e9251c1679c438b614760ea1`
- Reproducibility snapshot already used by the merged corpus: commit `167822980dc26ba65c5c14539c4ce12b81b0b8f3`
- The downloaded and existing snapshots contain the same 13,069 rows and 92 data columns. Their byte hashes differ because of checkout/line-ending serialization; value-wise comparison is exact to floating-point tolerance.
- `Component_molecular_weights.csv` explicitly defines the fixed helper components `Cholesterol` and `C14-PEG2000`. The active loader therefore resolves positive LNP_ML cholesterol rows to cholesterol, while leaving PEG structure unresolved because a product-average PEG2000 label does not define one exact molecular graph.

## LNP Atlas

- Archived dataset DOI: https://doi.org/10.5281/zenodo.17243732
- Data descriptor DOI: https://doi.org/10.1038/s41597-025-06456-w
- The retained raw table contains 1,092 formulations and is stored unchanged in `LNP_Atlas_Zenodo_17243732/LNP_Atlas_raw.csv`.

Raw files are never modified in place. Column normalization, endpoint parsing, filtering, and source-aware component resolution occur in the merge/loader layer and are covered by checks in `tests/test_unified_dataset.py`.
