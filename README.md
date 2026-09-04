# DeepLNP reproducibility package

This repository contains the code and machine-readable artifacts underlying the manuscript **“DeepLNP: role-aware dual-path learning for lipid nanoparticle delivery efficacy modelling and within-screen ranking.”**

## What is included

| Path | Contents |
|---|---|
| `merged_datasets/02_lnp_formulations/lnp_formulations_merged.csv` | Analysis-ready merged formulation table used by the reported experiments |
| `data/reproducibility/` | Row/group/fold assignments, feature schema, group summaries and endpoint-support audits |
| `deeplnp/` | Dataset, molecular-featurization and DeepLNP model code |
| `run_ablations.py` | Group-disjoint neural training, validation, held-screen evaluation and ablation runner |
| `results/selected_model/` | Frozen A115 fold summaries, aggregate metrics, integrity record and 12,692 held-screen OOF predictions |
| `results/manuscript_tables/` | Machine-readable values underlying the main and supplementary tables and quantitative figures |
| `bib_revision_analysis.py` | Regenerates the data-support, performance, application and robustness figures from the bundled artifacts |
| `figures/model_architecture.drawio` | Editable source for the DeepLNP architecture figure |

Historical checkpoints, third-party repository clones, temporary logs and superseded development experiments are intentionally excluded. They are not needed to verify the reported tables or regenerate the figures.

## Data lineage

The analysis-ready table integrates two upstream resources under their original terms:

1. **LNP_ML / LiON.** The source repository is <https://github.com/jswitten/LNP_ML>. The downloaded revision was `d4cb3dad295bbb78e9251c1679c438b614760ea1`; its 13,069 rows and 92 data columns were value-wise identical to the earlier snapshot used during reconstruction (`167822980dc26ba65c5c14539c4ce12b81b0b8f3`). The loader retained 12,692 delivery records.
2. **LNP Atlas v1.** The archived dataset is <https://doi.org/10.5281/zenodo.17243733> (concept DOI <https://doi.org/10.5281/zenodo.17243732>, CC BY 4.0). The unchanged source table contained 1,092 formulations; the loader retained 913 rows with a valid primary structure and at least one supported endpoint.

The resulting corpus contains 13,605 rows. Delivery labels are available for 12,692 LNP_ML rows across 38 experimental screens; the LNP Atlas rows supply physicochemical and bounded-toxicity observations but do not overlap delivery row-wise. The PEG2000 product label in LNP_ML is not treated as one exact molecular graph. Details and checksums are in [`data/SOURCE_PROVENANCE.md`](data/SOURCE_PROVENANCE.md) and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Environment

The reported run used Python 3.12.13, PyTorch 2.11.0+cu128, CUDA 12.8, RDKit 2026.03.4, NumPy 2.2.6, pandas 3.0.3, SciPy 1.18.0 and scikit-learn 1.9.0 on an NVIDIA GeForce RTX 4060 Ti. The complete tested version record is in [`environment-tested.txt`](environment-tested.txt). A minimal dependency specification is provided in [`requirements.txt`](requirements.txt).

When reproducing the Windows environment used for the manuscript, disable user-site packages so that unrelated packages do not shadow the selected environment:

```powershell
$env:PYTHONNOUSERSITE = '1'
python -m unittest discover -s tests
```

## Reproduce the selected DeepLNP run

The selected configuration is `A115_dual_rank_residual`. Each fold uses a fixed endpoint-balanced, group-disjoint partition and its own training seed:

```powershell
$seeds = 20260723, 20260724, 20260725, 20260726, 20260727
0..4 | ForEach-Object {
  python run_ablations.py --only A115_dual_rank_residual --epochs 6 `
    --seeds $seeds[$_] --sample-fraction 1.0 --split-seed 20260723 `
    --learning-rate 0.0005 --patience 2 --split-strategy balanced_group_kfold `
    --group-definition screen_context --group-folds 5 --group-fold-index $_ `
    --no-conformers
}
```

The frozen OOF predictions and the split assignments are bundled so that the manuscript’s reported metrics can be checked without retraining. Regenerate the quantitative analysis tables and figures with:

```powershell
python bib_revision_analysis.py
```

Outputs are written to `artifacts/`. The script asserts the 13,605-row corpus size, 12,692 unique A115 held-screen predictions and 38 held-out screens before producing results.

## Reconstruct from upstream data

The bundled analysis-ready table is the immutable input used for the manuscript. To reconstruct it from upstream sources, place the LNP Atlas v1 CSV at `method/LNP_Atlas-main/LNP_Atlas_DB_202509_v1.csv` and the LNP_ML table at `method/LNP_ML-main/data/all_data.csv`, then run `python merge_dataset.py`. Verify the resulting SHA-256 checksum against `REPRODUCIBILITY.md` before training. Raw upstream files must retain their original attribution and licence information.

## Citation

Please cite the manuscript and this repository. Citation metadata are provided in [`CITATION.cff`](CITATION.cff). A versioned GitHub release should be archived with Zenodo before final publication so that the code citation can use an immutable DOI.

## Licensing note

The upstream LNP_ML repository is distributed under the MIT License and LNP Atlas v1 is distributed under CC BY 4.0. Those terms continue to apply to the corresponding source-derived material. A separate licence for the original DeepLNP code has not yet been selected; the authors should add one before inviting third-party reuse.
