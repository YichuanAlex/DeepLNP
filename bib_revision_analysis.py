"""Generate BIB revision analyses and publication-ready figures.

This script performs data-support and endpoint-overlap audits on the exact
13,605-row corpus exposed by ``UnifiedLNPFormulationDataset`` and re-analyses
the frozen endpoint-balanced five-fold out-of-fold screen metrics. It does not
fit or tune a new predictive model and never uses held-out labels for model
selection.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from rdkit import Chem

from deeplnp.data.unified_dataset import (
    FORMULATION_COLUMNS,
    TASK_COLUMNS,
    UnifiedLNPFormulationDataset,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
FIG = OUT / "figures"
TAB = OUT / "tables"
SOURCE_AUDIT = Path(
    os.environ.get("DEEPLNP_DECISION_AUDIT_DIR", ROOT / "results" / "decision_audit")
)
SOURCE_METRICS = SOURCE_AUDIT / "per_screen.csv"
SOURCE_SUMMARY = SOURCE_AUDIT / "summary.csv"
SOURCE_MATCHED_NEURAL = ROOT / "results" / "selected_model"
SOURCE_OOF = SOURCE_MATCHED_NEURAL / "a115_held_screen_oof_predictions.csv.gz"

MODEL_LABELS = {
    "A115_dual_rank_residual": "DeepLNP",
    "Metadata only / Extra Trees": "Metadata Extra Trees",
    "Ionizable Morgan + metadata / Extra Trees": "Ionizable Morgan Extra Trees",
    "validation_selected_neural_tree_hybrid": "Validation-selected hybrid",
}
MODEL_ORDER = [
    "DeepLNP",
    "Metadata Extra Trees",
    "Ionizable Morgan Extra Trees",
    "Validation-selected hybrid",
]
MODEL_SHORT = {
    "DeepLNP": "DeepLNP",
    "Metadata Extra Trees": "Metadata ET",
    "Ionizable Morgan Extra Trees": "Morgan ET",
    "Validation-selected hybrid": "Hybrid",
}
COLORS = {
    "DeepLNP": "#D97930",
    "Metadata Extra Trees": "#4C9BE8",
    "Ionizable Morgan Extra Trees": "#2A9D8F",
    "Validation-selected hybrid": "#6F4E9C",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def _save_figure(fig: plt.Figure, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".svg", ".png"]:
        fig.savefig(
            FIG / f"{stem}{suffix}",
            dpi=600 if suffix == ".png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def _clean_series(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip()


def load_exact_corpus() -> tuple[UnifiedLNPFormulationDataset, pd.DataFrame]:
    dataset = UnifiedLNPFormulationDataset(
        str(ROOT / "merged_datasets"),
        sample_fraction=1.0,
        random_seed=20260723,
        use_spatial=False,
        include_source_context=False,
        group_definition="screen_context",
    )
    df = dataset.df.copy()
    df["_reported_biological_target"] = _raw_biological_target_evidence(df)
    return dataset, df


def _raw_biological_target_evidence(loaded: pd.DataFrame) -> np.ndarray:
    """Return source-reported target evidence aligned to the loaded corpus."""
    raw = pd.read_csv(
        ROOT
        / "merged_datasets"
        / "02_lnp_formulations"
        / "lnp_formulations_merged.csv",
        low_memory=False,
    )
    direct = _clean_series(
        raw.get("delivery_target", pd.Series("", index=raw.index))
    ).str.len().gt(0)
    profiles = _clean_series(
        raw.get("bioactivity_profile", pd.Series("", index=raw.index))
    )
    profile = profiles.map(
        lambda text: bool(
            UnifiedLNPFormulationDataset._profile_value(
                text, "biodistribution_result"
            )
            or UnifiedLNPFormulationDataset._profile_value(
                text, "gene_expression_result"
            )
        )
    )

    primary_smiles = _clean_series(raw["primary_smiles"])
    endpoint_columns = [
        column
        for columns in TASK_COLUMNS.values()
        for column in columns
        if column in raw.columns
    ]
    retained = primary_smiles.str.len().gt(0) & raw[endpoint_columns].apply(
        pd.to_numeric, errors="coerce"
    ).notna().any(axis=1)
    evidence = (direct | profile).loc[retained].to_numpy(dtype=bool)
    aligned_smiles = primary_smiles.loc[retained].to_numpy(dtype=str)
    if len(evidence) != len(loaded) or not np.array_equal(
        aligned_smiles, loaded["primary_smiles"].to_numpy(dtype=str)
    ):
        raise RuntimeError(
            "Raw evidence no longer aligns with the loaded corpus; update the "
            "retention audit before reporting coverage."
        )
    return evidence


def support_and_overlap_audit(
    dataset: UnifiedLNPFormulationDataset, df: pd.DataFrame
) -> dict[str, object]:
    endpoint_names = {
        "efficiency": "Delivery",
        "particle_size": "Size",
        "pdi": "PDI",
        "zeta_potential": "Zeta",
        "encapsulation": "Encapsulation",
        "toxicity": "Toxicity",
    }
    endpoint_masks = {
        task: df[f"target_{task}"].notna().to_numpy()
        for task in endpoint_names
    }

    role_masks: dict[str, np.ndarray] = {}
    for component, label in [
        ("ionizable", "Ionizable graph resolved"),
        ("helper", "Helper graph resolved"),
        ("cholesterol", "Sterol graph resolved"),
        ("peg", "PEG-lipid graph resolved"),
    ]:
        observed = np.fromiter(
            (
                dataset._component_info(row, component)[1]
                for _, row in df.iterrows()
            ),
            dtype=bool,
            count=len(df),
        )
        role_masks[label] = observed

    role_columns = {
        "ionizable": ["ionizable_lipid_smiles", "primary_smiles", "smiles"],
        "helper": ["helper_lipid_smiles", "helper_lipid", "helper_lipid_id"],
        "cholesterol": ["sterol_lipid_smiles", "sterol_lipid", "cholesterol"],
        "peg": ["peg_lipid_smiles", "peg_lipid"],
    }
    ratio_columns = {
        "ionizable": "cationic_lipid_mol_ratio",
        "helper": "phospholipid_mol_ratio",
        "cholesterol": "cholesterol_mol_ratio",
        "peg": "peg_lipid_mol_ratio",
    }
    support_rows = []
    for component, columns in role_columns.items():
        active = pd.to_numeric(df[ratio_columns[component]], errors="coerce").fillna(0).gt(0)
        identities: list[str] = []
        graphs: list[str] = []
        identity_known: list[bool] = []
        graph_resolved: list[bool] = []
        for _, row in df.iterrows():
            identity = next(
                (dataset._clean(row.get(column)) for column in columns if dataset._clean(row.get(column))),
                "",
            )
            if (
                component == "cholesterol"
                and not identity
                and dataset._clean(row.get("dataset_source")).lower() == "lnp_ml"
                and pd.to_numeric(row.get("cholesterol_mol_ratio"), errors="coerce") > 0
            ):
                identity = "Cholesterol (source-defined fixed role)"
            smiles, observed = dataset._component_info(row, component)
            canonical = ""
            if observed:
                molecule = dataset._mol(smiles)
                canonical = dataset._clean(smiles)
                if molecule is not None:
                    canonical = Chem.MolToSmiles(molecule, canonical=True)
            identities.append(identity)
            graphs.append(canonical)
            identity_known.append(bool(identity))
            graph_resolved.append(bool(observed))
        identity_mask = np.asarray(identity_known, dtype=bool)
        graph_mask = np.asarray(graph_resolved, dtype=bool)
        active_mask = active.to_numpy(dtype=bool)
        delivery_mask = endpoint_masks["efficiency"]
        identity_values = np.asarray(identities, dtype=object)
        graph_values = np.asarray(graphs, dtype=object)
        support_rows.append(
            {
                "role": component,
                "active_rows": int(active_mask.sum()),
                "identity_known_rows": int(identity_mask.sum()),
                "graph_resolved_rows": int(graph_mask.sum()),
                "active_and_graph_resolved_rows": int((active_mask & graph_mask).sum()),
                "unique_reported_identities": int(len({value for value in identities if value})),
                "unique_graphs": int(len({value for value in graphs if value})),
                "delivery_active_rows": int((delivery_mask & active_mask).sum()),
                "delivery_identity_known_rows": int((delivery_mask & identity_mask).sum()),
                "delivery_graph_resolved_rows": int((delivery_mask & graph_mask).sum()),
                "delivery_active_and_graph_resolved_rows": int(
                    (delivery_mask & active_mask & graph_mask).sum()
                ),
                "delivery_unique_reported_identities": int(
                    len({value for value in identity_values[delivery_mask] if value})
                ),
                "delivery_unique_graphs": int(
                    len({value for value in graph_values[delivery_mask] if value})
                ),
            }
        )
    component_support = pd.DataFrame(support_rows)

    process_columns = [
        name for name, _ in FORMULATION_COLUMNS if name in df.columns
    ]
    process_observed = (
        df[process_columns]
        .apply(pd.to_numeric, errors="coerce")
        .notna()
        .any(axis=1)
        .to_numpy()
    )
    target_observed = df["_reported_biological_target"].to_numpy(dtype=bool)
    group_observed = np.asarray(
        [bool(str(group).strip()) for group in dataset.groups], dtype=bool
    )
    feature_masks = {
        **role_masks,
        "Any reported composition/process field": process_observed,
        "Biological target": target_observed,
        "Experimental group": group_observed,
    }

    counts = pd.DataFrame(
        index=feature_masks.keys(), columns=endpoint_names.values(), dtype=int
    )
    coverage = counts.astype(float)
    endpoint_count_rows = []
    for task, endpoint_label in endpoint_names.items():
        endpoint_mask = endpoint_masks[task]
        endpoint_count = int(endpoint_mask.sum())
        endpoint_count_rows.append(
            {"endpoint": endpoint_label, "available_rows": endpoint_count}
        )
        for feature_label, feature_mask in feature_masks.items():
            joint = int(np.sum(endpoint_mask & feature_mask))
            counts.loc[feature_label, endpoint_label] = joint
            coverage.loc[feature_label, endpoint_label] = (
                joint / endpoint_count if endpoint_count else np.nan
            )

    endpoint_count_frame = pd.DataFrame(endpoint_count_rows)
    overlap = pd.DataFrame(
        index=endpoint_names.values(), columns=endpoint_names.values(), dtype=int
    )
    for task_a, label_a in endpoint_names.items():
        for task_b, label_b in endpoint_names.items():
            overlap.loc[label_a, label_b] = int(
                np.sum(endpoint_masks[task_a] & endpoint_masks[task_b])
            )

    all_four = np.logical_and.reduce(list(role_masks.values()))
    delivery = endpoint_masks["efficiency"]
    source = _clean_series(
        df.get("dataset_source", pd.Series("unknown", index=df.index))
    )
    summary = {
        "rows": int(len(df)),
        "delivery_rows": int(delivery.sum()),
        "all_four_structure_rows": int(all_four.sum()),
        "delivery_and_all_four_structure_rows": int(np.sum(delivery & all_four)),
        "efficacy_experimental_groups": int(
            len(set(np.asarray(dataset.groups, dtype=object)[delivery]))
        ),
        "source_counts": {
            str(key): int(value) for key, value in source.value_counts().items()
        },
        "support_definitions": {
            "Any reported composition/process field": (
                "logical OR across available composition and process fields; "
                "not complete process coverage"
            ),
            "Biological target": (
                "raw delivery_target or a non-empty biodistribution_result/"
                "gene_expression_result field in bioactivity_profile"
            ),
        },
    }

    TAB.mkdir(parents=True, exist_ok=True)
    counts.to_csv(TAB / "support_joint_counts.csv", index_label="feature_block")
    coverage.to_csv(
        TAB / "support_conditional_coverage.csv", index_label="feature_block"
    )
    endpoint_count_frame.to_csv(TAB / "endpoint_counts.csv", index=False)
    overlap.to_csv(TAB / "endpoint_overlap_counts.csv", index_label="endpoint")
    component_support.to_csv(TAB / "component_support_levels.csv", index=False)
    with (TAB / "corpus_support_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    plot_support_matrix(coverage, counts, overlap)
    plot_protocol()
    return {
        "summary": summary,
        "counts": counts,
        "coverage": coverage,
        "overlap": overlap,
    }


def plot_support_matrix(
    coverage: pd.DataFrame, counts: pd.DataFrame, overlap: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.7),
        gridspec_kw={"width_ratios": [1.18, 0.82]},
    )
    left, right = axes
    image = left.imshow(
        coverage.to_numpy(float), cmap="Blues", vmin=0, vmax=1, aspect="auto"
    )
    left.set_xticks(
        np.arange(coverage.shape[1]),
        coverage.columns,
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    left.set_yticks(np.arange(coverage.shape[0]), coverage.index)
    left.set_title("a  Endpoint-conditional input coverage", loc="left", fontweight="bold")
    for row in range(coverage.shape[0]):
        for col in range(coverage.shape[1]):
            value = float(coverage.iloc[row, col])
            count = int(counts.iloc[row, col])
            color = "white" if value >= 0.62 else "#222222"
            if np.isclose(value, 1.0):
                display = f"all\n({count:,})"
            elif np.isclose(value, 0.0):
                display = "none\n(0)"
            else:
                display = f"{100 * value:.1f}%\n({count:,})"
            left.text(
                col,
                row,
                display,
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )
    colorbar = fig.colorbar(image, ax=left, fraction=0.035, pad=0.02)
    colorbar.set_label("Fraction of endpoint-labelled rows")

    overlap_values = overlap.to_numpy(float)
    overlap_log = np.log10(overlap_values + 1)
    right.imshow(overlap_log, cmap="YlGnBu", aspect="equal")
    right.set_xticks(
        np.arange(overlap.shape[1]),
        overlap.columns,
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    right.set_yticks(np.arange(overlap.shape[0]), overlap.index)
    right.set_title("b  Endpoint co-observation", loc="left", fontweight="bold")
    maximum = overlap_log.max()
    for row in range(overlap.shape[0]):
        for col in range(overlap.shape[1]):
            value = int(overlap.iloc[row, col])
            color = "white" if overlap_log[row, col] > 0.58 * maximum else "#222222"
            right.text(
                col,
                row,
                f"{value:,}",
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )
    for axis in axes:
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.text(
        0.01,
        0.005,
        "Composition/process support is a logical OR across reported fields, not complete process coverage.",
        ha="left",
        va="bottom",
        fontsize=7,
        color="#455A64",
    )
    fig.tight_layout(w_pad=2.0, rect=(0, 0.055, 1, 1))
    _save_figure(fig, "data_support_and_endpoint_overlap")


def _draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    color: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.25,
        edgecolor=color,
        facecolor=color + "16",
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.025,
        y + height - 0.040,
        title,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=color,
    )
    ax.text(
        x + 0.025,
        y + height - 0.098,
        body,
        ha="left",
        va="top",
        fontsize=6.7,
        color="#263238",
        linespacing=1.18,
    )


def plot_protocol() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 3.65))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        (
            "1  Define the prediction task",
            "Set the delivery endpoint,\nheld-screen setting, and\ncandidate-ranking decision.",
            "#3B6FB6",
        ),
        (
            "2  Assemble formulation inputs",
            "Map lipid roles, ratios, process,\ntarget context, and endpoint\nsupport.",
            "#3B6FB6",
        ),
        (
            "3  Control shortcut signals",
            "Quantify source-linked\nmissingness and exclude explicit\nprovenance identifiers.",
            "#7A5195",
        ),
        (
            "4  Split experimental groups",
            "Assign complete groups first;\nfit preprocessing and select models\non training/validation groups.",
            "#7A5195",
        ),
        (
            "5  Train point and rank heads",
            "Optimize masked endpoints and\nwithin-screen ranking as separate\nbut coordinated outputs.",
            "#B55A30",
        ),
        (
            "6  Evaluate reliability",
            "Use screen bootstrap, size, and\nrisk--coverage analyses to define\nthe deployment scope.",
            "#2B8C6F",
        ),
    ]
    positions = []
    width, height = 0.285, 0.30
    xs = [0.025, 0.3575, 0.69]
    ys = [0.61, 0.10]
    for row, y in enumerate(ys):
        row_xs = xs if row % 2 == 0 else list(reversed(xs))
        for x in row_xs:
            positions.append((x, y))
    for index, ((title, body, color), (x, y)) in enumerate(zip(steps, positions)):
        _draw_box(ax, x, y, width, height, title, body, color)
        if index == len(steps) - 1:
            continue
        next_x, next_y = positions[index + 1]
        if abs(next_y - y) < 1e-6:
            start_x = x + width if next_x > x else x
            end_x = next_x if next_x > x else next_x + width
            start_y = end_y = y + height / 2
        else:
            start_x = end_x = x + width / 2
            start_y = y
            end_y = next_y + height
        ax.add_patch(
            FancyArrowPatch(
                (start_x, start_y),
                (end_x, end_y),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.0,
                color="#6B7280",
                shrinkA=5,
                shrinkB=5,
            )
        )
    _save_figure(fig, "leakage_aware_workflow")


def _bootstrap_interval(
    values: np.ndarray,
    weights: np.ndarray | None,
    rng: np.random.Generator,
    repeats: int = 5000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    values = values[valid]
    local_weights = None if weights is None else np.asarray(weights, dtype=float)[valid]
    if len(values) == 0:
        return float("nan"), float("nan")
    estimates = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        sample = rng.integers(0, len(values), size=len(values))
        if local_weights is None:
            estimates[repeat] = float(np.mean(values[sample]))
        else:
            estimates[repeat] = float(
                np.average(values[sample], weights=local_weights[sample])
            )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def group_size_sensitivity() -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(SOURCE_METRICS)
    frame = frame[
        (frame["split"] == "test") & frame["model"].isin(MODEL_LABELS)
    ].copy()
    frame["model_display"] = frame["model"].map(MODEL_LABELS)
    frame["size_stratum"] = pd.cut(
        frame["n"],
        bins=[0, 99, 499, np.inf],
        labels=["Small (<100)", "Medium (100-499)", "Large (>=500)"],
    )

    rng = np.random.default_rng(20260730)
    summary_rows = []
    stratum_rows = []
    for model, local in frame.groupby("model_display", sort=False):
        local = local[np.isfinite(local["rank_spearman"])].sort_values("group")
        values = local["rank_spearman"].to_numpy(float)
        sizes = local["n"].to_numpy(float)
        largest_group = str(local.loc[local["n"].idxmax(), "group"])
        keep = local["group"].astype(str).ne(largest_group).to_numpy()
        definitions = [
            ("Unweighted screens", values, None),
            ("Size-weighted rows", values, sizes),
            ("Exclude largest screen", values[keep], None),
        ]
        for analysis, metric_values, weights in definitions:
            estimate = (
                float(np.mean(metric_values))
                if weights is None
                else float(np.average(metric_values, weights=weights))
            )
            low, high = _bootstrap_interval(metric_values, weights, rng)
            summary_rows.append(
                {
                    "model": model,
                    "analysis": analysis,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "screens": int(len(metric_values)),
                    "largest_screen": largest_group,
                }
            )
        for stratum, subset in local.groupby("size_stratum", observed=True):
            values_local = subset["rank_spearman"].to_numpy(float)
            low, high = _bootstrap_interval(values_local, None, rng)
            stratum_rows.append(
                {
                    "model": model,
                    "size_stratum": str(stratum),
                    "screens": int(len(subset)),
                    "rows": int(subset["n"].sum()),
                    "mean_per_screen_spearman": float(np.mean(values_local)),
                    "ci_low": low,
                    "ci_high": high,
                }
            )

    summary = pd.DataFrame(summary_rows)
    strata = pd.DataFrame(stratum_rows)
    TAB.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TAB / "group_size_sensitivity.csv", index=False)
    strata.to_csv(TAB / "group_size_strata.csv", index=False)
    frame.to_csv(TAB / "held_out_screen_metrics.csv", index=False)
    plot_group_size_sensitivity(frame, summary, strata)
    return {"summary": summary, "strata": strata, "screens": frame}


def paired_model_bootstrap(repeats: int = 10000) -> pd.DataFrame:
    """Paired screen bootstrap for primary model differences on OOF test rows."""
    frame = pd.read_csv(SOURCE_METRICS)
    frame = frame[
        (frame["split"] == "test") & frame["model"].isin(MODEL_LABELS)
    ].copy()
    frame["model_display"] = frame["model"].map(MODEL_LABELS)
    pairs = [
        ("Validation-selected hybrid", "Metadata Extra Trees"),
        ("DeepLNP", "Ionizable Morgan Extra Trees"),
        ("Validation-selected hybrid", "DeepLNP"),
    ]
    rows = []
    for pair_index, (candidate, reference) in enumerate(pairs):
        left = frame[frame["model_display"] == candidate].set_index("group").sort_index()
        right = frame[frame["model_display"] == reference].set_index("group").sort_index()
        if not left.index.equals(right.index) or not np.array_equal(
            left["n"].to_numpy(int), right["n"].to_numpy(int)
        ):
            raise RuntimeError(f"Unpaired held-screen rows for {candidate} and {reference}")
        n = left["n"].to_numpy(float)
        left_sse = n * left["rmse"].to_numpy(float) ** 2
        right_sse = n * right["rmse"].to_numpy(float) ** 2
        left_rho = left["rank_spearman"].to_numpy(float)
        right_rho = right["rank_spearman"].to_numpy(float)
        rng = np.random.default_rng(20260902 + pair_index)
        rmse_delta = np.empty(repeats)
        rho_delta = np.empty(repeats)
        for repeat in range(repeats):
            sample = rng.integers(0, len(n), size=len(n))
            denominator = n[sample].sum()
            rmse_delta[repeat] = math.sqrt(left_sse[sample].sum() / denominator) - math.sqrt(
                right_sse[sample].sum() / denominator
            )
            rho_delta[repeat] = float(np.mean(left_rho[sample] - right_rho[sample]))
        rows.append(
            {
                "candidate": candidate,
                "reference": reference,
                "screens": int(len(n)),
                "oof_rmse_delta": math.sqrt(left_sse.sum() / n.sum())
                - math.sqrt(right_sse.sum() / n.sum()),
                "oof_rmse_delta_ci_low": float(np.quantile(rmse_delta, 0.025)),
                "oof_rmse_delta_ci_high": float(np.quantile(rmse_delta, 0.975)),
                "mean_screen_spearman_delta": float(np.mean(left_rho - right_rho)),
                "mean_screen_spearman_delta_ci_low": float(np.quantile(rho_delta, 0.025)),
                "mean_screen_spearman_delta_ci_high": float(np.quantile(rho_delta, 0.975)),
                "bootstrap_repeats": repeats,
                "bootstrap_unit": "experimental screen",
            }
        )
    result = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    result.to_csv(TAB / "paired_primary_model_bootstrap.csv", index=False)
    return result


def _candidate_name(row: pd.Series) -> str:
    for column in ["common_name", "lipid_name", "4cr_lipid_name", "formulation_id", "ionizable_lipid"]:
        value = str(row.get(column, "")).strip()
        if value and value.lower() not in {"nan", "none", "other"}:
            return value
    return f"record {int(row['idx'])}"


def _display_value(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return "Not reported"
    replacements = {
        "generic_cell": "Generic cell",
        "lung_epithelium": "Lung epithelium",
        "dendritic_cell": "Dendritic cell",
        "in_vitro": "In vitro",
        "intravenous": "Intravenous",
        "intramuscular": "Intramuscular",
        "intratracheal": "Intratracheal",
        "mRNA": "mRNA",
        "siRNA": "siRNA",
        "pDNA": "pDNA",
        "HeLa": "HeLa",
        "HBEC_ALI": "HBEC-ALI",
        "HEK293T": "HEK293T",
        "IGROV1": "IGROV1",
        "BMDM": "BMDM",
        "BDMC": "BDMC",
        "RAW264p7": "RAW264.7",
        "A549": "A549",
        "Mouse": "Mouse",
    }
    return replacements.get(text, text.replace("_", " ").title() if text else "Not reported")


def application_decision_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize how frozen DeepLNP OOF scores prioritize held-screen candidates."""
    prediction_columns = [
        "idx",
        "name",
        "split",
        "target",
        "prediction",
        "rank_score",
        "predicted_std",
        "group",
    ]
    predictions = pd.read_csv(SOURCE_OOF, usecols=prediction_columns)
    predictions = predictions[
        (predictions["name"] == "A115_dual_rank_residual")
        & (predictions["split"] == "test")
    ].copy()
    if len(predictions) != 12692 or predictions["idx"].nunique() != len(predictions):
        raise RuntimeError("DeepLNP OOF test predictions must cover 12,692 unique delivery rows")

    context_columns = [
        "delivery_target",
        "model_type",
        "route_of_administration",
        "cargo_type",
        "experiment_id",
        "library_id",
        "helper_lipid",
        "helper_lipid_id",
        "common_name",
        "lipid_name",
        "4cr_lipid_name",
        "formulation_id",
        "ionizable_lipid",
        "cationic_lipid_mol_ratio",
        "phospholipid_mol_ratio",
        "cholesterol_mol_ratio",
        "peg_lipid_mol_ratio",
    ]
    corpus = df.reset_index().rename(columns={"index": "idx"})
    predictions = predictions.merge(
        corpus[["idx", *context_columns]], on="idx", how="left", validate="one_to_one"
    )
    if predictions["group"].nunique() != 38 or not np.isfinite(predictions["target"]).all():
        raise RuntimeError("Application analysis requires 38 complete held-out delivery screens")

    def first_reported(values: pd.Series) -> str:
        clean = _clean_series(values)
        reported = clean[clean.ne("") & clean.str.lower().ne("nan")]
        return reported.iloc[0] if len(reported) else ""

    metadata = predictions.groupby("group", as_index=False).agg(
        delivery_target=("delivery_target", first_reported),
        model_type=("model_type", first_reported),
        route_of_administration=("route_of_administration", first_reported),
        cargo_type=("cargo_type", first_reported),
        experiment_id=("experiment_id", first_reported),
        library_id=("library_id", first_reported),
        prediction_rows=("idx", "size"),
    )
    screens = pd.read_csv(SOURCE_METRICS)
    screens = screens[
        (screens["split"] == "test")
        & (screens["model"] == "A115_dual_rank_residual")
    ].copy()
    screens = screens.merge(metadata, on="group", validate="one_to_one")
    if len(screens) != 38 or not np.array_equal(
        screens["n"].to_numpy(int), screens["prediction_rows"].to_numpy(int)
    ):
        raise RuntimeError("Per-screen metrics do not align with DeepLNP OOF predictions")

    rng = np.random.default_rng(20260903)
    budget_rows = []
    for fraction in [5, 10, 20]:
        for metric, label in [
            (f"hit_recall_{fraction}pct", "Top-set recall"),
            (f"normalized_regret_{fraction}pct", "Normalized regret"),
        ]:
            values = screens[metric].to_numpy(float)
            low, high = _bootstrap_interval(values, None, rng)
            budget_rows.append(
                {
                    "budget_percent": fraction,
                    "metric": label,
                    "mean": float(np.mean(values)),
                    "ci_low": low,
                    "ci_high": high,
                    "screens": len(values),
                }
            )
    budgets = pd.DataFrame(budget_rows)

    context_rows = []
    for context_type, column in [
        ("Delivery target", "delivery_target"),
        ("Administration route", "route_of_administration"),
        ("Cargo", "cargo_type"),
    ]:
        for category, local in screens.groupby(column, dropna=False, sort=True):
            values = local["rank_spearman"].to_numpy(float)
            low, high = (
                _bootstrap_interval(values, None, rng)
                if len(values) >= 2
                else (float("nan"), float("nan"))
            )
            context_rows.append(
                {
                    "context_type": context_type,
                    "category": _display_value(category),
                    "screens": int(len(local)),
                    "rows": int(local["n"].sum()),
                    "mean_screen_spearman": float(np.mean(values)),
                    "spearman_ci_low": low,
                    "spearman_ci_high": high,
                    "mean_hit_recall_10pct": float(local["hit_recall_10pct"].mean()),
                    "mean_normalized_regret_10pct": float(
                        local["normalized_regret_10pct"].mean()
                    ),
                }
            )
    contexts = pd.DataFrame(context_rows)

    eligible = screens[(screens["n"] >= 20) & np.isfinite(screens["rank_spearman"])].copy()
    median_rho = float(eligible["rank_spearman"].median())
    chosen = [
        ("Highest-ranking case", eligible.loc[eligible["rank_spearman"].idxmax()]),
        (
            "Median-ranking case",
            eligible.loc[(eligible["rank_spearman"] - median_rho).abs().idxmin()],
        ),
        ("Lowest-ranking case", eligible.loc[eligible["rank_spearman"].idxmin()]),
    ]
    experiment_labels = {
        "A549_form_screen": "A549 formulation screen",
        "LM_3CR": "LM three-component-reaction screen",
        "IR_4CR_ketone": "4CR ketone screen",
    }
    case_rows = []
    candidate_rows = []
    curve_rows = []
    case_colors = {
        "Highest-ranking case": "#2A9D8F",
        "Median-ranking case": "#4C78A8",
        "Lowest-ranking case": "#C75B39",
    }
    for code, (case, screen) in zip(["C1", "C2", "C3"], chosen):
        local = predictions[predictions["group"] == screen["group"]].copy()
        local["model_rank"] = local["rank_score"].rank(method="min", ascending=False).astype(int)
        local["candidate"] = local.apply(_candidate_name, axis=1)
        k = max(1, math.ceil(0.10 * len(local)))
        selected = local.nlargest(k, "rank_score")
        highest_ranked = selected.iloc[0]
        selected_best = selected.loc[selected["target"].idxmax()]
        observed_best = local.loc[local["target"].idxmax()]
        recovered = bool(observed_best["idx"] in set(selected["idx"]))
        screen_name = experiment_labels.get(
            str(screen["experiment_id"]), _display_value(screen["experiment_id"])
        )
        setting = " / ".join(
            [
                _display_value(screen["delivery_target"]),
                _display_value(screen["model_type"]),
                _display_value(screen["route_of_administration"]),
            ]
        )
        case_rows.append(
            {
                "case_code": code,
                "selection_rule": case,
                "screen": screen_name,
                "setting": setting,
                "candidates": int(len(local)),
                "top_decile_k": k,
                "screen_spearman": float(screen["rank_spearman"]),
                "hit_recall_10pct": float(screen["hit_recall_10pct"]),
                "normalized_regret_10pct": float(screen["normalized_regret_10pct"]),
                "best_selected_candidate": str(selected_best["candidate"]),
                "best_selected_target": float(selected_best["target"]),
                "observed_best_candidate": str(observed_best["candidate"]),
                "observed_best_target": float(observed_best["target"]),
                "observed_best_recovered": recovered,
                "group": str(screen["group"]),
                "color": case_colors[case],
            }
        )
        for role, row in [
            ("Highest model rank", highest_ranked),
            ("Best measured in selected top decile", selected_best),
            ("Best measured in complete screen", observed_best),
        ]:
            candidate_rows.append(
                {
                    "case_code": code,
                    "screen": screen_name,
                    "candidate_role": role,
                    "candidate": row["candidate"],
                    "helper_lipid": _display_value(
                        row["helper_lipid"]
                        if _display_value(row["helper_lipid"]) != "Not reported"
                        else row["helper_lipid_id"]
                    ),
                    "ionizable_mol_pct": row["cationic_lipid_mol_ratio"],
                    "helper_mol_pct": row["phospholipid_mol_ratio"],
                    "sterol_mol_pct": row["cholesterol_mol_ratio"],
                    "peg_lipid_mol_pct": row["peg_lipid_mol_ratio"],
                    "observed_target": float(row["target"]),
                    "predicted_point": float(row["prediction"]),
                    "rank_score": float(row["rank_score"]),
                    "predicted_std": float(row["predicted_std"]),
                    "model_rank": int(row["model_rank"]),
                    "model_rank_percentile": float(row["model_rank"] / len(local)),
                    "selected_at_10pct": bool(row["model_rank"] <= k),
                }
            )
        ordered = local.sort_values("rank_score", ascending=False)
        target_range = float(ordered["target"].max() - ordered["target"].min())
        running_best = ordered["target"].cummax().to_numpy(float)
        for inspected in range(1, math.ceil(0.25 * len(ordered)) + 1):
            curve_rows.append(
                {
                    "case_code": code,
                    "selection_rule": case,
                    "screen": screen_name,
                    "fraction_inspected": inspected / len(ordered),
                    "candidates_inspected": inspected,
                    "normalized_regret": (
                        float((ordered["target"].max() - running_best[inspected - 1]) / target_range)
                        if target_range > 0
                        else 0.0
                    ),
                    "color": case_colors[case],
                }
            )
    cases = pd.DataFrame(case_rows)
    candidates = pd.DataFrame(candidate_rows)
    curves = pd.DataFrame(curve_rows)

    screen_order = screens.sort_values(
        ["delivery_target", "rank_spearman", "group"], ascending=[True, False, True]
    ).reset_index(drop=True)
    screen_order.insert(0, "screen_code", [f"S{i:02d}" for i in range(1, len(screen_order) + 1)])
    for column in ["delivery_target", "model_type", "route_of_administration", "cargo_type"]:
        screen_order[f"{column}_display"] = screen_order[column].map(_display_value)

    TAB.mkdir(parents=True, exist_ok=True)
    budgets.to_csv(TAB / "application_budget_summary.csv", index=False)
    contexts.to_csv(TAB / "application_context_summary.csv", index=False)
    cases.to_csv(TAB / "application_representative_cases.csv", index=False)
    candidates.to_csv(TAB / "application_case_candidates.csv", index=False)
    curves.to_csv(TAB / "application_case_regret_curves.csv", index=False)
    screen_order.to_csv(TAB / "application_all_screens.csv", index=False)
    plot_application_results(budgets, contexts, cases, curves, screen_order)
    return {
        "budgets": budgets,
        "contexts": contexts,
        "cases": cases,
        "candidates": candidates,
        "curves": curves,
        "screens": screen_order,
    }


def plot_application_results(
    budgets: pd.DataFrame,
    contexts: pd.DataFrame,
    cases: pd.DataFrame,
    curves: pd.DataFrame,
    screens: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.0, 5.55),
        gridspec_kw={"height_ratios": [0.82, 1.18]},
    )
    for axis, metric, title, ylabel, color in [
        (axes[0, 0], "Top-set recall", "a  Recovery under finite budgets", "Top-set recall", "#6F4E9C"),
        (axes[0, 1], "Normalized regret", "b  Best-candidate regret", "Normalized regret", "#D97930"),
    ]:
        local = budgets[budgets["metric"] == metric].sort_values("budget_percent")
        mean = local["mean"].to_numpy(float)
        axis.errorbar(
            local["budget_percent"],
            mean,
            yerr=np.vstack(
                [mean - local["ci_low"].to_numpy(float), local["ci_high"].to_numpy(float) - mean]
            ),
            color=color,
            marker="o",
            markersize=5,
            linewidth=1.8,
            capsize=3,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        axis.set_xticks([5, 10, 20], ["5", "10", "20"])
        axis.set_xlim(4.2, 20.8)
        axis.set_xlabel("Candidates selected (%)")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylim(bottom=0)

    target = contexts[(contexts["context_type"] == "Delivery target") & (contexts["screens"] >= 2)]
    target = target.sort_values("mean_screen_spearman")
    target_screens = screens[screens["delivery_target_display"].isin(target["category"])].copy()
    rng = np.random.default_rng(20260903)
    for index, row in enumerate(target.itertuples(index=False)):
        values = target_screens.loc[
            target_screens["delivery_target_display"] == row.category, "rank_spearman"
        ].to_numpy(float)
        axes[1, 0].scatter(
            values,
            np.full(len(values), index) + rng.uniform(-0.10, 0.10, len(values)),
            s=17,
            color="#AAB4BE",
            edgecolor="white",
            linewidth=0.3,
            zorder=2,
        )
        axes[1, 0].errorbar(
            row.mean_screen_spearman,
            index,
            xerr=np.array(
                [[row.mean_screen_spearman - row.spearman_ci_low], [row.spearman_ci_high - row.mean_screen_spearman]]
            ),
            fmt="D",
            color="#263442",
            markerfacecolor="#2A9D8F",
            markersize=5,
            capsize=3,
            linewidth=1.1,
            zorder=3,
        )
    axes[1, 0].axvline(0, color="#7B8794", linewidth=0.8, linestyle="--")
    axes[1, 0].set_yticks(
        np.arange(len(target)),
        [f"{row.category} (n={row.screens})" for row in target.itertuples(index=False)],
    )
    axes[1, 0].set_xlabel("Per-screen Spearman")
    axes[1, 0].set_title("c  Performance by delivery target", loc="left", fontweight="bold")

    labels = {
        "Highest-ranking case": "Highest-ranking screen",
        "Median-ranking case": "Median screen",
        "Lowest-ranking case": "Lowest-ranking screen",
    }
    for case in cases.itertuples(index=False):
        local = curves[curves["case_code"] == case.case_code]
        axes[1, 1].step(
            100 * local["fraction_inspected"],
            local["normalized_regret"],
            where="post",
            color=case.color,
            linewidth=1.8,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=f"{labels[case.selection_rule]} ({case.case_code})",
        )
    axes[1, 1].axvline(10, color="#7B8794", linewidth=0.8, linestyle="--")
    axes[1, 1].set_xlim(0, 25)
    axes[1, 1].set_ylim(bottom=0)
    axes[1, 1].set_xlabel("Candidates inspected (%)")
    axes[1, 1].set_ylabel("Normalized regret")
    axes[1, 1].set_title("d  Retrospective candidate recovery", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False, loc="upper right")

    for axis in axes.flat:
        axis.grid(axis="y", color="#E1E6EA", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.55, w_pad=1.15, h_pad=1.05)
    _save_figure(fig, "application_screening_results")

    target_categories = screens["delivery_target_display"].drop_duplicates().tolist()
    palette = dict(
        zip(
            target_categories,
            ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D"],
        )
    )
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 9.0), sharey=True)
    y = np.arange(len(screens))
    colors = screens["delivery_target_display"].map(palette)
    for axis, column, title, limits, better in [
        (axes[0], "rank_spearman", "a  Screen Spearman", (-0.45, 0.85), "higher"),
        (axes[1], "hit_recall_10pct", "b  Hit@10%", (-0.02, 0.65), "higher"),
        (axes[2], "normalized_regret_10pct", "c  Normalized regret", (-0.02, 0.85), "lower"),
    ]:
        axis.scatter(screens[column], y, c=colors, s=22, edgecolor="white", linewidth=0.35)
        axis.set_xlim(*limits)
        axis.set_xlabel(f"{column.replace('_10pct', '').replace('_', ' ').title()} ({better})")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="x", color="#E1E6EA", linewidth=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].axvline(0, color="#7B8794", linewidth=0.8, linestyle="--")
    axes[0].set_yticks(y, screens["screen_code"], fontsize=7)
    axes[0].invert_yaxis()
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=category, markersize=5)
        for category, color in palette.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.055, 1, 1), w_pad=1.2)
    _save_figure(fig, "application_all_screens")


def plot_group_size_sensitivity(
    screens: pd.DataFrame, summary: pd.DataFrame, strata: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.65))
    model_order = list(COLORS)
    analyses = ["Unweighted screens", "Size-weighted rows", "Exclude largest screen"]
    offsets = np.linspace(-0.27, 0.27, len(model_order))
    for model, offset in zip(model_order, offsets):
        local = summary[summary["model"] == model].set_index("analysis").loc[analyses]
        y = local["estimate"].to_numpy(float)
        lower = y - local["ci_low"].to_numpy(float)
        upper = local["ci_high"].to_numpy(float) - y
        axes[0].errorbar(
            np.arange(len(analyses)) + offset,
            y,
            yerr=np.vstack([lower, upper]),
            fmt="o",
            color=COLORS[model],
            capsize=3,
            linewidth=1.1,
            markersize=4.5,
            label=MODEL_SHORT[model],
        )
    axes[0].axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    axes[0].set_xticks(
        np.arange(len(analyses)),
        ["Unweighted", "Size-\nweighted", "Without\nlargest"],
    )
    axes[0].set_ylabel("Mean per-screen Spearman")
    axes[0].set_title("a  Aggregation sensitivity", loc="left", fontweight="bold")

    for model in model_order:
        local = screens[screens["model_display"] == model]
        axes[1].scatter(
            local["n"],
            local["rank_spearman"],
            color=COLORS[model],
            s=18,
            alpha=0.68,
            edgecolors="white",
            linewidths=0.3,
            label=model,
        )
    axes[1].set_xscale("log")
    axes[1].axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("Held-out screen size (log scale)")
    axes[1].set_ylabel("Per-screen Spearman")
    axes[1].set_title("b  Screen-level heterogeneity", loc="left", fontweight="bold")

    strata_order = ["Small (<100)", "Medium (100-499)", "Large (>=500)"]
    for model, offset in zip(model_order, offsets):
        local = (
            strata[strata["model"] == model]
            .set_index("size_stratum")
            .reindex(strata_order)
        )
        means = local["mean_per_screen_spearman"].to_numpy(float)
        lower = means - local["ci_low"].to_numpy(float)
        upper = local["ci_high"].to_numpy(float) - means
        axes[2].errorbar(
            np.arange(len(strata_order)) + offset,
            means,
            yerr=np.vstack([lower, upper]),
            fmt="o",
            color=COLORS[model],
            linewidth=1.1,
            markersize=4.5,
            capsize=2.5,
        )
    axes[2].axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    axes[2].set_xticks(
        np.arange(len(strata_order)), ["<100", "100-499", ">=500"]
    )
    axes[2].set_xlabel("Rows per screen")
    axes[2].set_ylabel("Mean per-screen Spearman")
    axes[2].set_title("c  Size-stratified estimates", loc="left", fontweight="bold")

    for axis in axes:
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(model_order),
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.2)
    _save_figure(fig, "group_size_sensitivity")


def plot_primary_model_performance() -> None:
    """Visualize paired fold error and held-screen ranking heterogeneity."""
    folds = pd.read_csv(SOURCE_SUMMARY)
    folds = folds[
        (folds["split"] == "test") & folds["model"].isin(MODEL_LABELS)
    ].copy()
    folds["model_display"] = folds["model"].map(MODEL_LABELS)
    folds["model_short"] = folds["model_display"].map(MODEL_SHORT)
    folds = folds.sort_values(["fold_index", "model_display"])

    screens = pd.read_csv(SOURCE_METRICS)
    screens = screens[
        (screens["split"] == "test") & screens["model"].isin(MODEL_LABELS)
    ].copy()
    screens["model_display"] = screens["model"].map(MODEL_LABELS)
    screens["model_short"] = screens["model_display"].map(MODEL_SHORT)

    TAB.mkdir(parents=True, exist_ok=True)
    folds.to_csv(TAB / "primary_model_fold_metrics.csv", index=False)
    screens.to_csv(TAB / "primary_model_screen_metrics.csv", index=False)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.0, 3.75),
        gridspec_kw={"width_ratios": [1.08, 1.15, 1.0]},
    )
    x = np.arange(len(MODEL_ORDER))

    pivot = folds.pivot(index="fold_index", columns="model_display", values="rmse")
    pivot = pivot.reindex(columns=MODEL_ORDER)
    for _, row in pivot.iterrows():
        axes[0].plot(x, row.to_numpy(float), color="#BAC4CE", linewidth=0.9, zorder=1)
    for index, model in enumerate(MODEL_ORDER):
        values = pivot[model].to_numpy(float)
        axes[0].scatter(
            np.full(len(values), index),
            values,
            s=25,
            color=COLORS[model],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        axes[0].errorbar(
            index,
            float(np.mean(values)),
            yerr=float(np.std(values, ddof=1)),
            fmt="D",
            color="#1F2937",
            markerfacecolor=COLORS[model],
            markeredgecolor="#1F2937",
            markersize=5.2,
            capsize=3,
            linewidth=1.1,
            zorder=4,
        )
    axes[0].set_xticks(x, ["DeepLNP", "Metadata\nET", "Morgan\nET", "Hybrid"])
    axes[0].set_ylabel("Held-screen RMSE")
    axes[0].set_title("a  Paired five-fold point error", loc="left", fontweight="bold")
    axes[0].text(
        0.02,
        0.03,
        "diamonds: mean ± SD",
        transform=axes[0].transAxes,
        fontsize=7,
        color="#5F6B76",
    )

    rng = np.random.default_rng(20260730)
    bootstrap_rows = []
    for index, model in enumerate(MODEL_ORDER):
        values = screens.loc[
            screens["model_display"] == model, "rank_spearman"
        ].dropna().to_numpy(float)
        jitter = rng.uniform(-0.11, 0.11, size=len(values))
        axes[1].scatter(
            np.full(len(values), index) + jitter,
            values,
            s=14,
            alpha=0.58,
            color=COLORS[model],
            edgecolor="white",
            linewidth=0.25,
            zorder=2,
        )
        low, high = _bootstrap_interval(values, None, rng)
        mean = float(np.mean(values))
        axes[1].errorbar(
            index,
            mean,
            yerr=np.array([[mean - low], [high - mean]]),
            fmt="D",
            color="#1F2937",
            markerfacecolor=COLORS[model],
            markeredgecolor="#1F2937",
            markersize=5.2,
            capsize=3,
            linewidth=1.2,
            zorder=4,
        )
        bootstrap_rows.append(
            {
                "model": model,
                "screens": len(values),
                "mean_screen_spearman": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )
    pd.DataFrame(bootstrap_rows).to_csv(
        TAB / "primary_model_screen_bootstrap.csv", index=False
    )
    axes[1].axhline(0, color="#7B8794", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(x, ["DeepLNP", "Metadata\nET", "Morgan\nET", "Hybrid"])
    axes[1].set_ylabel("Per-screen Spearman")
    axes[1].set_title("b  Held-screen ranking", loc="left", fontweight="bold")
    axes[1].text(
        0.02,
        0.03,
        "n=38 screens; diamonds: mean and 95% screen bootstrap CI",
        transform=axes[1].transAxes,
        fontsize=6.7,
        color="#5F6B76",
    )

    aggregate = (
        folds.groupby("model_display")[["rmse", "within_group_spearman"]]
        .agg(["mean", "std"])
        .reindex(MODEL_ORDER)
    )
    offsets = {
        "DeepLNP": (5, -15),
        "Metadata Extra Trees": (-58, -15),
        "Ionizable Morgan Extra Trees": (5, 8),
        "Validation-selected hybrid": (5, -18),
    }
    for model in MODEL_ORDER:
        rmse = float(aggregate.loc[model, ("rmse", "mean")])
        rmse_sd = float(aggregate.loc[model, ("rmse", "std")])
        rank = float(aggregate.loc[model, ("within_group_spearman", "mean")])
        rank_sd = float(aggregate.loc[model, ("within_group_spearman", "std")])
        axes[2].errorbar(
            rmse,
            rank,
            xerr=rmse_sd,
            yerr=rank_sd,
            fmt="o",
            color=COLORS[model],
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=7,
            capsize=2.5,
            linewidth=1.0,
        )
        axes[2].annotate(
            MODEL_SHORT[model],
            (rmse, rank),
            xytext=offsets[model],
            textcoords="offset points",
            fontsize=7.2,
            color="#263442",
        )
    axes[2].set_xlabel("Held-screen RMSE")
    axes[2].set_ylabel("Mean screen Spearman")
    axes[2].set_title("c  Point--ranking trade-off", loc="left", fontweight="bold")

    for axis in axes:
        axis.grid(axis="y", color="#E2E8F0", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=1.35)
    _save_figure(fig, "primary_model_performance")


def plot_dual_path_ablation() -> None:
    """Show the matched progression from graph-only to dual-path ranking."""
    aggregate = pd.read_csv(SOURCE_MATCHED_NEURAL / "aggregate_metrics.csv")
    order = [
        "A79_validation_pareto_checkpoint",
        "A110_direct_morgan_replace",
        "A112_direct_morgan_rank_only",
        "A115_dual_rank_residual",
    ]
    labels = ["Graph only", "Direct replace", "Direct rank", "Dual residual"]
    frame = aggregate.set_index("name").loc[order].copy()
    frame.insert(0, "label", labels)
    frame.to_csv(TAB / "dual_path_ablation.csv", index_label="variant")

    metrics = [
        ("test_efficiency_rmse", "a  Point error", "RMSE", True),
        (
            "test_efficiency_within_group_spearman",
            "b  Screen-local ranking",
            "Mean screen Spearman",
            False,
        ),
        (
            "test_efficiency_within_group_hit_recall_at_10pct",
            "c  Top-decile recovery",
            "Hit@10%",
            False,
        ),
        (
            "test_efficiency_within_group_normalized_regret_at_10pct",
            "d  Selection regret",
            "Normalized regret@10%",
            True,
        ),
    ]
    colors = ["#8FA8B8", "#C9A27E", "#E68A3A", "#6F4E9C"]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.25), sharey=True)
    for axis, (prefix, title, xlabel, lower_is_better) in zip(axes, metrics):
        means = frame[f"{prefix}_mean"].to_numpy(float)
        errors = frame[f"{prefix}_std"].to_numpy(float)
        axis.plot(
            means,
            y,
            color="#C8D2DA",
            linewidth=1.0,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=1,
        )
        for index, (mean, error, color) in enumerate(zip(means, errors, colors)):
            axis.errorbar(
                mean,
                index,
                xerr=error,
                fmt="o",
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.5,
                markersize=6,
                capsize=2.5,
                linewidth=1.1,
                zorder=2,
            )
        axis.set_yticks(y)
        if axis is axes[0]:
            axis.set_yticklabels(labels)
        else:
            axis.tick_params(axis="y", labelleft=False)
        direction = "lower" if lower_is_better else "higher"
        axis.set_xlabel(f"{xlabel}\n({direction} is better)")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="x", color="#E1E7EC", linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        low = np.nanmin(means - errors)
        high = np.nanmax(means + errors)
        margin = max((high - low) * 0.07, 0.005)
        axis.set_xlim(low - margin, high + margin)
    axes[0].invert_yaxis()
    fig.tight_layout(pad=0.55, w_pad=0.9)
    _save_figure(fig, "dual_path_ablation")


def plot_fig1_performance_inset() -> None:
    """Create the compact, source-backed performance panel embedded in Fig. 1."""
    fig1_dir = ROOT / "oup-authoring-template" / "Fig"
    fig1_dir.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(SOURCE_SUMMARY)
    folds = folds[
        (folds["split"] == "test") & folds["model"].isin(MODEL_LABELS)
    ].copy()
    folds["model_display"] = folds["model"].map(MODEL_LABELS)
    aggregate = (
        folds.groupby("model_display")[["rmse", "within_group_spearman"]]
        .agg(["mean", "std"])
        .reindex(MODEL_ORDER)
    )
    rows = []
    for model in MODEL_ORDER:
        rows.append(
            {
                "model": MODEL_SHORT[model],
                "rmse_mean": float(aggregate.loc[model, ("rmse", "mean")]),
                "rmse_sd": float(aggregate.loc[model, ("rmse", "std")]),
                "screen_spearman_mean": float(
                    aggregate.loc[model, ("within_group_spearman", "mean")]
                ),
                "screen_spearman_sd": float(
                    aggregate.loc[model, ("within_group_spearman", "std")]
                ),
            }
        )
    inset_data = pd.DataFrame(rows)
    inset_data.to_csv(fig1_dir / "fig1_performance_inset_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.15, 2.0))
    label_offsets = [(5, -12), (-58, -12), (5, 7), (5, -18)]
    for (model, row), offset in zip(inset_data.set_index("model").iterrows(), label_offsets):
        full_model = MODEL_ORDER[list(MODEL_SHORT.values()).index(model)]
        ax.scatter(
            row["rmse_mean"],
            row["screen_spearman_mean"],
            color=COLORS[full_model],
            edgecolor="white",
            linewidth=0.45,
            s=38,
            zorder=3,
        )
        ax.annotate(
            model,
            (row["rmse_mean"], row["screen_spearman_mean"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.4,
            color="#263442",
        )
    ax.set_xlim(0.95, 1.065)
    ax.set_ylim(0.0, 0.36)
    ax.set_xlabel("RMSE  ← lower", fontsize=6.8)
    ax.set_ylabel("screen Spearman  → higher", fontsize=6.8)
    ax.tick_params(labelsize=6.2, length=2.5)
    ax.grid(color="#E6EBF0", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.5)
    for suffix in [".pdf", ".svg", ".png"]:
        fig.savefig(
            fig1_dir / f"fig1_performance_inset{suffix}",
            dpi=600 if suffix == ".png" else None,
            bbox_inches="tight",
            transparent=True,
        )
    plt.close(fig)


def model_result_table() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE_SUMMARY)
    frame = frame[
        (frame["split"] == "test") & frame["model"].isin(MODEL_LABELS)
    ].copy()
    frame["model_display"] = frame["model"].map(MODEL_LABELS)
    metrics = [
        "rmse",
        "within_group_spearman",
        "within_group_hit_recall_10pct",
        "within_group_normalized_regret_10pct",
    ]
    aggregate = (
        frame.groupby("model_display", sort=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.columns = [
        "model",
        "rmse_mean",
        "rmse_std",
        "screen_spearman_mean",
        "screen_spearman_std",
        "hit10_mean",
        "hit10_std",
        "nreg10_mean",
        "nreg10_std",
    ]
    interpretation = {
        "DeepLNP": "Dual-path neural model",
        "Metadata Extra Trees": "Lowest standalone point error",
        "Ionizable Morgan Extra Trees": "Fingerprint screening baseline",
        "Validation-selected hybrid": "Lowest joint point-ranking profile",
    }
    aggregate["interpretation"] = aggregate["model"].map(interpretation)
    aggregate.to_csv(TAB / "main_model_results.csv", index=False)
    return aggregate


def claim_evidence_table(support_summary: dict[str, object]) -> pd.DataFrame:
    rows = [
        {
            "claim": "Candidate ranking within a completed screen",
            "required_validation": "Held-out, screen-local ranking and selection metrics",
            "current_evidence": "Supported retrospectively",
            "permitted_interpretation": "Retrospective screening performance",
        },
        {
            "claim": "Relative endpoint prediction in a new experimental group",
            "required_validation": "Group-disjoint held-out screens",
            "current_evidence": f"Partial; {support_summary['efficacy_experimental_groups']} efficacy screens",
            "permitted_interpretation": "Limited prediction under group shift",
        },
        {
            "claim": "Absolute efficacy of an isolated new candidate",
            "required_validation": "Cross-screen commensurate, unstandardized outcomes",
            "current_evidence": "Not supported",
            "permitted_interpretation": "No absolute prospective calibration claim",
        },
        {
            "claim": "Joint optimization of four lipid structures",
            "required_validation": "Complete four-role structures with delivery labels",
            "current_evidence": (
                "Not supported; "
                f"{support_summary['delivery_and_all_four_structure_rows']} overlapping records"
            ),
            "permitted_interpretation": "Role-aware interface, not validated optimization",
        },
        {
            "claim": "Safety prediction",
            "required_validation": "Independent multi-group toxicity labels in test data",
            "current_evidence": "Not supported; sparse bounded labels",
            "permitted_interpretation": "Censored-endpoint diagnostic only",
        },
        {
            "claim": "Applicability-domain abstention",
            "required_validation": "Monotonic risk reduction as low-support cases are rejected",
            "current_evidence": "Not supported by frozen risk-coverage analysis",
            "permitted_interpretation": "No validated abstention rule",
        },
        {
            "claim": "Prospective discovery",
            "required_validation": "Blinded wet-lab validation after model freezing",
            "current_evidence": "Not tested",
            "permitted_interpretation": "No prospective discovery claim",
        },
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(TAB / "claim_evidence_matrix.csv", index=False)
    return frame


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dataset, df = load_exact_corpus()
    audit = support_and_overlap_audit(dataset, df)
    sensitivity = group_size_sensitivity()
    paired = paired_model_bootstrap()
    models = model_result_table()
    application = application_decision_analysis(df)
    plot_primary_model_performance()
    plot_dual_path_ablation()
    plot_fig1_performance_inset()
    claims = claim_evidence_table(audit["summary"])
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_source": str(
            ROOT
            / "merged_datasets"
            / "02_lnp_formulations"
            / "lnp_formulations_merged.csv"
        ),
        "screen_metric_source": str(SOURCE_METRICS),
        "fold_summary_source": str(SOURCE_SUMMARY),
        "model_selection_changed": True,
        "selected_neural_variant": "A115_dual_rank_residual",
        "held_out_labels_used_for_selection": False,
        "outputs": {
            "figures": sorted(path.name for path in FIG.glob("*")),
            "tables": sorted(path.name for path in TAB.glob("*")),
        },
        "support_summary": audit["summary"],
        "group_size_rows": int(len(sensitivity["screens"])),
        "model_rows": int(len(models)),
        "paired_comparison_rows": int(len(paired)),
        "application_screen_rows": int(len(application["screens"])),
        "application_case_rows": int(len(application["cases"])),
        "claim_rows": int(len(claims)),
    }
    with (OUT / "analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
