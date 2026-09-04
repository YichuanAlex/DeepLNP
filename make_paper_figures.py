#!/usr/bin/env python3
"""Generate reproducible vector figures for the LNP evaluation manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "arxivAuthorKit" / "figures"
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def box(ax, xy, width, height, text, color, fontsize=9):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.2, edgecolor=color, facecolor=color + "18",
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                linewidth=1.2, color="#44546a"))


def data_identifiability():
    """Visualize the component--endpoint support mismatch in the merged corpus."""
    columns = [
        "Ionizable\nstructure", "Helper\nstructure", "Sterol\nstructure",
        "PEG-lipid\nstructure", "Delivery\nlabel",
    ]
    rows = ["Delivery screens\n($n=12{,}692$)", "Literature records\n($n=913$)"]
    values = np.array([
        [1.000, 0.994, 0.000, 0.000, 1.000],
        [0.995, 0.962, 0.984, 0.853, 0.000],
    ])
    pd.DataFrame(values, index=rows, columns=columns).to_csv(
        OUT / "data_identifiability_plot_data.csv"
    )

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10.5, 4.7), gridspec_kw={"width_ratios": [1.18, 0.82]}
    )
    image = left.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    left.set_xticks(np.arange(len(columns)), labels=columns)
    left.set_yticks(np.arange(len(rows)), labels=rows)
    left.tick_params(axis="x", length=0, pad=7)
    left.tick_params(axis="y", length=0)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            left.text(
                col, row, f"{100 * value:.1f}%",
                ha="center", va="center",
                color="white" if value >= 0.58 else "#172033",
                fontweight="bold" if value in {0, 1} else "normal",
            )
    left.text(
        -0.12, 1.10, "a", transform=left.transAxes,
        fontsize=11, fontweight="bold", va="top",
    )
    left.text(
        -0.04, 1.10, "Source-conditioned input and endpoint support",
        transform=left.transAxes, fontsize=10, va="top",
    )
    colorbar = fig.colorbar(image, ax=left, fraction=0.035, pad=0.025)
    colorbar.set_label("Observed fraction")

    right.set_xlim(0, 1)
    right.set_ylim(0, 1)
    right.axis("off")
    right.text(
        -0.02, 1.10, "b", transform=right.transAxes,
        fontsize=11, fontweight="bold", va="top",
    )
    right.text(
        0.06, 1.10, "Component--endpoint overlap",
        transform=right.transAxes, fontsize=10, va="top",
    )
    box(right, (0.24, 0.78), 0.52, 0.13, "Merged corpus\n13,605 records", "#475569", 9.5)
    box(right, (0.03, 0.48), 0.40, 0.15, "Delivery labels\n12,692 records\n38 screens", "#2563eb", 9)
    box(right, (0.57, 0.48), 0.40, 0.15, "All four structures\n732 records", "#059669", 9)
    arrow(right, (0.42, 0.78), (0.23, 0.63))
    arrow(right, (0.58, 0.78), (0.77, 0.63))
    box(
        right, (0.23, 0.16), 0.54, 0.15,
        "Delivery labels +\nall four structures\n0 records",
        "#dc2626", 9.5,
    )
    arrow(right, (0.23, 0.48), (0.42, 0.31))
    arrow(right, (0.77, 0.48), (0.58, 0.31))
    fig.tight_layout(w_pad=2.2)
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"data_identifiability.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(10.5, 6.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    component_names = ["Ionizable\nlipid", "Helper\nphospholipid", "Sterol", "PEG-lipid"]
    colors = ["#2563eb", "#059669", "#d97706", "#9333ea"]
    xs = [0.02, 0.185, 0.35, 0.515]
    transformer_targets = [0.27, 0.38, 0.50, 0.61]
    for x, name, color, target_x in zip(xs, component_names, colors, transformer_targets):
        box(ax, (x, 0.82), 0.13, 0.12, name + "\nstructure + ratio", color)
        arrow(ax, (x + 0.065, 0.82), (x + 0.065, 0.75))
        box(ax, (x, 0.63), 0.13, 0.11, "Shared GNN", color, 8.5)
        arrow(ax, (x + 0.065, 0.63), (target_x, 0.54))
    box(ax, (0.72, 0.82), 0.25, 0.12,
        "Assay context\ncargo, model, route", "#475569")
    box(ax, (0.72, 0.63), 0.25, 0.11,
        "Biological target\n11-class query", "#dc2626")
    arrow(ax, (0.72, 0.88), (0.68, 0.49))
    arrow(ax, (0.72, 0.685), (0.68, 0.45))
    box(ax, (0.20, 0.40), 0.48, 0.14,
        "Role-aware component interaction\nshared graph tokens + target cross-attention",
        "#0f766e", 9)
    box(ax, (0.72, 0.40), 0.25, 0.14,
        "Formulation encoder\ncomposition, N/P, dose, mixing", "#64748b", 8.5)
    box(ax, (0.27, 0.22), 0.34, 0.10,
        "Three-expert gated fusion", "#0369a1", 9)
    box(ax, (0.72, 0.22), 0.25, 0.10,
        "Target-class auxiliary loss\nunconditional molecular state", "#b91c1c", 8)
    arrow(ax, (0.44, 0.40), (0.44, 0.32))
    arrow(ax, (0.72, 0.47), (0.61, 0.27))
    arrow(ax, (0.68, 0.44), (0.72, 0.27))
    heads = ["Delivery mean\n+ rank adapter", "Size / PDI", "Zeta / EE", "Censored\ntoxicity"]
    head_x = [0.02, 0.27, 0.52, 0.77]
    for x, name in zip(head_x, heads):
        box(ax, (x, 0.03), 0.20, 0.11, name, "#334155", 8.5)
        arrow(ax, (0.44, 0.22), (x + 0.10, 0.14))
    fig.tight_layout()
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"architecture.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluation_framework():
    fig, ax = plt.subplots(figsize=(10.5, 4.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        ("1. Data validity", "duplicate and label audit\ngroup-safe preprocessing"),
        ("2. Generalization", "group + scaffold splits\nleave-context-out"),
        ("3. Prediction", "RMSE, MAE, $R^2$\nPearson, Spearman"),
        ("6. Decision", "applicability domain\ntoxicity constraints"),
        ("5. Reliability", "group bootstrap CI\ncalibration, conformal"),
        ("4. Screening", "pair accuracy, NDCG\nhit recall, enrichment"),
    ]
    y_values = [0.73, 0.73, 0.73, 0.28, 0.28, 0.28]
    x_values = [0.03, 0.35, 0.67, 0.03, 0.35, 0.67]
    colors = ["#2563eb", "#0f766e", "#7c3aed", "#d97706", "#dc2626", "#475569"]
    for i, ((title, body), x, y, color) in enumerate(zip(stages, x_values, y_values, colors)):
        box(ax, (x, y), 0.27, 0.19, title + "\n" + body, color, 9)
        if i in [0, 1]:
            arrow(ax, (x + 0.27, y + 0.095), (x + 0.32, y + 0.095))
    arrow(ax, (0.805, 0.73), (0.805, 0.47))
    arrow(ax, (0.67, 0.375), (0.62, 0.375))
    arrow(ax, (0.35, 0.375), (0.30, 0.375))
    fig.tight_layout()
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"evaluation_framework.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def latest_ablation():
    labels = {
        "A56_group_centered_screen_objective": "Centered objective",
        "A61_full_no_target_conditioning": "w/o target conditioning",
        "A62_full_no_deterministic_descriptors": "w/o descriptors",
        "A63_full_no_process_observation_bits": "w/o process indicators",
        "A64_centered_rank_fully_detached": "Detached rank backbone",
        "A65_listnet_screen_objective": "ListNet",
        "A66_listmle_screen_objective": "ListMLE",
        "A77_centered_plus_light_listmle": "Centered + ListMLE",
    }
    expected_train = {42: 10728, 52: 10276, 62: 8171}
    rows = []
    for directory in sorted((ROOT / "ablation_results").iterdir()):
        metadata_path = directory / "metadata.json"
        results_path = directory / "results.json"
        if not metadata_path.exists() or not results_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        seed = metadata.get("split_seed")
        if (
            metadata.get("dataset_rows") != 13605
            or metadata.get("split_strategy") != "group"
            or seed not in expected_train
            or metadata.get("train_rows") != expected_train[seed]
        ):
            continue
        for item in json.loads(results_path.read_text(encoding="utf-8")):
            if item["name"] not in labels:
                continue
            metrics = item["val"]
            rows.append({
                "Configuration": labels[item["name"]],
                "Partition": seed,
                "RMSE": metrics["efficiency_rmse"],
                "Group Spearman": metrics["efficiency_within_group_spearman"],
                "Pair accuracy": metrics["efficiency_pair_accuracy"],
            })
    frame = pd.DataFrame(rows).drop_duplicates(
        ["Configuration", "Partition"], keep="last"
    )
    if frame.empty:
        return
    order = list(labels.values())
    frame.to_csv(OUT / "ablation_plot_data.csv", index=False)
    summary = frame.groupby("Configuration")[
        ["RMSE", "Group Spearman", "Pair accuracy"]
    ].agg(["mean", "std", "count"]).reindex(order)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 5.3))
    colors = [
        "#0072B2", "#999999", "#999999", "#999999",
        "#999999", "#999999", "#999999", "#009E73",
    ]
    y = np.arange(len(order))
    for axis, metric, direction in [
        (axes[0], "RMSE", "lower is better"),
        (axes[1], "Group Spearman", "higher is better"),
        (axes[2], "Pair accuracy", "higher is better"),
    ]:
        means = summary[(metric, "mean")].to_numpy(float)
        errors = summary[(metric, "std")].to_numpy(float)
        for row_index, (mean, error, color) in enumerate(
            zip(means, errors, colors)
        ):
            axis.errorbar(
                mean, row_index, xerr=error, fmt="o", color=color,
                ecolor=color, markersize=5, elinewidth=1.4, capsize=3,
                zorder=3,
            )
        axis.set_xlabel(f"Validation {metric}\n({direction}; mean ± SD, n=3)")
        axis.set_yticks(y)
        axis.set_yticklabels(order if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)
    axes[1].axvline(0, color="#777777", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"ablation_latest.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def reliability_and_ad():
    candidates = []
    for directory in (ROOT / "ablation_results").iterdir():
        metadata_path = directory / "metadata.json"
        results_path = directory / "results.json"
        if not metadata_path.exists() or not results_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("dataset_rows") != 13605
            or metadata.get("split_strategy") != "group"
            or metadata.get("split_seed") != 42
        ):
            continue
        for item in json.loads(results_path.read_text(encoding="utf-8")):
            if (
                item["name"] == "A77_centered_plus_light_listmle"
                and "efficiency_ad_joint_mean" in item["test"]
            ):
                candidates.append((directory.stat().st_mtime, item))
    if not candidates:
        return
    item = max(candidates, key=lambda pair: pair[0])[1]
    nominal = [0.50, 0.80, 0.90, 0.95]
    val = [item["val"][f"efficiency_coverage_{level}"] for level in [50, 80, 90, 95]]
    test = [item["test"][f"efficiency_coverage_{level}"] for level in [50, 80, 90, 95]]
    bins = ["[0.5, 0.7)", "[0.7, 0.85)", r"$\geq 0.85$"]
    rmse = [
        item["test"]["efficiency_ad_joint_0p5_0p7_rmse"],
        item["test"]["efficiency_ad_joint_0p7_0p85_rmse"],
        item["test"]["efficiency_ad_joint_ge_0p85_rmse"],
    ]
    counts = [
        int(item["test"]["efficiency_ad_joint_0p5_0p7_n"]),
        int(item["test"]["efficiency_ad_joint_0p7_0p85_n"]),
        int(item["test"]["efficiency_ad_joint_ge_0p85_n"]),
    ]
    pd.DataFrame({"nominal_coverage": nominal, "validation": val, "test": test}).to_csv(
        OUT / "coverage_plot_data.csv", index=False
    )
    pd.DataFrame({"joint_support_bin": bins, "n": counts, "test_rmse": rmse}).to_csv(
        OUT / "applicability_plot_data.csv", index=False
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].plot([0.45, 1.0], [0.45, 1.0], "--", color="#666666", label="Ideal")
    axes[0].plot(nominal, val, marker="o", color="#0072B2", label="Validation")
    axes[0].plot(nominal, test, marker="s", color="#D55E00", label="Held-out test")
    axes[0].set_xlabel("Nominal interval coverage")
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_xlim(0.45, 1.0)
    axes[0].set_ylim(0.15, 1.03)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    bars = axes[1].bar(bins, rmse, color=["#009E73", "#56B4E9", "#CC79A7"])
    axes[1].set_xlabel("Joint multi-role/context support")
    axes[1].set_ylabel("Held-out test RMSE")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, count in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                     f"n={count}", ha="center", va="bottom", fontsize=8)
    axes[1].set_ylim(0, max(rmse) * 1.18)
    fig.tight_layout()
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"reliability_ad.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def full_corpus_comparison():
    rows = []
    expected_train = {42: 10728, 52: 10276, 62: 8171}
    for directory in sorted((ROOT / "ablation_results").iterdir()):
        metadata_path = directory / "metadata.json"
        results_path = directory / "results.json"
        if not metadata_path.exists() or not results_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        seed = metadata.get("split_seed")
        if (
            metadata.get("dataset_rows") != 13605
            or metadata.get("split_strategy") != "group"
            or seed not in expected_train
            or metadata.get("train_rows") != expected_train[seed]
        ):
            continue
        results = {
            item["name"]: item
            for item in json.loads(results_path.read_text(encoding="utf-8"))
        }
        for internal, label in [
            ("A56_group_centered_screen_objective", "Centered objective"),
            ("A77_centered_plus_light_listmle", "Centered + ListMLE"),
        ]:
            if internal not in results:
                continue
            item = results[internal]["test"]
            rows.append({
                "Model": label, "Partition": seed,
                "RMSE": item["efficiency_rmse"],
                "Group Spearman": item["efficiency_within_group_spearman"],
                "Pair accuracy": item["efficiency_pair_accuracy"],
            })

    benchmark_root = ROOT / "method" / "benchmark_results"
    tree_directories = []
    for directory in benchmark_root.iterdir():
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists() or not (directory / "runs.csv").exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("split_strategy") == "group"
            and set(metadata.get("split_seeds", [])) >= {42, 52, 62}
            and {"random_forest", "extra_trees"}.issubset(
                set(metadata.get("families", []))
            )
        ):
            tree_directories.append(directory)
    tree_directory = max(tree_directories, key=lambda path: path.stat().st_mtime)
    trees = pd.read_csv(tree_directory / "runs.csv")
    trees = trees[
        trees["split_seed"].isin([42, 52, 62])
        & trees["family"].isin(["random_forest", "extra_trees"])
    ]
    for _, item in trees.iterrows():
        rows.append({
            "Model": (
                "Bae-style Random Forest"
                if item["family"] == "random_forest"
                else "Role-aware Extra Trees"
            ),
            "Partition": int(item["split_seed"]),
            "RMSE": item["test_rmse"],
            "Group Spearman": item["test_within_group_spearman"],
            "Pair accuracy": item["test_pair_accuracy"],
        })

    lift_directories = sorted(
        benchmark_root.glob("lift_same_corpus_*"),
        key=lambda path: path.stat().st_mtime,
    )
    if lift_directories:
        lift = pd.read_csv(lift_directories[-1] / "runs.csv")
        for _, item in lift[lift["split"] == "test"].iterrows():
            rows.append({
                "Model": "LIFT adaptation",
                "Partition": int(item["seed"]),
                "RMSE": item["rmse"],
                "Group Spearman": item["within_group_spearman"],
                "Pair accuracy": item["pair_accuracy"],
            })
    hybrid_directories = []
    for directory in benchmark_root.glob("dual_head_hybrid_*"):
        metadata_path = directory / "metadata.json"
        runs_path = directory / "runs.csv"
        if not metadata_path.exists() or not runs_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        run_frame = pd.read_csv(runs_path)
        test_partitions = run_frame[run_frame["split"] == "test"]
        if (
            metadata.get("deep_variant")
            == "A77_centered_plus_light_listmle"
            and metadata.get("selection_scope") == "per_partition"
            and metadata.get("rank_selection") == "group_spearman"
            and set(test_partitions["split_seed"].astype(int)) == {42, 52, 62}
            and len(test_partitions) == 3
        ):
            hybrid_directories.append(directory)
    if hybrid_directories:
        hybrid_directory = max(
            hybrid_directories, key=lambda path: path.stat().st_mtime
        )
        hybrid = pd.read_csv(hybrid_directory / "runs.csv")
        for _, item in hybrid[hybrid["split"] == "test"].iterrows():
            rows.append({
                "Model": "Validation-selected hybrid",
                "Partition": int(item["split_seed"]),
                "RMSE": item["rmse"],
                "Group Spearman": item["mean_group_spearman"],
                "Pair accuracy": item["pair_accuracy"],
            })
    frame = pd.DataFrame(rows).drop_duplicates(["Model", "Partition"], keep="last")
    frame.to_csv(OUT / "full_corpus_comparison_data.csv", index=False)
    order = [
        "Centered objective", "Centered + ListMLE",
        "Bae-style Random Forest", "Role-aware Extra Trees",
        "LIFT adaptation", "Validation-selected hybrid",
    ]
    summary = frame.groupby("Model")[["RMSE", "Group Spearman", "Pair accuracy"]].agg(
        ["mean", "std", "count"]
    ).reindex(order)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.7))
    palette = [
        "#0072B2", "#009E73", "#E69F00",
        "#D55E00", "#CC79A7", "#56B4E9",
    ]
    y = np.arange(len(order))
    for axis, metric, better in [
        (axes[0], "RMSE", "lower is better"),
        (axes[1], "Group Spearman", "higher is better"),
        (axes[2], "Pair accuracy", "higher is better"),
    ]:
        means = summary[(metric, "mean")].to_numpy(float)
        errors = summary[(metric, "std")].to_numpy(float)
        for row_index, (mean, error, color) in enumerate(
            zip(means, errors, palette)
        ):
            axis.errorbar(
                mean, row_index, xerr=error, fmt="o", color=color,
                ecolor=color, markersize=5, elinewidth=1.5, capsize=3,
                zorder=3,
            )
        axis.set_xlabel(f"{metric}\n({better}; mean ± SD, n=3)")
        axis.set_yticks(y)
        axis.set_yticklabels(order if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"full_corpus_comparison.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data_identifiability()
    architecture()
    evaluation_framework()
    latest_ablation()
    reliability_and_ad()
    full_corpus_comparison()
    print(f"Figures written to {OUT}")


if __name__ == "__main__":
    main()
