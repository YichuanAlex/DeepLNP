#!/usr/bin/env python3
"""Run the strictly matched efficacy-only ablation on five balanced folds.

This runner deliberately bypasses the automatic ablation ledger.  It writes a
standalone, timestamped benchmark bundle so that a reviewer-requested
confirmation experiment cannot rewrite the manuscript or ``ablation.md``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, TextIO

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr, t
from sklearn.metrics import ndcg_score

from deeplnp.data.unified_dataset import create_unified_dataloaders
from run_ablations import (
    A79_VALIDATION_PARETO_SWITCHES,
    A101_EFFICACY_ONLY_PARETO_SWITCHES,
    DEVICE,
    run_one,
)


ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = ROOT / "method" / "benchmark_results"
REFERENCE_A79_AUDIT = (
    BENCHMARK_ROOT / "decision_audit_20260723_180455" / "summary.csv"
)
REFERENCE_A79_PER_SCREEN = (
    BENCHMARK_ROOT / "decision_audit_20260723_180455" / "per_screen.csv"
)

PROTOCOL: Dict[str, Any] = {
    "experiment": "A101_efficacy_only_pareto",
    "reference": "A79_validation_pareto_checkpoint",
    "sample_fraction": 1.0,
    "split_strategy": "balanced_group_kfold",
    "split_seed": 20260723,
    "group_folds": 5,
    "group_definition": "screen_context",
    "fold_seeds": [20260723, 20260724, 20260725, 20260726, 20260727],
    "epochs": 6,
    "patience": 2,
    "learning_rate": 5e-4,
    "batch_size": 32,
    "use_conformers": False,
    "include_source_context": False,
    "use_mechanistic_descriptors": False,
    "target_encoding": "onehot",
    "balanced_sampling": True,
    "compute_final_ad": True,
}


class Tee:
    """Mirror fold progress to the terminal and an immutable text log."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def assert_strict_delta() -> Dict[str, Dict[str, Any]]:
    """Fail before training if A101 differs from A79 by anything unintended."""

    reference = copy.deepcopy(A79_VALIDATION_PARETO_SWITCHES)
    candidate = copy.deepcopy(A101_EFFICACY_ONLY_PARETO_SWITCHES)
    differing = {
        key: {"A79": reference.get(key), "A101": candidate.get(key)}
        for key in sorted(set(reference) | set(candidate))
        if reference.get(key) != candidate.get(key)
    }
    expected = {
        "enabled_tasks": {"A79": None, "A101": ["efficiency"]},
        "loss_target_aux_weight": {"A79": None, "A101": 0.0},
    }
    if differing != expected:
        raise AssertionError(
            "A101 is not the required two-key delta from A79: "
            + json.dumps(differing, ensure_ascii=False, sort_keys=True)
        )
    if candidate["checkpoint_selection"] != "efficacy_pareto":
        raise AssertionError("A101 checkpoint selection drifted from A79")
    if not candidate["group_pair_batches"]:
        raise AssertionError("A101 must retain the A79 group-pair sampler")
    return differing


def safe_spearman(prediction: np.ndarray, target: np.ndarray) -> float:
    if len(target) < 3 or np.std(prediction) < 1e-10 or np.std(target) < 1e-10:
        return float("nan")
    return float(spearmanr(prediction, target).statistic)


def per_screen_metrics(
    prediction_record: Dict[str, Iterable[Any]], fold_index: int, split: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(prediction_record)
    rows = []
    for group, local in frame.groupby("group", sort=True):
        prediction = local["prediction"].to_numpy(dtype=float)
        rank_score = local["rank_score"].to_numpy(dtype=float)
        target = local["target"].to_numpy(dtype=float)
        n = len(local)
        if n < 2:
            continue
        relevance = target - target.min() + 1e-8
        k = max(1, int(math.ceil(0.10 * n)))
        true_top = set(np.argsort(target)[-k:])
        predicted_top = np.argsort(rank_score)[-k:]
        hit = len(true_top.intersection(predicted_top)) / k
        regret = float(target.max() - target[predicted_top].max())
        target_range = float(target.max() - target.min())
        rows.append({
            "fold_index": fold_index,
            "seed": PROTOCOL["fold_seeds"][fold_index],
            "split": split,
            "screen": group,
            "n": n,
            "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
            "spearman": safe_spearman(rank_score, target),
            "ndcg": float(ndcg_score(
                relevance.reshape(1, -1), rank_score.reshape(1, -1)
            )),
            "hit_recall_at_10pct": hit,
            "normalized_regret_at_10pct": (
                regret / target_range if target_range > 1e-8 else 0.0
            ),
        })
    return pd.DataFrame(rows)


def prediction_frame(
    record: Dict[str, Iterable[Any]], fold_index: int, split: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(record)
    frame.insert(0, "split", split)
    frame.insert(0, "seed", PROTOCOL["fold_seeds"][fold_index])
    frame.insert(0, "fold_index", fold_index)
    return frame


def compact_fold_metrics(result: Dict[str, Any], fold_index: int) -> Dict[str, Any]:
    train = result["training_accounting"]
    row: Dict[str, Any] = {
        "model": result["name"],
        "fold_index": fold_index,
        "seed": result["seed"],
        "parameters": result["parameters"],
        "epochs_trained": result["epochs_trained"],
        "attempted_batches": train["attempted_batches"],
        "optimizer_steps": train["optimizer_steps"],
        "row_exposures": train["row_exposures"],
        "efficiency_label_exposures": train["efficiency_label_exposures"],
    }
    metric_keys = {
        "rmse": "efficiency_rmse",
        "mean_per_screen_spearman": "efficiency_within_group_spearman",
        "hit_recall_at_10pct": "efficiency_within_group_hit_recall_at_10pct",
        "normalized_regret_at_10pct": (
            "efficiency_within_group_normalized_regret_at_10pct"
        ),
        "screen_count": "efficiency_group_count",
        "n": "efficiency_n",
    }
    for split in ("val", "test"):
        metrics = result[split]
        for output_name, source_name in metric_keys.items():
            row[f"{split}_{output_name}"] = metrics.get(source_name, float("nan"))
    return row


def load_reference_a79() -> pd.DataFrame:
    if not REFERENCE_A79_AUDIT.exists():
        return pd.DataFrame()
    source = pd.read_csv(REFERENCE_A79_AUDIT)
    keep = (
        source["model"].eq("A79_validation_pareto_checkpoint")
        & source["split"].isin(["val", "test"])
    )
    reference = source.loc[keep].copy()
    return reference[[
        "fold_index", "seed", "split", "n", "groups", "rmse",
        "within_group_spearman", "within_group_hit_recall_10pct",
        "within_group_normalized_regret_10pct",
    ]].sort_values(["fold_index", "split"])


def write_summary(output_dir: Path) -> None:
    fold_files = sorted(output_dir.glob("fold_*/fold_metrics.json"))
    if not fold_files:
        return
    fold_rows = [
        json.loads(path.read_text(encoding="utf-8")) for path in fold_files
    ]
    folds = pd.DataFrame(fold_rows).sort_values("fold_index")
    folds.to_csv(output_dir / "fold_metrics.csv", index=False)

    metric_columns = [
        "test_rmse",
        "test_mean_per_screen_spearman",
        "test_hit_recall_at_10pct",
        "test_normalized_regret_at_10pct",
        "optimizer_steps",
        "row_exposures",
        "efficiency_label_exposures",
    ]
    summary_rows = []
    for metric in metric_columns:
        values = pd.to_numeric(folds[metric], errors="coerce").dropna()
        summary_rows.append({
            "metric": metric,
            "folds": int(len(values)),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
            "min": float(values.min()),
            "max": float(values.max()),
            "sum": float(values.sum()) if metric.endswith(
                ("steps", "exposures")
            ) else float("nan"),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)

    reference = load_reference_a79()
    if len(reference):
        reference.to_csv(output_dir / "reference_A79_fold_metrics.csv", index=False)
        test_reference = reference.loc[reference["split"].eq("test")].copy()
        test_reference = test_reference.rename(columns={
            "rmse": "A79_test_rmse",
            "within_group_spearman": "A79_test_mean_per_screen_spearman",
            "within_group_hit_recall_10pct": "A79_test_hit_recall_at_10pct",
            "within_group_normalized_regret_10pct": (
                "A79_test_normalized_regret_at_10pct"
            ),
        })
        comparison = folds.merge(
            test_reference.drop(columns=["split", "n", "groups"]),
            on=["fold_index", "seed"],
            how="left",
            validate="one_to_one",
        )
        for metric in (
            "rmse",
            "mean_per_screen_spearman",
            "hit_recall_at_10pct",
            "normalized_regret_at_10pct",
        ):
            comparison[f"A101_minus_A79_{metric}"] = (
                comparison[f"test_{metric}"] - comparison[f"A79_test_{metric}"]
            )
        comparison.to_csv(output_dir / "comparison_to_A79.csv", index=False)
        paired_rows = []
        for metric in (
            "rmse",
            "mean_per_screen_spearman",
            "hit_recall_at_10pct",
            "normalized_regret_at_10pct",
        ):
            difference = comparison[f"A101_minus_A79_{metric}"].dropna()
            paired_rows.append({
                "metric": metric,
                "folds": int(len(difference)),
                "A101_mean": float(comparison[f"test_{metric}"].mean()),
                "A101_std": float(comparison[f"test_{metric}"].std(ddof=1)),
                "A79_mean": float(comparison[f"A79_test_{metric}"].mean()),
                "A79_std": float(
                    comparison[f"A79_test_{metric}"].std(ddof=1)
                ),
                "A101_minus_A79_mean": float(difference.mean()),
                "A101_minus_A79_std": float(difference.std(ddof=1)),
                "A101_minus_A79_min": float(difference.min()),
                "A101_minus_A79_max": float(difference.max()),
                "paired_t95_ci_low": float(
                    difference.mean()
                    - t.ppf(0.975, df=len(difference) - 1)
                    * difference.std(ddof=1) / math.sqrt(len(difference))
                ),
                "paired_t95_ci_high": float(
                    difference.mean()
                    + t.ppf(0.975, df=len(difference) - 1)
                    * difference.std(ddof=1) / math.sqrt(len(difference))
                ),
                "inference_note": (
                    "Paired t interval over five prespecified folds; n=5 is "
                    "small, so treat the interval as descriptive."
                ),
            })
        pd.DataFrame(paired_rows).to_csv(
            output_dir / "paired_comparison_summary.csv", index=False
        )

    all_per_screen = []
    all_predictions = []
    for fold_dir in sorted(output_dir.glob("fold_*")):
        screen_path = fold_dir / "per_screen_metrics.csv"
        prediction_path = fold_dir / "predictions.csv"
        if screen_path.exists():
            all_per_screen.append(pd.read_csv(screen_path))
        if prediction_path.exists():
            all_predictions.append(pd.read_csv(prediction_path))
    if all_per_screen:
        pd.concat(all_per_screen, ignore_index=True).to_csv(
            output_dir / "per_screen_metrics.csv", index=False
        )
    if all_predictions:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        combined_predictions.to_csv(output_dir / "predictions.csv", index=False)
    else:
        combined_predictions = pd.DataFrame()

    combined_per_screen = (
        pd.concat(all_per_screen, ignore_index=True)
        if all_per_screen else pd.DataFrame()
    )
    test_predictions = (
        combined_predictions.loc[combined_predictions["split"].eq("test")]
        if len(combined_predictions) else pd.DataFrame()
    )
    test_screens = (
        combined_per_screen.loc[combined_per_screen["split"].eq("test")]
        if len(combined_per_screen) else pd.DataFrame()
    )
    integrity = {
        "completed_folds": int(len(folds)),
        "test_prediction_rows": int(len(test_predictions)),
        "test_unique_dataset_indices": (
            int(test_predictions["idx"].nunique()) if len(test_predictions) else 0
        ),
        "test_duplicate_dataset_indices": (
            int(test_predictions["idx"].duplicated().sum())
            if len(test_predictions) else 0
        ),
        "test_screen_rows": int(len(test_screens)),
        "test_unique_screens": (
            int(test_screens["screen"].nunique()) if len(test_screens) else 0
        ),
        "expected_complete_oof_rows": 12692,
        "expected_complete_oof_screens": 38,
    }
    integrity["complete_oof_pass"] = bool(
        integrity["completed_folds"] == 5
        and integrity["test_prediction_rows"] == 12692
        and integrity["test_unique_dataset_indices"] == 12692
        and integrity["test_duplicate_dataset_indices"] == 0
        and integrity["test_screen_rows"] == 38
        and integrity["test_unique_screens"] == 38
    )
    (output_dir / "integrity_checks.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if integrity["completed_folds"] == 5 and not integrity["complete_oof_pass"]:
        raise AssertionError(
            "Completed five folds but failed OOF integrity checks: "
            + json.dumps(integrity, sort_keys=True)
        )

    if (
        integrity["complete_oof_pass"]
        and REFERENCE_A79_PER_SCREEN.exists()
        and len(combined_per_screen)
    ):
        reference_screens = pd.read_csv(REFERENCE_A79_PER_SCREEN)
        reference_screens = reference_screens.loc[
            reference_screens["model"].eq("A79_validation_pareto_checkpoint")
            & reference_screens["split"].eq("test"),
            [
                "fold_index", "seed", "group", "n", "rank_spearman",
                "hit_recall_10pct", "normalized_regret_10pct",
            ],
        ].rename(columns={
            "group": "screen",
            "n": "A79_n",
            "rank_spearman": "A79_spearman",
            "hit_recall_10pct": "A79_hit_recall_at_10pct",
            "normalized_regret_10pct": "A79_normalized_regret_at_10pct",
        })
        candidate_screens = test_screens[[
            "fold_index", "seed", "screen", "n", "spearman",
            "hit_recall_at_10pct", "normalized_regret_at_10pct",
        ]].rename(columns={
            "n": "A101_n",
            "spearman": "A101_spearman",
            "hit_recall_at_10pct": "A101_hit_recall_at_10pct",
            "normalized_regret_at_10pct": (
                "A101_normalized_regret_at_10pct"
            ),
        })
        screen_comparison = candidate_screens.merge(
            reference_screens,
            on=["fold_index", "seed", "screen"],
            how="inner",
            validate="one_to_one",
        )
        if len(screen_comparison) != 38:
            raise AssertionError(
                "Expected 38 aligned A79/A101 held-out screens, got "
                f"{len(screen_comparison)}"
            )
        if not np.array_equal(
            screen_comparison["A101_n"].to_numpy(),
            screen_comparison["A79_n"].to_numpy(),
        ):
            raise AssertionError("A79/A101 screen sample counts are not aligned")

        screen_metric_names = [
            "spearman", "hit_recall_at_10pct",
            "normalized_regret_at_10pct",
        ]
        for metric in screen_metric_names:
            screen_comparison[f"A101_minus_A79_{metric}"] = (
                screen_comparison[f"A101_{metric}"]
                - screen_comparison[f"A79_{metric}"]
            )
        screen_comparison.to_csv(
            output_dir / "screen_comparison_to_A79.csv", index=False
        )

        rng = np.random.default_rng(20260730)
        bootstrap_rows = []
        bootstrap_repeats = 10000
        for metric in screen_metric_names:
            differences = screen_comparison[
                f"A101_minus_A79_{metric}"
            ].dropna().to_numpy(dtype=float)
            draws = rng.choice(
                differences,
                size=(bootstrap_repeats, len(differences)),
                replace=True,
            ).mean(axis=1)
            bootstrap_rows.append({
                "metric": metric,
                "screens": int(len(differences)),
                "A101_minus_A79_mean": float(differences.mean()),
                "screen_bootstrap_95_ci_low": float(
                    np.quantile(draws, 0.025)
                ),
                "screen_bootstrap_95_ci_high": float(
                    np.quantile(draws, 0.975)
                ),
                "bootstrap_repeats": bootstrap_repeats,
                "bootstrap_seed": 20260730,
                "bootstrap_unit": "held-out experimental screen",
            })
        pd.DataFrame(bootstrap_rows).to_csv(
            output_dir / "screen_paired_bootstrap.csv", index=False
        )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    effective_model_config = {
        **manifest["base_model_config"],
        **manifest["A101_switches"],
    }
    manifest["effective_model_config"] = effective_model_config
    manifest["config_resolution_checks"] = {
        "merge_rule": (
            "run_ablations.build_model deep-copies base_model_config then "
            "applies cfg.update(switches); ablation switches take precedence"
        ),
        "base_rank_detach_backbone": manifest["base_model_config"].get(
            "rank_detach_backbone"
        ),
        "A79_switch_rank_detach_backbone": manifest["A79_switches"].get(
            "rank_detach_backbone"
        ),
        "A101_switch_rank_detach_backbone": manifest["A101_switches"].get(
            "rank_detach_backbone"
        ),
        "effective_rank_detach_backbone": effective_model_config.get(
            "rank_detach_backbone"
        ),
        "override_pass": bool(
            effective_model_config.get("rank_detach_backbone") is False
        ),
    }
    manifest["completed_folds"] = folds["fold_index"].astype(int).tolist()
    manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["summary_files"] = [
        "fold_metrics.csv", "summary.csv", "per_screen_metrics.csv",
        "predictions.csv", "reference_A79_fold_metrics.csv",
        "comparison_to_A79.csv", "paired_comparison_summary.csv",
        "integrity_checks.json", "screen_comparison_to_A79.csv",
        "screen_paired_bootstrap.csv",
    ]
    (output_dir / "manifest.json").write_text(
        json.dumps(json_ready(manifest), ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )


def create_output_dir(path: Path | None, dry_run: bool) -> Path:
    if path is not None:
        output_dir = path.resolve()
    else:
        suffix = "_smoke" if dry_run else ""
        output_dir = (
            BENCHMARK_ROOT
            / f"matched_efficacy_ablation_{datetime.now():%Y%m%d_%H%M%S}{suffix}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one epoch on the requested fold(s) to validate the full path.",
    )
    parser.add_argument(
        "--skip-ad",
        action="store_true",
        help="Development-only speed option; do not use for the final matched run.",
    )
    args = parser.parse_args()

    if any(fold < 0 or fold >= PROTOCOL["group_folds"] for fold in args.folds):
        raise ValueError("fold indices must be in [0, 4]")
    if args.dry_run and len(args.folds) != 1:
        raise ValueError("dry-run must target exactly one fold")
    if not args.dry_run and args.skip_ad:
        raise ValueError("the final strict protocol requires applicability diagnostics")

    delta = assert_strict_delta()
    output_dir = create_output_dir(args.output_dir, args.dry_run)
    with open(ROOT / "configs" / "research_smoke_config.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    effective_protocol = {
        **PROTOCOL,
        "epochs": 1 if args.dry_run else PROTOCOL["epochs"],
        "patience": 0 if args.dry_run else PROTOCOL["patience"],
        "compute_final_ad": not args.skip_ad,
        "dry_run": bool(args.dry_run),
    }
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "protocol": effective_protocol,
        "strict_switch_delta": delta,
        "A79_switches": A79_VALIDATION_PARETO_SWITCHES,
        "A101_switches": A101_EFFICACY_ONLY_PARETO_SWITCHES,
        "base_model_config": config["model"],
        "device": str(DEVICE),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "reference_A79_audit": str(REFERENCE_A79_AUDIT),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(json_ready(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for fold_index in args.folds:
        fold_dir = output_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=False)
        log_path = fold_dir / "run.log"
        with log_path.open("w", encoding="utf-8") as log_handle:
            with redirect_stdout(Tee(sys.__stdout__, log_handle)):
                seed = PROTOCOL["fold_seeds"][fold_index]
                print(
                    f"Starting A101 fold={fold_index}, seed={seed}, "
                    f"device={DEVICE}, dry_run={args.dry_run}"
                )
                train_loader, val_loader, test_loader, dataset = (
                    create_unified_dataloaders(
                        str(ROOT / "merged_datasets"),
                        batch_size=PROTOCOL["batch_size"],
                        sample_fraction=PROTOCOL["sample_fraction"],
                        random_seed=PROTOCOL["split_seed"],
                        split_strategy=PROTOCOL["split_strategy"],
                        use_spatial=PROTOCOL["use_conformers"],
                        pin_memory=DEVICE.type == "cuda",
                        balanced_sampling=PROTOCOL["balanced_sampling"],
                        ensure_rare_task_holdout=False,
                        include_source_context=PROTOCOL["include_source_context"],
                        use_mechanistic_descriptors=(
                            PROTOCOL["use_mechanistic_descriptors"]
                        ),
                        target_encoding=PROTOCOL["target_encoding"],
                        group_definition=PROTOCOL["group_definition"],
                        group_folds=PROTOCOL["group_folds"],
                        group_fold_index=fold_index,
                    )
                )
                print(
                    "Split rows: "
                    f"{len(train_loader.dataset)}/"
                    f"{len(val_loader.dataset)}/"
                    f"{len(test_loader.dataset)}"
                )
                print(f"Prewarming {len(dataset)} deterministic molecular records")
                for index in range(len(dataset)):
                    dataset[index]
                    if (index + 1) % 1000 == 0:
                        print(f"  cached {index + 1}/{len(dataset)}")

                result = run_one(
                    "A101_efficacy_only_pareto",
                    copy.deepcopy(A101_EFFICACY_ONLY_PARETO_SWITCHES),
                    config["model"],
                    train_loader,
                    val_loader,
                    test_loader,
                    dataset,
                    seed,
                    effective_protocol["epochs"],
                    PROTOCOL["balanced_sampling"],
                    PROTOCOL["learning_rate"],
                    effective_protocol["patience"],
                    compute_final_ad=effective_protocol["compute_final_ad"],
                )
                if result["parameters"] != 390564:
                    raise AssertionError(
                        "Network drift detected: expected the frozen A79 "
                        f"parameter count 390564, got {result['parameters']}"
                    )

                pd.DataFrame(result["history"]).to_csv(
                    fold_dir / "history.csv", index=False
                )
                metrics = compact_fold_metrics(result, fold_index)
                (fold_dir / "fold_metrics.json").write_text(
                    json.dumps(
                        json_ready(metrics), ensure_ascii=False, indent=2,
                        allow_nan=True,
                    ),
                    encoding="utf-8",
                )
                (fold_dir / "val_metrics.json").write_text(
                    json.dumps(
                        json_ready(result["val"]), ensure_ascii=False, indent=2,
                        allow_nan=True,
                    ),
                    encoding="utf-8",
                )
                (fold_dir / "test_metrics.json").write_text(
                    json.dumps(
                        json_ready(result["test"]), ensure_ascii=False, indent=2,
                        allow_nan=True,
                    ),
                    encoding="utf-8",
                )
                predictions = pd.concat([
                    prediction_frame(
                        result["predictions"][split], fold_index, split
                    )
                    for split in ("val", "test")
                ], ignore_index=True)
                predictions.to_csv(fold_dir / "predictions.csv", index=False)
                per_screen = pd.concat([
                    per_screen_metrics(
                        result["predictions"][split], fold_index, split
                    )
                    for split in ("val", "test")
                ], ignore_index=True)
                per_screen.to_csv(
                    fold_dir / "per_screen_metrics.csv", index=False
                )
                print(
                    "Completed fold "
                    f"{fold_index}: epochs={metrics['epochs_trained']}, "
                    f"steps={metrics['optimizer_steps']}, "
                    f"efficacy_exposures={metrics['efficiency_label_exposures']}, "
                    f"test RMSE={metrics['test_rmse']:.6f}, "
                    "mean per-screen Spearman="
                    f"{metrics['test_mean_per_screen_spearman']:.6f}, "
                    f"Hit@10%={metrics['test_hit_recall_at_10pct']:.6f}, "
                    "normalized regret@10%="
                    f"{metrics['test_normalized_regret_at_10pct']:.6f}"
                )
        write_summary(output_dir)

    print(f"Saved matched benchmark to {output_dir}")


if __name__ == "__main__":
    main()
