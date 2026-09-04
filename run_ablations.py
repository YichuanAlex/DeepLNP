#!/usr/bin/env python3
"""Reproducible DeepLNP-SPACE ablations with group-safe evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, ndcg_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, RandomSampler, Sampler, Subset, WeightedRandomSampler

from deeplnp.data.unified_dataset import (
    CONTEXT_DIM, CONTEXT_VOCABS, FORMULATION_DIM, SPATIAL_DIM, TARGET_DIM, TASK_COLUMNS,
    create_unified_dataloaders, unified_collate_fn,
)
from deeplnp.models.predictor import LNPPredictor


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TASK_WEIGHTS = {
    "efficiency": 1.0, "particle_size": 0.5, "zeta_potential": 0.3,
    "pdi": 0.3, "encapsulation": 0.4, "toxicity": 0.7,
}

# Frozen endpoint-balanced reference used in the manuscript confirmation
# experiment.  Keep the matched efficacy-only ablation as a two-key delta
# from this dictionary so that sampler, architecture, objective, and
# checkpoint-selection drift can be checked mechanically.
A79_VALIDATION_PARETO_SWITCHES: Dict[str, Any] = {
    "use_component_transformer": True,
    "use_pair_bias": False,
    "use_spatial_features": True,
    "use_target_conditioning": True,
    "use_ratio_features": True,
    "use_structure_mask_features": False,
    "use_explicit_features": False,
    "use_separate_rank_head": True,
    "rank_head_mode": "residual",
    "rank_detach_backbone": False,
    "rank_gate_init": -1.0,
    "loss_ranking_weight": 1.0,
    "group_center_rank_weight": 0.5,
    "rank_margin": 0.1,
    "drop_source_context": True,
    "drop_mechanistic_descriptors": True,
    "group_pair_batches": True,
    "groups_per_batch": 4,
    "rows_per_group": 4,
    "checkpoint_selection": "efficacy_pareto",
}
A101_EFFICACY_ONLY_PARETO_SWITCHES: Dict[str, Any] = {
    **A79_VALIDATION_PARETO_SWITCHES,
    "enabled_tasks": ["efficiency"],
    "loss_target_aux_weight": 0.0,
}
A102_NO_RANK_ADAPTER_SWITCHES: Dict[str, Any] = {
    **A79_VALIDATION_PARETO_SWITCHES,
    "use_separate_rank_head": False,
    "loss_ranking_weight": 0.0,
    "group_center_rank_weight": 0.0,
}
A103_NO_COMPONENT_ATTENTION_SWITCHES: Dict[str, Any] = {
    **A79_VALIDATION_PARETO_SWITCHES,
    "use_component_transformer": False,
}
A104_NO_BIOLOGICAL_CONTEXT_SWITCHES: Dict[str, Any] = {
    **A79_VALIDATION_PARETO_SWITCHES,
    "use_target_conditioning": False,
    "drop_all_context_features": True,
    "drop_target_features": True,
}
A105_NO_PHYSICAL_ENDPOINTS_SWITCHES: Dict[str, Any] = {
    **A79_VALIDATION_PARETO_SWITCHES,
    "enabled_tasks": ["efficiency"],
}
A106_NO_TARGET_AUXILIARY_SWITCHES: Dict[str, Any] = {
    **A79_VALIDATION_PARETO_SWITCHES,
    "loss_target_aux_weight": 0.0,
}
A107_NO_SPATIAL_DESCRIPTORS_SWITCHES: Dict[str, Any] = {
    **A79_VALIDATION_PARETO_SWITCHES,
    "use_spatial_features": False,
}
A108_NO_STEROL_STRUCTURE_SWITCHES: Dict[str, Any] = {
    **A79_VALIDATION_PARETO_SWITCHES,
    "drop_component_structure_inputs": ["cholesterol"],
}
A109_IONIZABLE_ONLY_SWITCHES: Dict[str, Any] = {
    **A79_VALIDATION_PARETO_SWITCHES,
    "drop_component_roles": ["helper", "cholesterol", "peg"],
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_device(batch: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value.to(DEVICE, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def model_inputs(
    batch: Dict[str, Any], switches: Dict[str, Any] | None = None,
    training: bool = False,
) -> Dict[str, Any]:
    keys = [
        "ionizable_atom_features", "ionizable_edge_index", "ionizable_bond_features", "ionizable_batch",
        "helper_atom_features", "helper_edge_index", "helper_bond_features", "helper_batch",
        "cholesterol_atom_features", "cholesterol_edge_index", "cholesterol_bond_features", "cholesterol_batch",
        "peg_atom_features", "peg_edge_index", "peg_bond_features", "peg_batch",
        "ionizable_spatial_features", "helper_spatial_features", "cholesterol_spatial_features", "peg_spatial_features",
        "ionizable_fingerprint", "helper_fingerprint", "cholesterol_fingerprint", "peg_fingerprint",
        "molar_ratios", "formulation_features", "context_features", "target_features",
        "component_structure_mask", "component_active_mask",
    ]
    values = {key: batch.get(key) for key in keys}
    switches = switches or {}
    if switches.get("drop_source_context", False) and values["context_features"] is not None:
        context = values["context_features"].clone()
        source_width = len(CONTEXT_VOCABS["source"])
        context[:, -source_width:] = 0.0
        values["context_features"] = context
    if switches.get("drop_process_missingness", False) and values["formulation_features"] is not None:
        formulation = values["formulation_features"].clone()
        formulation[:, formulation.size(1) // 2:] = 0.0
        values["formulation_features"] = formulation
    if switches.get("drop_all_process_features", False) and values["formulation_features"] is not None:
        values["formulation_features"] = torch.zeros_like(values["formulation_features"])
    if switches.get("drop_all_context_features", False) and values["context_features"] is not None:
        values["context_features"] = torch.zeros_like(values["context_features"])
    if switches.get("drop_target_features", False) and values["target_features"] is not None:
        values["target_features"] = torch.zeros_like(values["target_features"])
    block_dropout = float(switches.get("input_block_dropout", 0.0))
    if training and block_dropout > 0:
        # Row-wise block dropout prevents the network from depending on a single
        # provenance-entangled metadata family while retaining those variables
        # whenever they carry transferable physical information.
        for key in ["formulation_features", "context_features", "target_features"]:
            value = values.get(key)
            if value is None:
                continue
            keep = (torch.rand(value.size(0), 1, device=value.device) >= block_dropout).to(value.dtype)
            values[key] = value * keep
    if switches.get("drop_mechanistic_descriptors", False):
        for component in ["ionizable", "helper", "cholesterol", "peg"]:
            key = f"{component}_spatial_features"
            if values[key] is not None:
                descriptor = values[key].clone()
                descriptor[:, 6:] = 0.0
                values[key] = descriptor
    if switches.get("drop_all_spatial_features", False):
        for component in ["ionizable", "helper", "cholesterol", "peg"]:
            key = f"{component}_spatial_features"
            if values[key] is not None:
                values[key] = torch.zeros_like(values[key])
    component_to_index = {
        "ionizable": 0, "helper": 1, "cholesterol": 2, "peg": 3,
    }
    for switch_name in [
        "drop_component_structure_inputs",
        "drop_component_ratio_values",
        "drop_component_ratio_observation_bits",
    ]:
        requested = list(switches.get(switch_name, []))
        unknown = sorted(set(requested).difference(component_to_index))
        if unknown:
            raise ValueError(f"Unknown component roles for {switch_name}: {unknown}")
        requested = list(dict.fromkeys(requested))
        if switch_name == "drop_component_structure_inputs":
            for component in requested:
                for suffix in [
                    "atom_features", "edge_index", "bond_features", "batch",
                    "spatial_features", "fingerprint",
                ]:
                    values[f"{component}_{suffix}"] = None
                if values["component_structure_mask"] is not None:
                    tensor = values["component_structure_mask"].clone()
                    tensor[:, component_to_index[component]] = 0.0
                    values["component_structure_mask"] = tensor
        elif values["formulation_features"] is not None:
            formulation = values["formulation_features"].clone()
            value_width = formulation.size(1) // 2
            for component in requested:
                role_index = component_to_index[component]
                for column in [role_index, role_index + 4]:
                    if switch_name == "drop_component_ratio_values":
                        formulation[:, column] = 0.0
                        if values["molar_ratios"] is not None and column == role_index:
                            ratios = values["molar_ratios"].clone()
                            ratios[:, role_index] = 0.0
                            values["molar_ratios"] = ratios
                    else:
                        formulation[:, value_width + column] = 0.0
            values["formulation_features"] = formulation
    dropped_roles = list(switches.get("drop_component_roles", []))
    if switches.get("drop_sterol_peg_roles", False):
        dropped_roles.extend(["cholesterol", "peg"])
    if dropped_roles:
        unknown = sorted(set(dropped_roles).difference(component_to_index))
        if unknown:
            raise ValueError(f"Unknown component roles requested for removal: {unknown}")
        dropped_roles = list(dict.fromkeys(dropped_roles))
        for component in dropped_roles:
            for suffix in ["atom_features", "edge_index", "bond_features", "batch", "spatial_features", "fingerprint"]:
                values[f"{component}_{suffix}"] = None
        for key in ["component_structure_mask", "component_active_mask", "molar_ratios"]:
            if values[key] is not None:
                tensor = values[key].clone()
                for component in dropped_roles:
                    tensor[:, component_to_index[component]] = 0.0
                values[key] = tensor
        # FORMULATION_COLUMNS begins with four molar ratios followed by four
        # mass ratios, then process variables; the second half stores their
        # observation indicators.  Remove the role-specific ratio channels
        # and their observation bits as well, otherwise a nominal "role
        # ablation" would still reveal that role through duplicated metadata.
        if values["formulation_features"] is not None:
            formulation = values["formulation_features"].clone()
            value_width = formulation.size(1) // 2
            for component in dropped_roles:
                role_index = component_to_index[component]
                for column in [role_index, role_index + 4]:
                    formulation[:, column] = 0.0
                    formulation[:, value_width + column] = 0.0
            values["formulation_features"] = formulation
    return values


def build_model(base: Dict[str, Any], switches: Dict[str, Any]) -> LNPPredictor:
    cfg = copy.deepcopy(base)
    cfg.update(switches)
    return LNPPredictor(
        mol_encoder_type=cfg["mol_encoder_type"], mol_feat_dim=cfg["mol_feat_dim"],
        atom_feat_dim=cfg["atom_feat_dim"], bond_feat_dim=cfg["bond_feat_dim"],
        num_gnn_layers=cfg["num_gnn_layers"], use_3d=cfg["use_3d"],
        struct_feat_dim=cfg["struct_feat_dim"], use_images=cfg["use_images"],
        image_feat_dim=cfg["image_feat_dim"], image_channels=cfg["image_channels"],
        image_size=cfg["image_size"], use_embeddings=cfg["use_embeddings"],
        embedding_feat_dim=cfg["embedding_feat_dim"], max_embedding_dim=cfg["max_embedding_dim"],
        formul_feat_dim=cfg["formul_feat_dim"], num_components=cfg["num_components"],
        formulation_input_dim=cfg.get("formulation_input_dim", FORMULATION_DIM),
        context_feat_dim=cfg.get("context_feat_dim", CONTEXT_DIM),
        target_feat_dim=cfg.get("target_feat_dim", TARGET_DIM),
        num_target_classes=cfg.get("num_target_classes", TARGET_DIM),
        spatial_feat_dim=cfg.get("spatial_feat_dim", SPATIAL_DIM),
        use_component_transformer=cfg.get("use_component_transformer", True),
        use_pair_bias=cfg.get("use_pair_bias", True),
        use_target_conditioning=cfg.get("use_target_conditioning", True),
        use_spatial_features=cfg.get("use_spatial_features", True),
        use_ratio_features=cfg.get("use_ratio_features", True),
        use_gaussian_ratio=cfg.get("use_gaussian_ratio", False),
        use_structure_mask_features=cfg.get("use_structure_mask_features", True),
        formulation_noise_std=cfg.get("formulation_noise_std", 0.0),
        use_explicit_features=cfg.get("use_explicit_features", False),
        fingerprint_input_dim=cfg.get("fingerprint_input_dim", 2048),
        explicit_fusion_mode=cfg.get("explicit_fusion_mode", "residual"),
        explicit_gate_init=cfg.get("explicit_gate_init", -4.0),
        use_direct_morgan_head=cfg.get("use_direct_morgan_head", False),
        direct_morgan_mode=cfg.get("direct_morgan_mode", "replace"),
        direct_morgan_gate_init=cfg.get("direct_morgan_gate_init", -1.0),
        use_separate_rank_head=cfg.get("use_separate_rank_head", False),
        rank_head_mode=cfg.get("rank_head_mode", "separate"),
        rank_detach_backbone=cfg.get("rank_detach_backbone", False),
        rank_gate_init=cfg.get("rank_gate_init", -1.0),
        physchem_feat_dim=cfg["physchem_feat_dim"], fusion_hidden_dim=cfg["fusion_hidden_dim"],
        num_heads=cfg["fusion_num_heads"], dropout=cfg["fusion_dropout"],
        num_experts=cfg.get("num_experts", 3), prediction_hidden_dim=cfg["prediction_hidden_dim"],
        num_tasks=cfg["num_tasks"], use_mc_dropout=cfg["use_mc_dropout"],
        mc_dropout_rate=cfg.get("mc_dropout_rate", 0.1),
    ).to(DEVICE)


class NoSupervisedEndpointError(RuntimeError):
    """Raised when a minibatch has no label for the enabled task subset."""


def multitask_loss(
    output: Dict[str, Any], batch: Dict[str, Any], ranking_weight: float = 0.5,
    target_aux_weight: float = 0.2, enabled_tasks: Iterable[str] | None = None,
    rank_margin: float = 0.1,
    robust_method: str = "none", robust_weight: float = 0.0,
    adaptive_rank_margin: bool = False,
    group_center_rank_weight: float = 0.0,
    listwise_method: str = "none", listwise_weight: float = 0.0,
    group_z_mse_weight: float = 0.0,
    pair_reduction: str = "global",
    toxicity_loss_mode: str = "censored_gaussian_nll",
    task_weights: Dict[str, float] | None = None,
) -> torch.Tensor:
    predictions = output["predictions"]
    terms: List[torch.Tensor] = []
    weights = TASK_WEIGHTS if task_weights is None else {**TASK_WEIGHTS, **task_weights}
    enabled = set(enabled_tasks) if enabled_tasks is not None else set(weights)
    for task, weight in weights.items():
        if task not in enabled:
            continue
        if task not in predictions:
            continue
        mask = batch[f"{task}_mask"].bool().reshape(-1)
        if not mask.any():
            continue
        pred = predictions[task].reshape(-1)[mask].float()
        target = batch[task].reshape(-1)[mask].float()
        log_var = predictions[f"{task}_log_variance"].reshape(-1)[mask].float()
        squared = (pred - target).pow(2)
        if task == "toxicity":
            censor = batch["toxicity_censor"].reshape(-1)[mask]
            if toxicity_loss_mode == "one_sided_mse":
                # For an upper-censored observation y <= c, only predictions
                # above c are penalized; the lower-censored case is symmetric.
                per_row = torch.where(censor.lt(0), F.relu(pred - target).pow(2), squared)
                per_row = torch.where(censor.gt(0), F.relu(target - pred).pow(2), per_row)
            elif toxicity_loss_mode == "censored_gaussian_nll":
                sigma = torch.exp(0.5 * log_var).clamp(min=1e-4)
                upper_cdf = torch.special.ndtr((target - pred) / sigma).clamp(min=1e-7)
                lower_cdf = torch.special.ndtr((pred - target) / sigma).clamp(min=1e-7)
                exact = 0.5 * torch.exp(-log_var) * squared + 0.5 * log_var
                per_row = torch.where(censor.lt(0), -upper_cdf.log(), exact)
                per_row = torch.where(censor.gt(0), -lower_cdf.log(), per_row)
            else:
                raise ValueError(f"Unknown toxicity_loss_mode={toxicity_loss_mode!r}")
        else:
            per_row = 0.5 * torch.exp(-log_var) * squared + 0.5 * log_var
        task_loss = per_row.mean()
        if task == "efficiency":
            full_pred = predictions[task].reshape(-1)
            full_target = batch[task].reshape(-1)
            group = batch["group_id"].reshape(-1)
            delta = full_target[:, None] - full_target[None, :]
            if adaptive_rank_margin:
                # Scale the exclusion band to each screen's observed label
                # spread.  This is computed inside a batch and does not use
                # validation/test labels during training.
                group_margin = torch.full_like(full_target, float(rank_margin))
                for group_value in torch.unique(group[mask]):
                    group_rows = mask & group.eq(group_value)
                    if int(group_rows.sum()) >= 2:
                        group_margin[group_rows] = (
                            full_target[group_rows].std(unbiased=False).clamp(min=0.05)
                            * float(rank_margin)
                        )
                pair_margin = group_margin[:, None]
            else:
                pair_margin = torch.full_like(delta, float(rank_margin))
            comparable = (
                mask[:, None] & mask[None, :] & group[:, None].eq(group[None, :])
                & torch.triu(torch.ones_like(delta, dtype=torch.bool), diagonal=1)
                & delta.abs().gt(pair_margin)
            )
            rank_pred = predictions.get("efficiency_rank_score", full_pred).reshape(-1)
            if comparable.any():
                signed = delta.sign() * (rank_pred[:, None] - rank_pred[None, :])
                if pair_reduction == "global":
                    pair_loss = F.softplus(-signed[comparable]).mean()
                elif pair_reduction in {"group_mean", "group_hard"}:
                    group_pair_losses = []
                    for group_value in torch.unique(group[mask]):
                        local_pairs = comparable & group[:, None].eq(group_value)
                        if not local_pairs.any():
                            continue
                        losses = F.softplus(-signed[local_pairs])
                        if pair_reduction == "group_hard":
                            hard_count = max(1, int(math.ceil(0.5 * len(losses))))
                            losses = torch.topk(losses, hard_count).values
                        group_pair_losses.append(losses.mean())
                    pair_loss = torch.stack(group_pair_losses).mean()
                else:
                    raise ValueError(f"Unknown pair_reduction={pair_reduction!r}")
                task_loss = task_loss + ranking_weight * pair_loss

            # Screen-local objectives are evaluated even when a batch has no
            # pair beyond the exclusion margin. This keeps their optimization
            # independent of the pairwise mining rule.
            centered_losses, listwise_losses, group_z_losses = [], [], []
            for group_value in torch.unique(group[mask]):
                local = mask & group.eq(group_value)
                if int(local.sum()) < 3:
                    continue
                local_pred = rank_pred[local]
                local_target = full_target[local]
                pred_centered = local_pred - local_pred.mean()
                target_centered = local_target - local_target.mean()
                if group_center_rank_weight > 0:
                    denominator = (
                        pred_centered.square().sum().sqrt()
                        * target_centered.square().sum().sqrt()
                    ).clamp(min=1e-6)
                    centered_losses.append(
                        1.0 - (pred_centered * target_centered).sum() / denominator
                    )
                if group_z_mse_weight > 0:
                    pred_z = pred_centered / local_pred.std(unbiased=False).clamp(min=1e-4)
                    target_z = target_centered / local_target.std(unbiased=False).clamp(min=1e-4)
                    group_z_losses.append(F.mse_loss(pred_z, target_z))
                if listwise_weight > 0:
                    if listwise_method == "listnet":
                        target_probability = F.softmax(target_centered.detach(), dim=0)
                        listwise_losses.append(
                            -(target_probability * F.log_softmax(local_pred, dim=0)).sum()
                        )
                    elif listwise_method == "listmle":
                        order = torch.argsort(local_target, descending=True)
                        sorted_scores = local_pred[order]
                        log_denominator = torch.stack([
                            torch.logsumexp(sorted_scores[position:], dim=0)
                            for position in range(len(sorted_scores))
                        ])
                        listwise_losses.append(
                            (log_denominator - sorted_scores).mean()
                        )
                    elif listwise_method in {"listce_sigmoid", "listce_softplus"}:
                        # Regression-compatible ListCE: the point-prediction
                        # mean, rather than a free ranking head, defines the
                        # list distribution.  Applying the same monotone,
                        # positive transform to predictions and real-valued
                        # targets keeps the Gaussian regression optimum
                        # compatible with the listwise optimum.
                        local_mean = full_pred[local]
                        if listwise_method == "listce_sigmoid":
                            predicted_relevance = torch.sigmoid(local_mean)
                            target_relevance = torch.sigmoid(local_target.detach())
                        else:
                            predicted_relevance = F.softplus(local_mean)
                            target_relevance = F.softplus(local_target.detach())
                        target_probability = target_relevance / target_relevance.sum().clamp(min=1e-8)
                        predicted_probability = predicted_relevance / predicted_relevance.sum().clamp(min=1e-8)
                        listwise_losses.append(
                            -(target_probability * predicted_probability.clamp(min=1e-8).log()).sum()
                        )
                    elif listwise_method != "none":
                        raise ValueError(f"Unknown listwise_method={listwise_method!r}")
            if centered_losses:
                task_loss = task_loss + group_center_rank_weight * torch.stack(
                    centered_losses
                ).mean()
            if group_z_losses:
                task_loss = task_loss + group_z_mse_weight * torch.stack(
                    group_z_losses
                ).mean()
            if listwise_losses:
                task_loss = task_loss + listwise_weight * torch.stack(
                    listwise_losses
                ).mean()

            # Group-robust penalties operate only on observed efficacy rows.
            group_risks = []
            for group_value in torch.unique(group[mask]):
                local = mask & group.eq(group_value)
                if local.any():
                    group_risks.append(per_row[local[mask]].mean())
            if len(group_risks) >= 2 and robust_weight > 0:
                risks = torch.stack(group_risks)
                if robust_method == "vrex":
                    task_loss = task_loss + robust_weight * risks.var(unbiased=False)
                elif robust_method == "group_cvar":
                    tail = max(1, int(math.ceil(0.5 * len(risks))))
                    task_loss = task_loss + robust_weight * torch.topk(risks, tail).values.mean()
                elif robust_method == "group_dro":
                    task_loss = task_loss + robust_weight * risks.max()
                elif robust_method == "irm":
                    scale = torch.ones((), device=full_pred.device, requires_grad=True)
                    penalties = []
                    full_log_var = predictions["efficiency_log_variance"].reshape(-1)
                    for group_value in torch.unique(group[mask]):
                        local = mask & group.eq(group_value)
                        if int(local.sum()) < 2:
                            continue
                        local_risk = (
                            0.5 * torch.exp(-full_log_var[local])
                            * (scale * full_pred[local] - full_target[local]).pow(2)
                            + 0.5 * full_log_var[local]
                        ).mean()
                        gradient = torch.autograd.grad(
                            local_risk, [scale], create_graph=True
                        )[0]
                        penalties.append(gradient.square())
                    if penalties:
                        task_loss = task_loss + robust_weight * torch.stack(penalties).mean()
        terms.append(weight * task_loss)

    target_mask = batch["target_class_mask"].bool().reshape(-1)
    if target_aux_weight > 0 and target_mask.any():
        terms.append(target_aux_weight * F.cross_entropy(
            predictions["target_logits"][target_mask], batch["target_class"].long()[target_mask]
        ))
    if robust_method == "coral" and robust_weight > 0 and "features" in output:
        features = output["features"]
        efficacy_mask = batch["efficiency_mask"].bool().reshape(-1)
        groups = batch["group_id"].reshape(-1)
        domains = []
        for group_value in torch.unique(groups[efficacy_mask]):
            local = efficacy_mask & groups.eq(group_value)
            if int(local.sum()) >= 2:
                domains.append(features[local])
        penalties = []
        for left, right in zip(domains[:-1], domains[1:]):
            mean_penalty = (left.mean(0) - right.mean(0)).pow(2).mean()
            left_centered = left - left.mean(0, keepdim=True)
            right_centered = right - right.mean(0, keepdim=True)
            left_cov = left_centered.T @ left_centered / max(len(left) - 1, 1)
            right_cov = right_centered.T @ right_centered / max(len(right) - 1, 1)
            covariance_penalty = (left_cov - right_cov).pow(2).mean()
            penalties.append(mean_penalty + covariance_penalty)
        if penalties:
            terms.append(robust_weight * torch.stack(penalties).mean())
    if robust_method == "group_orthogonal" and robust_weight > 0 and "features" in output:
        features = output["features"]
        efficacy_mask = batch["efficiency_mask"].bool().reshape(-1)
        groups = batch["group_id"].reshape(-1)
        if int(efficacy_mask.sum()) >= 3:
            local_features = features[efficacy_mask]
            _, inverse = torch.unique(groups[efficacy_mask], return_inverse=True)
            if int(inverse.max()) >= 1:
                one_hot = F.one_hot(inverse).to(local_features.dtype)
                local_features = local_features - local_features.mean(0, keepdim=True)
                one_hot = one_hot - one_hot.mean(0, keepdim=True)
                cross_covariance = local_features.T @ one_hot / max(len(local_features) - 1, 1)
                terms.append(robust_weight * cross_covariance.square().mean())
    if not terms:
        raise NoSupervisedEndpointError(
            "batch contains no supervised endpoint for the enabled tasks"
        )
    return torch.stack(terms).sum()


class GroupPairBatchSampler(Sampler[List[int]]):
    """Build batches with repeated efficacy groups plus sparse-task examples.

    Uniform row sampling often leaves no comparable within-screen pair in a
    minibatch. This sampler reserves half of each batch for several efficacy
    groups and fills the remainder from sparse multitask rows (or all rows when
    needed). Indices are relative to the training ``Subset`` and no validation
    or test labels are accessed.
    """

    def __init__(
        self, subset: Subset, batch_size: int, seed: int,
        groups_per_batch: int = 4, rows_per_group: int = 4,
        sparse_fill_fraction: float = 1.0,
    ) -> None:
        self.subset = subset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.groups_per_batch = int(groups_per_batch)
        self.rows_per_group = int(rows_per_group)
        self.sparse_fill_fraction = float(np.clip(sparse_fill_fraction, 0.0, 1.0))
        base = subset.dataset
        absolute = np.asarray(subset.indices, dtype=int)
        efficiency = base.df.iloc[absolute]["target_efficiency"].notna().to_numpy()
        self.group_rows: Dict[str, np.ndarray] = {}
        for position, absolute_idx in enumerate(absolute):
            if efficiency[position]:
                self.group_rows.setdefault(base.groups[int(absolute_idx)], []).append(position)
        self.group_rows = {
            group: np.asarray(rows, dtype=int)
            for group, rows in self.group_rows.items() if len(rows) >= 2
        }
        if not self.group_rows:
            raise RuntimeError("No training efficacy group contains at least two rows")
        sparse_columns = [f"target_{task}" for task in TASK_COLUMNS if task != "efficiency"]
        sparse = base.df.iloc[absolute][sparse_columns].notna().any(axis=1).to_numpy()
        self.sparse_rows = np.where(sparse)[0]
        self.all_rows = np.arange(len(subset), dtype=int)
        self.num_batches = max(1, math.ceil(len(subset) / self.batch_size))

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        group_names = np.asarray(list(self.group_rows), dtype=object)
        pair_slots = min(
            self.batch_size,
            self.groups_per_batch * self.rows_per_group,
        )
        for _ in range(self.num_batches):
            chosen_groups = rng.choice(
                group_names, size=self.groups_per_batch,
                replace=len(group_names) < self.groups_per_batch,
            )
            batch: List[int] = []
            for group in chosen_groups:
                rows = self.group_rows[str(group)]
                batch.extend(rng.choice(
                    rows, size=self.rows_per_group,
                    replace=len(rows) < self.rows_per_group,
                ).tolist())
            batch = batch[:pair_slots]
            remaining = self.batch_size - len(batch)
            if remaining:
                sparse_n = (
                    min(remaining, int(round(remaining * self.sparse_fill_fraction)))
                    if len(self.sparse_rows) else 0
                )
                if sparse_n:
                    batch.extend(rng.choice(
                        self.sparse_rows, size=sparse_n, replace=True
                    ).tolist())
                if remaining > sparse_n:
                    batch.extend(rng.choice(
                        self.all_rows, size=remaining - sparse_n, replace=True
                    ).tolist())
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches


def make_train_loader(
    template: DataLoader, seed: int, balanced: bool,
    switches: Dict[str, Any] | None = None,
) -> DataLoader:
    subset = template.dataset
    switches = switches or {}
    if switches.get("group_pair_batches", False):
        if not isinstance(subset, Subset):
            raise TypeError("group-pair batching requires a torch Subset")
        batch_sampler = GroupPairBatchSampler(
            subset, int(template.batch_size), seed,
            groups_per_batch=int(switches.get("groups_per_batch", 4)),
            rows_per_group=int(switches.get("rows_per_group", 4)),
            sparse_fill_fraction=float(switches.get("sparse_fill_fraction", 1.0)),
        )
        return DataLoader(
            subset, batch_sampler=batch_sampler, num_workers=0,
            pin_memory=torch.cuda.is_available(), collate_fn=unified_collate_fn,
        )
    generator = torch.Generator().manual_seed(seed)
    if balanced and isinstance(template.sampler, WeightedRandomSampler):
        sampler = WeightedRandomSampler(
            template.sampler.weights.clone(), len(subset), replacement=True, generator=generator
        )
    else:
        sampler = RandomSampler(subset, replacement=False, generator=generator)
    return DataLoader(
        subset, batch_size=template.batch_size, sampler=sampler, num_workers=0,
        pin_memory=torch.cuda.is_available(), collate_fn=unified_collate_fn,
    )


def safe_corr(fn, x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float("nan")
    return float(fn(x, y).statistic)


def group_bootstrap_rmse(
    pred: np.ndarray, target: np.ndarray, groups: np.ndarray, seed: int, repeats: int = 500,
) -> Tuple[float, float]:
    unique = np.unique(groups)
    if len(unique) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([np.where(groups == group)[0] for group in chosen])
        values.append(float(np.sqrt(np.mean((pred[rows] - target[rows]) ** 2))))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def bootstrap_mean_ci(
    values: Iterable[float], seed: int, repeats: int = 1000,
) -> Tuple[float, float]:
    """Percentile CI obtained by resampling experimental groups."""
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if len(finite) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        means[repeat] = rng.choice(finite, size=len(finite), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def jaccard_max(test_fp: torch.Tensor, train_fp: torch.Tensor, chunk: int = 128) -> np.ndarray:
    # Matrix-form binary Tanimoto avoids materializing a
    # [test_chunk, train, fingerprint_bit] tensor.
    train = train_fp.gt(0).float()
    train_norm = train.sum(dim=1)
    output = []
    for start in range(0, len(test_fp), chunk):
        test = test_fp[start:start + chunk].gt(0).float()
        intersection = test @ train.T
        union = test.sum(dim=1, keepdim=True) + train_norm.unsqueeze(0) - intersection
        output.append((intersection / union.clamp(min=1)).max(dim=1).values)
    return torch.cat(output).cpu().numpy()


def cosine_max(test_features: torch.Tensor, train_features: torch.Tensor, chunk: int = 256) -> np.ndarray:
    """Maximum cosine support after train-only standardization."""
    train = train_features.float()
    test = test_features.float()
    mean = train.mean(dim=0, keepdim=True)
    scale = train.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-4)
    train = (train - mean) / scale
    test = (test - mean) / scale
    train = F.normalize(train, dim=1, eps=1e-8)
    output = []
    for start in range(0, len(test), chunk):
        local = F.normalize(test[start:start + chunk], dim=1, eps=1e-8)
        output.append((local @ train.T).max(dim=1).values)
    return torch.cat(output).cpu().numpy()


def process_context_vector(sample: Dict[str, Any]) -> torch.Tensor:
    """Metadata used only for applicability-domain diagnostics."""
    context = sample["context_features"].clone()
    source_width = len(CONTEXT_VOCABS["source"])
    if source_width:
        context[-source_width:] = 0.0
    return torch.cat([
        sample["molar_ratios"].float() / 100.0,
        sample["formulation_features"].float(),
        sample["target_features"].float(),
        context.float(),
    ])


@torch.no_grad()
def evaluate_model(
    model: LNPPredictor, loader: DataLoader, dataset, train_indices: List[int], seed: int,
    ranking_weight: float = 0.5, target_aux_weight: float = 0.2,
    enabled_tasks: Iterable[str] | None = None,
    rank_margin: float = 0.1, input_switches: Dict[str, Any] | None = None,
    group_center_rank_weight: float = 0.0,
    listwise_method: str = "none", listwise_weight: float = 0.0,
    group_z_mse_weight: float = 0.0,
    pair_reduction: str = "global",
    toxicity_loss_mode: str = "censored_gaussian_nll",
    task_weights: Dict[str, float] | None = None,
    compute_ad: bool = True,
) -> Dict[str, float]:
    model.eval()
    records: Dict[str, Dict[str, List[np.ndarray]]] = {
        task: {key: [] for key in ["pred", "rank", "target", "std", "group", "idx", "censor"]}
        for task in TASK_COLUMNS
    }
    target_true, target_pred = [], []
    losses = []
    for raw in loader:
        batch = to_device(raw)
        output = model(**model_inputs(batch, input_switches))
        try:
            batch_loss = multitask_loss(
                output, batch, ranking_weight=ranking_weight,
                target_aux_weight=target_aux_weight, enabled_tasks=enabled_tasks,
                rank_margin=rank_margin,
                group_center_rank_weight=group_center_rank_weight,
                listwise_method=listwise_method, listwise_weight=listwise_weight,
                group_z_mse_weight=group_z_mse_weight,
                pair_reduction=pair_reduction,
                toxicity_loss_mode=toxicity_loss_mode, task_weights=task_weights,
            )
            losses.append(float(batch_loss.cpu()))
        except NoSupervisedEndpointError:
            pass
        predictions = output["predictions"]
        target_mask = batch["target_class_mask"].bool()
        if target_mask.any():
            target_true.extend(batch["target_class"][target_mask].cpu().tolist())
            target_pred.extend(predictions["target_logits"][target_mask].argmax(-1).cpu().tolist())
        for task in TASK_COLUMNS:
            if task not in predictions:
                continue
            mask = batch[f"{task}_mask"].bool()
            if not mask.any():
                continue
            stats = dataset.task_stats[task]
            pred = predictions[task][mask] * stats["std"] + stats["mean"]
            target = batch[task][mask] * stats["std"] + stats["mean"]
            std = torch.exp(0.5 * predictions[f"{task}_log_variance"][mask]) * stats["std"]
            values = records[task]
            values["pred"].append(pred.cpu().numpy())
            if task == "efficiency" and "efficiency_rank_score" in predictions:
                values["rank"].append(predictions["efficiency_rank_score"][mask].cpu().numpy())
            values["target"].append(target.cpu().numpy())
            values["std"].append(std.cpu().numpy())
            values["group"].append(batch["group_id"][mask].cpu().numpy())
            values["idx"].append(batch["idx"][mask].cpu().numpy())
            if task == "toxicity":
                values["censor"].append(batch["toxicity_censor"][mask].cpu().numpy())

    metrics: Dict[str, float] = {"multitask_loss": float(np.mean(losses))}
    for task, values in records.items():
        if not values["pred"]:
            continue
        pred = np.concatenate(values["pred"])
        target = np.concatenate(values["target"])
        std = np.concatenate(values["std"]).clip(1e-6)
        groups = np.concatenate(values["group"])
        metrics[f"{task}_n"] = int(len(target))
        if task == "toxicity" and values["censor"]:
            censor = np.concatenate(values["censor"])
            upper, lower, exact = censor < 0, censor > 0, censor == 0
            if upper.any():
                metrics["toxicity_upper_violation"] = float(np.mean(pred[upper] > target[upper]))
                metrics["toxicity_upper_excess"] = float(np.mean(np.maximum(pred[upper] - target[upper], 0)))
            if lower.any():
                metrics["toxicity_lower_violation"] = float(np.mean(pred[lower] < target[lower]))
            if not exact.any():
                continue
            pred, target, std, groups = pred[exact], target[exact], std[exact], groups[exact]
        error = pred - target
        metrics[f"{task}_mae"] = float(np.mean(np.abs(error)))
        metrics[f"{task}_rmse"] = float(np.sqrt(np.mean(error ** 2)))
        if len(target) > 1 and np.var(target) > 1e-10:
            metrics[f"{task}_r2"] = float(1 - np.sum(error ** 2) / np.sum((target - target.mean()) ** 2))
        metrics[f"{task}_pearson"] = safe_corr(pearsonr, pred, target)
        metrics[f"{task}_spearman"] = safe_corr(spearmanr, pred, target)
        metrics[f"{task}_nll"] = float(np.mean(0.5 * (error / std) ** 2 + np.log(std)))
        metrics[f"{task}_mean_predicted_std"] = float(np.mean(std))
        calibration_errors = []
        for level, z in [(50, 0.67449), (80, 1.28155), (90, 1.64485), (95, 1.95996)]:
            coverage = float(np.mean(np.abs(error) <= z * std))
            metrics[f"{task}_coverage_{level}"] = coverage
            calibration_errors.append(abs(coverage - level / 100.0))
        metrics[f"{task}_calibration_error"] = float(np.mean(calibration_errors))
        metrics[f"{task}_uncertainty_error_spearman"] = safe_corr(spearmanr, std, np.abs(error))
        low, high = group_bootstrap_rmse(pred, target, groups, seed)
        metrics[f"{task}_rmse_ci_low"] = low
        metrics[f"{task}_rmse_ci_high"] = high

        if task == "efficiency":
            rank_pred = np.concatenate(values["rank"]) if values["rank"] else pred
            comparable_by_margin = {margin: 0 for margin in [0.0, 0.05, 0.1, 0.2]}
            correct_by_margin = {margin: 0 for margin in comparable_by_margin}
            group_spearman = []
            group_ndcg = []
            group_hit_recall = {fraction: [] for fraction in [0.05, 0.10, 0.20]}
            group_regret = {fraction: [] for fraction in [0.05, 0.10, 0.20]}
            group_normalized_regret = {fraction: [] for fraction in [0.05, 0.10, 0.20]}
            group_sizes = []
            for group in np.unique(groups):
                rows = np.where(groups == group)[0]
                group_sizes.append(len(rows))
                if len(rows) >= 3:
                    corr = safe_corr(spearmanr, rank_pred[rows], target[rows])
                    if math.isfinite(corr):
                        group_spearman.append(corr)
                if len(rows) >= 2:
                    local_target = target[rows]
                    local_pred = rank_pred[rows]
                    local_relevance = local_target - local_target.min() + 1e-8
                    group_ndcg.append(float(ndcg_score(
                        local_relevance.reshape(1, -1),
                        local_pred.reshape(1, -1),
                    )))
                    target_range = float(local_target.max() - local_target.min())
                    for fraction in group_hit_recall:
                        local_k = max(1, int(math.ceil(fraction * len(rows))))
                        true_top = set(np.argsort(local_target)[-local_k:])
                        predicted_top = np.argsort(local_pred)[-local_k:]
                        group_hit_recall[fraction].append(
                            len(true_top.intersection(predicted_top)) / local_k
                        )
                        regret = float(
                            local_target.max() - local_target[predicted_top].max()
                        )
                        group_regret[fraction].append(regret)
                        group_normalized_regret[fraction].append(
                            regret / target_range if target_range > 1e-8 else 0.0
                        )
                delta_y = target[rows, None] - target[None, rows]
                delta_p = rank_pred[rows, None] - rank_pred[None, rows]
                for margin in comparable_by_margin:
                    triangle = np.triu(np.ones_like(delta_y, dtype=bool), 1) & (np.abs(delta_y) > margin)
                    comparable_by_margin[margin] += int(triangle.sum())
                    correct_by_margin[margin] += int(((delta_y * delta_p) > 0)[triangle].sum())
            metrics["efficiency_within_group_spearman"] = float(np.mean(group_spearman)) if group_spearman else float("nan")
            spearman_low, spearman_high = bootstrap_mean_ci(group_spearman, seed)
            metrics["efficiency_within_group_spearman_ci_low"] = spearman_low
            metrics["efficiency_within_group_spearman_ci_high"] = spearman_high
            metrics["efficiency_within_group_ndcg"] = (
                float(np.mean(group_ndcg)) if group_ndcg else float("nan")
            )
            ndcg_low, ndcg_high = bootstrap_mean_ci(group_ndcg, seed + 1)
            metrics["efficiency_within_group_ndcg_ci_low"] = ndcg_low
            metrics["efficiency_within_group_ndcg_ci_high"] = ndcg_high
            for fraction in group_hit_recall:
                label = str(int(round(100 * fraction)))
                recall_values = group_hit_recall[fraction]
                regret_values = group_regret[fraction]
                normalized_values = group_normalized_regret[fraction]
                metrics[f"efficiency_within_group_hit_recall_at_{label}pct"] = (
                    float(np.mean(recall_values)) if recall_values else float("nan")
                )
                recall_low, recall_high = bootstrap_mean_ci(
                    recall_values, seed + 10 + int(100 * fraction)
                )
                metrics[f"efficiency_within_group_hit_recall_at_{label}pct_ci_low"] = recall_low
                metrics[f"efficiency_within_group_hit_recall_at_{label}pct_ci_high"] = recall_high
                metrics[f"efficiency_within_group_selection_regret_at_{label}pct"] = (
                    float(np.mean(regret_values)) if regret_values else float("nan")
                )
                metrics[f"efficiency_within_group_normalized_regret_at_{label}pct"] = (
                    float(np.mean(normalized_values)) if normalized_values else float("nan")
                )
                regret_low, regret_high = bootstrap_mean_ci(
                    normalized_values, seed + 20 + int(100 * fraction)
                )
                metrics[f"efficiency_within_group_normalized_regret_at_{label}pct_ci_low"] = regret_low
                metrics[f"efficiency_within_group_normalized_regret_at_{label}pct_ci_high"] = regret_high
            metrics["efficiency_group_count"] = int(len(group_sizes))
            metrics["efficiency_group_count_ge3"] = int(np.sum(np.asarray(group_sizes) >= 3))
            metrics["efficiency_group_size_min"] = int(np.min(group_sizes))
            metrics["efficiency_group_size_median"] = float(np.median(group_sizes))
            metrics["efficiency_group_size_max"] = int(np.max(group_sizes))
            for margin in comparable_by_margin:
                label = str(margin).replace(".", "p")
                count = comparable_by_margin[margin]
                metrics[f"efficiency_pair_count_margin_{label}"] = count
                metrics[f"efficiency_pair_accuracy_margin_{label}"] = (
                    correct_by_margin[margin] / count if count else float("nan")
                )
            metrics["efficiency_pair_accuracy"] = metrics["efficiency_pair_accuracy_margin_0p1"]
            k = max(1, int(math.ceil(0.1 * len(target))))
            true_hits = set(np.argsort(target)[-k:])
            predicted_hits = set(np.argsort(rank_pred)[-k:])
            recall = len(true_hits & predicted_hits) / k
            metrics["efficiency_global_hit_recall_at_10pct"] = recall
            metrics["efficiency_global_enrichment_at_10pct"] = recall / (k / len(target))
            # sklearn NDCG requires non-negative relevance. Translation keeps
            # the ordering unchanged for screen-normalized z-scores.
            relevance = target - target.min() + 1e-8
            metrics["efficiency_global_ndcg"] = float(ndcg_score(
                relevance.reshape(1, -1), rank_pred.reshape(1, -1)
            ))

            if not compute_ad:
                continue
            idx = np.concatenate(values["idx"]).astype(int)
            test_samples = [dataset[i] for i in idx]
            train_samples = [dataset[i] for i in train_indices]
            role_similarities = []
            for role_index, role in enumerate(["ionizable", "helper", "cholesterol", "peg"]):
                test_observed = np.asarray([
                    bool(sample["component_structure_mask"][role_index]) for sample in test_samples
                ])
                train_observed = np.asarray([
                    bool(sample["component_structure_mask"][role_index]) for sample in train_samples
                ])
                role_similarity = np.full(len(test_samples), np.nan, dtype=float)
                if test_observed.any() and train_observed.any():
                    test_fp = torch.stack([
                        test_samples[position][f"{role}_fingerprint"]
                        for position in np.where(test_observed)[0]
                    ])
                    train_fp = torch.stack([
                        train_samples[position][f"{role}_fingerprint"]
                        for position in np.where(train_observed)[0]
                    ])
                    role_similarity[test_observed] = jaccard_max(test_fp, train_fp)
                role_similarities.append(role_similarity)
                metrics[f"efficiency_ad_{role}_coverage"] = float(test_observed.mean())
                if test_observed.any() and np.isfinite(role_similarity[test_observed]).any():
                    metrics[f"efficiency_ad_{role}_mean_max_tanimoto"] = float(
                        np.nanmean(role_similarity)
                    )
            role_matrix = np.stack(role_similarities, axis=1)
            role_count = np.sum(np.isfinite(role_matrix), axis=1)
            multi_role_similarity = np.divide(
                np.nansum(role_matrix, axis=1), role_count,
                out=np.zeros(len(role_matrix), dtype=float), where=role_count > 0,
            )
            test_process = torch.stack([process_context_vector(sample) for sample in test_samples])
            train_process = torch.stack([process_context_vector(sample) for sample in train_samples])
            process_similarity = (cosine_max(test_process, train_process) + 1.0) / 2.0
            joint_similarity = 0.5 * multi_role_similarity + 0.5 * process_similarity
            # Preserve the legacy ionizable-only key for backwards-compatible
            # tables while exposing stronger formulation-aware diagnostics.
            similarity = role_similarities[0]
            metrics["efficiency_mean_max_tanimoto"] = float(np.nanmean(similarity))
            metrics["efficiency_similarity_error_spearman"] = safe_corr(
                spearmanr, np.nan_to_num(similarity), -np.abs(error)
            )
            metrics["efficiency_ad_multi_role_mean"] = float(np.mean(multi_role_similarity))
            metrics["efficiency_ad_process_context_mean"] = float(np.mean(process_similarity))
            metrics["efficiency_ad_joint_mean"] = float(np.mean(joint_similarity))
            metrics["efficiency_ad_joint_error_spearman"] = safe_corr(
                spearmanr, joint_similarity, -np.abs(error)
            )
            # Support-based abstention diagnostic. Rows with the highest joint
            # support are retained first; a useful applicability score should
            # make RMSE decrease as coverage is reduced. This is diagnostic,
            # because no threshold is calibrated on held-out labels.
            support_order = np.argsort(joint_similarity)[::-1]
            risk_coverages, risk_values = [], []
            for coverage in np.linspace(0.1, 1.0, 10):
                retain_n = max(1, int(math.ceil(coverage * len(error))))
                retained = support_order[:retain_n]
                risk = float(np.sqrt(np.mean(error[retained] ** 2)))
                label = str(int(round(100 * coverage)))
                metrics[f"efficiency_ad_support_retain_{label}pct_rmse"] = risk
                risk_coverages.append(float(coverage))
                risk_values.append(risk)
            metrics["efficiency_ad_support_risk_coverage_auc"] = float(
                np.trapezoid(risk_values, risk_coverages)
            )
            metrics["efficiency_ad_support_abstention_gain_20pct"] = float(
                metrics["efficiency_rmse"]
                - metrics["efficiency_ad_support_retain_80pct_rmse"]
            )
            conditional_group_rmse = []
            conditional_group_support = []
            for group_value in np.unique(groups):
                rows = groups == group_value
                conditional_group_rmse.append(float(np.sqrt(np.mean(error[rows] ** 2))))
                conditional_group_support.append(float(np.mean(joint_similarity[rows])))
            metrics["efficiency_ad_group_support_error_spearman"] = safe_corr(
                spearmanr,
                np.asarray(conditional_group_support),
                -np.asarray(conditional_group_rmse),
            )
            for label, lower, upper in [
                ("joint_lt_0p5", -np.inf, 0.5),
                ("joint_0p5_0p7", 0.5, 0.7),
                ("joint_0p7_0p85", 0.7, 0.85),
                ("joint_ge_0p85", 0.85, np.inf),
            ]:
                rows = (joint_similarity >= lower) & (joint_similarity < upper)
                if rows.any():
                    metrics[f"efficiency_ad_{label}_n"] = int(rows.sum())
                    metrics[f"efficiency_ad_{label}_rmse"] = float(
                        np.sqrt(np.mean(error[rows] ** 2))
                    )
            for label, lower, upper in [
                ("sim_lt_0p3", -np.inf, 0.3),
                ("sim_0p3_0p5", 0.3, 0.5),
                ("sim_0p5_0p7", 0.5, 0.7),
                ("sim_ge_0p7", 0.7, np.inf),
            ]:
                rows = (similarity >= lower) & (similarity < upper)
                if rows.any():
                    metrics[f"efficiency_{label}_n"] = int(rows.sum())
                    metrics[f"efficiency_{label}_rmse"] = float(np.sqrt(np.mean(error[rows] ** 2)))
                    metrics[f"efficiency_{label}_coverage_90"] = float(
                        np.mean(np.abs(error[rows]) <= 1.64485 * std[rows])
                    )
            structure_complete = np.array([
                bool(dataset[i]["component_structure_mask"].bool().all()) for i in idx
            ])
            for label, rows in [("complete4", structure_complete), ("incomplete", ~structure_complete)]:
                if rows.any():
                    metrics[f"efficiency_{label}_n"] = int(rows.sum())
                    metrics[f"efficiency_{label}_rmse"] = float(np.sqrt(np.mean(error[rows] ** 2)))
            observed_count = np.array([
                int(dataset[i]["component_structure_mask"].bool().sum()) for i in idx
            ])
            for count in range(1, 5):
                rows = observed_count == count
                if rows.any():
                    metrics[f"efficiency_observed_components_{count}_n"] = int(rows.sum())
                    metrics[f"efficiency_observed_components_{count}_rmse"] = float(
                        np.sqrt(np.mean(error[rows] ** 2))
                    )

    if target_true:
        metrics["target_n"] = len(target_true)
        target_true_array = np.asarray(target_true)
        target_pred_array = np.asarray(target_pred)
        metrics["target_accuracy"] = float(
            np.mean(target_true_array == target_pred_array)
        )
        target_classes = np.unique(target_true_array)
        if len(target_classes) >= 2:
            # Average recall over classes represented in this held-out split.
            # sklearn's balanced_accuracy_score emits a warning when the model
            # predicts a training class absent from y_true; the manual form is
            # algebraically identical for represented classes and explicit
            # about the intended support.
            recalls = [
                float(np.mean(
                    target_pred_array[target_true_array == label] == label
                ))
                for label in target_classes
            ]
            metrics["target_balanced_accuracy"] = float(np.mean(recalls))
            metrics["target_macro_f1"] = float(
                f1_score(
                    target_true_array, target_pred_array,
                    average="macro", zero_division=0,
                )
            )
    return metrics


@torch.no_grad()
def collect_efficiency_predictions(
    model: LNPPredictor, loader: DataLoader, dataset,
    input_switches: Dict[str, Any] | None = None,
) -> Dict[str, List[Any]]:
    """Collect aligned held-out predictions for ensembles and conformal audit."""
    model.eval()
    rows: List[Dict[str, Any]] = []
    stats = dataset.task_stats["efficiency"]
    for raw in loader:
        batch = to_device(raw)
        output = model(**model_inputs(batch, input_switches))["predictions"]
        mask = batch["efficiency_mask"].bool()
        if not mask.any():
            continue
        idx = batch["idx"][mask].cpu().numpy().astype(int)
        pred = (output["efficiency"][mask] * stats["std"] + stats["mean"]).cpu().numpy()
        target = (batch["efficiency"][mask] * stats["std"] + stats["mean"]).cpu().numpy()
        std = (torch.exp(0.5 * output["efficiency_log_variance"][mask]) * stats["std"]).cpu().numpy()
        rank = output.get("efficiency_rank_score", output["efficiency"])[mask].cpu().numpy()
        for i, mu, y, sigma, score in zip(idx, pred, target, std, rank):
            rows.append({
                "idx": int(i), "prediction": float(mu), "target": float(y),
                "std": float(max(sigma, 1e-6)), "rank_score": float(score),
                "group": dataset.groups[int(i)],
            })
    rows.sort(key=lambda item: item["idx"])
    keys = ["idx", "prediction", "target", "std", "rank_score", "group"]
    return {key: [row[key] for row in rows] for key in keys}


def run_one(
    name: str, switches: Dict[str, Any], base_model: Dict[str, Any], train_template: DataLoader,
    val_loader: DataLoader, test_loader: DataLoader, dataset, seed: int, epochs: int,
    balanced: bool, learning_rate: float, patience: int, compute_final_ad: bool = True,
) -> Dict[str, Any]:
    seed_everything(seed)
    model = build_model(base_model, switches)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(switches.get("optimizer_learning_rate", learning_rate)),
        weight_decay=float(switches.get("optimizer_weight_decay", 1e-4)),
    )
    scaler = GradScaler("cuda", enabled=DEVICE.type == "cuda")
    train_loader = make_train_loader(train_template, seed, balanced, switches)
    best_score, best_state, history = float("inf"), None, []
    best_val_multitask_loss = float("inf")
    checkpoint_selection = str(
        switches.get("checkpoint_selection", "multitask_loss")
    )
    ranking_weight = float(switches.get("loss_ranking_weight", 0.5))
    rank_margin = float(switches.get("rank_margin", 0.1))
    robust_method = str(switches.get("robust_method", "none"))
    robust_weight = float(switches.get("robust_weight", 0.0))
    adaptive_rank_margin = bool(switches.get("adaptive_rank_margin", False))
    group_center_rank_weight = float(switches.get("group_center_rank_weight", 0.0))
    listwise_method = str(switches.get("listwise_method", "none"))
    listwise_weight = float(switches.get("listwise_weight", 0.0))
    group_z_mse_weight = float(switches.get("group_z_mse_weight", 0.0))
    pair_reduction = str(switches.get("pair_reduction", "global"))
    toxicity_loss_mode = str(switches.get(
        "toxicity_loss_mode", "censored_gaussian_nll"
    ))
    task_weights = switches.get("task_weights")
    target_aux_weight = float(switches.get("loss_target_aux_weight", 0.2))
    enabled_tasks = switches.get("enabled_tasks")
    epochs_without_improvement = 0
    total_attempted_batches = 0
    total_optimizer_steps = 0
    total_row_exposures = 0
    total_efficiency_label_exposures = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        epoch_attempted_batches = 0
        epoch_optimizer_steps = 0
        epoch_row_exposures = 0
        epoch_efficiency_label_exposures = 0
        for raw in train_loader:
            epoch_attempted_batches += 1
            total_attempted_batches += 1
            batch = to_device(raw)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=DEVICE.type == "cuda"):
                output = model(
                    **model_inputs(batch, switches, training=True),
                    return_features=robust_method in {"coral", "group_orthogonal"},
                )
                try:
                    loss = multitask_loss(
                        output, batch, ranking_weight=ranking_weight,
                        target_aux_weight=target_aux_weight,
                        enabled_tasks=enabled_tasks, rank_margin=rank_margin,
                        robust_method=robust_method, robust_weight=robust_weight,
                        adaptive_rank_margin=adaptive_rank_margin,
                        group_center_rank_weight=group_center_rank_weight,
                        listwise_method=listwise_method,
                        listwise_weight=listwise_weight,
                        group_z_mse_weight=group_z_mse_weight,
                        pair_reduction=pair_reduction,
                        toxicity_loss_mode=toxicity_loss_mode,
                        task_weights=task_weights,
                    )
                except NoSupervisedEndpointError:
                    continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
            rows_seen = int(batch["idx"].numel())
            efficiency_seen = int(batch["efficiency_mask"].bool().sum().item())
            epoch_optimizer_steps += 1
            epoch_row_exposures += rows_seen
            epoch_efficiency_label_exposures += efficiency_seen
            total_optimizer_steps += 1
            total_row_exposures += rows_seen
            total_efficiency_label_exposures += efficiency_seen
        val_metrics = evaluate_model(
            model, val_loader, dataset, train_template.dataset.indices, seed,
            ranking_weight=ranking_weight, target_aux_weight=target_aux_weight,
            enabled_tasks=enabled_tasks, rank_margin=rank_margin, input_switches=switches,
            group_center_rank_weight=group_center_rank_weight,
            listwise_method=listwise_method, listwise_weight=listwise_weight,
            group_z_mse_weight=group_z_mse_weight,
            pair_reduction=pair_reduction,
            toxicity_loss_mode=toxicity_loss_mode, task_weights=task_weights,
            compute_ad=False,
        )
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "attempted_batches": epoch_attempted_batches,
            "optimizer_steps": epoch_optimizer_steps,
            "row_exposures": epoch_row_exposures,
            "efficiency_label_exposures": epoch_efficiency_label_exposures,
            **val_metrics,
        })
        if checkpoint_selection in {"efficacy_pareto", "decision_pareto"}:
            def bounded_metric(key: str) -> float:
                value = float(val_metrics.get(key, 0.0))
                return value if math.isfinite(value) else 0.0

            if checkpoint_selection == "efficacy_pareto":
                # Legacy scalarization retained so historical runs remain
                # reproducible. The last term is explicitly global and is not
                # used by the corrected decision protocol below.
                selection_score = (
                    float(val_metrics.get("efficiency_rmse", float("inf")))
                    - 0.10 * bounded_metric("efficiency_spearman")
                    - 0.10 * bounded_metric("efficiency_within_group_spearman")
                    - 0.05 * bounded_metric("efficiency_pair_accuracy")
                    - 0.05 * bounded_metric("efficiency_global_ndcg")
                )
            else:
                # Validation-only screening scalarization. All ranking and
                # selection metrics are computed within an experimental
                # screen; lower normalized regret is preferred.
                selection_score = (
                    float(val_metrics.get("efficiency_rmse", float("inf")))
                    - 0.10 * bounded_metric("efficiency_within_group_spearman")
                    - 0.05 * bounded_metric("efficiency_pair_accuracy")
                    - 0.05 * bounded_metric("efficiency_within_group_ndcg")
                    - 0.05 * bounded_metric(
                        "efficiency_within_group_hit_recall_at_10pct"
                    )
                    + 0.05 * bounded_metric(
                        "efficiency_within_group_normalized_regret_at_10pct"
                    )
                )
        else:
            selection_score = float(val_metrics["multitask_loss"])
        history[-1]["checkpoint_selection_score"] = selection_score
        if selection_score < best_score:
            best_score = selection_score
            best_val_multitask_loss = float(val_metrics["multitask_loss"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if patience > 0 and epochs_without_improvement >= patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(DEVICE)
    val_metrics = evaluate_model(
        model, val_loader, dataset, train_template.dataset.indices, seed,
        ranking_weight=ranking_weight, target_aux_weight=target_aux_weight,
        enabled_tasks=enabled_tasks, rank_margin=rank_margin, input_switches=switches,
        group_center_rank_weight=group_center_rank_weight,
        listwise_method=listwise_method, listwise_weight=listwise_weight,
        group_z_mse_weight=group_z_mse_weight,
        pair_reduction=pair_reduction,
        toxicity_loss_mode=toxicity_loss_mode, task_weights=task_weights,
        compute_ad=compute_final_ad,
    )
    test_metrics = evaluate_model(
        model, test_loader, dataset, train_template.dataset.indices, seed,
        ranking_weight=ranking_weight, target_aux_weight=target_aux_weight,
        enabled_tasks=enabled_tasks, rank_margin=rank_margin, input_switches=switches,
        group_center_rank_weight=group_center_rank_weight,
        listwise_method=listwise_method, listwise_weight=listwise_weight,
        group_z_mse_weight=group_z_mse_weight,
        pair_reduction=pair_reduction,
        toxicity_loss_mode=toxicity_loss_mode, task_weights=task_weights,
        compute_ad=compute_final_ad,
    )
    return {
        "name": name, "seed": seed, "switches": switches, "balanced_sampling": balanced,
        "best_val_loss": best_val_multitask_loss,
        "best_selection_score": best_score,
        "checkpoint_selection": checkpoint_selection,
        "history": history,
        "val": val_metrics, "test": test_metrics,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "epochs_trained": len(history),
        "training_accounting": {
            "attempted_batches": total_attempted_batches,
            "optimizer_steps": total_optimizer_steps,
            "row_exposures": total_row_exposures,
            "efficiency_label_exposures": total_efficiency_label_exposures,
        },
        "predictions": {
            "val": collect_efficiency_predictions(model, val_loader, dataset, switches),
            "test": collect_efficiency_predictions(model, test_loader, dataset, switches),
        },
    }


def aggregate(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {"name": result["name"], "seed": result["seed"], "parameters": result["parameters"]}
        row.update({f"val_{key}": value for key, value in result["val"].items()})
        row.update({f"test_{key}": value for key, value in result["test"].items()})
        rows.append(row)
    raw = pd.DataFrame(rows)
    numeric = [column for column in raw.columns if column not in {"name", "seed"}]
    grouped = raw.groupby("name", sort=False)[numeric].agg(["mean", "std"])
    grouped.columns = [f"{column}_{stat}" for column, stat in grouped.columns]
    return raw, grouped.reset_index()


def update_ablation_markdown(
    summary: pd.DataFrame,
    results: List[Dict[str, Any]],
    output_dir: Path,
    metadata: Dict[str, Any],
) -> None:
    path = ROOT / "ablation.md"
    marker = "<!-- AUTO_ABLATION_RESULTS -->"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# DeepLNP 消融实验记录\n\n"
    prefix = existing.split(marker)[0].rstrip() + "\n\n"
    primary = [
        "name", "parameters_mean", "val_multitask_loss_mean", "val_efficiency_rmse_mean",
        "val_efficiency_spearman_mean", "val_efficiency_pair_accuracy_mean",
        "test_efficiency_rmse_mean", "test_efficiency_spearman_mean",
        "test_efficiency_within_group_spearman_mean",
        "test_efficiency_within_group_ndcg_mean",
        "test_efficiency_within_group_hit_recall_at_10pct_mean",
        "test_efficiency_within_group_normalized_regret_at_10pct_mean",
        "test_efficiency_calibration_error_mean",
        "test_toxicity_upper_violation_mean",
    ]
    available = [column for column in primary if column in summary]
    table = summary[available].copy()
    for column in table.columns[1:]:
        table[column] = table[column].map(lambda value: "NA" if pd.isna(value) else f"{value:.4f}")
    lines = [
        marker,
        f"## 自动消融结果（{datetime.now().isoformat(timespec='seconds')}）",
        "",
        f"设备：`{DEVICE}`；每个配置种子数：`{len(set(r['seed'] for r in results))}`；原始结果：`{output_dir.relative_to(ROOT)}`。",
        "",
        table.to_markdown(index=False),
        "",
        "### 自动判读",
        "",
    ]
    lines.insert(
        4,
        "- Run protocol: "
        f"rows={metadata['dataset_rows']}, sample_fraction={metadata['sample_fraction']}, "
        f"epochs={metadata['epochs']}, split={metadata['split_strategy']}, "
        f"ETKDG_conformers={metadata['use_conformers']}.",
    )
    selection = (
        summary["val_efficiency_rmse_mean"].astype(float)
        - 0.10 * summary.get("val_efficiency_spearman_mean", 0.0).fillna(0.0)
        - 0.10 * summary.get(
            "val_efficiency_within_group_spearman_mean", 0.0
        ).fillna(0.0)
        - 0.05 * summary.get(
            "val_efficiency_pair_accuracy_mean", 0.0
        ).fillna(0.0)
        - 0.05 * summary.get(
            "val_efficiency_within_group_ndcg_mean", 0.0
        ).fillna(0.0)
        - 0.05 * summary.get(
            "val_efficiency_within_group_hit_recall_at_10pct_mean", 0.0
        ).fillna(0.0)
        + 0.05 * summary.get(
            "val_efficiency_within_group_normalized_regret_at_10pct_mean", 0.0
        ).fillna(0.0)
    )
    best_composite = summary.loc[selection.idxmin(), "name"]
    lines.append(
        f"- 按预设验证集点误差/筛选复合准则选择：`{best_composite}`。"
        "不同目标函数的原始 loss 尺度不可横向比较；测试集不用于选择模型。"
    )
    if "val_efficiency_spearman_mean" in summary:
        finite = summary[np.isfinite(summary["val_efficiency_spearman_mean"])]
        if len(finite):
            best_rank = finite.loc[finite["val_efficiency_spearman_mean"].idxmax(), "name"]
            lines.append(f"- 按验证集效力 Spearman 排序最佳：`{best_rank}`。")
    lines.extend([
        "- 单个稀疏终点少于 20 个封存测试标签时，只报告描述性结果，不据此宣称模块优劣。",
        "- 模块结论必须同时参考均值、种子间标准差、验证指标和适用域分层；测试集仅做一次诊断性核验。",
        "",
    ])
    path.write_text(prefix + "\n".join(lines), encoding="utf-8")


def register_ablation_round(
    output_dir: Path, summary: pd.DataFrame, metadata: Dict[str, Any],
) -> None:
    """Guarantee that every completed result directory enters the master ledger.

    Human-authored decisions may subsequently refine this generated entry, but
    a successful run can no longer be omitted from ablation.md by accident.
    """
    manifest_path = ROOT / "ablation_manifest.yaml"
    manifest = (
        yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        if manifest_path.exists()
        else {}
    )
    manifest.setdefault("rounds", [])
    directory = output_dir.name
    if any(item.get("directory") == directory for item in manifest.get("rounds", [])):
        return
    round_numbers = [int(str(item.get("id", "R0"))[1:]) for item in manifest.get("rounds", [])]
    round_id = f"R{max(round_numbers, default=0) + 1:02d}"
    selection = (
        summary["val_efficiency_rmse_mean"].astype(float)
        - 0.10 * summary.get("val_efficiency_spearman_mean", 0.0).fillna(0.0)
        - 0.10 * summary.get(
            "val_efficiency_within_group_spearman_mean", 0.0
        ).fillna(0.0)
        - 0.05 * summary.get(
            "val_efficiency_pair_accuracy_mean", 0.0
        ).fillna(0.0)
        - 0.05 * summary.get(
            "val_efficiency_within_group_ndcg_mean", 0.0
        ).fillna(0.0)
        - 0.05 * summary.get(
            "val_efficiency_within_group_hit_recall_at_10pct_mean", 0.0
        ).fillna(0.0)
        + 0.05 * summary.get(
            "val_efficiency_within_group_normalized_regret_at_10pct_mean", 0.0
        ).fillna(0.0)
    )
    best_composite = summary.loc[selection.idxmin(), "name"]
    finite_rank = summary[np.isfinite(summary.get("val_efficiency_spearman_mean", np.nan))]
    best_rank = (
        finite_rank.loc[finite_rank["val_efficiency_spearman_mean"].idxmax(), "name"]
        if len(finite_rank) else "NA"
    )
    smoke_only = (
        float(metadata.get("sample_fraction", 1.0)) < 0.05
        or int(metadata.get("epochs", 0)) <= 1
        or len(metadata.get("seeds", [])) < 1
    )
    status = "smoke-only" if smoke_only else "candidate"
    decisions = {}
    for _, row in summary.iterrows():
        decisions[row["name"]] = (
            "自动记录：val loss=" + fmt_metric(row.get("val_multitask_loss_mean"))
            + ", RMSE=" + fmt_metric(row.get("val_efficiency_rmse_mean"))
            + ", Spearman=" + fmt_metric(row.get("val_efficiency_spearman_mean"))
            + ", pair accuracy=" + fmt_metric(row.get("val_efficiency_pair_accuracy_mean"))
            + "；待结合多 split 证据复核。"
        )
    manifest.setdefault("rounds", []).append({
        "id": round_id,
        "directory": directory,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "stage": "自动登记消融轮次",
        "objective": "比较本轮命令行请求的模型配置，并防止完成结果遗漏出总账。",
        "protocol": (
            f"{metadata['dataset_rows']} rows; {metadata['split_strategy']} split "
            f"{metadata['train_rows']}/{metadata['val_rows']}/{metadata['test_rows']}; "
            f"up to {metadata['epochs']} epochs; seeds {metadata['seeds']}; "
            f"ETKDG {'enabled' if metadata['use_conformers'] else 'disabled'}; "
            f"lr={metadata['learning_rate']}; patience={metadata['early_stopping_patience']}."
        ),
        "baseline": str(summary.iloc[0]["name"]),
        "verdict": (
            "仅用于检查代码路径和数值稳定性；样本或训练轮数不足，不进行科学选型。"
            if smoke_only else
            f"自动初判：预设验证点误差/筛选复合准则最佳为 {best_composite}；"
            f"验证效率 Spearman 最佳为 {best_rank}。不同目标函数的原始 loss "
            f"尺度不可横向比较；测试集不参与该选择。"
        ),
        "status": status,
        "decisions": decisions,
    })
    manifest_path.write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )


def append_standard_ablation_record(
    summary: pd.DataFrame, output_dir: Path, metadata: Dict[str, Any],
) -> None:
    """Append one schema-stable record without rewriting experiment history."""
    path = ROOT / "ablation.md"
    output_dir = output_dir.resolve()
    relative = output_dir.relative_to(ROOT).as_posix()
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if f"`{relative}`" in existing:
        return
    columns = [
        "name",
        "val_efficiency_rmse_mean",
        "val_efficiency_within_group_spearman_mean",
        "val_efficiency_within_group_ndcg_mean",
        "val_efficiency_within_group_hit_recall_at_10pct_mean",
        "val_efficiency_within_group_normalized_regret_at_10pct_mean",
        "test_efficiency_rmse_mean",
        "test_efficiency_within_group_spearman_mean",
        "test_efficiency_within_group_ndcg_mean",
        "test_efficiency_within_group_hit_recall_at_10pct_mean",
        "test_efficiency_within_group_normalized_regret_at_10pct_mean",
    ]
    table = summary[[column for column in columns if column in summary]].copy()
    table = table.rename(columns={
        "name": "Variant",
        "val_efficiency_rmse_mean": "Val RMSE",
        "val_efficiency_within_group_spearman_mean": "Val group rho",
        "val_efficiency_within_group_ndcg_mean": "Val screen NDCG",
        "val_efficiency_within_group_hit_recall_at_10pct_mean": "Val Hit@10%",
        "val_efficiency_within_group_normalized_regret_at_10pct_mean": "Val NReg@10%",
        "test_efficiency_rmse_mean": "Test RMSE (diag.)",
        "test_efficiency_within_group_spearman_mean": "Test group rho (diag.)",
        "test_efficiency_within_group_ndcg_mean": "Test screen NDCG (diag.)",
        "test_efficiency_within_group_hit_recall_at_10pct_mean": "Test Hit@10% (diag.)",
        "test_efficiency_within_group_normalized_regret_at_10pct_mean": "Test NReg@10% (diag.)",
    })
    for column in table.columns[1:]:
        table[column] = table[column].map(
            lambda value: "NA" if pd.isna(value) else f"{float(value):.4f}"
        )
    decision = (
        summary["val_efficiency_rmse_mean"].astype(float)
        - 0.10 * summary.get(
            "val_efficiency_within_group_spearman_mean", 0.0
        ).fillna(0.0)
        - 0.05 * summary.get(
            "val_efficiency_pair_accuracy_mean", 0.0
        ).fillna(0.0)
        - 0.05 * summary.get(
            "val_efficiency_within_group_ndcg_mean", 0.0
        ).fillna(0.0)
        - 0.05 * summary.get(
            "val_efficiency_within_group_hit_recall_at_10pct_mean", 0.0
        ).fillna(0.0)
        + 0.05 * summary.get(
            "val_efficiency_within_group_normalized_regret_at_10pct_mean", 0.0
        ).fillna(0.0)
    )
    best = str(summary.loc[decision.idxmin(), "name"])
    lines = [
        "",
        f"## Ablation v2 — {output_dir.name}",
        "",
        "- Purpose: reviewer-driven model and component evaluation.",
        f"- Artifact: `{relative}`.",
        (
            f"- Protocol: rows={metadata['dataset_rows']}; "
            f"split={metadata['split_strategy']}; split seed={metadata['split_seed']}; "
            f"epochs≤{metadata['epochs']}; train seeds={metadata['seeds']}; "
            f"device={metadata['device']}."
        ),
        "- Selection policy: validation groups only; test columns are sealed diagnostics.",
        "- Ranking scope: NDCG, Hit@10%, and normalized regret are averaged over experimental screens.",
        f"- Validation decision: `{best}` has the lowest prespecified decision score in this round.",
        "",
        table.to_markdown(index=False),
        "",
        "- Limitation: a single split or smoke subset cannot promote a module; confirmation requires repeated disjoint groups.",
        "",
    ]
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def fmt_metric(value: Any) -> str:
    return "NA" if value is None or pd.isna(value) else f"{float(value):.4f}"


def _aligned_ensemble(runs: List[Dict[str, Any]], split: str) -> Dict[str, np.ndarray]:
    records = [run["predictions"][split] for run in runs]
    reference = np.asarray(records[0]["idx"], dtype=int)
    for record in records[1:]:
        if not np.array_equal(reference, np.asarray(record["idx"], dtype=int)):
            raise RuntimeError(f"unaligned {split} predictions across ensemble members")
    means = np.stack([np.asarray(record["prediction"], dtype=float) for record in records])
    variances = np.stack([np.asarray(record["std"], dtype=float) ** 2 for record in records])
    mean = means.mean(0)
    total_variance = np.mean(variances + means ** 2, axis=0) - mean ** 2
    return {
        "idx": reference,
        "target": np.asarray(records[0]["target"], dtype=float),
        "prediction": mean,
        "std": np.sqrt(np.maximum(total_variance, 1e-12)),
        "rank_score": np.stack([np.asarray(record["rank_score"], dtype=float) for record in records]).mean(0),
        "group": np.asarray(records[0]["group"], dtype=object),
    }


def _ensemble_point_metrics(record: Dict[str, np.ndarray]) -> Dict[str, float]:
    target, pred, std = record["target"], record["prediction"], record["std"]
    error = pred - target
    denom = np.sum((target - target.mean()) ** 2)
    result = {
        "n": int(len(target)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1 - np.sum(error ** 2) / denom) if denom > 1e-10 else float("nan"),
        "pearson": safe_corr(pearsonr, pred, target),
        "spearman": safe_corr(spearmanr, pred, target),
        "uncertainty_error_spearman": safe_corr(spearmanr, std, np.abs(error)),
        "nll": float(np.mean(0.5 * (error / std) ** 2 + np.log(std))),
    }
    return result


def ensemble_and_conformal_tables(results: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensemble_rows, conformal_rows, group_rows = [], [], []
    for name in dict.fromkeys(result["name"] for result in results):
        members = [result for result in results if result["name"] == name]
        if len(members) < 2:
            continue
        val = _aligned_ensemble(members, "val")
        test = _aligned_ensemble(members, "test")
        row = {"name": name, "members": len(members)}
        row.update({f"val_{key}": value for key, value in _ensemble_point_metrics(val).items()})
        row.update({f"test_{key}": value for key, value in _ensemble_point_metrics(test).items()})
        ensemble_rows.append(row)

        calibration_scores = np.abs(val["target"] - val["prediction"]) / val["std"]
        n_cal = len(calibration_scores)
        for nominal in [0.5, 0.8, 0.9, 0.95]:
            quantile_level = min(math.ceil((n_cal + 1) * nominal) / n_cal, 1.0)
            quantile = float(np.quantile(calibration_scores, quantile_level, method="higher"))
            for split_name, record in [("validation", val), ("test", test)]:
                half_width = quantile * record["std"]
                covered = np.abs(record["target"] - record["prediction"]) <= half_width
                conformal_rows.append({
                    "name": name, "split": split_name, "nominal": nominal,
                    "calibration_n": n_cal, "quantile": quantile,
                    "coverage": float(np.mean(covered)),
                    "mean_interval_width": float(np.mean(2 * half_width)),
                })

        for split_name, record in [("validation", val), ("test", test)]:
            group_names = sorted(set(record["group"].tolist()))
            for group in group_names:
                keep = record["group"] == group
                if int(keep.sum()) < 2:
                    continue
                group_rows.append({
                    "name": name, "split": split_name, "group": group,
                    "n": int(keep.sum()),
                    "spearman": safe_corr(spearmanr, record["rank_score"][keep], record["target"][keep]),
                    "ndcg": float(ndcg_score(
                        (record["target"][keep] - record["target"][keep].min() + 1e-8)[None, :],
                        record["rank_score"][keep][None, :],
                    )),
                })
    return pd.DataFrame(ensemble_rows), pd.DataFrame(conformal_rows), pd.DataFrame(group_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--sample-fraction", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Seed used only to construct train/validation/test groups.")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--only", nargs="+", default=None,
                        help="Run only named ablations (for iterative experiments).")
    parser.add_argument("--no-conformers", action="store_true",
                        help="Use deterministic 2D descriptors and skip ETKDG prewarming.")
    parser.add_argument(
        "--split-strategy",
        choices=["group", "group_kfold", "balanced_group_kfold", "random"],
        default="group",
                        help="Group is the primary leakage-safe protocol; random is diagnostic only.")
    parser.add_argument("--include-source-context", action="store_true",
                        help="Expose source one-hot only for the provenance ablation.")
    parser.add_argument("--mechanistic-descriptors", action="store_true",
                        help="Use six deterministic mechanism-motivated 2D descriptor proxies.")
    parser.add_argument("--target-encoding", choices=["onehot", "taxonomy"], default="onehot",
                        help="Categorical target identity or transferable biological taxonomy.")
    parser.add_argument(
        "--group-definition",
        choices=["screen_context", "screen_id", "library"],
        default="screen_context",
        help="Experimental grouping granularity for leakage-safe splitting.",
    )
    parser.add_argument("--group-folds", type=int, default=5)
    parser.add_argument("--group-fold-index", type=int, default=0)
    parser.add_argument("--skip-ad", action="store_true",
                        help="Skip final applicability-domain diagnostics during development screening.")
    parser.add_argument(
        "--ensure-rare-task-holdout",
        action="store_true",
        help=(
            "Diagnostic-only: move one complete toxicity group into test when the "
            "ordinary group split has no toxicity labels. Disabled by default so "
            "efficacy comparisons use exactly the same frozen split as external baselines."
        ),
    )
    args = parser.parse_args()
    with open(ROOT / "configs" / "research_smoke_config.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    train_loader, val_loader, test_loader, dataset = create_unified_dataloaders(
        str(ROOT / "merged_datasets"), batch_size=32, sample_fraction=args.sample_fraction,
        random_seed=args.split_seed, split_strategy=args.split_strategy, use_spatial=not args.no_conformers,
        pin_memory=DEVICE.type == "cuda",
        balanced_sampling=True,
        ensure_rare_task_holdout=(
            args.ensure_rare_task_holdout and args.split_strategy == "group"
        ),
        include_source_context=args.include_source_context,
        use_mechanistic_descriptors=args.mechanistic_descriptors,
        target_encoding=args.target_encoding,
        group_definition=args.group_definition,
        group_folds=args.group_folds,
        group_fold_index=args.group_fold_index,
    )
    print(f"Prewarming deterministic molecular features for {len(dataset)} rows...")
    for idx in range(len(dataset)):
        dataset[idx]
        if (idx + 1) % 1000 == 0:
            print(f"  cached {idx + 1}/{len(dataset)}")

    ablations = [
        ("A0_weighted_pool", {"use_component_transformer": False, "formulation_noise_std": 0.0}, True),
        ("A1_set_attention", {"use_component_transformer": True, "use_pair_bias": False,
                              "use_spatial_features": False, "use_target_conditioning": False,
                              "use_gaussian_ratio": False, "formulation_noise_std": 0.0}, True),
        ("A2_spatial_pair", {"use_component_transformer": True, "use_pair_bias": True,
                             "use_spatial_features": True, "use_target_conditioning": False,
                             "use_gaussian_ratio": False, "formulation_noise_std": 0.0}, True),
        ("A3_target_conditioned", {"use_component_transformer": True, "use_pair_bias": True,
                                   "use_spatial_features": True, "use_target_conditioning": True,
                                   "use_gaussian_ratio": False, "formulation_noise_std": 0.0}, True),
        ("A4_rbf_noise_full", {"use_component_transformer": True, "use_pair_bias": True,
                               "use_spatial_features": True, "use_target_conditioning": True,
                               "use_gaussian_ratio": True, "formulation_noise_std": 0.10}, True),
        ("A5_no_pair_bias", {"use_component_transformer": True, "use_pair_bias": False,
                             "use_spatial_features": True, "use_target_conditioning": True,
                             "use_gaussian_ratio": True, "formulation_noise_std": 0.10}, True),
        ("A6_no_spatial", {"use_component_transformer": True, "use_pair_bias": True,
                           "use_spatial_features": False, "use_target_conditioning": True,
                           "use_gaussian_ratio": True, "formulation_noise_std": 0.10}, True),
        ("A7_no_target", {"use_component_transformer": True, "use_pair_bias": True,
                          "use_spatial_features": True, "use_target_conditioning": False,
                          "use_gaussian_ratio": True, "formulation_noise_std": 0.10}, True),
        ("A8_unbalanced_full", {"use_component_transformer": True, "use_pair_bias": True,
                                "use_spatial_features": True, "use_target_conditioning": True,
                                "use_gaussian_ratio": True, "formulation_noise_std": 0.10}, False),
        ("A9_rbf_only", {"use_component_transformer": True, "use_pair_bias": True,
                         "use_spatial_features": True, "use_target_conditioning": True,
                         "use_gaussian_ratio": True, "formulation_noise_std": 0.0}, True),
        ("A10_noise_only", {"use_component_transformer": True, "use_pair_bias": True,
                            "use_spatial_features": True, "use_target_conditioning": True,
                            "use_gaussian_ratio": False, "formulation_noise_std": 0.10}, True),
        ("A11_no_ratio", {"use_component_transformer": True, "use_pair_bias": True,
                          "use_spatial_features": True, "use_target_conditioning": True,
                          "use_ratio_features": False, "use_gaussian_ratio": False,
                          "formulation_noise_std": 0.0}, True),
        ("A12_count_morgan_hybrid", {"use_component_transformer": True, "use_pair_bias": True,
                                      "use_spatial_features": True, "use_target_conditioning": True,
                                      "use_ratio_features": True, "use_gaussian_ratio": False,
                                      "formulation_noise_std": 0.0, "use_explicit_features": True}, True),
        ("A13_count_morgan_no_spatial", {"use_component_transformer": True, "use_pair_bias": True,
                                          "use_spatial_features": False, "use_target_conditioning": True,
                                          "use_ratio_features": True, "use_gaussian_ratio": False,
                                          "formulation_noise_std": 0.0, "use_explicit_features": True}, True),
        ("A14_count_morgan_residual", {"use_component_transformer": True, "use_pair_bias": True,
                                        "use_spatial_features": True, "use_target_conditioning": True,
                                        "use_ratio_features": True, "use_gaussian_ratio": False,
                                        "formulation_noise_std": 0.0, "use_explicit_features": True}, True),
        ("A15_count_morgan_replace", {"use_component_transformer": True, "use_pair_bias": True,
                                       "use_spatial_features": True, "use_target_conditioning": True,
                                       "use_ratio_features": True, "use_gaussian_ratio": False,
                                       "formulation_noise_std": 0.0, "use_explicit_features": True,
                                       "explicit_fusion_mode": "replace"}, True),
        ("A16_residual_rank_0p1", {"use_component_transformer": True, "use_pair_bias": True,
                                    "use_spatial_features": True, "use_target_conditioning": True,
                                    "use_ratio_features": True, "use_gaussian_ratio": False,
                                    "formulation_noise_std": 0.0, "use_explicit_features": True,
                                    "explicit_fusion_mode": "residual", "loss_ranking_weight": 0.1}, True),
        ("A17_residual_rank_1p0", {"use_component_transformer": True, "use_pair_bias": True,
                                    "use_spatial_features": True, "use_target_conditioning": True,
                                    "use_ratio_features": True, "use_gaussian_ratio": False,
                                    "formulation_noise_std": 0.0, "use_explicit_features": True,
                                    "explicit_fusion_mode": "residual", "loss_ranking_weight": 1.0}, True),
        ("A18_residual_gate_0p27", {"use_component_transformer": True, "use_pair_bias": True,
                                     "use_spatial_features": True, "use_target_conditioning": True,
                                     "use_ratio_features": True, "use_gaussian_ratio": False,
                                     "formulation_noise_std": 0.0, "use_explicit_features": True,
                                     "explicit_fusion_mode": "residual", "explicit_gate_init": -1.0}, True),
        ("A19_graph_rank_0p1", {"use_component_transformer": True, "use_pair_bias": True,
                                 "use_spatial_features": True, "use_target_conditioning": True,
                                 "use_ratio_features": True, "use_gaussian_ratio": False,
                                 "formulation_noise_std": 0.0, "use_explicit_features": False,
                                 "loss_ranking_weight": 0.1}, True),
        ("A20_graph_rank_1p0", {"use_component_transformer": True, "use_pair_bias": True,
                                 "use_spatial_features": True, "use_target_conditioning": True,
                                 "use_ratio_features": True, "use_gaussian_ratio": False,
                                 "formulation_noise_std": 0.0, "use_explicit_features": False,
                                 "loss_ranking_weight": 1.0}, True),
        ("A21_graph_rank_2p0", {"use_component_transformer": True, "use_pair_bias": True,
                                 "use_spatial_features": True, "use_target_conditioning": True,
                                 "use_ratio_features": True, "use_gaussian_ratio": False,
                                 "formulation_noise_std": 0.0, "use_explicit_features": False,
                                 "loss_ranking_weight": 2.0}, True),
        ("A22_graph_rank_1p0_low_lr", {"use_component_transformer": True, "use_pair_bias": True,
                                        "use_spatial_features": True, "use_target_conditioning": True,
                                        "use_ratio_features": True, "use_gaussian_ratio": False,
                                        "formulation_noise_std": 0.0, "use_explicit_features": False,
                                        "loss_ranking_weight": 1.0,
                                        "optimizer_learning_rate": 2e-4}, True),
        ("A23_dual_head_graph", {"use_component_transformer": True, "use_pair_bias": True,
                                  "use_spatial_features": True, "use_target_conditioning": True,
                                  "use_ratio_features": True, "use_gaussian_ratio": False,
                                  "formulation_noise_std": 0.0, "use_explicit_features": False,
                                  "use_separate_rank_head": True, "loss_ranking_weight": 0.5}, True),
        ("A24_dual_head_residual", {"use_component_transformer": True, "use_pair_bias": True,
                                     "use_spatial_features": True, "use_target_conditioning": True,
                                     "use_ratio_features": True, "use_gaussian_ratio": False,
                                     "formulation_noise_std": 0.0, "use_explicit_features": True,
                                     "explicit_fusion_mode": "residual",
                                     "use_separate_rank_head": True, "loss_ranking_weight": 0.1}, True),
        ("A25_dual_head_residual_rank_1p0", {"use_component_transformer": True, "use_pair_bias": True,
                                              "use_spatial_features": True, "use_target_conditioning": True,
                                              "use_ratio_features": True, "use_gaussian_ratio": False,
                                              "formulation_noise_std": 0.0, "use_explicit_features": True,
                                              "explicit_fusion_mode": "residual",
                                              "use_separate_rank_head": True,
                                              "loss_ranking_weight": 1.0}, True),
        ("A26_rank_adapter_graph", {"use_component_transformer": True, "use_pair_bias": True,
                                     "use_spatial_features": True, "use_target_conditioning": True,
                                     "use_ratio_features": True, "use_gaussian_ratio": False,
                                     "formulation_noise_std": 0.0, "use_explicit_features": False,
                                     "use_separate_rank_head": True, "rank_head_mode": "residual",
                                     "rank_detach_backbone": True, "rank_gate_init": -1.0,
                                     "loss_ranking_weight": 1.0}, True),
        ("A27_rank_adapter_graph_2p0", {"use_component_transformer": True, "use_pair_bias": True,
                                        "use_spatial_features": True, "use_target_conditioning": True,
                                        "use_ratio_features": True, "use_gaussian_ratio": False,
                                        "formulation_noise_std": 0.0, "use_explicit_features": False,
                                        "use_separate_rank_head": True, "rank_head_mode": "residual",
                                        "rank_detach_backbone": True, "rank_gate_init": -1.0,
                                        "loss_ranking_weight": 2.0}, True),
        ("A28_rank_adapter_graph_gate_0p5", {"use_component_transformer": True, "use_pair_bias": True,
                                             "use_spatial_features": True, "use_target_conditioning": True,
                                             "use_ratio_features": True, "use_gaussian_ratio": False,
                                             "formulation_noise_std": 0.0, "use_explicit_features": False,
                                             "use_separate_rank_head": True, "rank_head_mode": "residual",
                                             "rank_detach_backbone": True, "rank_gate_init": 0.0,
                                             "loss_ranking_weight": 1.0}, True),
        ("A29_full_without_pair_bias", {"use_component_transformer": True, "use_pair_bias": False,
                                         "use_spatial_features": True, "use_target_conditioning": True,
                                         "use_ratio_features": True, "use_gaussian_ratio": False,
                                         "formulation_noise_std": 0.0, "use_explicit_features": False,
                                         "use_separate_rank_head": True, "rank_head_mode": "residual",
                                         "rank_detach_backbone": True, "rank_gate_init": -1.0,
                                         "loss_ranking_weight": 2.0}, True),
        ("A30_full_without_spatial_descriptors", {"use_component_transformer": True, "use_pair_bias": True,
                                                   "use_spatial_features": False, "use_target_conditioning": True,
                                                   "use_ratio_features": True, "use_gaussian_ratio": False,
                                                   "formulation_noise_std": 0.0, "use_explicit_features": False,
                                                   "use_separate_rank_head": True, "rank_head_mode": "residual",
                                                   "rank_detach_backbone": True, "rank_gate_init": -1.0,
                                                   "loss_ranking_weight": 2.0}, True),
        ("A31_full_without_target_conditioning", {"use_component_transformer": True, "use_pair_bias": True,
                                                   "use_spatial_features": True, "use_target_conditioning": False,
                                                   "use_ratio_features": True, "use_gaussian_ratio": False,
                                                   "formulation_noise_std": 0.0, "use_explicit_features": False,
                                                   "use_separate_rank_head": True, "rank_head_mode": "residual",
                                                   "rank_detach_backbone": True, "rank_gate_init": -1.0,
                                                   "loss_ranking_weight": 2.0}, True),
        ("A32_full_without_explicit_missingness", {"use_component_transformer": True, "use_pair_bias": True,
                                                    "use_spatial_features": True, "use_target_conditioning": True,
                                                    "use_ratio_features": True, "use_gaussian_ratio": False,
                                                    "use_structure_mask_features": False,
                                                    "formulation_noise_std": 0.0, "use_explicit_features": False,
                                                    "use_separate_rank_head": True, "rank_head_mode": "residual",
                                                    "rank_detach_backbone": True, "rank_gate_init": -1.0,
                                                    "loss_ranking_weight": 2.0}, True),
        ("A33_efficiency_only", {"use_component_transformer": True, "use_pair_bias": True,
                                  "use_spatial_features": True, "use_target_conditioning": True,
                                  "use_ratio_features": True, "use_gaussian_ratio": False,
                                  "formulation_noise_std": 0.0, "use_explicit_features": False,
                                  "use_separate_rank_head": True, "rank_head_mode": "residual",
                                  "rank_detach_backbone": True, "rank_gate_init": -1.0,
                                  "loss_ranking_weight": 2.0, "loss_target_aux_weight": 0.0,
                                  "enabled_tasks": ["efficiency"]}, True),
        ("A34_streamlined_without_pair_or_missingness", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_gaussian_ratio": False,
            "use_structure_mask_features": False,
            "formulation_noise_std": 0.0, "use_explicit_features": False,
            "use_separate_rank_head": True, "rank_head_mode": "residual",
            "rank_detach_backbone": True, "rank_gate_init": -1.0,
            "loss_ranking_weight": 2.0,
        }, True),
        ("A35_provenance_retained", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_gaussian_ratio": False,
            "use_structure_mask_features": False, "use_explicit_features": False,
            "use_separate_rank_head": True, "rank_head_mode": "residual",
            "rank_detach_backbone": True, "rank_gate_init": -1.0,
            "loss_ranking_weight": 2.0, "rank_margin": 0.1,
        }, True),
        ("A36_provenance_removed", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_gaussian_ratio": False,
            "use_structure_mask_features": False, "use_explicit_features": False,
            "use_separate_rank_head": True, "rank_head_mode": "residual",
            "rank_detach_backbone": True, "rank_gate_init": -1.0,
            "loss_ranking_weight": 2.0, "rank_margin": 0.1,
            "drop_source_context": True,
        }, True),
        ("A37_provenance_and_process_masks_removed", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_gaussian_ratio": False,
            "use_structure_mask_features": False, "use_explicit_features": False,
            "use_separate_rank_head": True, "rank_head_mode": "residual",
            "rank_detach_backbone": True, "rank_gate_init": -1.0,
            "loss_ranking_weight": 2.0, "rank_margin": 0.1,
            "drop_source_context": True, "drop_process_missingness": True,
        }, True),
        ("A38_mechanistic_descriptors_removed", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_gaussian_ratio": False,
            "use_structure_mask_features": False, "use_explicit_features": False,
            "use_separate_rank_head": True, "rank_head_mode": "residual",
            "rank_detach_backbone": True, "rank_gate_init": -1.0,
            "loss_ranking_weight": 2.0, "rank_margin": 0.1,
            "drop_source_context": True, "drop_mechanistic_descriptors": True,
        }, True),
        ("A39_rank_margin_zero", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_gaussian_ratio": False,
            "use_structure_mask_features": False, "use_explicit_features": False,
            "use_separate_rank_head": True, "rank_head_mode": "residual",
            "rank_detach_backbone": True, "rank_gate_init": -1.0,
            "loss_ranking_weight": 2.0, "rank_margin": 0.0,
            "drop_source_context": True,
        }, True),
        ("A40_rank_margin_0p2", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_gaussian_ratio": False,
            "use_structure_mask_features": False, "use_explicit_features": False,
            "use_separate_rank_head": True, "rank_head_mode": "residual",
            "rank_detach_backbone": True, "rank_gate_init": -1.0,
            "loss_ranking_weight": 2.0, "rank_margin": 0.2,
            "drop_source_context": True,
        }, True),
        ("A41_delivery_four_roles", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": True,
            "rank_gate_init": -1.0, "loss_ranking_weight": 2.0,
            "rank_margin": 0.1, "drop_source_context": True,
            "enabled_tasks": ["efficiency"], "loss_target_aux_weight": 0.0,
        }, True),
        ("A42_delivery_without_sterol_peg", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": True,
            "rank_gate_init": -1.0, "loss_ranking_weight": 2.0,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_sterol_peg_roles": True,
            "enabled_tasks": ["efficiency"], "loss_target_aux_weight": 0.0,
        }, True),
        ("A43_source_free_without_rank_adapter", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": False,
            "loss_ranking_weight": 0.0, "drop_source_context": True,
            "drop_mechanistic_descriptors": True,
        }, True),
        ("A44_source_free_attached_rank_adapter", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True,
        }, True),
        ("A45_source_free_adaptive_rank_margin", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": True,
            "rank_gate_init": -1.0, "loss_ranking_weight": 2.0,
            "rank_margin": 0.25, "adaptive_rank_margin": True,
            "drop_source_context": True, "drop_mechanistic_descriptors": True,
        }, True),
        ("A46_source_free_vrex", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": True,
            "rank_gate_init": -1.0, "loss_ranking_weight": 2.0,
            "drop_source_context": True, "drop_mechanistic_descriptors": True,
            "robust_method": "vrex", "robust_weight": 0.1,
        }, True),
        ("A47_source_free_group_cvar", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": True,
            "rank_gate_init": -1.0, "loss_ranking_weight": 2.0,
            "drop_source_context": True, "drop_mechanistic_descriptors": True,
            "robust_method": "group_cvar", "robust_weight": 0.25,
        }, True),
        ("A48_source_free_coral", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": True,
            "rank_gate_init": -1.0, "loss_ranking_weight": 2.0,
            "drop_source_context": True, "drop_mechanistic_descriptors": True,
            "robust_method": "coral", "robust_weight": 0.01,
        }, True),
        ("A49_source_free_deeper_gnn", {
            "mol_feat_dim": 96, "num_gnn_layers": 4,
            "fusion_hidden_dim": 96, "prediction_hidden_dim": 96,
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": True,
            "rank_gate_init": -1.0, "loss_ranking_weight": 2.0,
            "drop_source_context": True, "drop_mechanistic_descriptors": True,
        }, True),
        ("A50_no_process_or_context", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": True,
            "rank_gate_init": -1.0, "loss_ranking_weight": 2.0,
            "drop_source_context": True, "drop_mechanistic_descriptors": True,
            "drop_all_process_features": True, "drop_all_context_features": True,
        }, True),
        ("A51_metadata_block_dropout", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": True,
            "rank_gate_init": -1.0, "loss_ranking_weight": 2.0,
            "drop_source_context": True, "drop_mechanistic_descriptors": True,
            "input_block_dropout": 0.3,
        }, True),
        ("A52_block_dropout_vrex", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": True,
            "rank_gate_init": -1.0, "loss_ranking_weight": 2.0,
            "drop_source_context": True, "drop_mechanistic_descriptors": True,
            "input_block_dropout": 0.3,
            "robust_method": "vrex", "robust_weight": 0.1,
        }, True),
        ("A53_chemistry_ratio_only", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": False,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": True,
            "rank_gate_init": -1.0, "loss_ranking_weight": 2.0,
            "drop_source_context": True, "drop_mechanistic_descriptors": True,
            "drop_all_process_features": True, "drop_all_context_features": True,
            "drop_target_features": True,
        }, True),
        ("A54_group_paired_rank_batches", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": True,
            "rank_gate_init": -1.0, "loss_ranking_weight": 2.0,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
        }, True),
        ("A55_attached_group_paired_rank", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
        }, True),
        ("A56_group_centered_screen_objective", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
        }, True),
        ("A57_group_dro", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": False,
            "loss_ranking_weight": 0.0, "drop_source_context": True,
            "drop_mechanistic_descriptors": True,
            "robust_method": "group_dro", "robust_weight": 0.25,
            "group_pair_batches": True, "groups_per_batch": 4,
            "rows_per_group": 4,
        }, True),
        ("A58_irm_regression", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": False,
            "loss_ranking_weight": 0.0, "drop_source_context": True,
            "drop_mechanistic_descriptors": True,
            "robust_method": "irm", "robust_weight": 0.1,
            "group_pair_batches": True, "groups_per_batch": 4,
            "rows_per_group": 4,
        }, True),
        ("A59_light_group_pair_sampling", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": True,
            "rank_gate_init": -1.0, "loss_ranking_weight": 2.0,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 2, "rows_per_group": 4,
            "sparse_fill_fraction": 0.5,
        }, True),
        ("A60_light_group_centered_objective", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 2, "rows_per_group": 4,
            "sparse_fill_fraction": 0.5,
        }, True),
        ("A61_full_no_target_conditioning", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": False,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
        }, True),
        ("A62_full_no_deterministic_descriptors", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": False, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
        }, True),
        ("A63_full_no_process_observation_bits", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_process_missingness": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
        }, True),
        ("A64_centered_rank_fully_detached", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": True,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
        }, True),
        ("A65_listnet_screen_objective", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 0.0,
            "listwise_method": "listnet", "listwise_weight": 0.5,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
        }, True),
        ("A66_listmle_screen_objective", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 0.0,
            "listwise_method": "listmle", "listwise_weight": 0.5,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
        }, True),
        ("A67_group_zscore_mse", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 0.0,
            "group_z_mse_weight": 0.5,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
        }, True),
        ("A68_group_orthogonalized_centered", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5,
            "robust_method": "group_orthogonal", "robust_weight": 0.01,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
        }, True),
        ("A69_toxicity_one_sided_mse", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5,
            "toxicity_loss_mode": "one_sided_mse",
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
        }, True),
        ("A70_centered_plus_listmle", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5,
            "listwise_method": "listmle", "listwise_weight": 0.25,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
        }, True),
        ("A75_group_balanced_pair_loss", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5,
            "pair_reduction": "group_mean",
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
        }, True),
        ("A76_group_hard_pair_loss", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5,
            "pair_reduction": "group_hard",
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
        }, True),
        ("A77_centered_plus_light_listmle", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5,
            "listwise_method": "listmle", "listwise_weight": 0.1,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
        }, True),
        ("A78_centered_plus_trace_listmle", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5,
            "listwise_method": "listmle", "listwise_weight": 0.05,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
        }, True),
        (
            "A79_validation_pareto_checkpoint",
            copy.deepcopy(A79_VALIDATION_PARETO_SWITCHES),
            True,
        ),
        (
            "A101_efficacy_only_pareto",
            copy.deepcopy(A101_EFFICACY_ONLY_PARETO_SWITCHES),
            True,
        ),
        (
            "A102_no_rank_adapter_matched",
            copy.deepcopy(A102_NO_RANK_ADAPTER_SWITCHES),
            True,
        ),
        (
            "A103_no_component_attention_matched",
            copy.deepcopy(A103_NO_COMPONENT_ATTENTION_SWITCHES),
            True,
        ),
        (
            "A104_no_biological_context_matched",
            copy.deepcopy(A104_NO_BIOLOGICAL_CONTEXT_SWITCHES),
            True,
        ),
        (
            "A105_no_physical_endpoints_matched",
            copy.deepcopy(A105_NO_PHYSICAL_ENDPOINTS_SWITCHES),
            True,
        ),
        (
            "A106_no_target_auxiliary_matched",
            copy.deepcopy(A106_NO_TARGET_AUXILIARY_SWITCHES),
            True,
        ),
        (
            "A107_no_spatial_descriptors_matched",
            copy.deepcopy(A107_NO_SPATIAL_DESCRIPTORS_SWITCHES),
            True,
        ),
        (
            "A108_no_sterol_structure_matched",
            copy.deepcopy(A108_NO_STEROL_STRUCTURE_SWITCHES),
            True,
        ),
        (
            "A109_ionizable_only_matched",
            copy.deepcopy(A109_IONIZABLE_ONLY_SWITCHES),
            True,
        ),
        ("A110_direct_morgan_replace", {
            **copy.deepcopy(A79_VALIDATION_PARETO_SWITCHES),
            "use_direct_morgan_head": True,
            "direct_morgan_mode": "replace",
            "use_separate_rank_head": False,
        }, True),
        ("A111_direct_morgan_residual", {
            **copy.deepcopy(A79_VALIDATION_PARETO_SWITCHES),
            "use_direct_morgan_head": True,
            "direct_morgan_mode": "residual",
            "direct_morgan_gate_init": -1.0,
            "use_separate_rank_head": False,
        }, True),
        ("A112_direct_morgan_rank_only", {
            **copy.deepcopy(A79_VALIDATION_PARETO_SWITCHES),
            "use_direct_morgan_head": True,
            "direct_morgan_mode": "rank_only",
            "use_separate_rank_head": False,
        }, True),
        ("A113_direct_rank_no_biological_context", {
            **copy.deepcopy(A79_VALIDATION_PARETO_SWITCHES),
            "use_direct_morgan_head": True,
            "direct_morgan_mode": "rank_only",
            "use_separate_rank_head": False,
            "drop_all_context_features": True,
            "drop_target_features": True,
        }, True),
        ("A114_direct_rank_no_descriptors", {
            **copy.deepcopy(A79_VALIDATION_PARETO_SWITCHES),
            "use_direct_morgan_head": True,
            "direct_morgan_mode": "rank_only",
            "use_separate_rank_head": False,
            "use_spatial_features": False,
            "drop_all_spatial_features": True,
        }, True),
        ("A115_dual_rank_residual", {
            **copy.deepcopy(A79_VALIDATION_PARETO_SWITCHES),
            "use_direct_morgan_head": True,
            "direct_morgan_mode": "rank_residual",
            "direct_morgan_gate_init": -1.0,
            "use_separate_rank_head": True,
        }, True),
        ("A80_centered_morgan_residual", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": True,
            "explicit_fusion_mode": "residual", "explicit_gate_init": -4.0,
            "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
        }, True),
        ("A81_full_no_helper_role", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
            "drop_component_roles": ["helper"],
        }, True),
        ("A82_full_no_sterol_role", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
            "drop_component_roles": ["cholesterol"],
        }, True),
        ("A83_full_no_peg_role", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
            "drop_component_roles": ["peg"],
        }, True),
        ("A84_ionizable_only_roles", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
            "drop_component_roles": ["helper", "cholesterol", "peg"],
        }, True),
        ("A85_listwise_morgan_pareto", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": True,
            "explicit_fusion_mode": "residual", "explicit_gate_init": -4.0,
            "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5,
            "listwise_method": "listmle", "listwise_weight": 0.1,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
            "checkpoint_selection": "efficacy_pareto",
        }, True),
        ("A86_decision_pareto_centered", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5,
            "rank_margin": 0.1, "drop_source_context": True,
            "drop_mechanistic_descriptors": True, "group_pair_batches": True,
            "groups_per_batch": 4, "rows_per_group": 4,
            "checkpoint_selection": "decision_pareto",
        }, True),
        ("A87_rcr_sigmoid_w0p05", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": False,
            "loss_ranking_weight": 0.0, "group_center_rank_weight": 0.0,
            "listwise_method": "listce_sigmoid", "listwise_weight": 0.05,
            "drop_source_context": True, "drop_mechanistic_descriptors": True,
            "group_pair_batches": True, "groups_per_batch": 4, "rows_per_group": 4,
            "checkpoint_selection": "decision_pareto",
        }, True),
        ("A88_rcr_sigmoid_w0p10", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": False,
            "loss_ranking_weight": 0.0, "group_center_rank_weight": 0.0,
            "listwise_method": "listce_sigmoid", "listwise_weight": 0.10,
            "drop_source_context": True, "drop_mechanistic_descriptors": True,
            "group_pair_batches": True, "groups_per_batch": 4, "rows_per_group": 4,
            "checkpoint_selection": "decision_pareto",
        }, True),
        ("A89_rcr_softplus_w0p10", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": False,
            "loss_ranking_weight": 0.0, "group_center_rank_weight": 0.0,
            "listwise_method": "listce_softplus", "listwise_weight": 0.10,
            "drop_source_context": True, "drop_mechanistic_descriptors": True,
            "group_pair_batches": True, "groups_per_batch": 4, "rows_per_group": 4,
            "checkpoint_selection": "decision_pareto",
        }, True),
        ("A90_sterol_peg_no_ratio_values", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5, "rank_margin": 0.1,
            "drop_source_context": True, "drop_mechanistic_descriptors": True,
            "drop_component_ratio_values": ["cholesterol", "peg"],
            "group_pair_batches": True, "groups_per_batch": 4, "rows_per_group": 4,
            "checkpoint_selection": "decision_pareto",
        }, True),
        ("A91_sterol_peg_no_ratio_observation", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5, "rank_margin": 0.1,
            "drop_source_context": True, "drop_mechanistic_descriptors": True,
            "drop_component_ratio_observation_bits": ["cholesterol", "peg"],
            "group_pair_batches": True, "groups_per_batch": 4, "rows_per_group": 4,
            "checkpoint_selection": "decision_pareto",
        }, True),
        ("A92_sterol_peg_no_structure_identity", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5, "rank_margin": 0.1,
            "drop_source_context": True, "drop_mechanistic_descriptors": True,
            "drop_component_structure_inputs": ["cholesterol", "peg"],
            "group_pair_batches": True, "groups_per_batch": 4, "rows_per_group": 4,
            "checkpoint_selection": "decision_pareto",
        }, True),
        ("A93_no_process_missingness", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5, "rank_margin": 0.1,
            "drop_source_context": True, "drop_mechanistic_descriptors": True,
            "drop_process_missingness": True,
            "group_pair_batches": True, "groups_per_batch": 4, "rows_per_group": 4,
            "checkpoint_selection": "decision_pareto",
        }, True),
        ("A94_no_biological_context", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5, "rank_margin": 0.1,
            "drop_source_context": True, "drop_mechanistic_descriptors": True,
            "drop_all_context_features": True, "drop_target_features": True,
            "group_pair_batches": True, "groups_per_batch": 4, "rows_per_group": 4,
            "checkpoint_selection": "decision_pareto",
        }, True),
        ("A95_efficacy_only_reference", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5, "rank_margin": 0.1,
            "drop_source_context": True, "drop_mechanistic_descriptors": True,
            "group_pair_batches": True, "groups_per_batch": 4, "rows_per_group": 4,
            "checkpoint_selection": "decision_pareto",
            "enabled_tasks": ["efficiency"], "loss_target_aux_weight": 0.0,
        }, True),
        ("A96_efficacy_only_no_sterol_peg_ratio", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5, "rank_margin": 0.1,
            "drop_source_context": True, "drop_mechanistic_descriptors": True,
            "drop_component_ratio_values": ["cholesterol", "peg"],
            "group_pair_batches": True, "groups_per_batch": 4, "rows_per_group": 4,
            "checkpoint_selection": "decision_pareto",
            "enabled_tasks": ["efficiency"], "loss_target_aux_weight": 0.0,
        }, True),
        ("A97_efficacy_only_no_sterol_peg_observation", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5, "rank_margin": 0.1,
            "drop_source_context": True, "drop_mechanistic_descriptors": True,
            "drop_component_ratio_observation_bits": ["cholesterol", "peg"],
            "group_pair_batches": True, "groups_per_batch": 4, "rows_per_group": 4,
            "checkpoint_selection": "decision_pareto",
            "enabled_tasks": ["efficiency"], "loss_target_aux_weight": 0.0,
        }, True),
        ("A98_efficacy_only_no_sterol_peg_structure", {
            "use_component_transformer": True, "use_pair_bias": False,
            "use_spatial_features": True, "use_target_conditioning": True,
            "use_ratio_features": True, "use_structure_mask_features": False,
            "use_explicit_features": False, "use_separate_rank_head": True,
            "rank_head_mode": "residual", "rank_detach_backbone": False,
            "rank_gate_init": -1.0, "loss_ranking_weight": 1.0,
            "group_center_rank_weight": 0.5, "rank_margin": 0.1,
            "drop_source_context": True, "drop_mechanistic_descriptors": True,
            "drop_component_structure_inputs": ["cholesterol", "peg"],
            "group_pair_batches": True, "groups_per_batch": 4, "rows_per_group": 4,
            "checkpoint_selection": "decision_pareto",
            "enabled_tasks": ["efficiency"], "loss_target_aux_weight": 0.0,
        }, True),
    ]
    if args.only:
        requested = set(args.only)
        ablations = [entry for entry in ablations if entry[0] in requested]
        missing = requested.difference(entry[0] for entry in ablations)
        if missing:
            raise ValueError(f"Unknown ablation names: {sorted(missing)}")
    results = []
    for name, switches, balanced in ablations:
        for seed in args.seeds:
            print(f"Running {name}, seed={seed}, balanced={balanced}")
            result = run_one(
                name, switches, config["model"], train_loader, val_loader, test_loader,
                dataset, seed, args.epochs, balanced, args.learning_rate, args.patience,
                compute_final_ad=not args.skip_ad,
            )
            results.append(result)
            print(
                f"  val_loss={result['val']['multitask_loss']:.4f}, "
                f"val_eff_rmse={result['val'].get('efficiency_rmse', float('nan')):.4f}, "
                f"test_eff_rmse={result['test'].get('efficiency_rmse', float('nan')):.4f}"
            )

    output_dir = ROOT / "ablation_results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw, summary = aggregate(results)
    raw.to_csv(output_dir / "runs.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    ensemble, conformal, per_group = ensemble_and_conformal_tables(results)
    ensemble.to_csv(output_dir / "ensemble_metrics.csv", index=False)
    conformal.to_csv(output_dir / "conformal_metrics.csv", index=False)
    per_group.to_csv(output_dir / "per_group_metrics.csv", index=False)
    with open(output_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2, allow_nan=True)
    metadata = {
        "dataset_rows": len(dataset),
        "train_rows": len(train_loader.dataset),
        "val_rows": len(val_loader.dataset),
        "test_rows": len(test_loader.dataset),
        "sample_fraction": args.sample_fraction,
        "split_seed": args.split_seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "early_stopping_patience": args.patience,
        "seeds": args.seeds,
        "split_strategy": args.split_strategy,
        "use_conformers": not args.no_conformers,
        "include_source_context": args.include_source_context,
        "use_mechanistic_descriptors": args.mechanistic_descriptors,
        "target_encoding": args.target_encoding,
        "group_definition": args.group_definition,
        "group_folds": (
            args.group_folds
            if args.split_strategy in {"group_kfold", "balanced_group_kfold"}
            else None
        ),
        "group_fold_index": (
            args.group_fold_index
            if args.split_strategy in {"group_kfold", "balanced_group_kfold"}
            else None
        ),
        "ensure_rare_task_holdout": bool(args.ensure_rare_task_holdout),
        "device": str(DEVICE),
        "torch_version": torch.__version__,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    append_standard_ablation_record(summary, output_dir, metadata)
    register_ablation_round(output_dir, summary, metadata)
    # Replace the latest-only block with the complete immutable experiment
    # ledger. Completed rounds are registered automatically above.
    try:
        from sync_ablation_history import sync_history
        sync_history()
    except Exception as exc:
        print(f"Warning: complete ablation ledger sync failed: {exc}")
    concise = [
        column for column in [
            "name", "val_efficiency_rmse_mean",
            "val_efficiency_within_group_spearman_mean",
            "test_efficiency_rmse_mean",
            "test_efficiency_within_group_spearman_mean",
        ] if column in summary
    ]
    print(summary[concise].to_string(index=False))
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
