#!/usr/bin/env python3
"""
DeepLNP 模型评估脚本

功能:
1. 加载最佳模型权重
2. 在测试集上评估
3. 生成详细评估报告
4. 可视化预测结果
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from deeplnp.data.unified_dataset import (
    CONTEXT_DIM,
    FORMULATION_DIM,
    SPATIAL_DIM,
    TARGET_DIM,
    TASK_COLUMNS,
    create_unified_dataloaders,
)
from deeplnp.models.predictor import LNPPredictor


def project_path(path_value: str, default_relative: str) -> Path:
    raw = path_value or default_relative
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        if count > 1:
            names = ", ".join(torch.cuda.get_device_name(i) for i in range(count))
            print(f"✅ 检测到 CUDA 多卡：{count} 张 GPU ({names})，评估阶段使用 cuda:0 单进程")
        else:
            print(f"✅ 使用 CUDA 单卡评估：{torch.cuda.get_device_name(0)}")
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("✅ 使用 Apple Silicon MPS GPU 评估")
        return torch.device("mps")
    print("⚠️  未检测到 GPU，使用 CPU 评估")
    return torch.device("cpu")


class Evaluator:
    """DeepLNP 模型评估器"""
    
    def __init__(
        self,
        model_path: str,
        dataset_name: str,
        device: torch.device,
        config: Dict[str, Any],
    ):
        self.model_path = Path(model_path)
        self.dataset_name = dataset_name
        self.device = device
        self.config = config
        
        # 加载模型
        self.model = self._load_model()
        
        # 结果保存目录
        self.result_dir = project_path(config.get('evaluation', {}).get('result_dir'), "evaluation_results")
        self.result_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_model(self) -> LNPPredictor:
        """加载模型权重"""
        print(f"📥 加载模型：{self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # 优先使用检查点中保存的配置
        if 'config' in checkpoint:
            model_config = checkpoint['config']['model']
            self.config = checkpoint['config']
            print(f"✅ 使用检查点中的配置 (Epoch {checkpoint['epoch']})")
        else:
            model_config = self.config['model']
            print("⚠️ 使用配置文件中的配置")
        
        # 创建模型
        model = LNPPredictor(
            mol_encoder_type=model_config['mol_encoder_type'],
            mol_feat_dim=model_config['mol_feat_dim'],
            atom_feat_dim=model_config['atom_feat_dim'],
            bond_feat_dim=model_config['bond_feat_dim'],
            num_gnn_layers=model_config['num_gnn_layers'],
            use_3d=model_config['use_3d'],
            struct_feat_dim=model_config['struct_feat_dim'],
            use_images=model_config['use_images'],
            image_feat_dim=model_config['image_feat_dim'],
            image_channels=model_config['image_channels'],
            image_size=model_config['image_size'],
            use_embeddings=model_config['use_embeddings'],
            embedding_feat_dim=model_config['embedding_feat_dim'],
            max_embedding_dim=model_config['max_embedding_dim'],
            formul_feat_dim=model_config['formul_feat_dim'],
            num_components=model_config['num_components'],
            formulation_input_dim=model_config.get('formulation_input_dim', FORMULATION_DIM),
            context_feat_dim=model_config.get('context_feat_dim', CONTEXT_DIM),
            target_feat_dim=model_config.get('target_feat_dim', TARGET_DIM),
            num_target_classes=model_config.get('num_target_classes', TARGET_DIM),
            spatial_feat_dim=model_config.get('spatial_feat_dim', SPATIAL_DIM),
            use_component_transformer=model_config.get('use_component_transformer', True),
            use_pair_bias=model_config.get('use_pair_bias', True),
            use_target_conditioning=model_config.get('use_target_conditioning', True),
            use_spatial_features=model_config.get('use_spatial_features', True),
            use_ratio_features=model_config.get('use_ratio_features', True),
            use_gaussian_ratio=model_config.get('use_gaussian_ratio', False),
            use_structure_mask_features=model_config.get('use_structure_mask_features', True),
            formulation_noise_std=model_config.get('formulation_noise_std', 0.0),
            use_explicit_features=model_config.get('use_explicit_features', False),
            explicit_fusion_mode=model_config.get('explicit_fusion_mode', 'residual'),
            explicit_gate_init=model_config.get('explicit_gate_init', -3.0),
            fingerprint_input_dim=self.config.get('dataset', {}).get('fingerprint_dim', 2048),
            use_separate_rank_head=model_config.get('use_separate_rank_head', False),
            rank_head_mode=model_config.get('rank_head_mode', 'separate'),
            rank_detach_backbone=model_config.get('rank_detach_backbone', False),
            rank_gate_init=model_config.get('rank_gate_init', -1.0),
            physchem_feat_dim=model_config['physchem_feat_dim'],
            fusion_hidden_dim=model_config['fusion_hidden_dim'],
            num_heads=model_config['fusion_num_heads'],
            dropout=model_config['fusion_dropout'],
            num_experts=model_config.get('num_experts', 3),
            prediction_hidden_dim=model_config['prediction_hidden_dim'],
            num_tasks=model_config['num_tasks'],
            use_mc_dropout=model_config['use_mc_dropout'],
            mc_dropout_rate=model_config.get('mc_dropout_rate', 0.1),
        )
        
        # 加载权重，兼容旧 DataParallel/DDP 保存的 module. 前缀。
        # 评估阶段保持单进程单设备，避免 DataParallel 按第 0 维切分 edge_index。
        state_dict = checkpoint['model_state_dict']
        if any(key.startswith("module.") for key in state_dict):
            state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        epoch = checkpoint.get('epoch', 'unknown')
        score = checkpoint.get('best_score', checkpoint.get('val_loss', 0.0))
        print(f"✅ 模型加载成功 (Epoch {epoch}, Score: {score:.4f})")
        
        # 从检查点读取标准化参数（如果存在）
        self.target_mean = checkpoint.get('target_mean', 0.0)
        self.target_std = checkpoint.get('target_std', 1.0)
        self.task_stats = checkpoint.get('task_stats', {'efficiency': {'mean': self.target_mean, 'std': self.target_std}})
        print(f"📊 检查点中的标准化参数: mean={self.target_mean:.4f}, std={self.target_std:.4f}")
        
        return model
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        """在测试集上评估"""
        print("\n📊 加载测试数据...")
        
        # 使用 merged_datasets 目录
        merged_datasets_dir = project_path(self.config['dataset'].get('merged_datasets_dir'), 'merged_datasets')
        
        _, _, test_loader, dataset = create_unified_dataloaders(
            merged_datasets_dir=str(merged_datasets_dir),
            batch_size=self.config['training']['batch_size'],
            num_workers=0,
            sample_fraction=self.config['dataset'].get('sample_fraction', 1.0),
            random_seed=self.config['dataset'].get('random_seed', 42),
            split_strategy=self.config['dataset'].get('split_strategy', 'group'),
            use_spatial=self.config['dataset'].get('use_spatial', True),
            pin_memory=self.device.type == 'cuda',
            include_source_context=self.config['dataset'].get('include_source_context', False),
            use_mechanistic_descriptors=self.config['dataset'].get('use_mechanistic_descriptors', False),
            target_encoding=self.config['dataset'].get('target_encoding', 'onehot'),
        )
        # The checkpoint scaling is authoritative and was fitted on its train split.
        dataset.task_stats = self.task_stats
        dataset._sample_cache.clear()
        
        # 获取标准化参数（优先使用检查点中的参数）
        if hasattr(self, 'target_mean') and hasattr(self, 'target_std'):
            target_mean = self.target_mean
            target_std = self.target_std
        else:
            target_mean = getattr(dataset, 'target_mean', 0.0)
            target_std = getattr(dataset, 'target_std', 1.0)
        print(f"📊 使用标准化参数: mean={target_mean:.4f}, std={target_std:.4f}")
        
        print(f"✅ 封存测试集：{len(test_loader.dataset)} 条样本")
        
        # 评估
        print("\n🔍 开始评估...")
        
        all_predictions = {task: [] for task in TASK_COLUMNS}
        all_rank_scores = []
        all_targets = {task: [] for task in TASK_COLUMNS}
        all_indices = {task: [] for task in TASK_COLUMNS}
        all_censors = {task: [] for task in TASK_COLUMNS}
        
        for batch in test_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 预测 - 支持四组分配方
            outputs = self.model(
                ionizable_atom_features=batch.get('ionizable_atom_features', batch.get('atom_features')),
                ionizable_edge_index=batch.get('ionizable_edge_index', batch.get('edge_index')),
                ionizable_bond_features=batch.get('ionizable_bond_features', batch.get('bond_features')),
                ionizable_batch=batch.get('ionizable_batch'),
                helper_atom_features=batch.get('helper_atom_features'),
                helper_edge_index=batch.get('helper_edge_index'),
                helper_bond_features=batch.get('helper_bond_features'),
                helper_batch=batch.get('helper_batch'),
                cholesterol_atom_features=batch.get('cholesterol_atom_features'),
                cholesterol_edge_index=batch.get('cholesterol_edge_index'),
                cholesterol_bond_features=batch.get('cholesterol_bond_features'),
                cholesterol_batch=batch.get('cholesterol_batch'),
                peg_atom_features=batch.get('peg_atom_features'),
                peg_edge_index=batch.get('peg_edge_index'),
                peg_bond_features=batch.get('peg_bond_features'),
                peg_batch=batch.get('peg_batch'),
                ionizable_spatial_features=batch.get('ionizable_spatial_features'),
                helper_spatial_features=batch.get('helper_spatial_features'),
                cholesterol_spatial_features=batch.get('cholesterol_spatial_features'),
                peg_spatial_features=batch.get('peg_spatial_features'),
                molar_ratios=batch.get('molar_ratios', torch.zeros(batch['target'].size(0), 4).to(self.device)),
                embeddings=batch.get('embedding', batch.get('embeddings')),
                images=batch.get('image', batch.get('images')),
                formulation_features=batch.get('formulation_features'),
                context_features=batch.get('context_features'),
                target_features=batch.get('target_features'),
                component_structure_mask=batch.get('component_structure_mask'),
                component_active_mask=batch.get('component_active_mask'),
            )
            
            # 收集预测和真实值
            if 'predictions' in outputs:
                for task in TASK_COLUMNS:
                    if task not in outputs['predictions'] or task not in batch:
                        continue
                    mask = batch[f'{task}_mask'].bool()
                    if not mask.any():
                        continue
                    stats = self.task_stats.get(task, {'mean': 0.0, 'std': 1.0})
                    pred = outputs['predictions'][task][mask].cpu() * stats['std'] + stats['mean']
                    target = batch[task][mask].cpu() * stats['std'] + stats['mean']
                    all_predictions[task].append(pred)
                    if task == 'efficiency' and 'efficiency_rank_score' in outputs['predictions']:
                        rank_score = (
                            outputs['predictions']['efficiency_rank_score'][mask].cpu()
                            * stats['std'] + stats['mean']
                        )
                        all_rank_scores.append(rank_score)
                    all_targets[task].append(target)
                    all_indices[task].append(batch['idx'][mask].cpu())
                    if task == 'toxicity':
                        all_censors[task].append(batch['toxicity_censor'][mask].cpu())
        
        # 合并结果
        predictions_np = {
            task: torch.cat(values).numpy() for task, values in all_predictions.items() if values
        }
        targets_np = {
            task: torch.cat(all_targets[task]).numpy() for task in predictions_np
        }
        indices_np = {
            task: torch.cat(all_indices[task]).numpy() for task in predictions_np
        }
        rank_scores_np = torch.cat(all_rank_scores).numpy() if all_rank_scores else None
        if not predictions_np:
            print("⚠️ 警告：没有收集到任何预测结果")
            return {'error': 'No predictions collected'}
        
        # 计算指标
        censors_np = {
            task: torch.cat(values).numpy() for task, values in all_censors.items() if values
        }
        metrics = self._compute_metrics(predictions_np, targets_np, censors_np)
        
        # 保存结果
        results = {
            'model_path': str(self.model_path),
            'dataset': self.dataset_name,
            'test_size': len(test_loader.dataset),
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'predictions': {task: values.tolist() for task, values in predictions_np.items()},
            'efficiency_rank_scores': rank_scores_np.tolist() if rank_scores_np is not None else None,
            'targets': {task: values.tolist() for task, values in targets_np.items()},
        }
        
        # 保存为 JSON
        result_file = self.result_dir / f"evaluation_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 评估结果已保存到：{result_file}")
        
        # 生成 Top 20 候选 LNP 配方
        if 'efficiency' in predictions_np:
            self._generate_top_candidates(
                dataset,
                predictions_np['efficiency'],
                targets_np['efficiency'],
                indices_np['efficiency'],
                ranking_scores=rank_scores_np,
            )
        
        # 可视化
        self._visualize_results(predictions_np, targets_np)
        
        return results
    
    def _compute_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        censors: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """计算评估指标"""
        metrics = {}
        
        for task in predictions:
            pred = predictions[task]
            target = targets[task]
            metrics[f'{task}_count'] = int(len(target))

            if task == 'toxicity' and task in censors:
                censor = censors[task]
                upper = censor < 0
                lower = censor > 0
                if upper.any():
                    metrics['toxicity_upper_bound_violation_rate'] = float(np.mean(pred[upper] > target[upper]))
                    metrics['toxicity_mean_upper_excess'] = float(np.mean(np.maximum(pred[upper] - target[upper], 0.0)))
                if lower.any():
                    metrics['toxicity_lower_bound_violation_rate'] = float(np.mean(pred[lower] < target[lower]))
                exact = censor == 0
                if not exact.any():
                    continue
                pred, target = pred[exact], target[exact]
            
            # MSE
            mse = np.mean((pred - target) ** 2)
            metrics[f'{task}_mse'] = float(mse)
            
            # MAE
            mae = np.mean(np.abs(pred - target))
            metrics[f'{task}_mae'] = float(mae)
            
            # RMSE
            rmse = np.sqrt(mse)
            metrics[f'{task}_rmse'] = float(rmse)
            
            # R²
            ss_res = np.sum((target - pred) ** 2)
            ss_tot = np.sum((target - np.mean(target)) ** 2)
            if len(target) > 1 and ss_tot > 1e-8:
                r2 = 1 - (ss_res / ss_tot)
                metrics[f'{task}_r2'] = float(r2)
            
            # Pearson 相关系数
            if len(target) > 1 and np.std(target) > 1e-8 and np.std(pred) > 1e-8:
                pearson = np.corrcoef(pred, target)[0, 1]
                metrics[f'{task}_pearson'] = float(pearson)
        
        return metrics
    
    def _generate_top_candidates(
        self,
        dataset,
        predictions: np.ndarray,
        targets: np.ndarray,
        row_indices: np.ndarray,
        ranking_scores: Optional[np.ndarray] = None,
    ):
        """Rank held-out observations; this is not prospective generation."""
        print("\n📊 排序 Top 20 封存测试观测（仅用于诊断，不是新候选）...")
        
        # 获取完整的 LNP 配方信息
        ionizable_smiles_list = [dataset.smiles_list[int(idx)] for idx in row_indices]
        
        # 尝试获取其他脂质成分的 SMILES 或名称
        helper_smiles_list = []
        cholesterol_smiles_list = []
        peg_smiles_list = []
        helper_name_list = []
        cholesterol_name_list = []
        peg_name_list = []
        molar_ratio_list = []
        
        if hasattr(dataset, 'df'):
            # 从配方数据中提取脂质信息
            for idx in range(len(predictions)):
                if idx < len(dataset.df):
                    row = dataset.df.iloc[int(row_indices[idx])]
                    helper_identity = row.get('helper_lipid_id', row.get('helper_lipid', ''))
                    sterol_identity = row.get('sterol_lipid', 'cholesterol')
                    peg_identity = row.get('peg_lipid', '')
                    helper_name_list.append(str(helper_identity) if pd.notna(helper_identity) else '')
                    cholesterol_name_list.append(str(sterol_identity) if pd.notna(sterol_identity) else 'cholesterol')
                    peg_name_list.append(str(peg_identity) if pd.notna(peg_identity) else '')
                    # Export exactly the same validated/fallback structures
                    # consumed by the model, not merely the raw source cells.
                    resolver = getattr(dataset, '_component_smiles', None)
                    helper_smiles_list.append(resolver(row, 'helper') if resolver else str(row.get('helper_lipid_smiles', '') or ''))
                    cholesterol_smiles_list.append(resolver(row, 'cholesterol') if resolver else str(row.get('sterol_lipid_smiles', '') or ''))
                    peg_smiles_list.append(resolver(row, 'peg') if resolver else str(row.get('peg_lipid_smiles', '') or ''))
                    # 摩尔比
                    molar_ratios = {}
                    if 'cationic_lipid_mol_ratio' in row and pd.notna(row['cationic_lipid_mol_ratio']):
                        molar_ratios['ionizable'] = row['cationic_lipid_mol_ratio']
                    if 'phospholipid_mol_ratio' in row and pd.notna(row['phospholipid_mol_ratio']):
                        molar_ratios['helper'] = row['phospholipid_mol_ratio']
                    if 'cholesterol_mol_ratio' in row and pd.notna(row['cholesterol_mol_ratio']):
                        molar_ratios['cholesterol'] = row['cholesterol_mol_ratio']
                    if 'peg_lipid_mol_ratio' in row and pd.notna(row['peg_lipid_mol_ratio']):
                        molar_ratios['peg'] = row['peg_lipid_mol_ratio']
                    molar_ratio_list.append(molar_ratios if molar_ratios else '')
                else:
                    helper_smiles_list.append('')
                    helper_name_list.append('')
                    cholesterol_smiles_list.append('')
                    cholesterol_name_list.append('')
                    peg_smiles_list.append('')
                    peg_name_list.append('')
                    molar_ratio_list.append('')
        else:
            helper_smiles_list = [''] * len(predictions)
            helper_name_list = [''] * len(predictions)
            cholesterol_smiles_list = [''] * len(predictions)
            cholesterol_name_list = [''] * len(predictions)
            peg_smiles_list = [''] * len(predictions)
            peg_name_list = [''] * len(predictions)
            molar_ratio_list = [''] * len(predictions)
        
        # 创建结果 DataFrame（包含完整 LNP 配方）
        results_df = pd.DataFrame({
            'rank': range(1, len(predictions) + 1),
            'predicted_efficiency': predictions,
            'ranking_score': ranking_scores if ranking_scores is not None else predictions,
            'actual_efficiency': targets,
            'ionizable_lipid_smiles': ionizable_smiles_list,
            'helper_lipid': helper_name_list,
            'helper_lipid_smiles': helper_smiles_list,
            'sterol_lipid': cholesterol_name_list,
            'sterol_lipid_smiles': cholesterol_smiles_list,
            'peg_lipid': peg_name_list,
            'peg_lipid_smiles': peg_smiles_list,
        })
        
        # 添加摩尔比信息
        if molar_ratio_list:
            results_df['molar_ratios'] = molar_ratio_list
        
        # The selected model uses a gradient-isolated residual adapter for
        # screening order. Point predictions remain the calibrated values
        # reported to the user, while ranking_score controls only ordering.
        results_df = results_df.sort_values('ranking_score', ascending=False)
        results_df['rank'] = range(1, len(results_df) + 1)
        
        # 选择 Top 20
        top20 = results_df.head(20).reset_index(drop=True)
        
        # 保存为 CSV
        top20_file = self.result_dir / "top20_heldout_predictions.csv"
        top20.to_csv(top20_file, index=False)
        
        print(f"✅ Top 20 封存测试观测已保存到：{top20_file}")
        print("\n📊 Top 20 封存测试观测预览:")
        print(top20.to_string(index=False))
    
    def _visualize_results(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
    ):
        """可视化预测结果"""
        print("\n📈 生成可视化图表...")
        
        # 设置样式
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 12
        
        # 为每个任务创建图表
        for task in predictions:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # 散点图
            ax1 = axes[0]
            ax1.scatter(targets[task], predictions[task], alpha=0.5, s=20)
            ax1.plot([targets[task].min(), targets[task].max()], 
                    [targets[task].min(), targets[task].max()], 
                    'r--', linewidth=2)
            ax1.set_xlabel('True Values')
            ax1.set_ylabel('Predictions')
            ax1.set_title(f'{task.capitalize()} - Parity Plot')
            ax1.grid(True, alpha=0.3)
            
            # 残差图
            ax2 = axes[1]
            residuals = predictions[task] - targets[task]
            ax2.scatter(targets[task], residuals, alpha=0.5, s=20)
            ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
            ax2.set_xlabel('True Values')
            ax2.set_ylabel('Residuals')
            ax2.set_title(f'{task.capitalize()} - Residual Plot')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存
            plot_file = self.result_dir / f"plot_{task}_{self.dataset_name}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ {task} 可视化：{plot_file}")
        
        print(f"\n💾 所有图表已保存到：{self.result_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DeepLNP 模型评估')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='最佳模型路径',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='agile',
        help='数据集名称',
    )
    
    args = parser.parse_args()
    
    # 默认配置
    config = {
        'dataset': {
            'merged_datasets_dir': 'merged_datasets',
            'modalities': ['formulation'],
            'max_atoms': 256,
            'fingerprint_dim': 2048,
            'image_size': 128,
        },
        'model': {
            'mol_encoder_type': 'graph',
            'mol_feat_dim': 128,
            'atom_feat_dim': 39,
            'bond_feat_dim': 4,
            'num_gnn_layers': 3,
            'use_3d': True,
            'struct_feat_dim': 256,
            'use_images': True,
            'image_feat_dim': 256,
            'image_channels': 3,
            'image_size': 128,
            'use_embeddings': True,
            'embedding_feat_dim': 128,
            'max_embedding_dim': 2048,
            'formul_feat_dim': 128,
            'num_components': 4,
            'physchem_feat_dim': 128,
            'fusion_hidden_dim': 128,
            'fusion_num_heads': 2,
            'fusion_dropout': 0.0,
            'num_experts': 3,
            'prediction_hidden_dim': 256,
            'num_tasks': 4,
            'use_mc_dropout': True,
            'mc_dropout_rate': 0.1,
        },
        'training': {
            'batch_size': 16,
        },
        'evaluation': {
            'result_dir': 'evaluation_results',
        },
    }
    
    # 检测设备
    device = detect_device()
    print(f"🔍 使用设备：{device}")
    
    # 创建评估器
    evaluator = Evaluator(
        model_path=args.model,
        dataset_name=args.dataset,
        device=device,
        config=config,
    )
    
    # 运行评估
    results = evaluator.evaluate()
    
    # 打印结果
    print("\n" + "="*80)
    print("评估结果汇总")
    print("="*80)
    
    for metric, value in results['metrics'].items():
        print(f"{metric:30s}: {value:.4f}")
    
    print("="*80)


if __name__ == '__main__':
    main()


def evaluate_and_select_candidates(
    model: LNPPredictor,
    test_loader: DataLoader,
    device: torch.device,
    top_k: int = 20,
    config: Dict[str, Any] = None,
) -> list:
    """
    评估模型并筛选 Top K 候选 LNP 配方（完整四组分）
    
    Args:
        model: 训练好的模型
        test_loader: 测试集 DataLoader
        device: 设备
        top_k: 选择多少个候选分子
        config: 配置字典
    
    Returns:
        List of candidate dictionaries with predictions and properties
    """
    print(f"\n🔍 开始评估并筛选 Top {top_k} 候选 LNP 配方（完整四组分）...")
    
    model.eval()
    candidates = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # 准备数据
            targets = batch['target'].to(device)
            
            # 准备输入特征（使用与 train.py 相同的四组分特征）
            ionizable_atom_features = batch.get('ionizable_atom_features').to(device) if 'ionizable_atom_features' in batch else None
            ionizable_edge_index = batch.get('ionizable_edge_index').to(device) if 'ionizable_edge_index' in batch else None
            ionizable_bond_features = batch.get('ionizable_bond_features').to(device) if 'ionizable_bond_features' in batch else None
            ionizable_batch = batch.get('ionizable_batch').to(device) if 'ionizable_batch' in batch else None
            
            helper_atom_features = batch.get('helper_atom_features').to(device) if 'helper_atom_features' in batch else None
            helper_edge_index = batch.get('helper_edge_index').to(device) if 'helper_edge_index' in batch else None
            helper_bond_features = batch.get('helper_bond_features').to(device) if 'helper_bond_features' in batch else None
            helper_batch = batch.get('helper_batch').to(device) if 'helper_batch' in batch else None
            
            cholesterol_atom_features = batch.get('cholesterol_atom_features').to(device) if 'cholesterol_atom_features' in batch else None
            cholesterol_edge_index = batch.get('cholesterol_edge_index').to(device) if 'cholesterol_edge_index' in batch else None
            cholesterol_bond_features = batch.get('cholesterol_bond_features').to(device) if 'cholesterol_bond_features' in batch else None
            cholesterol_batch = batch.get('cholesterol_batch').to(device) if 'cholesterol_batch' in batch else None
            
            peg_atom_features = batch.get('peg_atom_features').to(device) if 'peg_atom_features' in batch else None
            peg_edge_index = batch.get('peg_edge_index').to(device) if 'peg_edge_index' in batch else None
            peg_bond_features = batch.get('peg_bond_features').to(device) if 'peg_bond_features' in batch else None
            peg_batch = batch.get('peg_batch').to(device) if 'peg_batch' in batch else None
            
            molar_ratios = batch.get('molar_ratios').to(device) if 'molar_ratios' in batch else None
            
            formulation_features = batch.get('formulation_features').to(device) if 'formulation_features' in batch else None
            context_features = batch.get('context_features').to(device) if 'context_features' in batch else None
            
            # 可选模态
            fingerprints = batch.get('ionizable_fingerprint', batch.get('fingerprint')).to(device) if 'ionizable_fingerprint' in batch or 'fingerprint' in batch else None
            embeddings = batch.get('embedding', batch.get('embeddings')).to(device) if 'embedding' in batch or 'embeddings' in batch else None
            atom_features = batch.get('atom_features').to(device) if 'atom_features' in batch else None
            
            # ========== 特征优先级逻辑（严格按优先级选择） ==========
            # 第一优先级：四组分特征（ionizable_atom_features 等）
            # 第二优先级：embeddings（3D 构象或预训练嵌入）
            # 第三优先级：atom_features（单分子图特征）
            # 第四优先级：fingerprint（分子指纹）
            
            # 检查是否有四组分特征（只要 ionizable_atom_features 存在就使用四组分模式）
            has_four_component = (ionizable_atom_features is not None)
            
            if has_four_component:
                # 使用四组分特征，不需要 fallback
                pass
            elif embeddings is not None:
                # 第二优先级：使用 embeddings（3D 构象或预训练嵌入）
                pass
            elif atom_features is not None:
                # 第三优先级：使用 atom_features（单分子图特征）
                ionizable_atom_features = atom_features
            elif fingerprints is not None:
                # 第四优先级：使用 fingerprint 作为最后备选
                embeddings = fingerprints
            
            # 模型预测
            try:
                # 根据特征优先级调用模型
                if has_four_component:
                    output = model(
                        ionizable_atom_features=ionizable_atom_features,
                        ionizable_edge_index=ionizable_edge_index,
                        ionizable_bond_features=ionizable_bond_features,
                        ionizable_batch=ionizable_batch,
                        helper_atom_features=helper_atom_features,
                        helper_edge_index=helper_edge_index,
                        helper_bond_features=helper_bond_features,
                        helper_batch=helper_batch,
                        cholesterol_atom_features=cholesterol_atom_features,
                        cholesterol_edge_index=cholesterol_edge_index,
                        cholesterol_bond_features=cholesterol_bond_features,
                        cholesterol_batch=cholesterol_batch,
                        peg_atom_features=peg_atom_features,
                        peg_edge_index=peg_edge_index,
                        peg_bond_features=peg_bond_features,
                        peg_batch=peg_batch,
                        molar_ratios=molar_ratios,
                        formulation_features=formulation_features,
                        ionizable_spatial_features=batch.get('ionizable_spatial_features').to(device) if 'ionizable_spatial_features' in batch else None,
                        helper_spatial_features=batch.get('helper_spatial_features').to(device) if 'helper_spatial_features' in batch else None,
                        cholesterol_spatial_features=batch.get('cholesterol_spatial_features').to(device) if 'cholesterol_spatial_features' in batch else None,
                        peg_spatial_features=batch.get('peg_spatial_features').to(device) if 'peg_spatial_features' in batch else None,
                        context_features=context_features,
                    )
                else:
                    # 使用 embeddings（可能是原始 embeddings、atom_features 或 fallback 的 fingerprints）
                    output = model(
                        embeddings=embeddings,
                        formulation_features=formulation_features,
                    )
                
                # 获取预测值（处理不同的输出格式）
                predictions = None
                if isinstance(output, dict):
                    if 'efficiency' in output:
                        # 直接输出 efficiency
                        eff = output['efficiency']
                        predictions = eff.cpu().numpy()
                        if predictions.ndim == 0:
                            predictions = predictions.reshape(1)
                        predictions = predictions.flatten()
                    elif 'predictions' in output:
                        pred = output['predictions']
                        if isinstance(pred, dict):
                            # predictions 是 dict，提取 efficiency
                            if 'efficiency' in pred:
                                eff = pred['efficiency']
                                predictions = eff.cpu().numpy()
                                if predictions.ndim == 0:
                                    predictions = predictions.reshape(1)
                                predictions = predictions.flatten()
                            else:
                                # 获取第一个可用的预测值
                                first_key = list(pred.keys())[0]
                                first_val = pred[first_key]
                                predictions = first_val.cpu().numpy()
                                if predictions.ndim == 0:
                                    predictions = predictions.reshape(1)
                                predictions = predictions.flatten()
                        else:
                            # predictions 是 tensor
                            predictions = pred.cpu().numpy()
                            if predictions.ndim == 0:
                                predictions = predictions.reshape(1)
                            predictions = predictions.flatten()
                    else:
                        # 尝试获取第一个值
                        first_val = list(output.values())[0]
                        predictions = first_val.cpu().numpy()
                        if predictions.ndim == 0:
                            predictions = predictions.reshape(1)
                        predictions = predictions.flatten()
                else:
                    # output 是 tensor
                    predictions = output.cpu().numpy()
                    if predictions.ndim == 0:
                        predictions = predictions.reshape(1)
                    predictions = predictions.flatten()
                
                targets_np = targets.cpu().numpy().flatten()
                efficiency_mask = batch.get('efficiency_mask')
                if isinstance(efficiency_mask, torch.Tensor):
                    valid_positions = np.flatnonzero(efficiency_mask.cpu().numpy().astype(bool).flatten())
                    predictions = predictions[valid_positions]
                    targets_np = targets_np[valid_positions]
                else:
                    valid_positions = np.arange(len(targets_np))
                
                # 验证形状是否匹配
                if len(predictions) != len(targets_np):
                    print(f"  ⚠️  警告：预测值长度 ({len(predictions)}) 与目标值长度 ({len(targets_np)}) 不匹配")
                    print(f"     这可能是因为模型输出形状不正确或数据加载问题")
                    # 使用较小的长度，避免索引越界
                    min_len = min(len(predictions), len(targets_np))
                    predictions = predictions[:min_len]
                    targets_np = targets_np[:min_len]
                
                # 获取 SMILES 和完整 LNP 配方信息
                smiles_list = batch.get('smiles', [None] * len(targets))
                
                # 获取完整 LNP 配方信息（从 dataset 的 df 中）
                subset = test_loader.dataset
                dataset = getattr(subset, 'dataset', subset)
                has_df = hasattr(dataset, 'df')
                
                # 收集候选分子
                for i in range(len(targets_np)):
                    batch_i = int(valid_positions[i])
                    # 从 batch 索引获取原始数据索引
                    sample_idx = int(batch['idx'][batch_i]) if 'idx' in batch else batch_idx * test_loader.batch_size + batch_i
                    
                    candidate = {
                        'smiles': smiles_list[batch_i] if batch_i < len(smiles_list) else None,
                        'predicted_efficiency': float(predictions[i]),
                        'actual_efficiency': float(targets_np[i]),
                        'score': float(predictions[i]),  # 默认使用效率作为得分
                    }
                    
                    # 添加完整 LNP 配方信息
                    if has_df and sample_idx < len(dataset.df):
                        row = dataset.df.iloc[sample_idx]
                        
                        # 可电离脂质 SMILES
                        if 'ionizable_lipid_smiles' in row and pd.notna(row['ionizable_lipid_smiles']):
                            candidate['ionizable_lipid_smiles'] = row['ionizable_lipid_smiles']
                        elif 'smiles' in row and pd.notna(row['smiles']):
                            candidate['ionizable_lipid_smiles'] = row['smiles']
                        else:
                            candidate['ionizable_lipid_smiles'] = ''
                        
                        # 辅助脂质（使用 helper_lipid_id）
                        helper_identity = row.get('helper_lipid_id', row.get('helper_lipid', 'DSPC'))
                        candidate['helper_lipid'] = str(helper_identity) if pd.notna(helper_identity) else 'DSPC'
                        
                        # 使用数据加载阶段已解析/回退后的真实结构，避免把
                        # Atlas 中不同 sterol/PEG 配方重新写成统一默认值。
                        candidate['helper_lipid_smiles'] = batch.get('helper_smiles', [''])[batch_i]
                        candidate['cholesterol_smiles'] = batch.get('cholesterol_smiles', [''])[batch_i]
                        candidate['peg_lipid_smiles'] = batch.get('peg_smiles', [''])[batch_i]
                        sterol_identity = row.get('sterol_lipid', 'cholesterol')
                        peg_identity = row.get('peg_lipid', '')
                        candidate['cholesterol'] = str(sterol_identity) if pd.notna(sterol_identity) else 'cholesterol'
                        candidate['peg_lipid'] = str(peg_identity) if pd.notna(peg_identity) else ''
                        
                        # 摩尔比
                        molar_ratios = {}
                        if 'cationic_lipid_mol_ratio' in row and pd.notna(row['cationic_lipid_mol_ratio']):
                            molar_ratios['ionizable'] = float(row['cationic_lipid_mol_ratio'])
                        if 'phospholipid_mol_ratio' in row and pd.notna(row['phospholipid_mol_ratio']):
                            molar_ratios['helper'] = float(row['phospholipid_mol_ratio'])
                        if 'cholesterol_mol_ratio' in row and pd.notna(row['cholesterol_mol_ratio']):
                            molar_ratios['cholesterol'] = float(row['cholesterol_mol_ratio'])
                        if 'peg_lipid_mol_ratio' in row and pd.notna(row['peg_lipid_mol_ratio']):
                            molar_ratios['peg'] = float(row['peg_lipid_mol_ratio'])
                        
                        # 如果数据集中没有摩尔比，使用默认值
                        if not molar_ratios:
                            molar_ratios = {
                                'ionizable': 50.0,
                                'helper': 10.0,
                                'cholesterol': 38.5,
                                'peg': 1.5
                            }
                        
                        candidate['molar_ratios'] = molar_ratios
                    else:
                        # 如果没有数据集信息，使用默认值
                        candidate['ionizable_lipid_smiles'] = candidate.get('smiles', '')
                        candidate['helper_lipid'] = 'DSPC'
                        candidate['cholesterol'] = 'cholesterol'
                        candidate['peg_lipid'] = 'DMG-PEG2000'
                        candidate['molar_ratios'] = {
                            'ionizable': 50.0,
                            'helper': 10.0,
                            'cholesterol': 38.5,
                            'peg': 1.5
                        }
                    
                    candidates.append(candidate)
                    
            except Exception as e:
                print(f"⚠️  Batch {batch_idx} 预测出错：{e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 按预测效率排序，选择 Top K
    candidates.sort(key=lambda x: x['score'], reverse=True)
    top_candidates = candidates[:top_k]
    
    if len(top_candidates) > 0:
        print(f"✅ 共评估 {len(candidates)} 个分子，选出 Top {top_k} 候选分子")
        print(f"   最佳预测效率：{top_candidates[0]['score']:.4f}")
        print(f"   最低预测效率：{top_candidates[-1]['score']:.4f}")
    else:
        print(f"⚠️  没有收集到任何候选分子，请检查数据加载和模型输出")
    
    return top_candidates
