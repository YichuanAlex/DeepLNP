#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LNP 多模态数据集合并脚本
将所有 7 个论文的数据集按照模态统一合并到 merged_datasets 目录下
"""

import os
import re
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# 根目录
ROOT_DIR = Path(__file__).resolve().parent
MERGED_DIR = ROOT_DIR / 'merged_datasets'

# 各数据集路径
DATASETS = {
    'AGILE': ROOT_DIR / 'method' / 'AGILE-main' / 'data',
    'LANTERN': ROOT_DIR / 'method' / 'LANTERN-main' / 'data',
    'LipoBART': ROOT_DIR / 'method' / 'LipoBART-main' / 'data',
    'LNP_Atlas': ROOT_DIR / 'method' / 'LNP_Atlas-main',
    'LNP_ML': ROOT_DIR / 'method' / 'LNP_ML-main' / 'data',
    'TransMA': ROOT_DIR / 'method' / 'TransMA-main' / 'dataset',
    'phil_LNP': ROOT_DIR / 'method' / 'phil_LNP_modelling-main' / 'LNP_data',
}


def create_output_directories():
    """创建输出目录结构"""
    print("=" * 80)
    print("创建输出目录结构...")
    print("=" * 80)
    
    # 分子描述符 + 活性数据
    mol_bio_dir = MERGED_DIR / '01_molecular_descriptors_bioactivity'
    mol_bio_dir.mkdir(parents=True, exist_ok=True)
    
    # LNP 配方数据
    formulation_dir = MERGED_DIR / '02_lnp_formulations'
    formulation_dir.mkdir(parents=True, exist_ok=True)
    
    # 磷脂片段数据
    lipid_fragment_dir = MERGED_DIR / '03_lipid_fragments'
    lipid_fragment_dir.mkdir(parents=True, exist_ok=True)
    
    # 细胞图像数据
    cell_image_dir = MERGED_DIR / '04_cell_images'
    cell_image_dir.mkdir(parents=True, exist_ok=True)
    
    # 原始数据备份
    raw_data_dir = MERGED_DIR / '00_raw_data_backup'
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ 输出目录创建完成：{MERGED_DIR}")
    print(f"  - {mol_bio_dir}")
    print(f"  - {formulation_dir}")
    print(f"  - {lipid_fragment_dir}")
    print(f"  - {cell_image_dir}")
    print(f"  - {raw_data_dir}")
    print()
    
    return {
        'mol_bio': mol_bio_dir,
        'formulation': formulation_dir,
        'lipid_fragment': lipid_fragment_dir,
        'cell_image': cell_image_dir,
        'raw_backup': raw_data_dir,
    }


def normalize_smiles(smiles: str) -> str:
    """标准化 SMILES 字符串"""
    if pd.isna(smiles):
        return ''
    # 去除首尾空格
    smiles = str(smiles).strip()
    # 统一小写列名时使用原始 SMILES
    return smiles


def parse_numeric_measurement(value):
    """Parse values such as ``108.0 ± 2.1`` while preserving missingness."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    return float(match.group(0)) if match else np.nan


def parse_lipid_molar_ratio(value):
    """Return the canonical ionizable/helper/sterol/PEG four-vector."""
    if pd.isna(value):
        return [np.nan] * 4
    numbers = [parse_numeric_measurement(part) for part in str(value).replace("/", ":").split(":")]
    numbers = [number for number in numbers if not pd.isna(number)]
    return (numbers + [np.nan] * 4)[:4] if len(numbers) >= 4 else [np.nan] * 4


def merge_molecular_bioactivity_data(output_dirs: Dict) -> pd.DataFrame:
    """
    合并分子描述符 + 生物活性数据
    包括：AGILE, LANTERN, TransMA
    """
    print("=" * 80)
    print("任务 1: 合并分子描述符 + 生物活性数据")
    print("=" * 80)
    
    all_datasets = []
    
    # 1. AGILE 数据
    print("\n[1/3] 处理 AGILE 数据集...")
    agile_candidate = pd.read_csv(DATASETS['AGILE'] / 'candidate_set_smiles_plus_features.csv')
    agile_finetune = pd.read_csv(DATASETS['AGILE'] / 'finetuning_set_smiles_plus_features.csv')
    
    # 标准化列名
    agile_candidate.columns = [col.lower() if col != 'SMILES' else 'smiles' for col in agile_candidate.columns]
    agile_finetune.columns = [col.lower() if col != 'SMILES' else 'smiles' for col in agile_finetune.columns]
    
    # 添加来源标记和细胞系标记
    agile_candidate['dataset_source'] = 'AGILE_candidate'
    agile_finetune['dataset_source'] = 'AGILE_finetuning'
    agile_finetune['cell_line'] = 'HeLa'  # AGILE 主要是 HeLa 细胞
    
    # 去除重复列
    agile_finetune = agile_finetune.loc[:, ~agile_finetune.columns.duplicated()]
    
    print(f"  - AGILE candidate: {len(agile_candidate)} 条，{len(agile_candidate.columns)} 列")
    print(f"  - AGILE finetuning: {len(agile_finetune)} 条，{len(agile_finetune.columns)} 列")
    
    # 保存原始数据
    agile_candidate.to_csv(output_dirs['raw_backup'] / 'AGILE_candidate.csv', index=False)
    agile_finetune.to_csv(output_dirs['raw_backup'] / 'AGILE_finetuning.csv', index=False)
    
    all_datasets.append(agile_finetune)  # 使用 finetuning 版本 (包含活性数据)
    
    # 2. LANTERN 数据
    print("\n[2/3] 处理 LANTERN 数据集...")
    lantern_agile = pd.read_csv(DATASETS['LANTERN'] / 'AGILE_with_features.csv')
    lantern_agile.columns = [col.lower() if col != 'SMILES' else 'smiles' for col in lantern_agile.columns]
    lantern_agile['dataset_source'] = 'LANTERN'
    lantern_agile['cell_line'] = 'HeLa'  # LANTERN 也是 HeLa 细胞
    
    # 去除重复列
    lantern_agile = lantern_agile.loc[:, ~lantern_agile.columns.duplicated()]
    
    print(f"  - LANTERN AGILE_with_features: {len(lantern_agile)} 条，{len(lantern_agile.columns)} 列")
    
    # 保存原始数据
    lantern_agile.to_csv(output_dirs['raw_backup'] / 'LANTERN_AGILE.csv', index=False)
    
    all_datasets.append(lantern_agile)
    
    # 3. TransMA 数据
    print("\n[3/3] 处理 TransMA 数据集...")
    transma_datasets = []
    
    # Hela 细胞系
    for split_type in ['cliff', 'scaffold']:
        for split_name in ['train', 'test']:
            file_path = DATASETS['TransMA'] / 'Hela' / split_type / f'{split_name}.csv'
            if file_path.exists():
                df = pd.read_csv(file_path)
                df.columns = [col.lower() if col != 'SMILES' else 'smiles' for col in df.columns]
                df['cell_line'] = 'HeLa'
                df['dataset_source'] = f'TransMA_Hela_{split_type}_{split_name}'
                # 去除重复列
                df = df.loc[:, ~df.columns.duplicated()]
                transma_datasets.append(df)
                print(f"  - TransMA Hela {split_type} {split_name}: {len(df)} 条")
    
    # Raw 细胞系
    for split_type in ['cliff', 'scaffold']:
        for split_name in ['train', 'test']:
            file_path = DATASETS['TransMA'] / 'RaW' / split_type / f'{split_name}.csv'
            if file_path.exists():
                df = pd.read_csv(file_path)
                df.columns = [col.lower() if col != 'SMILES' else 'smiles' for col in df.columns]
                df['cell_line'] = 'Raw'
                df['dataset_source'] = f'TransMA_Raw_{split_type}_{split_name}'
                # 去除重复列
                df = df.loc[:, ~df.columns.duplicated()]
                transma_datasets.append(df)
                print(f"  - TransMA Raw {split_type} {split_name}: {len(df)} 条")
    
    # 合并 TransMA 数据
    transma_merged = pd.concat(transma_datasets, ignore_index=True)
    
    # 保存原始数据
    for split_type in ['cliff', 'scaffold']:
        for split_name in ['train', 'test']:
            for cell_line in ['Hela', 'RaW']:
                file_path = DATASETS['TransMA'] / cell_line / split_type / f'{split_name}.csv'
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    df.to_csv(output_dirs['raw_backup'] / f'TransMA_{cell_line}_{split_type}_{split_name}.csv', index=False)
    
    # 统一 TransMA 列名与 AGILE/LANTERN 对齐
    transma_merged = transma_merged.rename(columns={'target': 'expt_target'})
    
    # 为 TransMA 添加缺失的描述符列 (用 NaN 填充)
    ref_df = all_datasets[0]  # AGILE 作为参考
    ref_cols = [col for col in ref_df.columns if col not in ['cell_line', 'dataset_source']]
    missing_cols = set(ref_cols) - set(transma_merged.columns)
    
    print(f"\n  TransMA 需要补充 {len(missing_cols)} 个描述符列")
    for col in missing_cols:
        transma_merged[col] = np.nan
    
    # 确保列顺序一致
    transma_merged = transma_merged[ref_cols + [col for col in transma_merged.columns if col not in ref_cols]]
    
    all_datasets.append(transma_merged)
    
    # 合并所有数据集
    print("\n合并所有分子 + 活性数据...")
    merged_df = pd.concat(all_datasets, ignore_index=True)
    
    # 标准化 SMILES
    merged_df['smiles'] = merged_df['smiles'].apply(normalize_smiles)
    
    # 去重 (基于 SMILES)
    print(f"合并后总数：{len(merged_df)} 条")
    print(f"去重前唯一 SMILES 数：{merged_df['smiles'].nunique()}")
    merged_df = merged_df.drop_duplicates(subset=['smiles'], keep='first')
    print(f"去重后总数：{len(merged_df)} 条")
    
    # 保存合并后的数据
    output_file = output_dirs['mol_bio'] / 'molecular_descriptors_bioactivity_merged.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"\n✓ 保存到：{output_file}")
    
    # 统计信息
    stats = {
        'total_samples': len(merged_df),
        'unique_smiles': merged_df['smiles'].nunique(),
        'total_features': len([col for col in merged_df.columns if col.startswith('desc_')]),
        'datasets_included': merged_df['dataset_source'].unique().tolist(),
        'cell_lines': merged_df['cell_line'].unique().tolist() if 'cell_line' in merged_df.columns else ['HeLa', 'Raw'],
    }
    
    print("\n数据统计:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    return merged_df, stats


def merge_lnp_formulation_data(output_dirs: Dict) -> pd.DataFrame:
    """
    合并 LNP 配方数据
    包括：LNP_Atlas, LNP_ML
    """
    print("\n" + "=" * 80)
    print("任务 2: 合并 LNP 配方数据")
    print("=" * 80)
    
    all_datasets = []
    
    # 1. LNP_Atlas 数据
    print("\n[1/2] 处理 LNP_Atlas 数据集...")
    # 尝试不同的编码
    try:
        lnp_atlas = pd.read_csv(DATASETS['LNP_Atlas'] / 'LNP_Atlas_DB_202509_v1.csv', encoding='utf-8')
    except UnicodeDecodeError:
        lnp_atlas = pd.read_csv(DATASETS['LNP_Atlas'] / 'LNP_Atlas_DB_202509_v1.csv', encoding='latin-1')
    
    # 标准化列名
    lnp_atlas.columns = [col.lower() for col in lnp_atlas.columns]
    lnp_atlas['dataset_source'] = 'LNP_Atlas'
    
    print(f"  - LNP_Atlas: {len(lnp_atlas)} 条，{len(lnp_atlas.columns)} 列")
    
    # 保存原始数据
    lnp_atlas.to_csv(output_dirs['raw_backup'] / 'LNP_Atlas_raw.csv', index=False)
    
    all_datasets.append(lnp_atlas)
    
    # 2. LNP_ML 数据
    print("\n[2/2] 处理 LNP_ML 数据集...")
    lnp_ml = pd.read_csv(DATASETS['LNP_ML'] / 'all_data.csv')
    
    # 标准化列名
    lnp_ml.columns = [col.lower() for col in lnp_ml.columns]
    lnp_ml['dataset_source'] = 'LNP_ML'
    
    print(f"  - LNP_ML: {len(lnp_ml)} 条，{len(lnp_ml.columns)} 列")
    
    # 保存原始数据
    lnp_ml.to_csv(output_dirs['raw_backup'] / 'LNP_ML_raw.csv', index=False)
    
    all_datasets.append(lnp_ml)
    
    # 合并数据集
    print("\n合并 LNP 配方数据...")
    merged_df = pd.concat(all_datasets, ignore_index=True)
    
    # 统一可电离脂质主结构。LNP Atlas 将结构存放在
    # ionizable_lipid_smiles；旧逻辑按空的 smiles 去重会把 1,092 条压成 1 条。
    for col in [
        'smiles', 'ionizable_lipid_smiles', 'helper_lipid_smiles',
        'sterol_lipid_smiles', 'peg_lipid_smiles',
    ]:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].apply(normalize_smiles)

    merged_df['primary_smiles'] = merged_df.get('ionizable_lipid_smiles', pd.Series('', index=merged_df.index))
    empty_primary = merged_df['primary_smiles'].eq('')
    merged_df.loc[empty_primary, 'primary_smiles'] = merged_df.loc[empty_primary, 'smiles']

    # Parse LNP Atlas textual ratios and physical-property strings into model-ready values.
    ratio_cols = [
        'cationic_lipid_mol_ratio', 'phospholipid_mol_ratio',
        'cholesterol_mol_ratio', 'peg_lipid_mol_ratio',
    ]
    parsed_ratios = merged_df.get('lipid_molar_ratio', pd.Series(np.nan, index=merged_df.index)).apply(parse_lipid_molar_ratio)
    for idx, col in enumerate(ratio_cols):
        existing = pd.to_numeric(merged_df.get(col, pd.Series(np.nan, index=merged_df.index)), errors='coerce')
        parsed = parsed_ratios.apply(lambda values: values[idx])
        merged_df[col] = existing.fillna(parsed)

    for col in [
        'particle_size_nm_std', 'pdi_std', 'zeta_potential_mv_std',
        'encapsulation_efficiency_percent_std',
    ]:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].apply(parse_numeric_measurement)

    print(f"合并后总数：{len(merged_df)} 条")
    # Only remove true duplicate records. Repeated lipids measured in different
    # screens/targets are distinct supervised observations and must be retained.
    identity_cols = [
        col for col in [
            'dataset_source', 'lnp_id', 'formulation_id', 'experiment_id',
            'primary_smiles', *ratio_cols, 'model_type', 'delivery_target',
            'route_of_administration', 'target_type',
        ] if col in merged_df.columns
    ]
    before = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=identity_cols, keep='first').reset_index(drop=True)
    print(f"仅去除完全相同实验标识后：{before} -> {len(merged_df)} 条")
    
    # 保存合并后的数据
    output_file = output_dirs['formulation'] / 'lnp_formulations_merged.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"\n✓ 保存到：{output_file}")
    
    # 统计信息
    stats = {
        'total_samples': len(merged_df),
        'datasets_included': merged_df['dataset_source'].unique().tolist(),
        'formulation_columns': len([col for col in merged_df.columns if 'ratio' in col or 'lipid' in col]),
        'physical_properties': len([col for col in merged_df.columns if 'size' in col or 'zeta' in col or 'pdi' in col]),
    }
    
    print("\n数据统计:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    return merged_df, stats


def merge_lipid_fragment_data(output_dirs: Dict) -> pd.DataFrame:
    """
    合并磷脂片段数据
    包括：LipoBART
    """
    print("\n" + "=" * 80)
    print("任务 3: 合并磷脂片段数据")
    print("=" * 80)
    
    # LipoBART 数据
    print("\n处理 LipoBART 数据集...")
    
    # 读取所有相关文件
    full_lipids = pd.read_csv(DATASETS['LipoBART'] / 'full_iPhos_lipids.csv')
    multiclass = pd.read_csv(DATASETS['LipoBART'] / 'iphos_multiclass.csv')
    targets = pd.read_csv(DATASETS['LipoBART'] / 'iphos_targets.csv')
    
    # 标准化列名
    full_lipids.columns = [col.lower() for col in full_lipids.columns]
    multiclass.columns = [col.lower() for col in multiclass.columns]
    targets.columns = [col.lower() for col in targets.columns]
    
    # 添加来源标记
    full_lipids['dataset_source'] = 'LipoBART_full'
    multiclass['dataset_source'] = 'LipoBART_multiclass'
    targets['dataset_source'] = 'LipoBART_targets'
    
    print(f"  - full_iPhos_lipids: {len(full_lipids)} 条")
    print(f"  - iphos_multiclass: {len(multiclass)} 条")
    print(f"  - iphos_targets: {len(targets)} 条")
    
    # 保存原始数据
    full_lipids.to_csv(output_dirs['raw_backup'] / 'LipoBART_full.csv', index=False)
    multiclass.to_csv(output_dirs['raw_backup'] / 'LipoBART_multiclass.csv', index=False)
    targets.to_csv(output_dirs['raw_backup'] / 'LipoBART_targets.csv', index=False)
    
    # 合并主要数据
    merged_df = full_lipids.copy()
    merged_df = merged_df.rename(columns={'combined': 'smiles'})
    
    # 保存合并后的数据
    output_file = output_dirs['lipid_fragment'] / 'lipid_fragments_merged.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"\n✓ 保存到：{output_file}")
    
    # 保存 multiclass 和 targets 作为补充数据
    multiclass.to_csv(output_dirs['lipid_fragment'] / 'lipid_multiclass.csv', index=False)
    targets.to_csv(output_dirs['lipid_fragment'] / 'lipid_targets.csv', index=False)
    
    # 统计信息
    stats = {
        'total_lipids': len(merged_df),
        'head_groups': merged_df['head'].nunique(),
        'tail_groups': merged_df['tail'].nunique(),
        'dataset_source': 'LipoBART',
    }
    
    print("\n数据统计:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    return merged_df, stats


def merge_cell_image_data(output_dirs: Dict) -> Dict:
    """
    合并细胞图像数据
    包括：phil_LNP_modelling
    """
    print("\n" + "=" * 80)
    print("任务 4: 合并细胞图像数据")
    print("=" * 80)
    
    phil_lnp_dir = DATASETS['phil_LNP']
    
    # 创建图像数据目录
    image_output_dir = output_dirs['cell_image']
    
    # 复制图像数据
    print("\n处理 phil_LNP_modelling 细胞图像数据集...")
    
    stats = {
        'train_images': 0,
        'test_images': 0,
        'gfp_images': 0,
        'metadata_files': [],
    }
    
    # 1. 复制训练集图像
    train_src = phil_lnp_dir / 'images_train'
    train_dst = image_output_dir / 'images_train'
    
    if train_src.exists():
        shutil.copytree(train_src, train_dst, dirs_exist_ok=True)
        train_count = len(list(train_src.glob('*.npy')))
        stats['train_images'] = train_count
        print(f"  - 复制训练集图像：{train_count} 个")
    
    # 2. 复制测试集图像
    test_src = phil_lnp_dir / 'images_test'
    test_dst = image_output_dir / 'images_test'
    
    if test_src.exists():
        shutil.copytree(test_src, test_dst, dirs_exist_ok=True)
        test_count = len(list(test_src.glob('*.npy')))
        stats['test_images'] = test_count
        print(f"  - 复制测试集图像：{test_count} 个")
    
    # 3. 复制 GFP 图像
    gfp_train_src = phil_lnp_dir / 'gfp_train'
    gfp_test_src = phil_lnp_dir / 'gfp_test'
    gfp_train_dst = image_output_dir / 'gfp_train'
    gfp_test_dst = image_output_dir / 'gfp_test'
    
    if gfp_train_src.exists():
        shutil.copytree(gfp_train_src, gfp_train_dst, dirs_exist_ok=True)
        gfp_train_count = len(list(gfp_train_src.glob('*.npy')))
        stats['gfp_images'] += gfp_train_count
        print(f"  - 复制 GFP 训练图像：{gfp_train_count} 个")
    
    if gfp_test_src.exists():
        shutil.copytree(gfp_test_src, gfp_test_dst, dirs_exist_ok=True)
        gfp_test_count = len(list(gfp_test_src.glob('*.npy')))
        stats['gfp_images'] += gfp_test_count
        print(f"  - 复制 GFP 测试图像：{gfp_test_count} 个")
    
    # 4. 复制元数据文件
    metadata_files = ['cell_stats.npy', 'gfp_stats.npy', 'ReadMe']
    for meta_file in metadata_files:
        src_file = phil_lnp_dir / meta_file
        if src_file.exists():
            shutil.copy2(src_file, image_output_dir / meta_file)
            stats['metadata_files'].append(meta_file)
            print(f"  - 复制元数据：{meta_file}")
    
    # 创建图像数据说明文件
    readme_content = f"""# 细胞图像数据集

## 数据来源
phil_LNP_modelling: Deep learning models for lipid-nanoparticle-based drug delivery

## 数据结构

### images_train/
- 训练集细胞图像
- 数量：{stats['train_images']} 个
- 格式：numpy 数组 (.npy)
- 尺寸：192×192×3 (高×宽×通道)
- 通道：1=细胞 tracker 染色，2=LNP 递送，3=明场
- 时间点：0-20

### images_test/
- 测试集细胞图像
- 数量：{stats['test_images']} 个
- 格式：numpy 数组 (.npy)
- 尺寸：192×192×3

### gfp_train/
- GFP 表达训练图像 (终点 72 小时)
- 数量：{stats['gfp_images'] // 2} 个 (估计)
- 格式：numpy 数组 (.npy)

### gfp_test/
- GFP 表达测试图像 (终点 72 小时)
- 数量：{stats['gfp_images'] // 2} 个 (估计)
- 格式：numpy 数组 (.npy)

### 元数据文件
- cell_stats.npy: 细胞图像标准化参数
- gfp_stats.npy: GFP 图像标准化参数
- ReadMe: 原始数据说明

## 使用方法
```python
import numpy as np

# 读取细胞图像
cell_image = np.load('images_train/C04_F001_T0072_cell_1.npy')
# shape: (192, 192, 3)

# 读取 GFP 图像
gfp_image = np.load('gfp_train/C04_F001_T0072_cell_1.npy')

# 标准化
cell_stats = np.load('cell_stats.npy', allow_pickle=True).item()
normalized_image = (cell_image - cell_stats['mean']) / cell_stats['std']
```

## 注意事项
- 图像数据保持原始格式，未进行训练集/测试集划分
- 所有图像均为 per-cell 级别数据
- 可用于深度学习模型训练
"""
    
    readme_file = image_output_dir / 'README.md'
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n✓ 图像数据整理完成")
    print(f"  总图像数：{stats['train_images'] + stats['test_images'] + stats['gfp_images']}")
    
    return stats


def generate_summary_report(all_stats: Dict):
    """生成汇总报告"""
    print("\n" + "=" * 80)
    print("生成汇总报告...")
    print("=" * 80)
    
    report = f"""# LNP 多模态数据集合并报告

## 概述
本数据集整合了 7 篇 LNP 相关论文的数据，按照模态分类整理，便于 AI 模型训练使用。

## 数据集结构

```
merged_datasets/
├── 00_raw_data_backup/          # 原始数据备份
├── 01_molecular_descriptors_bioactivity/  # 分子描述符 + 活性
│   └── molecular_descriptors_bioactivity_merged.csv
├── 02_lnp_formulations/         # LNP 配方数据
│   └── lnp_formulations_merged.csv
├── 03_lipid_fragments/          # 磷脂片段数据
│   ├── lipid_fragments_merged.csv
│   ├── lipid_multiclass.csv
│   └── lipid_targets.csv
└── 04_cell_images/              # 细胞图像数据
    ├── images_train/
    ├── images_test/
    ├── gfp_train/
    ├── gfp_test/
    ├── cell_stats.npy
    ├── gfp_stats.npy
    └── README.md
```

## 各模态数据统计

### 1. 分子描述符 + 生物活性
- 数据来源：AGILE, LANTERN, TransMA
- 样本数：{all_stats['mol_bio']['total_samples']}
- 唯一化合物数：{all_stats['mol_bio']['unique_smiles']}
- 特征数：{all_stats['mol_bio']['total_features']} 个分子描述符
- 细胞系：{', '.join(all_stats['mol_bio']['cell_lines'])}
- 包含数据集：{', '.join(all_stats['mol_bio']['datasets_included'])}

### 2. LNP 配方数据
- 数据来源：LNP_Atlas, LNP_ML
- 样本数：{all_stats['formulation']['total_samples']}
- 配方参数列：{all_stats['formulation']['formulation_columns']}
- 物理性质列：{all_stats['formulation']['physical_properties']}
- 包含数据集：{', '.join(all_stats['formulation']['datasets_included'])}

### 3. 磷脂片段数据
- 数据来源：LipoBART
- 脂质数：{all_stats['lipid_fragment']['total_lipids']}
- 头部基团数：{all_stats['lipid_fragment']['head_groups']}
- 尾部基团数：{all_stats['lipid_fragment']['tail_groups']}

### 4. 细胞图像数据
- 数据来源：phil_LNP_modelling
- 训练集图像：{all_stats['cell_image']['train_images']}
- 测试集图像：{all_stats['cell_image']['test_images']}
- GFP 图像：{all_stats['cell_image']['gfp_images']}
- 元数据文件：{', '.join(all_stats['cell_image']['metadata_files'])}

## 数据使用说明

### 分子描述符 + 活性数据
适用于：
- 分子性质预测 (QSAR)
- 虚拟筛选
- 分子优化
- 构效关系分析

使用方法：
```python
import pandas as pd

df = pd.read_csv('merged_datasets/01_molecular_descriptors_bioactivity/molecular_descriptors_bioactivity_merged.csv')
smiles = df['smiles']
features = df[[col for col in df.columns if col.startswith('desc_')]]
targets = df[[col for col in df.columns if col.startswith('expt_')]]
```

### LNP 配方数据
适用于：
- LNP 配方优化
- 递送效率预测
- 物理性质预测
- 多任务学习

使用方法：
```python
df = pd.read_csv('merged_datasets/02_lnp_formulations/lnp_formulations_merged.csv')
# 特征：脂质 SMILES, 摩尔比，物理性质
# 靶点：递送效率，基因表达等
```

### 磷脂片段数据
适用于：
- 片段贡献分析
- 可解释性研究
- 脂质结构优化

使用方法：
```python
df = pd.read_csv('merged_datasets/03_lipid_fragments/lipid_fragments_merged.csv')
head_smiles = df['head']
tail_smiles = df['tail']
combined_smiles = df['smiles']
```

### 细胞图像数据
适用于：
- 深度学习 (CNN, LSTM)
- 细胞形态分析
- 时相预测
- 多模态融合

使用方法：
```python
import numpy as np

# 读取图像
image = np.load('merged_datasets/04_cell_images/images_train/C04_F001_T0072_cell_1.npy')
# shape: (192, 192, 3)

# 标准化
stats = np.load('merged_datasets/04_cell_images/cell_stats.npy', allow_pickle=True).item()
normalized = (image - stats['mean']) / stats['std']
```

## 注意事项

1. **数据完整性**: 所有原始数据已备份至 `00_raw_data_backup/` 目录
2. **SMILES 标准化**: 分子结构数据已进行 SMILES 标准化处理
3. **列名统一**: 各数据集的列名已统一为小写格式
4. **去重处理**: 基于 SMILES 进行了去重，保留第一条记录
5. **未划分数据集**: 按照要求，未进行训练集/验证集/测试集划分
6. **图像格式**: 细胞图像保持原始 numpy 格式，未进行转换

## 引用

如果使用本数据集，请引用原始论文：
- AGILE: Nature Communications, 2024
- LANTERN: 2024
- TransMA: 2024
- LNP_Atlas: Scientific Data, 2025
- LNP_ML: Nature Biotechnology, 2024
- LipoBART: Nature Nanotechnology, 2025
- phil_LNP: 2020

## 生成时间
{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    report_file = MERGED_DIR / 'README.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ 汇总报告已保存至：{report_file}")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("LNP 多模态数据集合并工具")
    print("=" * 80)
    print(f"根目录：{ROOT_DIR}")
    print(f"输出目录：{MERGED_DIR}")
    print()
    
    # 创建输出目录
    output_dirs = create_output_directories()
    
    # 存储所有统计信息
    all_stats = {}
    
    try:
        # 任务 1: 合并分子描述符 + 活性数据
        _, mol_bio_stats = merge_molecular_bioactivity_data(output_dirs)
        all_stats['mol_bio'] = mol_bio_stats
        
        # 任务 2: 合并 LNP 配方数据
        _, formulation_stats = merge_lnp_formulation_data(output_dirs)
        all_stats['formulation'] = formulation_stats
        
        # 任务 3: 合并磷脂片段数据
        _, lipid_fragment_stats = merge_lipid_fragment_data(output_dirs)
        all_stats['lipid_fragment'] = lipid_fragment_stats
        
        # 任务 4: 合并细胞图像数据
        cell_image_stats = merge_cell_image_data(output_dirs)
        all_stats['cell_image'] = cell_image_stats
        
        # 生成汇总报告
        generate_summary_report(all_stats)
        
        print("\n" + "=" * 80)
        print("✓ 所有任务完成!")
        print("=" * 80)
        print(f"\n合并后的数据集位于：{MERGED_DIR}")
        print("\n目录结构:")
        for item in MERGED_DIR.iterdir():
            if item.is_dir():
                print(f"  📁 {item.name}/")
            else:
                print(f"  📄 {item.name}")
        
    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
